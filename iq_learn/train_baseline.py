import datetime
import os
import random
import time
import types
from collections import deque
from itertools import count

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from agent import make_agent
from dataset.memory import Memory
from make_envs import make_env
from utils.logger import Logger
from utils.utils import (average_dicts, eval_mode, evaluate,
                         get_concat_samples, hard_update, soft_update)
from wrappers.atari_wrapper import LazyFrames
from utils.utils import EarlyStopper
from typing import IO, Any, Dict
import pickle
from bc import BehaviorCloningLossCalculator
from encoder.utils import cheetah_full_loader
from torch.optim import Adam
import sys

# clustering report
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
from scipy.optimize import linear_sum_assignment
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(2)

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    cfg.env_name = cfg.env_name
    cfg.agent_name = cfg.agent_name
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    if args.wandb:
        if args.cond_type!="none":
            if args.experimental != "none":
                exp_name = args.env.cond
                wandb.init(
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=f"baseline {args.env.short_name} experimental {args.experimental}"
                )
            elif args.method.bc_init:
                exp_name = args.env.cond
                wandb.init(
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=f"baseline {args.env.short_name} bc_init{args.method.bc_init} level{args.num_levels}"
                )
            else:
                level = "" if args.num_levels < 3 else f"level{args.num_levels} "
                exp_name = f"baseline {args.env.short_name} bc_init{args.method.bc_init} {level}actor{args.agent.actor_lr} critic{args.agent.critic_lr}"
                wandb.init(
                    project="hil_iq", 
                    sync_tensorboard=True, 
                    reinit=True, 
                    config=args, 
                    name=exp_name
                )
        else: 
            exp_name = args.env.demo
            wandb.init(
                project="hil_iq", 
                sync_tensorboard=True, 
                reinit=True, 
                config=args, 
                name=f"{args.env.name} iq-learn only"
            )
    
    if (not args.method.bc_init):
        print(f'[Critic]: use {args.additional_loss}*{args.cql_coef} ({args.num_random} randoms) loss')

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    env.seed(args.seed) 
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent = make_agent(env, args)
    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    conds = None
    cond_location = hydra.utils.to_absolute_path(f'cond/{args.env.cond}')
    if os.path.isfile(cond_location):
        # Load data from single file.
        with open(cond_location, 'rb') as f:
            conds = read_file(cond_location, f)
    else:
        raise ValueError(f"Condition file {cond_location} not found")
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              cond_dim=args.cond_dim,
                              cond_type=args.cond_type,
                              conds=conds)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    # Load trajs as encoder input
    expert_location = hydra.utils.to_absolute_path(f'experts/{args.env.demo}')
    if os.path.isfile(expert_location):
        with open(expert_location, 'rb') as f:
            experts = read_file(expert_location, f)
    else:
        raise ValueError(f"Expert file {expert_location} not found")
    

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir,
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=args.agent.name)

    steps = 0

    # track mean reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    # Encoder Init
    # print work dir
    # change work dir to encoder dir
    # os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    # if sac
    if args.agent.name == "sac":
        agent.loss_calculator = BehaviorCloningLossCalculator(
                ent_weight=1e-3,  # args.method.bc_ent_weight,
                l2_weight=0.0,  # args.method.bc_l2_weight,
                kld_weight=args.method.kld_alpha,  # args.method.bc_kld_weight,
                agent_name=args.agent.name
            )
    else: 
        # TODO: add bc loss calculator for softq
        agent.loss_calculator = BehaviorCloningLossCalculator(
                ent_weight=1e-3,  # args.method.bc_ent_weight,
                l2_weight=0.0,  # args.method.bc_l2_weight,
                kld_weight=args.method.kld_alpha,  # args.method.bc_kld_weight,
                agent_name=args.agent.name
            )
    agent.bc_alpha = args.method.bc_alpha

    exp_dir = args.exp_dir
    os.makedirs(exp_dir, exist_ok=True)

    # BC initialization
    if args.method.bc_init:
        
        # sys.path.append('/home/zichang/proj/IQ-Learn/iq_learn/encoder')
        print("Current working directory: ", os.getcwd())
        # Save the current working directory
        previous_dir = os.getcwd()

        # Change to the desired directory
        target_dir = '/home/zichang/proj/IQ-Learn/iq_learn/encoder'
        os.chdir(target_dir)
        print("Current working directory: ", os.getcwd())
        sys.path.append(target_dir)
        from encoder.baseline_model import BaselineModel
        # Append the target directory to sys.path if needed
        

        encoder = BaselineModel(
            dist_size=args.cond_dim,        # provide the dist_size argument
            seq_len=1000,     # default seq_len value or customize
            state_space=env.observation_space.shape[0],  # state space dimension
            action_space=env.action_space.shape[0],      # action space dimension
            key_hidden_size=256,             # hidden size for key projection
            value_hidden_size=256            # hidden size for value projection
        )

        encoder.train()
        encoder.to(device)

        # Change back to the previous directory
        os.chdir(previous_dir)
        print("Reverted back to previous directory: ", os.getcwd())
        last_layers_to_unfreeze = ['linear1', 'linear2', 'linear3', 'm_feat', 'transformer', 'compact_last', 'mu_layer','logvar_layer']

        # Freeze all parameters first
        for param in encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze specific last layers by their names
        for name, param in encoder.named_parameters():
            if any(last_layer_name in name for last_layer_name in last_layers_to_unfreeze):
                param.requires_grad = True

        # Print all layers and sub-layers
        # print("\nModel layers using model.modules():")
        # for layer in encoder.modules():
        #     print(layer)


        encoder_optimizer = Adam(params=encoder.parameters(), lr=args.en_lr, amsgrad=True)
        print(f"Baseline encoder initialized, optimizer ready")
        
    # BC initialization
    if args.method.bc_init:
        agent.bc_update = types.MethodType(bc_update, agent)
        first_mu_0 = 0
        early_stopper = EarlyStopper(patience=3, min_delta=0)
        for learn_steps_bc in count():
            # print(f"BC step: {learn_steps_bc}")
            expert_batch = expert_memory_replay.get_samples(
                agent.batch_size, agent.device
            )
            expert_obs, _, expert_action, __, ___, expert_cond_detached, true_traj_idx = expert_batch
            emb_list = get_new_cond(
                encoder, 
                experts, 
                device,
                true_traj_idx)
            unique = []
            dist_params_list = []
            for i in true_traj_idx:
                if i not in unique:
                    unique.append(i)
                    dist_params_list.append(emb_list["dist_params"][i])
            mu = [dist_params[0] for dist_params in dist_params_list]     
            log_var = [dist_params[1] for dist_params in dist_params_list]
            mu = torch.stack(mu, dim=0)
            log_var = torch.stack(log_var, dim=0)
           
            expert_cond_list = emb_list["emb"]
            expert_cond = torch.stack(expert_cond_list, dim=0)
            
            epsilon = 1e-6  # Small value to avoid numerical issues
            if args.agent.name=="sac" and (expert_action.min() < -1.0 or expert_action.max() > 1.0):
                print(f"Action out of bound: {expert_action.min()}, {expert_action.max()}") 
                print("Force normalizing expert action to [-1, 1]")
                expert_action = torch.clamp(expert_action, min=-1.0 + epsilon, max=1.0 - epsilon)
                print(expert_action.min(), expert_action.max())
            clip_value = 5.0
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_value)
            losses = agent.bc_update(
                expert_obs,
                expert_action,
                expert_cond,
                logger,
                learn_steps_bc,
                args.cond_type,
                mu,
                log_var,
                encoder_optimizer
            )

            # for name, param in encoder.named_parameters():
            #     if any(last_layer_name in name for last_layer_name in last_layers_to_unfreeze):
            #         print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")

            # log losses
            
            if learn_steps_bc % 10 == 0:  # args.log_interval == 0:
                # test: log the l2 norm between the latest latent mean (traj_id=0) and the first latent mean every 10 steps
                states = torch.tensor(experts["states"][0],dtype=torch.float32).to(device)
                actions = torch.tensor(experts["actions"][0],dtype=torch.float32).to(device)
                mu_0, _ = encoder(states, actions)
                mu_0 = mu_0.detach().cpu().numpy()
                l2_norm = np.linalg.norm(mu_0 - first_mu_0)
                
                # print(f"Step: {learn_steps_bc}, L2 Norm: {l2_norm.item()}")
                losses["mean_l2_norm"] = l2_norm.item()
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_steps_bc)
                
                logger.dump(learn_steps_bc)
                # print (f"Step: {learn_steps_bc}, L2 Norm: {l2_norm.item()}")
                # TODO: early stopping
                # if early_stopper.early_stop(-l2_norm.item()):  
                #     unique_encoder_file = f"prob-encoder_dim{args.cond_dim}_kld_alpha{args.method.kld_alpha}_betaB_step_{learn_steps_bc}.ckpt"
                #     save_dir = os.path.join(exp_dir, unique_encoder_file)
                #     torch.save(encoder, save_dir)
                #     print(f"Encoder saved at {save_dir}") 
                #     print("Early stopping at step: ", learn_steps_bc)          
                #     break

            # eval every n steps
            if learn_steps_bc % args.env.eval_interval == 0:
                eval_num = 0 
                for eval_index in range(eval_num):
                    # low ability level
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental="none")
                    returns = np.mean(eval_returns)
                    logger.log(f'eval/episode_reward_low{eval_index}', returns, learn_steps)
                    # high ability level
                    high_index = eval_index + args.expert.demos//2
                    eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=high_index, experimental="none")
                    returns = np.mean(eval_returns)
                    logger.log(f'eval/episode_reward_high{high_index}', returns, learn_steps)
                # logger.log('eval/bc_episode_reward', returns, learn_steps_bc)
                logger.dump(learn_steps_bc, ty="eval")
                # refresh expert memory with new cond and new dist_params
                emb_list = update_expert_memory(
                    encoder, 
                    experts, 
                    device)
                expert_memory_replay.clear()
                expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              cond_dim=args.cond_dim,
                              cond_type=args.cond_type,
                              conds=emb_list)
                print(f'--> New expert memory size: {expert_memory_replay.size()}')
                logdir = str(hydra.utils.to_absolute_path(f'baseline/plot/{args.env.short_name}'))
                logname = os.path.join(logdir, f"stage1_b{learn_steps_bc}.log")
                os.makedirs(os.path.dirname(logname), exist_ok=True)
                cluster_mse = clustering_report(emb_list, logname, 3)  # Adjust n_features as needed
                logger.log('eval/cluster_mse', cluster_mse, learn_steps_bc)
                logger.dump(learn_steps_bc, ty="eval")
            
                if cluster_mse<=0.0001:
                    exit_save(encoder, exp_dir, experts, device, learn_steps_bc, args)
                    print(f"Training has converged with cluster_mse {cluster_mse} Exiting...")
                    wandb.finish()
                    sys.exit(0)
            # save the encoder every 500 steps
            # if (learn_steps_bc % args.bc_save_interval == 0 or learn_steps_bc==10) and learn_steps_bc > 0:
            #     unique_encoder_file = f"prob-encoder_dim{args.cond_dim}_kld_alpha{args.method.kld_alpha}_betaB_step_{learn_steps_bc}.ckpt"
            #     save_dir = os.path.join(exp_dir, unique_encoder_file)
            #     torch.save(encoder, save_dir)
            #     print(f"Encoder saved at {save_dir}")
            # test: calculate the l2 norm between the latest latent mean and the first latent mean every 10 steps
            if learn_steps_bc == args.bc_steps:
                learn_steps_bc += 1
                print("Finished BC!")
                break
            if learn_steps_bc == 50:
                exit_save(encoder, exp_dir, experts, device, learn_steps_bc, args) 
                print(f"Early save at step {learn_steps_bc}")
    exit_save(encoder, exp_dir, experts, device, learn_steps_bc, args) 
    wandb.finish()   
    

def exit_save(encoder, exp_dir, experts,device, learn_steps_bc, args):
    create_emb = True 
    if create_emb:
        encoder.eval()
        emb_list = update_expert_memory(encoder, experts, device)
        # save the emb_list to a file
        emb_dir = hydra.utils.to_absolute_path(f'cond/{args.env.short_name}/baseline/')
        os.makedirs(emb_dir, exist_ok=True)
        emb_file = os.path.join(emb_dir, f"cond_{args.exp_id}.pkl")
        with open(emb_file, 'wb') as f:
            pickle.dump(emb_list, f)
        print(f"Baseline conditions saved at {emb_file}")
    torch.save(encoder, os.path.join(exp_dir, f"prob-encoder_dim{args.cond_dim}_kld_alpha{args.method.kld_alpha}_betaB_step_{learn_steps_bc}.ckpt"))
    print("Encoder saved at ", os.path.join(exp_dir, "baseline.ckpt"))    

def get_new_cond(encoder, experts, device, traj_idx_list):
    new_emb_list = {"emb": [], "dist_params":[]}
    for i in traj_idx_list:
        states = torch.tensor(experts["states"][i],dtype=torch.float32).to(device)
        actions = torch.tensor(experts["actions"][i],dtype=torch.float32).to(device)
        mu, logvar = encoder(states,actions)
        new_emb_list["emb"].append(mu)
        new_emb_list["dist_params"].append((mu, logvar))
    return new_emb_list

def update_expert_memory(encoder, experts, device):
    emb_list = {"emb": [], "dist_params":[]}
    for i in range(len(experts["states"])):
        states = torch.tensor(experts["states"][i],dtype=torch.float32).to(device)
        actions = torch.tensor(experts["actions"][i],dtype=torch.float32).to(device)
        mean, std = encoder(states, actions)
        emb = reparameterize(mean, std)
        emb_list["emb"].append(emb)
        emb_list["dist_params"].append((mean, std))
    ## --> Normalize the emb using z score normalization
    # emb_list["emb"] = zscore(emb_list["emb"])
    # save numpy arrays for file
    emb_list["emb"] = [i.detach().cpu().numpy() for i in emb_list["emb"]]
    emb_list["dist_params"] = [(i[0].detach().cpu().numpy(), i[1].detach().cpu().numpy()) for i in emb_list["dist_params"]]
    return emb_list

def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu

# cond_type: 1 for real indexed, 0 for fixed index 0, -1 for [-1]*cond_dim
def get_random_cond(cond_dim, cond_type, conds):
    conds = conds["emb"]
    # select random index from conds length
    index = random.randint(0, len(conds)-1)
    if cond_type=="random" or cond_type=="debug" :
        cond = conds[index][:cond_dim]
    elif cond_type=="none" or cond_type=="dummy":
        cond = [-1]*cond_dim
    else:
        # throw error that cond_type is not recognized
        raise ValueError("cond_type is not recognized. Use 'random', 'debug', 'dummy', or 'none'")
    return cond

def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """Read file from the input path. Assumes the file stores dictionary data.

    Args:
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data

def bc_update(self, observation, action, condition, logger, step, cond_type, mu, log_var, encoder_optimizer):
    # SAC version
    if self.actor:
        if cond_type == "none":
            training_metrics = self.loss_calculator(self, observation, action)
        else:
            training_metrics = self.loss_calculator(self, (observation, condition), action, mu, log_var)

        loss = training_metrics["loss/bc_actor"]
        # print(training_metrics)

        # optimize actor
        self.actor_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        encoder_optimizer.step()

        # update critic
        # if step % self.actor_update_frequency == 0:
        #     if cond_type == 'none':
        #         critic_losses = self.bc_update_critic(observation, logger, step)
        #     else:
        #         critic_losses = self.bc_update_critic((observation, condition), logger, step)

        #     training_metrics.update(critic_losses)

        # log
        for key, loss in training_metrics.items():
            if "bc_" in key.split("/")[1]:
                _key = f"train/{key.split('/')[1]}"
            else:
                _key = f"train/bc_{key.split('/')[1]}"
            logger.log(_key, loss, step)

    # Q-Learning version
    else:
        if cond_type == "none":
            training_metrics = self.loss_calculator(self, observation, action)
        else:
            training_metrics = self.loss_calculator(self, (observation, condition), action, mu, log_var)

        loss = training_metrics["loss/bc_actor"]

        # optimize Q function
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # Optionally update target Q networks if using target networks
        if self.use_target_networks:
            self.update_target_networks()

        # log
        for key, loss in training_metrics.items():
            if "bc_" in key.split("/")[1]:
                _key = f"train/{key.split('/')[1]}"
            else:
                _key = f"train/bc_{key.split('/')[1]}"
            logger.log(_key, loss, step)

    # else:
    #     raise NotImplementedError("q learning loop has yet implemented")

    # if step % self.critic_target_update_frequency == 0:
    #     if self.args.train.soft_update:
    #         soft_update(self.critic_net, self.critic_target_net,
    #                     self.critic_tau)
    #     else:
    #         hard_update(self.critic_net, self.critic_target_net)

    return training_metrics

def clustering_report(emb_list, logname, n_features):
    # compare traj embeddings
    # option 1. k_means num_clusters = n_proficiency_levels
    # expected the same level in the same cluster
    
    # define dataset
    X, _ = make_classification(n_samples=1000, n_features=n_features, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
    # define the model
    model = KMeans(n_clusters=n_features, n_init=10)
    # fit the model
    emb_list["emb"] = [np.squeeze(i) for i in emb_list["emb"]]
    # check if the emb is 1D
    if np.array(emb_list["emb"]).ndim == 1:
        emb_list["emb"] = np.array(emb_list["emb"]).reshape(-1,1)
    model.fit(emb_list["emb"])
    # assign a cluster to each example
    yhat = model.predict(emb_list["emb"])
    # retrieve unique clusters
    clusters = unique(yhat)
    # create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = where(yhat == cluster)
        # create scatter of these samples
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
    # show the plot
    # pyplot.show()
    # pyplot.savefig(f'plot/{exp_name}_kmeans.png')
    # calculate the ratio
    pred_acc = [[0 for i in range(n_features)] for i in range(n_features)]
    if "level" not in emb_list.keys(): # temporary fix for baseline embs missing level
        # 0, 1, 2 for proficiency levels corresponding to 0to9, 10to19, 20to29
        cluster_num = len(emb_list["emb"])/n_features
        emb_list["level"] = [int(i/cluster_num) for i in range(len(emb_list["emb"]))]
    for index, cluster in enumerate(yhat):
        proficiency_level = emb_list["level"][index]
        pred_acc[cluster][proficiency_level] += 1
    
    with open(logname, 'w') as f:
        f.write("#" * 80 + "\n")
        for index_i, i in enumerate(pred_acc):
            f.write(f" >>> Cluster {index_i} | Total Count: {sum(i)}\n")
            for index_j, j in enumerate(i):
                if sum(i) > 0:
                    f.write(f"Proficiency Level {index_j}: {100 * j / sum(i):.2f}% | Count: {j}\n")
                else:
                    f.write(f"Proficiency Level {index_j}: 0.00% | Count: 0\n")
            f.write("\n")
        f.write("#" * 80 + "\n")
    print(f"Log saved at {logname}")

    real_acc = np.eye(n_features, n_features)
    pred_acc = pred_acc / np.sum(pred_acc, axis=1, keepdims=True)
    # Calculate optimal assignment between real_acc and pred_acc using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-np.dot(real_acc, pred_acc.T))
    # Rearrange pred_acc rows according to optimal assignment
    pred_acc_matched = pred_acc[col_ind]

    # Calculate Mean Squared Error (MSE) based on optimal assignment
    mse = np.mean((real_acc - pred_acc_matched) ** 2)
    print(f"Clustering Mean Squared Error (MSE): {mse}. Detailed log saved at {logname}.")

    return mse
if __name__ == "__main__":
    main()