"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import pickle
import random
import time
import types
from collections import deque
from itertools import count
from typing import IO, Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

import wandb
from agent import make_agent
from bc import BehaviorCloningLossCalculator
from dataset.memory import Memory
from iq import iq_loss
from make_envs import make_env
from utils.logger import Logger
from utils.utils import (
    average_dicts,
    eval_mode,
    evaluate,
    get_concat_samples,
    hard_update,
    soft_update,
)

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
    wandb.init(
        project="hil_iq",
        sync_tensorboard=True,
        reinit=True,
        config=args,
        name=f"{args.env.cond} expert{args.expert.demos} temp{args.agent.init_temp} {args.method.loss}",
    )
    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    env.seed(args.seed)  # TODO: uncomment this to enable seed on gym
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
    expert_memory_replay = Memory(REPLAY_MEMORY // 2, args.seed)
    expert_memory_replay.load(
        hydra.utils.to_absolute_path(f"experts/{args.env.demo}"),
        num_trajs=args.expert.demos,
        sample_freq=args.expert.subsample_freq,
        seed=args.seed + 42,
        cond_dim=args.cond_dim,
        cond_type=args.cond_type,
        cond_location=hydra.utils.to_absolute_path(f"cond/{args.env.cond}"),
    )
    print(f"--> Expert memory size: {expert_memory_replay.size()}")

    online_memory_replay = Memory(REPLAY_MEMORY // 2, args.seed + 1)

    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"--> Saving logs at: {log_dir}")
    logger = Logger(
        args.log_dir,
        log_frequency=args.log_interval,
        writer=writer,
        save_tb=True,
        agent=args.agent.name,
    )

    steps = 0

    # track mean reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    # Sample initial states from env
    # state_0 = [env.reset()] * INITIAL_STATES
    # if isinstance(state_0[0], LazyFrames):
    #     state_0 = np.array(state_0) / 255.0
    #     print("lazy frames detected")
    # state_0 = torch.FloatTensor(np.array(state_0,dtype=np.float32)).to(args.device)

    # BC initialization
    if args.method.bc_init:
        agent.bc_update = types.MethodType(bc_update, agent)
        agent.loss_calculator = BehaviorCloningLossCalculator(
            ent_weight=1e-3,  # args.method.bc_ent_weight,
            l2_weight=0.0,  # args.method.bc_l2_weight,
        )

        for learn_steps_bc in count():
            expert_batch = expert_memory_replay.get_samples(
                agent.batch_size, agent.device
            )
            expert_obs, _, expert_action, __, ___, expert_cond = expert_batch
            losses = agent.bc_update(
                expert_obs,
                expert_action,
                expert_cond,
                logger,
                learn_steps_bc,
                args.cond_dim,
            )

            # log losses
            if learn_steps_bc % 10 == 0:  # args.log_interval == 0:
                for key, loss in losses.items():
                    writer.add_scalar(key, loss, global_step=learn_steps_bc)

                logger.dump(learn_steps_bc)

            # eval every n steps
            if learn_steps_bc % 100 == 0:  # args.env.eval_interval == 0:
                for eval_index in range(args.expert.demos):
                    eval_returns, eval_timesteps = evaluate(
                        agent,
                        eval_env,
                        hydra.utils.to_absolute_path(f"cond/{args.env.cond}"),
                        num_episodes=args.eval.eps,
                        cond_dim=args.cond_dim,
                        random_index=args.cond_type,
                        eval_index=eval_index,
                    )
                    returns = np.mean(eval_returns)
                    logger.log(
                        f"eval/bc_episode_reward{eval_index}", returns, learn_steps_bc
                    )
                # logger.log('eval/bc_episode_reward', returns, learn_steps_bc)
                logger.dump(learn_steps_bc, ty="eval")

            if learn_steps_bc == args.bc_steps:
                learn_steps_bc += 1
                print("Finished BC!")
                break

    for epoch in count():  # n of episodes
        state = env.reset()
        episode_reward = 0
        done = False
        cond = get_random_cond(
            args.cond_dim,
            args.cond_type,
            hydra.utils.to_absolute_path(f"cond/{args.env.cond}"),
        )
        start_time = time.time()
        for episode_step in range(EPISODE_STEPS):  # n of steps
            if steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    if args.cond_type == "none":
                        action = agent.choose_action(state, sample=True)
                    else:
                        action = agent.choose_action((state, cond), sample=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                learn_steps += 1  # To prevent repeated eval at timestep 0
                if args.cond_type == "debug":
                    for eval_index in range(args.expert.demos):
                        eval_returns, eval_timesteps = evaluate(
                            agent,
                            eval_env,
                            hydra.utils.to_absolute_path(f"cond/{args.env.cond}"),
                            num_episodes=args.eval.eps,
                            cond_dim=args.cond_dim,
                            cond_type=args.cond_type,
                            eval_index=eval_index,
                        )
                        returns = np.mean(eval_returns)
                        logger.log(
                            f"eval/episode_reward{eval_index}", returns, learn_steps
                        )
                else:
                    eval_returns, eval_timesteps = evaluate(
                        agent,
                        eval_env,
                        hydra.utils.to_absolute_path(f"cond/{args.env.cond}"),
                        num_episodes=args.eval.eps,
                        cond_dim=args.cond_dim,
                        cond_type=args.cond_type,
                    )
                    returns = np.mean(eval_returns)
                    logger.log("eval/episode_reward", returns, learn_steps)
                logger.log("eval/episode", epoch, learn_steps)
                logger.dump(learn_steps, ty="eval")
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if returns > best_eval_returns:
                    # Store best eval returns
                    best_eval_returns = returns
                    if wandb.run:
                        wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, epoch, args, output_dir="results_best")

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            done_no_lim = done
            if (
                str(env.__class__.__name__).find("TimeLimit") >= 0
                and episode_step + 1 == env._max_episode_steps
            ):
                done_no_lim = 0
            # if type(state) == np.ndarray:
            # online_memory_replay.add((state, next_state, action, reward, done_no_lim, cond))
            online_memory_replay.add(
                (state, next_state, action, reward, done_no_lim, cond)
            )

            if online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if begin_learn is False:
                    print("Learn begins!")
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print("Finished!")
                    wandb.finish()
                    return

                ######
                # IQ-Learn Modification
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
                losses = agent.iq_update(
                    online_memory_replay,
                    expert_memory_replay,
                    logger,
                    learn_steps,
                    args.cond_type,
                )
                ######

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        logger.log("train/episode", epoch, learn_steps)
        logger.log("train/episode_reward", episode_reward, learn_steps)
        logger.log("train/duration", time.time() - start_time, learn_steps)
        logger.dump(learn_steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir="results")


# cond_type: 1 for real indexed, 0 for fixed index 0, -1 for [-1]*cond_dim
def get_random_cond(cond_dim, cond_type, cond_location):
    if os.path.isfile(cond_location):
        # Load data from single file.
        with open(cond_location, "rb") as f:
            conds = read_file(cond_location, f)
    conds = conds["emb"]
    # select random index from conds length
    index = random.randint(0, len(conds) - 1)
    if cond_type == "random" or cond_type == "debug":
        cond = conds[index][:cond_dim]
    elif cond_type == "none" or cond_type == "dummy":
        cond = [-1] * cond_dim
    else:
        # throw error that cond_type is not recognized
        raise ValueError(
            "cond_type is not recognized. Use 'random', 'debug', 'dummy', or 'none'"
        )
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


def save(agent, epoch, args, output_dir="results"):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f"sqil_{args.env.name}"
        else:
            name = f"iq_{args.env.name}"

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f"{output_dir}/{args.agent.name}_{name}")


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    # expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
    (
        policy_obs,
        policy_next_obs,
        policy_action,
        policy_reward,
        policy_done,
        policy_cond,
    ) = policy_batch
    (
        expert_obs,
        expert_next_obs,
        expert_action,
        expert_reward,
        expert_done,
        expert_cond,
    ) = expert_batch

    if args.only_expert_states:
        # expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
        expert_batch = (
            expert_obs,
            expert_next_obs,
            policy_action,
            expert_reward,
            expert_done,
            expert_cond,
        )

    # obs, next_obs, action, reward, done, is_expert = get_concat_samples(
    #     policy_batch, expert_batch, args)
    obs, next_obs, action, reward, done, cond, is_expert = get_concat_samples(
        policy_batch, expert_batch, args
    )

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    # current_Q = self.critic(obs, action)
    # y = (1 - done) * self.gamma * self.getV(next_obs)
    current_Q = self.critic((obs, action, cond))
    y = (1 - done) * self.gamma * self.getV((next_obs, cond))
    if args.train.use_target:
        with torch.no_grad():
            # y = (1 - done) * self.gamma * self.get_targetV(next_obs)
            y = (1 - done) * self.gamma * self.get_targetV((next_obs, cond))

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    # value_loss = (self.getV(obs) - y).mean()
    value_loss = (self.getV((obs, cond)) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1 / (4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step, cond_type):
    args = self.args
    # policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    # expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
    (
        policy_obs,
        policy_next_obs,
        policy_action,
        policy_reward,
        policy_done,
        policy_cond,
    ) = policy_batch
    (
        expert_obs,
        expert_next_obs,
        expert_action,
        expert_reward,
        expert_done,
        expert_cond,
    ) = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        # expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done
        expert_batch = (
            expert_obs,
            expert_next_obs,
            policy_action,
            expert_reward,
            expert_done,
            expert_cond,
        )

    batch = get_concat_samples(policy_batch, expert_batch, args)
    # obs, next_obs, action = batch[0:3]
    obs, next_obs, action, reward, done, cond, is_expert = batch

    agent = self
    # current_V = self.getV(obs)
    if cond_type == "none":
        current_V = self.getV(obs)
    else:
        current_V = self.getV((obs, cond))
    if args.train.use_target:
        with torch.no_grad():
            # next_V = self.get_targetV(next_obs)
            if cond_type == "none":
                next_V = self.get_targetV(next_obs)
            else:
                next_V = self.get_targetV((next_obs, cond))
    else:
        # next_V = self.getV(next_obs)
        if cond_type == "none":
            next_V = self.get_targetV(next_obs)
        else:
            next_V = self.get_targetV((next_obs, cond))
        # next_V = self.getV((next_obs, cond))

    if "DoubleQ" in self.args.q_net._target_:
        # current_Q1, current_Q2 = self.critic(obs, action, both=True)
        if cond_type == "none":
            current_Q1, current_Q2 = self.critic(obs, action, both=True)
        else:
            current_Q1, current_Q2 = self.critic((obs, action, cond), both=True)
        q1_loss, loss_dict1 = iq_loss(
            agent, current_Q1, current_V, next_V, batch, cond_type
        )
        q2_loss, loss_dict2 = iq_loss(
            agent, current_Q2, current_V, next_V, batch, cond_type
        )
        critic_loss = 1 / 2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        # current_Q = self.critic(obs, action)
        if cond_type == "none":
            current_Q = self.critic(obs, action)
        elif args.agent.name == "softq":
            current_Q = self.critic((obs, cond), action)
        else:
            current_Q = self.critic((obs, action, cond))
        critic_loss, loss_dict = iq_loss(
            agent, current_Q, current_V, next_V, batch, cond_type
        )
    logger.log("train/critic_loss", critic_loss, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step, cond_type):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, logger, step, cond_type)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            def change_shape(online, expert):
                shape = online.shape
                expert = torch.reshape(expert, shape)
                return expert

            expert_item = change_shape(policy_batch[0], expert_batch[0])
            if self.args.offline:
                obs = expert_batch[0]
                cond = expert_batch[-1]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)
                cond = torch.cat([policy_batch[-1], expert_batch[-1]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    # actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
                    if cond_type == "none":
                        actor_alpha_losses = self.update_actor_and_alpha(
                            obs, logger, step
                        )
                    else:
                        actor_alpha_losses = self.update_actor_and_alpha(
                            (obs, cond), logger, step
                        )

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net, self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


def bc_update(self, observation, action, condition, logger, step, cond_dim):
    # SAC version
    if self.actor:
        if cond_dim == -2:
            training_metrics = self.loss_calculator(self, observation, action)
        else:
            training_metrics = self.loss_calculator(
                self, (observation, condition), action
            )

        loss = training_metrics["loss/bc_actor"]

        # optimize actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # update critic
        # if step % self.actor_update_frequency == 0:
        #     if cond_dim==-2:
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
        raise NotImplementedError("q learning loop has yet implemented")

    # if step % self.critic_target_update_frequency == 0:
    #     if self.args.train.soft_update:
    #         soft_update(self.critic_net, self.critic_target_net,
    #                     self.critic_tau)
    #     else:
    #         hard_update(self.critic_net, self.critic_target_net)

    return training_metrics


if __name__ == "__main__":
    main()
