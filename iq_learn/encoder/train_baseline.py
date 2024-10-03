

import os
import argparse
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from grid_world import grid
# from gym_miniworld import miniworld
# from world3d import world3d
import utils
import modules
from modules import (
    GridDecoder,
    GridActionEncoder,
    LinearLayer
)
from datetime import datetime
import wandb
from scipy.stats import zscore
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
LOGGER = logging.getLogger(__name__)
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
def parse_args():

    parser = argparse.ArgumentParser(description="vta agr parser")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--name", type=str, default="st")

    # data size
    parser.add_argument("--dataset-path", type=str, default="./data/demos")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seq-size", type=int, default=6)
    parser.add_argument("--init-size", type=int, default=1)
    parser.add_argument("--hil-seq-size", type=int, default=1000)

    # model size
    parser.add_argument("--state-size", type=int, default=8)
    parser.add_argument("--belief-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--latent-n", type=int, default=10)

    # observation distribution
    parser.add_argument("--obs-std", type=float, default=1.0)
    parser.add_argument("--obs-bit", type=int, default=5)

    # optimization
    parser.add_argument("--learn-rate", type=float, default=0.0005)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--max-iters", type=int, default=100000)

    # subsequence prior params
    parser.add_argument("--seg-num", type=int, default=100)
    parser.add_argument("--seg-len", type=int, default=100)

    # gumbel params
    parser.add_argument("--max-beta", type=float, default=1.0)
    parser.add_argument("--min-beta", type=float, default=0.1)
    parser.add_argument("--beta-anneal", type=float, default=100)

    # log dir
    parser.add_argument("--log-dir", type=str, default="./asset/log/")
    parser.add_argument("--save_interval", type=int, default=500)

    # coding length params
    parser.add_argument("--kl_coeff", type=float, default=1.0)
    parser.add_argument("--rec_coeff", type=float, default=1.0)
    parser.add_argument("--use_abs_pos_kl", type=float, default=0)
    parser.add_argument("--coding_len_coeff", type=float, default=1.0)
    parser.add_argument("--max_coding_len_coeff", type=float, default=0.0001)
    parser.add_argument("--use_min_length_boundary_mask", action="store_true")

    # baselines
    parser.add_argument("--ddo", action="store_true")
    parser.add_argument("--expert_file", type=str, default="")
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--eval_expert_file", type=str, default="")
    return parser.parse_args()

def main():
     # parse arguments
    args = parse_args()

    if not args.wandb:
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "disabled"

    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # set logger
    log_format = "[%(asctime)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format, stream=sys.stderr)

    # set size
    init_size = args.init_size

    # set device as gpu
    # device = torch.device("cuda", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # set writer
    def date_str():
        s = str(datetime.now())
        d, t = s.split(" ")
        t = "-".join(t.split(":")[:-1])
        return d + "-" + t
    exp_name = args.name + "_" + date_str()

    wandb.init(
        project="Stage_1",
        entity="zichang_team",
        name=exp_name,
        sync_tensorboard=False,
        settings=wandb.Settings(start_method="fork"),
    )

    LOGGER.info("EXP NAME: " + exp_name)
    LOGGER.info(">" * 80)
    LOGGER.info(args)
    LOGGER.info(">" * 80)

    # load dataset
    if "compile" in args.dataset_path:
        train_loader, test_loader = utils.compile_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size,
        )
        encoder = modules.CompILEGridEncoder(feat_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
    elif "miniworld" in args.dataset_path:
        train_loader, test_loader = utils.miniworld_loader(args.batch_size)
        action_encoder = GridActionEncoder(
            action_size=train_loader.dataset.action_size,
            embedding_size=args.belief_size,
        )
        encoder = modules.MiniWorldEncoderPano(input_dim=3)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = False
    elif "cheetah" in args.dataset_path:
        train_loader, test_loader = utils.hil_loader(args.batch_size, args.hil_seq_size)
        full_loader = utils.cheetah_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "cartpole" in args.dataset_path:
        train_loader, test_loader = utils.cartpole_loader(args.batch_size, args.hil_seq_size)
        full_loader = utils.cartpole_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn")
    elif "hopper" in args.dataset_path:
        train_loader, test_loader = utils.hopper_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.hopper_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "walker" in args.dataset_path:
        train_loader, test_loader = utils.walker_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.walker_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "ant" in args.dataset_path:
        train_loader, test_loader = utils.ant_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.ant_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "humanoid" in args.dataset_path:
        train_loader, test_loader = utils.humanoid_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.humanoid_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "swimmer" in args.dataset_path:
        train_loader, test_loader = utils.swimmer_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.swimmer_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "invertedp" in args.dataset_path:
        train_loader, test_loader = utils.invertedp_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.invertedp_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    elif "lunarlander" in args.dataset_path:
        train_loader, test_loader = utils.lunarlander_loader(args.batch_size, args.hil_seq_size, expert_file=args.expert_file)
        full_loader = utils.lunarlander_full_loader(1, args.eval_expert_file)
        action_encoder = LinearLayer(
            input_size=train_loader.dataset.action_size,
            output_size=args.belief_size)
        encoder = LinearLayer(
            input_size=train_loader.dataset.obs_size,
            output_size=args.belief_size)
        decoder = GridDecoder(
            input_size=args.belief_size,
            action_size=train_loader.dataset.action_size,
            feat_size=args.belief_size,
        )
        output_normal = True
        os.chdir("/home/zichang/proj/IQ-Learn/iq_learn/encoder")
    else:
        raise ValueError(f"Unrecognize dataset_path {args.dataset_path}")

    # if args.dataset_path in ["cheetah","cartpole","lunar"]:
    if True: # TODO: improve logic
        from hssm_rl_hil import EnvModel
    else:
        from hssm_rl import EnvModel
    seq_size = train_loader.dataset.seq_size

    use_abs_pos_kl = args.use_abs_pos_kl == 1.0
    model = EnvModel(
        action_encoder=action_encoder,
        encoder=encoder,
        decoder=decoder,
        belief_size=args.belief_size,
        state_size=args.state_size,
        num_layers=args.num_layers,
        max_seg_len=args.seg_len,
        max_seg_num=args.seg_num,
        latent_n=args.latent_n,
        kl_coeff=args.kl_coeff,
        rec_coeff=args.rec_coeff,
        use_abs_pos_kl=use_abs_pos_kl,
        coding_len_coeff=args.coding_len_coeff,
        use_min_length_boundary_mask=args.use_min_length_boundary_mask,
        ddo=args.ddo,
        output_normal=output_normal
    )
    model = nn.DataParallel(model)
    model = model.to(device)
    LOGGER.info("Model initialized")

    # init optimizer
    optimizer = Adam(params=model.parameters(), lr=args.learn_rate, amsgrad=True)

    pre_test_full_state_list, pre_test_full_action_list, pre_test_full_level_list = next(iter(test_loader))
    pre_test_full_state_list = pre_test_full_state_list.to(device)
    pre_test_full_action_list = pre_test_full_action_list.to(device)

    exp_dir = os.path.join("experiments", args.name)
    os.makedirs(exp_dir, exist_ok=True)

    # for each iter
    torch.autograd.set_detect_anomaly(False)
    b_idx = 0
    train_loss_list =[]

    early_stopper = EarlyStopper(patience=3, min_delta=0)
    while b_idx <= args.max_iters:
        for train_obs_list, train_action_list, train_level_list in train_loader:
                b_idx += 1
                # mask temp annealing
                if args.beta_anneal:
                    model.module.state_model.mask_beta = (
                        args.max_beta - args.min_beta
                    ) * 0.999 ** (b_idx / args.beta_anneal) + args.min_beta
                else:
                    model.module.state_model.mask_beta = args.max_beta

                ##############
                # train time #
                ##############
                train_obs_list = train_obs_list.to(device)
                train_action_list = train_action_list.to(device)

                # run model with train mode
                model.module.train()
                optimizer.zero_grad()
                results = model.module(
                    train_obs_list, train_action_list, seq_size, init_size, args.obs_std
                )

                if args.coding_len_coeff > 0:
                    if results["obs_cost"].mean() < 0.02:
                        model.module.coding_len_coeff += 0.00000001
                    elif b_idx > 0:
                        model.module.coding_len_coeff -= 0.00000001

                    model.module.coding_len_coeff = min(args.max_coding_len_coeff, model.module.coding_len_coeff)
                    model.module.coding_len_coeff = max(0.000000, model.module.coding_len_coeff)
                    results["coding_len_coeff"] = model.module.coding_len_coeff
                
                # get train loss and backward update
                train_total_loss = results["train_loss"]
                train_loss_list.append(train_total_loss.detach().cpu())
                train_total_loss.backward()
                if args.grad_clip > 0.0:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.module.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()

                # get num of skills in traj
                num_skills = [len(i) for i in results["unique_z_list"]]

                # log
                if b_idx % 5 == 0:
                    results["grad_norm"] = grad_norm
                    train_stats, log_str, log_data = utils.log_train(results, None, b_idx)
                    train_stats["train/loss"] = train_total_loss.detach().cpu()
                    train_stats["num_skills"] = sum(num_skills)/len(num_skills)
                    LOGGER.info(log_str, *log_data)
                    # LOGGER.info("ep: {:08}, training loss: {}".format(b_idx,train_total_loss.detach().cpu()))
                    # emb_list = run_exp(model.module.state_model, full_loader, "det", device)
                    logname = os.path.join("result_clustering", args.name, args.exp_id, f"stage1_b{b_idx}.log")
                    os.makedirs(os.path.dirname(logname), exist_ok=True)
                    # cluster_mse = clustering_report(emb_list, logname, 3)  # Adjust n_features as needed
                    # train_stats["cluster_mse"] = cluster_mse
                    # wandb.log(train_stats, step=b_idx)
                
                    # if cluster_mse<=0.0001 and sum(num_skills)/len(num_skills) < 200:
                    #     exp_dir = os.path.join("experiments", args.name, args.exp_id)
                    #     os.makedirs(exp_dir, exist_ok=True)
                    #     torch.save(
                    #         model.module.state_model, os.path.join(exp_dir, f"model-{b_idx}.ckpt")
                    #     )
                    #     print(f"Training has converged with cluster_mse {cluster_mse} Exiting...")
                    #     sys.exit(0)
                        
                np.set_printoptions(threshold=100000)
                torch.set_printoptions(threshold=100000)
                    
                if b_idx % args.save_interval == 0 or b_idx==10 or b_idx==30:
                    exp_dir = os.path.join("experiments", args.name, args.exp_id)
                    os.makedirs(exp_dir, exist_ok=True)
                    torch.save(
                        model.module.state_model, os.path.join(exp_dir, f"model-{b_idx}.ckpt")
                    )
                    
                    
                #############
                # test time #
                #############
                if b_idx % 100 == 0:
                    with torch.no_grad():
                        ##################
                        # test data elbo #
                        ##################
                        model.module.eval()
                        results = model.module(
                            pre_test_full_state_list,
                            pre_test_full_action_list,
                            seq_size,
                            init_size,
                            args.obs_std,
                        )
                        # log
                        test_stats, log_str, log_data = utils.log_test(results, None, b_idx)
                        LOGGER.info(log_str, *log_data)
                        wandb.log(test_stats, step=b_idx)

if __name__ == "__main__":
    main()
