"""
Evaluating the Decoder

This document outlines the steps to evaluate the decoder using the results obtained from our experiments.

Steps to Evaluate the Decoder:
1. Access Results in WandB:
   - Navigate to the WandB web interface and locate the `_content.hydra_base_dir`.

2. Transfer Results:
   - In the Origami environment, change directories to `_content.hydra_base_dir`.
   - Use the SCP command to copy the `results/` directory to `server2` at the following path: 
     `decoder/checkpoints/[env name]/[exp type]/`.

3. Run Evaluation:
   - On `server2`, modify the relevant arguments in `evaluate_decoder.py` and execute the script with the following parameters:

   | Argument   | Value                                                                                   |
   |------------|-----------------------------------------------------------------------------------------|
   | env        | Short name of the environment (e.g., `Hopper-v2`)                                      |
   | pretrain   | Path to the pre-trained model (e.g., `decoder/checkpoints/[env name]/[exp type]/results/results/sac_iq_Hopper-v2_actor`) |
   | env.cond   | Path to the condition file (e.g., `cheetah/no_id/HalfCheetah-v2_30_4208r+6301r+10775r_dummy.pkl`)               |
   | **exp_id** | **Identifier for the experiment (e.g., `s1` for our method seed 1, `d1` for dummy condition seed 1)**                                         |

By following these steps, you can successfully evaluate the decoder and analyze its performance on the specified tasks.
"""

import os
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
import random
from agent import make_agent
from dataset.memory import Memory
from iq import iq_loss
from make_envs import make_env
from utils.logger import Logger
from utils.utils import (average_dicts, eval_mode, evaluate,
                         get_concat_samples, hard_update, soft_update)
from typing import IO, Any, Dict
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    cfg.env_name = cfg.env_name
    cfg.agent_name = cfg.agent_name
    print(OmegaConf.to_yaml(cfg))
    return cfg

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

def evaluate_indexed(agent, eval_env, conds, indexes, args):
    means = []
    stds = []
    all_returns = []
    for eval_index in tqdm(indexes):
        # print(f"eval_index={eval_index}")
        eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental="none")
        
        # Calculate mean and std of returns
        mean_returns = np.mean(eval_returns)
        std_returns = np.std(eval_returns)
        
        means.append(mean_returns)
        stds.append(std_returns)
        all_returns.append(eval_returns)
        # print(f"Mean={mean_returns}, Std={std_returns}")
    return means, stds, all_returns

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

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
    agent = make_agent(env, args)

    if args.pretrain:
        pretrain_paths = OmegaConf.to_container(args.pretrain)  # Convert ListConfig to a regular list
        if len(pretrain_paths) == 2:
            actor_path, critic_path = pretrain_paths
            actor_path = hydra.utils.to_absolute_path(actor_path)
            critic_path = hydra.utils.to_absolute_path(critic_path)

            if os.path.isfile(actor_path) and os.path.isfile(critic_path):
                print("=> loading pretrain '{}' and '{}'".format(actor_path, critic_path))
                agent.load(actor_path, critic_path)
            else:
                print("[Attention]: One or both checkpoints not found: {} and {}".format(actor_path, critic_path))
        else:
            print("[Error]: Invalid number of paths provided. Expected two paths in a list.")
    
    conds = None
    cond_location = hydra.utils.to_absolute_path(f'cond/{args.env.cond}')
    if os.path.isfile(cond_location):
        # Load data from single file.
        with open(cond_location, 'rb') as f:
            conds = read_file(cond_location, f)
    else:
        raise ValueError(f"Condition file {cond_location} not found")

    mean_per_level = []
    std_per_level = []
    if "none" in args.experimental:
        print(f"Evaluate decoder, eval.eps={args.eval.eps}")
        means = []
        stds = []
        all_returns = []
        full_indexes = range(len(conds['emb'])) 
        for i in range(3):
            print(f"Condition from {i*len(full_indexes)//3} to {(i+1)*len(full_indexes)//3}")
            indexes = full_indexes[i*len(full_indexes)//3:(i+1)*len(full_indexes)//3]
            mean, std, returns = evaluate_indexed(agent, eval_env, conds, indexes, args)
            means.extend(mean)
            stds.extend(std)
            all_returns.extend(returns)

            mean_returns = np.mean(returns)
            std_returns = np.std(returns)
            mean_per_level.append(mean_returns)
            std_per_level.append(std_returns)
            print(f"Mean={mean_returns}, Std={std_returns}")
    # Convert lists to numpy arrays for easier handling
    means = np.array(means)
    stds = np.array(stds)

    def convert_to_list(item):
        return item.tolist() if isinstance(item, np.ndarray) else item
    # Save the mean, std, and returns to a CSV file
    data = {
        'Mean': means,
        'Std': stds,
        'Returns': [convert_to_list(r) for r in all_returns]
    }

    result_last_dir = os.path.join(args.exp_dir, "csv", args.env.short_name)
    df = pd.DataFrame(data)
    if not os.path.exists(result_last_dir):
        # print(f"Please create directory {result_last_dir} first")
        os.makedirs(result_last_dir)
        print(f"Created directory {result_last_dir}")
    else:
        print(f"Directory {result_last_dir} already exists")
    csv_filename = os.path.join(result_last_dir, f'{args.exp_id}.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Full data has been saved to '{csv_filename}'.")


    # print mean_per_level and std_per_level to a file
    dir = os.path.join(args.exp_dir, "result", args.env.short_name)
    if not os.path.exists(dir):
        # print(f"Please create directory {dir} first")
        os.makedirs(dir)
        print(f"Created directory {dir}")
    else:
        print(f"Directory {dir} already exists")
    data = {
        'Mean': mean_per_level,
        'Std': std_per_level
    }
    df = pd.DataFrame(data)
    csv_filename = os.path.join(dir, f'{args.exp_id}.csv') 
    df.to_csv(csv_filename, index=False)
    print(f"Result has been saved to '{csv_filename}'.")
    

if __name__ == '__main__':
    main()
