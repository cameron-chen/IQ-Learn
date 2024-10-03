"""
Ablation study, options by args.experimental:
1. weighted: weighted condition on two conditions
2. perturb: perturbing a single dimension of the condition
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

    if "weighted" in args.experimental:
        print(f"{args.experimental}: alpha*low_condition+(1-alpha)*high_condition, eval.eps={args.eval.eps}")
        print("Episode_reward:")
        alphas = []
        means = []
        stds = []
        all_returns = []
        custom_index = range(-10, 1, 2) # custom index to replace for loop for more ablation study
        # for i in custom_index:
        for i in range(10, -1, -1):
            alpha = "{:.2f}".format(i*0.1)
            alphas.append(alpha)
            experimental_indexed = args.experimental + alpha
            eval_index = 0 # placeholder
            eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental=experimental_indexed)
            
            # Calculate mean and std of returns
            mean_returns = np.mean(eval_returns)
            std_returns = np.std(eval_returns)
            
            means.append(mean_returns)
            stds.append(std_returns)
            all_returns.append(eval_returns)
            print(f'Alpha={alpha}:', mean_returns)
    elif "perturb" in args.experimental:
        print(f"{args.experimental}: -2.5 to 2.5, eval.eps={args.eval.eps}") 
        alphas = []
        means = []
        stds = []
        all_returns = []
        dim = args.experimental.replace("perturb", "")
        dim = int(dim)
        original_value = conds["emb"][20][dim] 
        print(f"Original value: {original_value}")
        print("Episode_reward:")
        for i in range(-25, 26, 5):
            perturb_value = "{:.2f}".format(original_value + i*0.1)
            alphas.append(perturb_value)
            experimental_indexed = args.experimental + "_" + perturb_value
            eval_index = 0 # placeholder
            eval_returns, eval_timesteps = evaluate(agent, eval_env, conds, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental=experimental_indexed)
            
            # Calculate mean and std of returns
            mean_returns = np.mean(eval_returns)
            std_returns = np.std(eval_returns)
            
            means.append(mean_returns)
            stds.append(std_returns)
            all_returns.append(eval_returns)
            print(f'Perturb value={perturb_value}:', mean_returns)
    # Convert lists to numpy arrays for easier handling
    alphas = np.array(alphas, dtype=float)
    means = np.array(means)
    stds = np.array(stds)

    # Calculate mean + std and mean - std
    mean_plus_std = means + stds
    mean_minus_std = means - stds

    # Plot the graphs in the same plot
    plt.figure(figsize=(10, 6))

    plt.plot(alphas, means, marker='o', label='Mean')
    plt.plot(alphas, mean_plus_std, marker='o', linestyle='--', label='Mean + Std')
    plt.plot(alphas, mean_minus_std, marker='o', linestyle='--', label='Mean - Std')

    plt.title('Mean, Mean + Std, and Mean - Std of Returns')
    plt.xlabel('Alpha')
    plt.ylabel('Return')
    plt.legend()

    exp_dir = args.exp_dir
    result_last_dir = os.path.join(exp_dir, args.env.short_name)

    if not os.path.exists(result_last_dir):
        print(f"Please create directory {result_last_dir} first")
        # os.makedirs(result_last_dir)
        # print(f"Created directory {result_last_dir}")
    else:
        print(f"Directory {result_last_dir} already exists")
    # Save the plot as a PNG file
    plot_filename = os.path.join(result_last_dir, f'{args.env.short_name}_{args.experimental}.png')
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    plt.show()

    def convert_to_list(item):
        return item.tolist() if isinstance(item, np.ndarray) else item
    # Save the mean, std, and returns to a CSV file
    data = {
        'Alpha': alphas,
        'Mean': means,
        'Std': stds,
        'Returns': [convert_to_list(r) for r in all_returns]
    }

    df = pd.DataFrame(data)
    result_last_dir = os.path.join(result_last_dir, "csv")
    if not os.path.exists(result_last_dir):
        print(f"Please create directory {result_last_dir} first")
        # os.makedirs(result_last_dir)
        # print(f"Created directory {result_last_dir}")
    else:
        print(f"Directory {result_last_dir} already exists")
    csv_filename = os.path.join(result_last_dir, f'{args.env.short_name}_{args.experimental}.csv')
    df.to_csv(csv_filename, index=False)
    print(f"Data has been saved to '{csv_filename}'.")




if __name__ == '__main__':
    main()
