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

    if args.experimental == "weighted":
        print(f"Weighted Condition: alpha*low_condition+(1-alpha)*high_condition, eval.eps={args.eval.eps}")
        print("Episode_reward:")
        for i in range(10, -1, -1):
            alpha = "{:.2f}".format(i*0.1)
            experimental_indexed = args.experimental + alpha
            eval_index = 0
            eval_returns, eval_timesteps = evaluate(agent, eval_env, cond_location, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental=experimental_indexed)
            returns = np.mean(eval_returns)
            print(f'Alpha={alpha}:', returns)
    elif "weighted" in args.experimental:
        eval_index = 0
        eval_returns, eval_timesteps = evaluate(agent, eval_env, cond_location, num_episodes=args.eval.eps, cond_dim=args.cond_dim, cond_type=args.cond_type, eval_index=eval_index, experimental=args.experimental)
        returns = np.mean(eval_returns)
        print(f'episode_reward:', returns)




if __name__ == '__main__':
    main()
