# !pip install gym=0.26 mujoco==2.3.3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # before import torch, keras
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'  # Add this as well
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
from render_browser import render_browser
import time

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

# @render_browser
def rollout(actor, env, perturb_condition, num_episodes=3, video_name="perturb"):
    """Evaluates the policy and render a video
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    # state = env.reset()    
    traj = []
    total_timesteps = []
    total_returns = []
    actions = []
    dones = 0

    # while len(total_returns) < num_episodes and dones < num_episodes:
    state, _ = env.reset()
    done = False
    terminated = False
    truncated = False
    cond = perturb_condition
    with eval_mode(actor):
        while not done and not terminated and not truncated:
            # yield env.render(mode='rgb_array')
            # env.render()
            action = actor.choose_action((state,cond), sample=False)
            next_state, reward, terminated, truncated, info= env.step(action)
            done = truncated or terminated 
            state = next_state
            # env.render()
    
            if 'episode' in info.keys():
                total_returns.append(info['episode']['r'])
                total_timesteps.append(info['episode']['l'])
            if done:
                dones += 1
            traj.append((state, next_state, action, reward, done))
            actions.append(action)

    return actions 

   
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

    # env = make_env(args)
    eval_env = make_env(args, render=True)
    agent = make_agent(eval_env, args)

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

    condition = conds["emb"][20]
    if args.experimental == "perturb":
        print(f"Perturbed Condition:  Perturb one dimension of the condition")
        print_condition = [f"{x:.2f}" for x in condition]
        print(f"(High) Original Condition: {print_condition}")
        print(f"(Medium):{[f"{x:.2f}" for x in conds["emb"][10]]}")
        print(f"(Low):{[f"{x:.2f}" for x in conds["emb"][0]]}")
        for dim in range(2,10,1):
            print(f"Perturbing dimension {dim}:")
            for value in range(-6,6,1):
                perturb_value = value/2
                video_folder = f'/home/zichang/proj/IQ-Learn/iq_learn/video/{args.env.short_name}'
                eval_env = make_env(args, render=True, video_folder=video_folder, video_name=f"perturb_dim{dim}_val{perturb_value}")
                perturb_condition = condition.copy()
                perturb_condition[dim] = perturb_value
                # each dim to :2f for printing
                print_condition = [f"{x:.2f}" for x in perturb_condition]
                print(f"Current Condition: {print_condition}")
                try:
                    rollout(agent, eval_env, perturb_condition, num_episodes=args.eval.eps, video_name=f"perturb_dim{dim}_val{value}")
                finally:
                    if eval_env is not None:
                        eval_env.close_video_recorder()
                        eval_env.close()
                        print("Environment closed after rollouts.\n")
                
    # Clean up meta and other related files
    video_folder = f"/home/zichang/proj/IQ-Learn/iq_learn/video/{args.env.short_name}"
    for filename in os.listdir(video_folder):
        if filename.endswith(".meta.json"):
            os.remove(os.path.join(video_folder, filename))
            print(f"Removed {filename}")

if __name__ == '__main__':
    main()
