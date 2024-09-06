import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # before import torch, keras
os.environ['PYOPENGL_PLATFORM'] = 'egl'
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

import gym
from gym import wrappers
from gym.wrappers import Monitor
from pyvirtualdisplay import Display
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
    # vidsavedir = "./video"
    # env = gym.make("Hopper-v2")
    # env = wrappers.Monitor(env, vidsavedir, force=True, video_callable=lambda episode_id: True)
    # env.reset()
    # env = gym.wrappers.RecordVideo(env=env, 
    #                                video_folder="./video", 
    #                                name_prefix=video_name, 
    #                                episode_trigger=lambda x: x % 2 == 0)
    # env = wrappers.Monitor(env, "./video", codec='mpeg4')
    # env = wrappers.Monitor(env, "./video", video_callable=lambda episode_id: True, force=True)
    # env = RecordVideo(env=env, video_path="./video/video.mp4")
    
    
    state = env.reset()    
    traj = []
    total_timesteps = []
    total_returns = []
    actions = []
    # env.start_video_recorder()
    display = Display(visible=0, size=(1400, 900, 24))
    display.start()
    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False
        terminated = False
        cond = perturb_condition
        with eval_mode(actor):
            while not done and not terminated:
                # yield env.render(mode='rgb_array')
                # env.render()
                action = actor.choose_action((state,cond), sample=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                env.render()
        
                if 'episode' in info.keys():
                    total_returns.append(info['episode']['r'])
                    total_timesteps.append(info['episode']['l'])
                traj.append((state, next_state, action, reward, done))
                actions.append(action)
    
    env.close()
    display.stop()
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

    # display = Display(visible=0, size=(1400, 900, 24))
    # display.start()
    try:
        condition = conds["emb"][20]
        if args.experimental == "perturb":
            print(f"Perturbed Condition:  Perturb one dimension of the condition")
            for dim in range(2):
                for value in range(-5,-4,1):
                    perturb_condition = condition.copy()
                    perturb_condition[dim] = value/2
                    print(f"Dimension {dim} perturbed to {value}")
                    actions = rollout(agent, eval_env, perturb_condition, num_episodes=args.eval.eps, video_name=f"perturb_dim{dim}_val{value}")
                    # render(eval_env, actions, "./video")
    except KeyboardInterrupt:
        pass
    finally:
        # eval_env.close()
        # display.stop()
        pass


if __name__ == '__main__':
    main()
