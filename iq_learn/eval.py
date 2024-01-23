import random
from itertools import count
from collections import defaultdict
# import gymnasium as gym
import gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
# from tensorboardX import SummaryWriter

from agent import make_agent
from make_envs import make_env, is_atari
from dataset.memory import Memory
import pickle
from stable_baselines3.common.evaluation import evaluate_policy
def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

    env = make_env(args)

    from stable_baselines3 import SAC
    agent = SAC(
        "MlpPolicy", 
        env=env,
        verbose=1
    )
    path = '/home/zichang/proj/IQ-Learn/iq_learn/iq.para/HalfCheetah-v3_trained.zip'
    agent = SAC.load(path, env=env)
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(
        agent,
        env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

if __name__ == '__main__':
    main()