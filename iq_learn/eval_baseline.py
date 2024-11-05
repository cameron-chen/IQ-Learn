import argparse
import os
import torch
import numpy as np
import sys
import logging
import pickle
import gym
from typing import IO, Any, Dict
from tqdm import tqdm
from scipy.stats import zscore

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("exp_name", help="name of the experiment to run")
    arg_parser.add_argument("expert_file", help="full path of expert demo")
    arg_parser.add_argument("checkpoint", type=str, default="encoder/encoder.pt")
    arg_parser.add_argument("--obs-std", type=float, default=1.0)
    arg_parser.add_argument("--batchsize", type=int, default=8)
    arg_parser.add_argument("--n_features", type=int, default=3)
    arg_parser.add_argument("--embed_mode", type=str, default="det", help="det, mean, dummy or prob", choices=["det", "mean", "dummy", "prob", "z_logit"])
    arg_parser.add_argument("--exp_id", type=str, default="no_id", help="experiment id for saving")
    arg_parser.add_argument("--cond_dim", type=int, default=10, help="condition dimension")
    args = arg_parser.parse_args()
    mycwd = os.getcwd()
    target_dir = '/home/zichang/proj/IQ-Learn/iq_learn/encoder'
    os.chdir(target_dir)
    print("Current working directory: ", os.getcwd())
    sys.path.append(target_dir)
    from encoder.baseline_model import BaselineModel
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    encoder = torch.load(args.checkpoint).cpu()
    encoder.eval()
    os.chdir(mycwd)
    dataset_paths = args.expert_file.split(",")
    state = []
    action = []
    level = []
    for idx, dataset_path in enumerate(dataset_paths):
        state_i, action_i = open_dataset(dataset_path)
        level_i = [idx for i in range(len(state_i))]
        state.extend(state_i)
        action.extend(action_i)
        level.extend(level_i)
    # os.chdir(mycwd) 
    state = np.array(state, dtype='float32')
    action = np.array(action, dtype='float32')
    level = np.array(level, dtype='int')
    emb_list = {"num_m":[],"emb": [], "level":[], "num_z":[], "z":[], "logit_m":[]}
    for s, a, l in tqdm(zip(state, action, level)):
        s = torch.tensor(s,dtype=torch.float32).cpu()
        a = torch.tensor(a,dtype=torch.float32).cpu()
        mu_0, _ = encoder(s, a)
        mu_0 = mu_0.detach().cpu().numpy()
        emb_list["emb"].append(mu_0)
        emb_list["level"].append(l)
    emb_list["emb"] = zscore(emb_list["emb"])
    emb_list["emb"] = zscore(emb_list["emb"])
    datafile = f"/home/zichang/proj/IQ-Learn/iq_learn/encoder/data/baseline/{args.exp_name}.pkl"
    os.makedirs(os.path.dirname(datafile), exist_ok=True)
    with open(datafile, 'wb') as f:
        pickle.dump(emb_list, f)
    print(f"Baseline conditions saved at {datafile}")    
    return emb_list

def open_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    state = [np.array(x, dtype=type) for x in trajectories['states']]
    action = [np.array(x, dtype=type) for x in trajectories['actions']]
    return state, action
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

if __name__ == "__main__":
    main()