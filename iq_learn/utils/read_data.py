#%% --> Import
# read the 25 trajs and list their states/return.
# read from experts/HalfCheetah-v2_25.pkl
from typing import Any, Dict, IO, List, Tuple
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
# with open('experts/HalfCheetah-v3_20_Gemini_2k+6k.pkl', 'rb') as f:
#     data = pickle.load(f)
# # with open('experts/HalfCheetah-v2_25.pkl', 'rb') as f:
# #     data = pickle.load(f)

# # read rewards and calculate average return
# returns = [sum(i) for i in data["rewards"][10:]]
# print('len', len(returns))
# print(f"max mean min: {max(returns)} {np.mean(returns)} {min(returns)}")
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

def main():
    expert_location = 'experts/LunarLander-v2_100_230r.pkl'
    with open(expert_location, 'rb') as f:
        trajs = read_file(expert_location, f)

# with open('experts/CartPole-v1_1000.pkl', 'rb') as f:
#     data = pickle.load(f)
#%% --> Demos
    # check the lengths for official data
    lengths = [len(i) for i in trajs["states"]]
    print('len', lengths)
    
    # check the length for rollout
    lengths = trajs['lengths']
    print('lengths of trajs', lengths)

    # check returns
    # returns = [sum(i) for i in trajs["rewards"][:]]
    # print('number of trajs', len(returns))
    # print(f"max mean min: {max(returns)} {np.mean(returns)} {min(returns)}")
    # # check the min mean and max of states along axis 0
    # states = trajs["states"]
    # min = np.min(states, axis=0)
    # mean = np.mean(states, axis=0)
    # max = np.max(states, axis=0)
    # print('min', min)
    # print('mean', mean)
    # print('max', max)
    
#%% --> Embeddings
    
# emb = data['emb']
# print('len', len(emb))
# print('shape', emb[0].shape)
# print(f"value: {emb[0]}")

# # find mean and variance of the embeddings, round to 2 decimals
# emb = np.array(emb)
# print('mean', np.mean(emb, axis=0).round(2))
# print('var', np.var(emb, axis=0).round(2))



# print(f"value: {returns[0]}")

# states = data["states"]
# print('len', len(states[0]))
# print('shape', states[0][0].shape)
# print(f"value: {states[0][0]}")
 

# actions = data["actions"]
# print('len', len(actions[0]))
# print('shape', actions[0][0].shape)
# print(f"value: {actions[0][0]}")


# def sort()
#     new_actions = []
#     for i in range(len(actions)):
#         list = []
#         for j in range(len(actions[i])):
#             item = np.array(actions[i][j][0],dtype=np.float32)
#             list.append(item)
#             # exit()
#         new_actions.append(list)
#     print('len', len(new_actions[0]))
#     print('shape', new_actions[0][0].shape)

#     new_states = []
#     for i in range(len(states)):
#         list = []
#         for j in range(len(states[i])):
#             item = np.array(states[i][j][0],dtype=np.float32)
#             list.append(item)
#             # exit()
#         new_states.append(list)
#     print('len', len(new_states[0]))
#     print('shape', new_states[0][0].shape)

#     data["states"] = new_states
#     data["actions"] = new_actions
#     # dump the new data to /HalfCheetah-v3_long.pkl
#     with open('experts/HalfCheetah-v3_long.pkl', 'wb') as f:
#         pickle.dump(data, f)
    # plot the states and save in a file

    # sort the states and list the index from high to low

if __name__=="__main__":
    main()     
