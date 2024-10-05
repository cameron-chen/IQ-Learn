#%% --> Import
# read the 25 trajs and list their states/return.
# read from experts/HalfCheetah-v2_25.pkl
from typing import Any, Dict, IO, List, Tuple
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import argparse
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

def evolution():
    # plot the first 2 dims of latent mean from step 2 to 10
    prefix = 'cond/test_meanAsEmb_cond_dim10_kld_alpha1_step'
    dim0_list = []
    dim1_list = []
    for i in range(2, 10, 2):
        expert_loc = prefix + str(i) + '.pkl'
        with open(expert_loc, 'rb') as f:
            emb_list = read_file(expert_loc, f)
        dim0_list.append(emb_list['emb'][0][0])
        dim1_list.append(emb_list['emb'][0][1])
    # plot dim0_list and dim1_list on the same plot
    # x axis should be 2, 4, 6, 8, 10
    # y axis should be the values of dim0 and dim1
    x = [2, 4, 6, 8]
    plt.plot(x, dim0_list, label='dim0')
    plt.plot(x, dim1_list, label='dim1')
    plt.legend()
    plt.show()
    savepic = "test.png"
    plt.savefig(savepic)


def main():
    # make expert_location an argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str, help="name of the file")
    args = arg_parser.parse_args()
    expert_location = args.file
    # expert_location = 'experts/LunarLander-v2_100_230r.pkl'
    with open(expert_location, 'rb') as f:
        trajs = read_file(expert_location, f)
    
    # truncate each value length to 30:
    for key in trajs:
        trajs[key] = trajs[key][:30]
    # save the truncated data to a new file

    save_file = 'experts/antmaze/antmaze_30.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(trajs, f)
    print(f"Saved the truncated data to {save_file}")

    # Calculate mean and std for each skill level
    # results = []
    # skill_levels = ['Low', 'Medium', 'High']
    # import numpy as np
    # for i in range(3):
    #     skill_rewards = trajs['rewards'][i * 10:(i + 1) * 10]  # Get 10 rewards for each skill level
    #     total_rewards = [sum(reward) for reward in skill_rewards]  # Sum the 1000 steps for each reward
    #     mean = np.mean(total_rewards)
    #     std = np.std(total_rewards)
    #     results.append(f"& ${mean:.1f}\\pm{std:.1f}$ ")

    # # # Print the results
    # for i, result in enumerate(results):
    #     skill_level = skill_levels[i]
    #     print(f"{skill_level}: {result}")

#%% --> Print the keys of the trajs
    # print(f"Type of trajs: {type(trajs)}")
    # print(f"Keys of trajs: {trajs.keys()}")
    # print(f"Length of trajs: {len(trajs)}")
    # print(f"Length of states: {len(trajs['states'])}")
    # print(f"Length of actions: {len(trajs['actions'])}")
    # print(f"Length of rewards: {len(trajs['rewards'])}")
    # print(f"Length of lengths: {len(trajs['lengths'])}")
    # print(f"trajs['states'] shape: {trajs['states'][0].shape}")
    # print(f"trajs['next_states'] shape: {trajs['next_states'][0].shape}")
    # print(f"trajs['actions'] shape: {trajs['actions'][0].shape}")
    # print(f"trajs['rewards'] shape: {trajs['rewards'][0].shape}")
    # print(f"trajs['lengths'] shape: {trajs['lengths'][0]}")
    # print(f"traj['lengths']: {trajs['lengths']}")
#%% --> Compute distance between embeddings
    from scipy.stats import wasserstein_distance
    import numpy as np

    # Assume trajs["emb"] is a numpy array with shape (30, 10), where the first 10 rows are for skill level 1,
    # the next 10 rows are for skill level 2, and the last 10 rows are for skill level 3.

    # Split the embeddings based on skill levels
    skill_level_1 = trajs["emb"][:10]  # First 10 embeddings
    skill_level_2 = trajs["emb"][10:20]  # Next 10 embeddings
    skill_level_3 = trajs["emb"][20:30]  # Last 10 embeddings

    # Compute the Wasserstein distance between the distributions of different skill levels
    dist_1_2 = [wasserstein_distance(skill_level_1[:, i], skill_level_2[:, i]) for i in range(skill_level_1.shape[1])]
    dist_1_3 = [wasserstein_distance(skill_level_1[:, i], skill_level_3[:, i]) for i in range(skill_level_1.shape[1])]
    dist_2_3 = [wasserstein_distance(skill_level_2[:, i], skill_level_3[:, i]) for i in range(skill_level_1.shape[1])]

    # Average the distances across all dimensions
    average_dist_1_2 = np.mean(dist_1_2)
    average_dist_1_3 = np.mean(dist_1_3)
    average_dist_2_3 = np.mean(dist_2_3)

    print("Wasserstein Distance between skill level 1 and 2:", average_dist_1_2)
    print("Wasserstein Distance between skill level 1 and 3:", average_dist_1_3)
    print("Wasserstein Distance between skill level 2 and 3:", average_dist_2_3)


# with open('experts/CartPole-v1_1000.pkl', 'rb') as f:
#     data = pickle.load(f)
#%% --> Demos
    # check the lengths for official data
    lengths = [len(i) for i in trajs["emb"]]
    print('len', len(trajs["emb"]))
    
    emb = trajs["emb"]
    new_emb = []
    # for i in range(0, len(emb), 2):
    #     new_emb.append(emb[i])
    # print('new len:', len(new_emb))
    # trajs["emb"] = new_emb
    # save_file = 'cond/test_hopperEmb_tripple_dim10_kld1.pkl'
    # with open(save_file, 'wb') as f:
    #     pickle.dump(trajs, f)
    # emb_sample = emb[0]
    # print('emb sample',emb_sample)

    # mean = [i[0] for i in trajs["dist_params"]]
    # mean_sample = mean[0]
    # print('mean sample',mean_sample)

    # std = [i[1] for i in trajs["dist_params"]]
    # std_sample = std[0]
    # print('std sample',std_sample)
    # check the length for rollout
    # lengths = trajs['lengths']
    # print('lengths of trajs', lengths)

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
