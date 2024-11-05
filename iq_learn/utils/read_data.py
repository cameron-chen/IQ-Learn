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
    import numpy as np
    # make expert_location an argument
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("file", type=str, help="name of the file")
    args = arg_parser.parse_args()
    expert_location = args.file
    print(f"Reading file {expert_location}")
    with open(expert_location, 'rb') as f:
        trajs = read_file(expert_location, f)
    
    print(trajs.keys())
    

#%% --> Print the mean std of rewards for [0:10][10:20][20:30]
    # print the mean std of rewards for [0:10][10:20][20:30]
    # with open(file, 'rb') as f:
    #     trajs = read_file(file, f)
    # rewards.extend(trajs['rewards'])
    # rewards = []
    # for file in expert_location.split(','):
    #     with open(file, 'rb') as f:
    #         trajs = read_file(file, f)
    #     rewards.extend(trajs['rewards'])
    # for i in range(3):
    #     temp = rewards[i * 100:(i + 1) * 100]
    #     total_rewards = [sum(reward) for reward in temp]
    #     avg = np.mean(total_rewards)
    #     std = np.std(total_rewards)
    #     print(f"Skill level {i + 1}: {avg:.1f}±{std:.1f}")
    #     print(f"Latex format: ${avg:.1f} {{\\text{{\scriptsize $\pm {std:.1f}$}}}}$")

    
    # # print the keys of the trajs
    # print(f"Keys of trajs: {trajs.keys()}")
    
    # # print the rewards when 1=dones for each traj
    # dones = trajs['dones']
    # rewards = trajs['rewards']
    # for i in [1,10,20]:
    #     for j in range(len(dones[i])):
    #         if dones[i][j] == 1:
    #             print(f"Traj {i} has reward {rewards[i][j]}")
    
    # make the trajs['emb'] 's every value to 0
#%% --> Print the keys of the trajs
    # new = []
    # for i in range(100):
    #     new.append(np.zeros(trajs['emb'][0].shape))
    # trajs['emb'] = new
    # print(len(trajs['emb']))
    # # save the modified data to a new file
    # save_file = 'cond/kitchen/no_id/identical_cond.pkl'
    # with open(save_file, 'wb') as f:
    #     pickle.dump(trajs, f)

    # # print the max min mean of returns
    # returns = [sum(i) for i in trajs["rewards"]]
    # print(f"Cumulated return max mean min: {max(returns)} {np.mean(returns)} {min(returns)}")

    # max_returns = [max(i) for i in trajs["rewards"]]
    # print(f"Max reward max mean min: {max(max_returns)} {np.mean(max_returns)} {min(max_returns)}")
    # # calculate the number of different values in max_returns
    # unique_values = np.unique(max_returns)
    # print(f"Number of unique values in max_returns: {len(unique_values)}")
    # # count the number of occurences of each unique value
    # occurences = {value: max_returns.count(value) for value in unique_values}
    # print(f"Occurences of each unique value in max_returns: {occurences}")
    # return

    # # truncate each value length to 30:
    # for key in trajs:
    #     trajs[key] = trajs[key][:30]
    # # save the truncated data to a new file

    # save_file = 'experts/kitchen/kitchen_30.pkl'
    # with open(save_file, 'wb') as f:
    #     pickle.dump(trajs, f)
    # print(f"Saved the truncated data to {save_file}")

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
    import numpy as np
    import ot
    # Assume trajs["emb"] is a numpy array with shape (30, 10), where the first 10 rows are for skill level 1,
    # the next 10 rows are for skill level 2, and the last 10 rows are for skill level 3.

    # Split the embeddings based on skill levels
    skill_level_1 = trajs["emb"][:20]  # First 10 embeddings
    skill_level_2 = trajs["emb"][100:120]  # Next 10 embeddings
    skill_level_3 = trajs["emb"][200:220]  # Last 10 embeddings
    vectors = [skill_level_1, skill_level_2, skill_level_3]
    print("\na: (10, 10) 10 trajectories embeddings, each of dim 10")
    print("\n Measure between levels")
    dist_between_levels = []
    for level in range(3):

        n = 50  # nb samples

        # mu_s = np.array([0, 0])
        # cov_s = np.array([[1, 0], [0, 1]])

        # mu_t = np.array([4, 4])
        # cov_t = np.array([[1, -.8], [-.8, 1]])

        # xs = ot.datasets.make_2D_samples_gauss(n, mu_s, cov_s)
        # xt = ot.datasets.make_2D_samples_gauss(n, mu_t, cov_t)
        xs = vectors[(level)%3][:10]
        xt = vectors[(level+1)%3][:10]

        # xs = vectors[(level)%3][:5]
        # xt = vectors[(level)%3][5:]

        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        # loss matrix
        M = ot.dist(xs, xt)

        # EMD
        Wd = ot.emd2(a, b, M)

        first_level = (level)%3
        second_level = (level+1)%3
        # print(f'xs: {xs.shape}, xt: {xt.shape}, Wd: {Wd}')
        print(f'Distance between skill level {first_level} and {second_level}: {Wd:.2f}')
        dist_between_levels.append(Wd)

    print("\n Measure within level")
    dist_within_levels = []
    for level in range(3):

        n = 50  

        xs = vectors[(level)%3][:10]
        xt = vectors[(level)%3][10:20]

        a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

        # loss matrix
        M = ot.dist(xs, xt)

        # EMD
        Wd = ot.emd2(a, b, M)

        first_level = (level)%3
        second_level = (level+1)%3
        # print(f'xs: {xs.shape}, xt: {xt.shape}, Wd: {Wd}')
        print(f'Distance within skill level {first_level}: {Wd:.2f}')
        dist_within_levels.append(Wd)

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sample confusion matrix data (3x3)
    confusion_matrix = np.array([[dist_within_levels[0], 0, 0],
                                [dist_between_levels[0], dist_within_levels[1], 0],
                                [dist_between_levels[2], dist_between_levels[1], dist_within_levels[2]]])

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(confusion_matrix, dtype=bool), k=1)  # k=1 ignores diagonal

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 7))
    labels = ['Low', 'Medium', 'Expert']
    fontsize = 20
    # Draw the heatmap
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt=".2f", cmap='OrRd', mask=mask,
            cbar=True, linewidths=0.5, vmin=0, vmax=10, xticklabels=labels, yticklabels=labels, annot_kws={"size": fontsize})

    # Add titles and labels
    # Add title and adjust title font size
    plt.title('Walker2D', fontsize=fontsize)  # Set title font size here

    # Set x and y axis labels with specific font size
    # plt.xlabel('Ability Level', fontsize=fontsize)  # Optional: Add this line if you need an x-label
    # plt.ylabel('Ability Level', fontsize=fontsize)  # Optional: Add this line if you need a y-label
    # Set the font size for ticks
    plt.tick_params(axis='both', labelsize=fontsize)  # Set the ticks font size
    # Adjust the color bar properties
    colorbar = heatmap.collections[0].colorbar  # Access the color bar
    colorbar.ax.tick_params(labelsize=fontsize)  # Set the ticks font size for the color bar
    # colorbar.set_label('Legend', fontsize=fontsize)  # Set the color bar label and font size
    # plt.xlabel('Ability Level')
    # plt.ylabel('Ability Level')

    # # Adjust axes limits to create a staircase effect
    # plt.xlim(-0.5, 2.5)
    # plt.ylim(2.5, -0.5)

    # Show the plot
    plt.show()
    # savepic = 'utils/policy_dif/hopper_confusion_matrix.png'
    savepic = 'utils/policy_dif/walker_confusion_matrix.png'
    # savepic = 'utils/policy_dif/cheetah_confusion_matrix.png'

    plt.savefig(savepic)
    print(f"Saved to {savepic}")

#%% --> Demos Deprecated policy difference

    # print("\nMeasure between each vector: a[i,:] and b[i,:]")
    # # Compute the Wasserstein distance between the distributions of different skill levels
    # for level in range(3):
    #     dists = []
    #     for i in range(10):
    #         for j in range(10):
    #             dists.append(wasserstein_distance(vectors[level][i], vectors[(level+1)%3][j]))
    #     avg_dist = np.mean(dists)
    #     first_level = (level+1)%4
    #     second_level = (level+2)%4
    #     print(f"Distance between skill level {first_level} and {second_level}: {avg_dist:.2f}")

    # # measure for [:,i]    
    # print("\nMeasure for each dimension: a[:,i] and b[:,i]")
    # dist_1_2 = [wasserstein_distance(skill_level_1[:, i], skill_level_2[:, i]) for i in range(skill_level_1.shape[1])]
    # dist_1_3 = [wasserstein_distance(skill_level_1[:, i], skill_level_3[:, i]) for i in range(skill_level_1.shape[1])]
    # dist_2_3 = [wasserstein_distance(skill_level_2[:, i], skill_level_3[:, i]) for i in range(skill_level_1.shape[1])]

    # # Average the distances across all dimensions
    # average_dist_1_2 = np.mean(dist_1_2)
    # average_dist_1_3 = np.mean(dist_1_3)
    # average_dist_2_3 = np.mean(dist_2_3)

    # print(f"Distance between skill level 1 and 2: {average_dist_1_2:.2f}", )
    # print(f"Distance between skill level 2 and 3: {average_dist_2_3:.2f}", )
    # print(f"Distance between skill level 3 and 1: {average_dist_1_3:.2f}", )
    


# with open('experts/CartPole-v1_1000.pkl', 'rb') as f:
#     data = pickle.load(f)
#%% --> Demos
    # check the lengths for official data
    # lengths = [len(i) for i in trajs["emb"]]
    # print('len', len(trajs["emb"]))
    
    # emb = trajs["emb"]
    # new_emb = []
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
