# read the 25 trajs and list their rewards/return.
# read from experts/HalfCheetah-v2_25.pkl

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

with open('experts/HalfCheetah-v3_150000_steps_100_trajs.pkl', 'rb') as f:
    data = pickle.load(f)

for i in range(100):
    print('traj', i, 'reward', np.sum(data['rewards'][i]))

# plot the rewards and save in a file

plt.plot(np.sum(data['rewards'], axis=1))

# sort the rewards and list the index from high to low

sorted_rewards = np.sort(np.sum(data['rewards'], axis=1))
sorted_index = np.argsort(np.sum(data['rewards'], axis=1))
print(sorted_rewards)