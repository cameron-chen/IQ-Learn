# read the 25 trajs and list their states/return.
# read from experts/HalfCheetah-v2_25.pkl

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# with open('experts/HalfCheetah-v3_20_Gemini_2k+6k.pkl', 'rb') as f:
#     data = pickle.load(f)
# # with open('experts/HalfCheetah-v2_25.pkl', 'rb') as f:
# #     data = pickle.load(f)

# # read rewards and calculate average return
# returns = [sum(i) for i in data["rewards"][10:]]
# print('len', len(returns))
# print(f"max mean min: {max(returns)} {np.mean(returns)} {min(returns)}")

with open('cond/cheetah_20_dummy.pkl', 'rb') as f:
    data = pickle.load(f)

emb = data['emb']
print('len', len(emb))
print('shape', emb[0].shape)
print(f"value: {emb[0]}")


# print(f"value: {returns[0]}")

# states = data["states"]
# print('len', len(states[0]))
# print('shape', states[0][0].shape)
# print(f"value: {states[0][0]}")
 

# actions = data["actions"]
# print('len', len(actions[0]))
# print('shape', actions[0][0].shape)
# print(f"value: {actions[0][0]}")

exit()
new_actions = []
for i in range(len(actions)):
    list = []
    for j in range(len(actions[i])):
        item = np.array(actions[i][j][0],dtype=np.float32)
        list.append(item)
        # exit()
    new_actions.append(list)
print('len', len(new_actions[0]))
print('shape', new_actions[0][0].shape)

new_states = []
for i in range(len(states)):
    list = []
    for j in range(len(states[i])):
        item = np.array(states[i][j][0],dtype=np.float32)
        list.append(item)
        # exit()
    new_states.append(list)
print('len', len(new_states[0]))
print('shape', new_states[0][0].shape)

data["states"] = new_states
data["actions"] = new_actions
# dump the new data to /HalfCheetah-v3_long.pkl
with open('experts/HalfCheetah-v3_long.pkl', 'wb') as f:
    pickle.dump(data, f)
# plot the states and save in a file

# sort the states and list the index from high to low