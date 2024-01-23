# Step 1 : read from experts/HalfCheetah-v3_8.pkl and experts/HalfCheetah-v3_9.pkl
# Step 2: combine the two files into one file
# Step 3: save it in experrts/ HalfCheetah-v3_Gemini.pkl

# Step 1 : read from experts/HalfCheetah-v3_8.pkl and experts/HalfCheetah-v3_9.pkl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

with open('experts/HalfCheetah-v3_8.pkl', 'rb') as f:
    data1 = pickle.load(f)
with open('experts/HalfCheetah-v3_6_2327r.pkl', 'rb') as f:
    data2 = pickle.load(f)

# Step 2: combine the two files into one file
# for each file, take three items out of each value list and combine them into a new list
def combine_value(data1, data2):
    new_data = {}
    for key in data1.keys():
        new_data[key] = []
        new_data[key].extend(data1[key][:3])
        new_data[key].extend(data2[key][:3])
    return new_data

combined_data = combine_value(data1, data2)
# Step 3: save it in experts/ HalfCheetah-v3_Gemini.pkl
with open('experts/HalfCheetah-v3_Gemini_2k+6k.pkl', 'wb') as f:
    pickle.dump(combined_data, f)

