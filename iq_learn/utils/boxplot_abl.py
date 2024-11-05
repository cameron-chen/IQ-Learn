import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import pandas as pd
import numpy as np
import ast
import glob
import os
import torch
import pickle
from typing import IO, Any, Dict

def convert_to_list(str_list):
    """ Convert string representation of list to an actual list """
    return ast.literal_eval(str_list)

def aggregate_results_and_calculate_l2(files, expert_returns):

    l2_norm = []
    # Process each file to aggregate returns
    for file in files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        
        # Extract the Returns column (which are saved as lists)
        returns = df['Returns'].apply(convert_to_list)

        # Aggregate returns from each level (first 10 rows, next 10 rows, and last 10 rows)
        for i in range(30):  # Assuming 30 rows per file
            for ret in returns[i]:
                dif = np.sqrt((ret - expert_returns[i]) ** 2 / expert_returns[i])
                l2_norm.append(dif)
            #  # Calculate the mean of the 20 learnt returns for this trajectory
            # learnt_mean_return = np.mean(returns[i])
            # # Calculate the squared difference
            # l2_norm.append(np.sqrt((learnt_mean_return - expert_returns[i]) ** 2))

     # Finally, return the square root of the mean squared difference (i.e., L2 norm)
    return np.mean(l2_norm), np.std(l2_norm)

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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("expert_loc", type=str, help="expert file")
    arg_parser.add_argument("dir", type=str, help="directory containing the CSV files")
    arg_parser.add_argument("pattern", type=str, help="file name pattern (e.g., 's*.csv' or 'd*.csv')")
    arg_parser.add_argument("pattern2", type=str, help="file name pattern (e.g., 's*.csv' or 'd*.csv')")

    args = arg_parser.parse_args()
    files = glob.glob(os.path.join(args.dir, args.pattern))
    if not files:
        print(f"No files found matching pattern {args.pattern} in directory {args.dir}")
        return
    files2 = glob.glob(os.path.join(args.dir, args.pattern2))
    if not files2:
        print(f"No files found matching pattern {args.pattern2} in directory {args.dir}")
        return

    # Assuming conds["rewards"] is a list of lists where each element contains 1000 rewards
    # For this example, we'll just simulate expert returns as 30 random trajectories with 1000 rewards each.
    # Replace this with your actual expert returns in practice.
    experts = None
    if os.path.isfile(args.expert_loc):
        # Load data from single file.
        with open(args.expert_loc, 'rb') as f:
            experts = read_file(args.expert_loc, f)
    else:
        raise ValueError(f"Expert file {args.expert_loc} not found") 
    expert_returns = [np.sum(traj) for traj in experts["rewards"]]
    
    file_flat = [[] for i in range(3)]
    # Process each file to aggregate returns
    for file in files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        
        # Extract the Returns column (which are saved as lists)
        returns = df['Returns'].apply(convert_to_list)

        for j in range(3):
            # Aggregate returns from each level (first 10 rows, next 10 rows, and last 10 rows)
            for i in range(j*10, (j+1)*10):  # Assuming 30 rows per file
                for ret in returns[i]:
                    file_flat[j].append(ret)
    print(len(file_flat[0]))

    file_flat2 = [[] for i in range(3)]
    # Process each file to aggregate returns
    for file in files2:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        
        # Extract the Returns column (which are saved as lists)
        returns = df['Returns'].apply(convert_to_list)

        for j in range(3):
            # Aggregate returns from each level (first 10 rows, next 10 rows, and last 10 rows)
            for i in range(j*10, (j+1)*10):  # Assuming 30 rows per file
                for ret in returns[i]:
                    file_flat2[j].append(ret)
    print(len(file_flat2[0]))
    # Set up the matplotlib figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Create a 1x3 grid of subplots

    labels = ['Low', 'Medium', 'Expert']
    # Loop through the subplots
    fontsize = 18
    for i, ax in enumerate(axes):
        # Prepare data for the current subplot
        data = [file_flat[i], file_flat2[i]]
        
        # Create a boxplot for the current subplot
        sns.boxplot(data=data, ax=ax, palette="Set2")
        
        # Set title and labels
        ax.set_title(f'{labels[i]}', fontsize=fontsize+2)
        ax.set_xticklabels(['Skill Pooling', 'VTE'], fontsize=fontsize)
        if i==0:
            ax.set_ylabel('Rewards', fontsize=fontsize)

        # Add a main title for the entire figure
        # plt.suptitle('Comparison of File Flat and File Flat 2', fontsize=20)

        # Show the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the main title
        plt.show()
    # save it to plot/boxplot_abl.pdf
    savepic = "utils/plot/boxplot_abl.pdf"
    plt.savefig(savepic)
    print(f"Saved to {savepic}")

if __name__ == "__main__":
    main()