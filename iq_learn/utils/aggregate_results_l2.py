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

             # Calculate the mean of the 20 learnt returns for this trajectory
            learnt_mean_return = np.mean(returns[i])
            # Calculate the squared difference
            l2_norm.append(np.sqrt((learnt_mean_return - expert_returns[i]) ** 2))

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
    args = arg_parser.parse_args()

    # Get list of files matching the pattern in the specified directory
    files = glob.glob(os.path.join(args.dir, args.pattern))

    if not files:
        print(f"No files found matching pattern {args.pattern} in directory {args.dir}")
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

    # Aggregate results for all files collectively and calculate avg L2 norm
    avg, std = aggregate_results_and_calculate_l2(files, expert_returns)
    print(f"Average L2 norm between learnt and expert returns: {avg:.3f}+-{std:.3f}")

    # Prepare a DataFrame to hold aggregated results with formatted mean and std
    aggregated_df = pd.DataFrame({
        'Avg l2 norm': [avg],
        'Std l2 norm': [std]
    })

    # Save the aggregated results to a CSV file
    suffix = 's' if args.pattern.startswith('s') else 'd'
    aggregated_csv_path = os.path.join(args.dir, f'aggregated_l2_{suffix}.csv')
    aggregated_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated results saved to {aggregated_csv_path}")

if __name__ == '__main__':
    main()
