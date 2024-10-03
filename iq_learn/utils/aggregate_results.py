"""
This script aggregates return values from multiple CSV files and computes the mean and standard deviation for three levels of returns. Each CSV file contains 30 rows, with each row holding a list of 20 return values. The first 10 rows correspond to "Level 1", the next 10 to "Level 2", and the final 10 to "Level 3". The script reads multiple files matching a given pattern (e.g., 's*.csv' or 'd*.csv'), extracts the return lists, combines them by level, and calculates the mean and standard deviation for each level across the entire dataset.

Usage:
    python script_name.py <directory> <file_pattern>

Example:
    python script_name.py decoder/result/hopper "s*.csv"
    
Arguments:
    - directory: Path to the folder containing the CSV files.
    - file_pattern: Pattern for file names (e.g., 's*.csv', 'd*.csv').

"""
import argparse
import pandas as pd
import numpy as np
import ast
import glob
import os

def convert_to_list(str_list):
    """ Convert string representation of list to an actual list """
    return ast.literal_eval(str_list)

def aggregate_results(files):
    # Initialize lists to store combined returns for each level
    level_1_returns = []
    level_2_returns = []
    level_3_returns = []

    # Process each file to aggregate returns
    for file in files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        
        # Extract the Returns column (which are saved as lists)
        returns = df['Returns'].apply(convert_to_list)

        # Aggregate returns from each level (first 10 rows, next 10 rows, and last 10 rows)
        for i in range(30):  # Assuming 30 rows per file
            if i < 10:
                level_1_returns.extend(returns[i])
            elif i < 20:
                level_2_returns.extend(returns[i])
            else:
                level_3_returns.extend(returns[i])

    # Calculate the mean and std for each level
    level_1_mean = np.mean(level_1_returns)
    level_1_std = np.std(level_1_returns)

    level_2_mean = np.mean(level_2_returns)
    level_2_std = np.std(level_2_returns)

    level_3_mean = np.mean(level_3_returns)
    level_3_std = np.std(level_3_returns)

    return {
        'Level 1': {'Mean': level_1_mean, 'Std': level_1_std},
        'Level 2': {'Mean': level_2_mean, 'Std': level_2_std},
        'Level 3': {'Mean': level_3_mean, 'Std': level_3_std}
    }

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("dir", type=str, help="directory containing the CSV files")
    arg_parser.add_argument("pattern", type=str, help="file name pattern (e.g., 's*.csv' or 'd*.csv')")
    args = arg_parser.parse_args()

    # Get list of files matching the pattern in the specified directory
    files = glob.glob(os.path.join(args.dir, args.pattern))

    if not files:
        print(f"No files found matching pattern {args.pattern} in directory {args.dir}")
        return

    # Aggregate results for all files collectively
    result = aggregate_results(files)
    # Print overall mean and std from the aggregated results
    print("\nOverall Results:")
    for level in ['Level 1', 'Level 2', 'Level 3']:
        mean_value = round(result[level]['Mean'], 1)
        std_value = round(result[level]['Std'], 1)
        print(f"{level} -> Mean: {mean_value}, Std: {std_value}")


    # Prepare a DataFrame to hold aggregated results with formatted mean and std
    aggregated_df = pd.DataFrame({
        'Level': ['Level 1', 'Level 2', 'Level 3'],
        'Mean': [round(result['Level 1']['Mean'], 1), round(result['Level 2']['Mean'], 1), round(result['Level 3']['Mean'], 1)],
        'Std': [round(result['Level 1']['Std'], 1), round(result['Level 2']['Std'], 1), round(result['Level 3']['Std'], 1)]
    })

    # Save the aggregated results to a CSV file
        # Determine the suffix based on the file pattern
    suffix = 's' if args.pattern.startswith('s') else 'd'
    aggregated_csv_path = os.path.join(args.dir, f'aggregated_{suffix}.csv')
    aggregated_df.to_csv(aggregated_csv_path, index=False)
    print(f"Aggregated results saved to {aggregated_csv_path}")

if __name__ == '__main__':
    main()