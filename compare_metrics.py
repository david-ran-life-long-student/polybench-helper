import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Import compute_trimmed_mean from analysis_helper in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
from analysis_helper import compute_trimmed_mean

def extract_size(n_str):
    if pd.isna(n_str):
        return 0
    match = re.search(r'-DN=(\d+)', str(n_str))
    if match:
        return int(match.group(1))
    return n_str

def main(df1_path, df2_path, metric_name):

    label1 = "Eval Lab"
    label2 = "gemm ikj"
    output_png = f"./data/compare_{metric_name}.png"

    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    if metric_name not in df1.columns or metric_name not in df2.columns:
        print(f"Error: Metric '{metric_name}' not found in one of the dataframes.")
        print("Available metrics df1:", df1.columns.tolist())
        print("Available metrics df2:", df2.columns.tolist())
        sys.exit(1)
        
    # Extract integer problem size for sorting and x-axis plotting
    if 'N' in df1.columns:
        df1['ProblemSize'] = df1['N'].apply(extract_size)
    else:
        print("Warning: Column 'N' not found in first dataframe. Using index as ProblemSize.")
        df1['ProblemSize'] = df1.index
        
    if 'N' in df2.columns:
        df2['ProblemSize'] = df2['N'].apply(extract_size)
    else:
        print("Warning: Column 'N' not found in second dataframe. Using index as ProblemSize.")
        df2['ProblemSize'] = df2.index
    
    # Compute trimmed mean
    df1_mean = compute_trimmed_mean(df1, target_col=metric_name, group_cols=['ProblemSize']).reset_index()
    df2_mean = compute_trimmed_mean(df2, target_col=metric_name, group_cols=['ProblemSize']).reset_index()
    
    # Sort by ProblemSize
    df1_mean = df1_mean.sort_values('ProblemSize')
    df2_mean = df2_mean.sort_values('ProblemSize')
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(df1_mean['ProblemSize'], df1_mean[metric_name], label=label1, marker='o')
    plt.plot(df1_mean['ProblemSize'], df1_mean[metric_name], linestyle='-', alpha=0.5)
    
    plt.scatter(df2_mean['ProblemSize'], df2_mean[metric_name], label=label2, marker='x')
    plt.plot(df2_mean['ProblemSize'], df2_mean[metric_name], linestyle='--', alpha=0.5)
    
    plt.xlabel('Problem Size (N)')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} over Problem Size')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")

if __name__ == '__main__':
    for each in ['IPC','vIPC','%L1m','%L2m','L1AC','vL1AC','%Stall']:
        main("./data/eval_lab_result.csv", "./data/better_lab_result.csv", each)
