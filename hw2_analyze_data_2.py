import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_trimmed_mean(df, target_col='Result'):
    """
    Groups the dataframe by the specified columns, removes the single highest
    and single lowest value in the target column for each group,
    and calculates the average of the remaining values.
    """
    def _trimmed_mean(series):
        # If a group somehow has 2 or fewer runs, dropping the max and min
        # would leave no data. Safely fall back to a regular mean.
        if len(series) <= 2:
            return series.mean()

        # Sort the series, slice off the first (min) and last (max) elements,
        # and compute the mean of the rest.
        return series.sort_values().iloc[1:-1].mean()

    group_cols = list(set(df.columns) - {target_col, 'Run_Iteration'})

    # Group by the configuration columns and apply the custom function
    trimmed_df = df.groupby(group_cols)[target_col].agg(_trimmed_mean).reset_index()

    return trimmed_df

def plot_strict_speedup_scaling(csv_filepath, target_opt_level='-O2'):
    # 1. Load data
    df = pd.read_csv(csv_filepath)

    # 3. Average the Iterations
    # Group by all configuration parameters to get the stable mean runtime
    grouped_df = compute_trimmed_mean(df)

    # 2. Extract and clean parameters
    # Parse the exact integer N from the '-DN=...' string
    grouped_df['N'] = grouped_df['-DN=512'].apply(lambda x: int(str(x).split('=')[1]))
    grouped_df['Threads'] = grouped_df['OMP_NUM_THREADS']

    # Standardize the optimization column name
    grouped_df.rename(columns={'-O0': 'Opt_Level'}, inplace=True)

    # 4. Isolate the Strict Serial Baseline
    # Baseline: No OpenMP (-fopenmp == False) and No Vectorization (-fno-vectorize == True)
    baseline_mask = (grouped_df['-fopenmp'] == False) & (grouped_df['-fno-vectorize'] == True)
    baseline_df = grouped_df[baseline_mask]

    # Create a dictionary mapping the tuple (N, Opt_Level) to the Baseline Runtime
    baseline_map = baseline_df.set_index(['N', 'Opt_Level'])['Result'].to_dict()

    # 5. Filter for the parallel runs we want to evaluate
    parallel_mask = (
            (grouped_df['-fopenmp'] == True) &
            (grouped_df['Opt_Level'] == target_opt_level) &
            (grouped_df['-fno-vectorize'] == False) # <-- Added this filter!
    )
    parallel_df = grouped_df[parallel_mask].copy()

    # 6. Compute Absolute Speedup: T_serial / T_parallel
    parallel_df['Speedup'] = parallel_df.apply(
        lambda row: baseline_map.get((row['N'], row['Opt_Level'])) / row['Result']
        if (row['N'], row['Opt_Level']) in baseline_map else None,
        axis=1
    )

    # Drop any rows where we didn't have a matching strict baseline
    parallel_df = parallel_df.dropna(subset=['Speedup'])

    if parallel_df.empty:
        print(f"No parallel data found for optimization level {target_opt_level} with matching baselines.")
        return

    # 7. Plotting (Pure Matplotlib)
    plt.figure(figsize=(10, 6))

    # Get unique problem sizes to loop through
    unique_ns = sorted(parallel_df['N'].unique())

    # Generate distinct colors for each 'N' using the viridis colormap
    colors = cm.viridis(np.linspace(0, 1, len(unique_ns)))

    # Plot each problem size as a separate line
    for i, n_val in enumerate(unique_ns):
        # Isolate data for this specific N and sort by Threads to ensure the line draws left-to-right
        subset = parallel_df[parallel_df['N'] == n_val].sort_values(by='Threads')

        plt.plot(
            subset['Threads'],
            subset['Speedup'],
            marker='o',
            linewidth=2,
            color=colors[i],
            label=f"{n_val}"  # Used for the legend
        )

    # Add the "Ideal Speedup" reference line (y = x)
    max_threads = parallel_df['Threads'].max()
    plt.plot(
        [1, max_threads], [1, max_threads],
        color='red', linestyle='--', label='Ideal Speedup ($y=x$)'
    )

    # 8. Formatting
    plt.title(f'Absolute Strong Scaling Speedup ({target_opt_level})', fontsize=14, pad=15)
    plt.xlabel('Number of Threads', fontsize=12, labelpad=10)
    plt.ylabel('Speedup ($T_{serial} / T_{parallel}$)', fontsize=12, labelpad=10)

    # Formatting the grid and axes
    plt.xticks(range(1, int(max_threads) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)

    # Move the legend outside the plot area
    plt.legend(title='Problem Size (N)', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the output
    plt.tight_layout()
    output_filename = f'strict_speedup_scaling_{target_opt_level.strip("-")}.png'
    plt.savefig(output_filename, dpi=300)
    plt.show()
    print(f"Saved strict speedup plot to '{output_filename}'")

def print_optimal_configurations(csv_filepath, target_opt_level='-O3'):
    # 1. Load data
    df = pd.read_csv(csv_filepath)

    df = compute_trimmed_mean(df)

    # 2. Extract and clean parameters
    df['N'] = df['-DN=512'].apply(lambda x: int(str(x).split('=')[1]))
    df['Threads'] = df['OMP_NUM_THREADS']

    # Standardize the optimization column name
    df.rename(columns={'-O0': 'Opt_Level'}, inplace=True)

    # Filter by a specific optimization level to ensure a fair comparison
    df = df[df['Opt_Level'] == target_opt_level]

    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return

    optimal_rows = []

    # 4. Print Table Header
    print(f"\nOptimal Binary Configurations for {target_opt_level}:")
    print(f"{'Size (N)':<10} | {'Min Time (s)':<15} | {'Binary Version':<18} | {'Threads':<8} | {'Vectorized?'}")
    print("-" * 73)

    # 5. Find the best configuration for each problem size N
    for n_val in sorted(df['N'].unique()):

        # Isolate all permutations tested for this specific matrix size
        subset = df[df['N'] == n_val]

        # Find the index of the absolute minimum execution time
        best_idx = subset['Result'].idxmin()
        best_row = subset.loc[best_idx]

        optimal_rows.append(best_row)

        # Format the output variables
        exec_time = best_row['Result']
        is_omp = "OpenMP" if best_row['-fopenmp'] else "Serial (No OMP)"

        # If it's the serial version, threading doesn't matter (tool usually defaults to 1)
        threads = int(best_row['Threads']) if best_row['-fopenmp'] else "N/A"

        # Interpret the inverted flag (False = Vectorized)
        is_vec = "Yes" if not best_row['-fno-vectorize'] else "No"

        print(f"{n_val:<10} | {exec_time:<15.6f} | {is_omp:<18} | {str(threads):<8} | {is_vec}")

    # Return the filtered dataframe in case you want to plot it later
    return pd.DataFrame(optimal_rows)

if __name__ == "__main__":
    # Point this to your generated CSV
    print_optimal_configurations("./data/gemm_omp_ikj.csv", target_opt_level='-O2')
    plot_strict_speedup_scaling("./data/gemm_omp_ikj.csv", target_opt_level='-O0')
