import pandas as pd



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

def check_vectorization_impact(csv_filepath, target_opt_level='-O3'):
    # 1. Load data
    df = pd.read_csv(csv_filepath)

    # 2. Extract and clean parameters
    df['N'] = df['-DN=512'].apply(lambda x: int(str(x).split('=')[1]))
    df['Threads'] = df['OMP_NUM_THREADS']
    df.rename(columns={'-O0': 'Opt_Level'}, inplace=True)

    # Filter by the optimization level we want to investigate
    df = df[df['Opt_Level'] == target_opt_level]

    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return

    # 3. Get stable runtimes using the trimmed mean function
    grouped_df = compute_trimmed_mean(df, target_col='Result')

    # 4. Separate the data into Vectorized and Unvectorized
    # We set a multi-index so we can easily join them side-by-side
    index_cols = ['N', 'Threads', '-fopenmp']

    vec_df = grouped_df[grouped_df['-fno-vectorize'] == False].set_index(index_cols)
    unvec_df = grouped_df[grouped_df['-fno-vectorize'] == True].set_index(index_cols)

    # 5. Join them to compare the exact same configurations side-by-side
    compare_df = unvec_df[['Result']].join(vec_df[['Result']], lsuffix='_unvec', rsuffix='_vec')
    compare_df.rename(columns={'Result_unvec': 'Time_Unvectorized', 'Result_vec': 'Time_Vectorized'}, inplace=True)

    # Drop any rows that don't have a matching pair
    compare_df = compare_df.dropna()

    # 6. Calculate how much Vectorization sped up the code
    # Speedup > 1.0 means Vectorization helped. Speedup < 1.0 means it hurt.
    compare_df['Vec_Speedup'] = compare_df['Time_Unvectorized'] / compare_df['Time_Vectorized']

    # 7. Aggregate the summary to make it readable
    # We average the speedup across all thread counts for a given problem size and execution type
    summary = compare_df.groupby(['N', '-fopenmp'])['Vec_Speedup'].mean().reset_index()

    # 8. Print the Results
    print(f"\nAverage Vectorization Impact Analysis ({target_opt_level}):")
    print(f"{'Size (N)':<10} | {'Execution Type':<16} | {'Avg Speedup':<13} | {'Verdict'}")
    print("-" * 70)

    for _, row in summary.iterrows():
        n_val = int(row['N'])
        exec_type = "OpenMP" if row['-fopenmp'] else "Serial"
        speedup = row['Vec_Speedup']

        # Classify the impact (using a 5% margin of error for "No Difference")
        if speedup > 1.05:
            verdict = "Helped (Faster)"
        elif speedup < 0.95:
            verdict = "Hurt (Slower)"
        else:
            verdict = "No Meaningful Difference"

        print(f"{n_val:<10} | {exec_type:<16} | {speedup:<13.2f} | {verdict}")

    # Return the raw, unaggregated comparison dataframe in case you want to plot it
    return compare_df.reset_index()

if __name__ == "__main__":
    # Point this to your generated CSV
    impact_df = check_vectorization_impact("./data/gemm_omp_ikj.csv", target_opt_level='-O3')
    print_optimal_configurations("./data/gemm_omp_ikj.csv", target_opt_level='-O2')