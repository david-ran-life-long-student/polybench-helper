import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def compute_trimmed_mean(df, target_col='Result', group_cols=None):
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

    if group_cols is None:
        group_cols = list(set(df.columns) - {target_col, 'Run_Iteration'})
        # In case Run_Iteration wasn't in columns, ensure it's not in group_cols
        group_cols = [c for c in group_cols if c != 'Run_Iteration']

    if not group_cols:
        return pd.DataFrame()

    # Group by the configuration columns and apply the custom function
    trimmed_df = df.groupby(group_cols)[target_col].agg(_trimmed_mean).reset_index()
    return trimmed_df

def print_optimal_configurations(df, size_col='N', time_col='Result', opt_col='Opt_Level', target_opt_level='-O3', omp_col='-fopenmp', vec_col='-fno-vectorize', threads_col='Threads'):
    """
    Prints the best configurations for each problem size at a specific optimization level.
    """
    df = df.copy()
    if opt_col in df.columns and target_opt_level:
        df = df[df[opt_col] == target_opt_level]

    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return pd.DataFrame()

    optimal_rows = []

    print(f"\nOptimal Binary Configurations for {target_opt_level}:")
    print(f"{'Size':<10} | {'Min Time (s)':<15} | {'Binary Version':<18} | {'Threads':<8} | {'Vectorized?'}")
    print("-" * 73)

    for n_val in sorted(df[size_col].unique()):
        subset = df[df[size_col] == n_val]
        if subset.empty:
            continue

        best_idx = subset[time_col].idxmin()
        best_row = subset.loc[best_idx]
        optimal_rows.append(best_row)

        exec_time = best_row[time_col]
        is_omp = "OpenMP" if omp_col in best_row and best_row[omp_col] else "Serial (No OMP)"
        threads = int(best_row[threads_col]) if omp_col in best_row and best_row[omp_col] and threads_col in best_row and pd.notna(best_row[threads_col]) else "N/A"
        is_vec = "Yes" if vec_col in best_row and not best_row[vec_col] else ("No" if vec_col in best_row else "Unknown")

        print(f"{n_val:<10} | {exec_time:<15.6f} | {is_omp:<18} | {str(threads):<8} | {is_vec}")

    return pd.DataFrame(optimal_rows)

def check_vectorization_impact(df, size_col='N', time_col='Result', opt_col='Opt_Level', target_opt_level='-O3', omp_col='-fopenmp', vec_col='-fno-vectorize', threads_col='Threads'):
    """
    Analyzes the speedup (or slowdown) caused by vectorization.
    """
    df = df.copy()
    if opt_col in df.columns and target_opt_level:
        df = df[df[opt_col] == target_opt_level]

    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return pd.DataFrame()

    if vec_col not in df.columns:
        print(f"Vectorization column '{vec_col}' not found.")
        return pd.DataFrame()

    # Get stable runtimes using the trimmed mean function
    grouped_df = compute_trimmed_mean(df, target_col=time_col)

    index_cols = [size_col]
    if threads_col in grouped_df.columns:
        index_cols.append(threads_col)
    if omp_col in grouped_df.columns:
        index_cols.append(omp_col)

    # Separate the data into Vectorized and Unvectorized
    vec_df = grouped_df[grouped_df[vec_col] == False].set_index(index_cols)
    unvec_df = grouped_df[grouped_df[vec_col] == True].set_index(index_cols)

    # Join them to compare the exact same configurations side-by-side
    compare_df = unvec_df[[time_col]].join(vec_df[[time_col]], lsuffix='_unvec', rsuffix='_vec')
    compare_df.rename(columns={f'{time_col}_unvec': 'Time_Unvectorized', f'{time_col}_vec': 'Time_Vectorized'}, inplace=True)
    compare_df = compare_df.dropna()

    if compare_df.empty:
        print("No matching vectorized/unvectorized pairs found.")
        return pd.DataFrame()

    # Calculate how much Vectorization sped up the code
    compare_df['Vec_Speedup'] = compare_df['Time_Unvectorized'] / compare_df['Time_Vectorized']

    group_by_cols = [size_col]
    if omp_col in compare_df.index.names:
        group_by_cols.append(omp_col)

    summary = compare_df.groupby(group_by_cols)['Vec_Speedup'].mean().reset_index()

    print(f"\nAverage Vectorization Impact Analysis ({target_opt_level}):")
    print(f"{'Size':<10} | {'Execution Type':<16} | {'Avg Speedup':<13} | {'Verdict'}")
    print("-" * 70)

    for _, row in summary.iterrows():
        n_val = int(row[size_col])
        exec_type = "OpenMP" if omp_col in row and row[omp_col] else "Serial"
        speedup = row['Vec_Speedup']

        if speedup > 1.05:
            verdict = "Helped (Faster)"
        elif speedup < 0.95:
            verdict = "Hurt (Slower)"
        else:
            verdict = "No Meaningful Difference"

        print(f"{n_val:<10} | {exec_type:<16} | {speedup:<13.2f} | {verdict}")

    return compare_df.reset_index()

def plot_3d_performance_surface(df, x_col='Threads', y_col='N', z_col='Result', opt_col='Opt_Level', target_opt_level='-O3', vec_col='-fno-vectorize', output_file='3d_performance_surface.png'):
    """
    Plots a 3D surface mapping Threads and Problem Size to Runtime.
    """
    df = df.copy()
    if opt_col in df.columns and target_opt_level:
        df = df[df[opt_col] == target_opt_level]
    if vec_col in df.columns:
        df = df[df[vec_col] == False]

    if df.empty:
        print("No data available to plot 3D surface after filtering.")
        return

    if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
        print(f"Required columns ({x_col}, {y_col}, {z_col}) not found in dataframe.")
        return

    grouped_df = df.groupby([y_col, x_col])[z_col].mean().reset_index()
    pivot_table = grouped_df.pivot(index=y_col, columns=x_col, values=z_col)

    x_data = pivot_table.columns.values
    y_data = pivot_table.index.values

    X, Y = np.meshgrid(x_data, y_data)
    Z = pivot_table.values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)

    ax.set_title('3D Performance Surface: Runtime vs. Threads & Problem Size', fontsize=14)
    ax.set_xlabel(x_col, fontsize=12, labelpad=10)
    ax.set_ylabel(y_col, fontsize=12, labelpad=10)
    ax.set_zlabel(z_col, fontsize=12, labelpad=10)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=z_col)
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved 3D surface plot to '{output_file}'")
    plt.show()

def plot_strict_speedup_scaling(df, size_col='N', time_col='Result', opt_col='Opt_Level', target_opt_level='-O2', omp_col='-fopenmp', vec_col='-fno-vectorize', threads_col='Threads', output_file='strict_speedup_scaling.png'):
    """
    Plots the absolute strong scaling speedup comparing parallel execution to a strict serial baseline.
    """
    df = df.copy()
    grouped_df = compute_trimmed_mean(df, target_col=time_col)

    if omp_col not in grouped_df.columns or vec_col not in grouped_df.columns:
        print(f"Required columns '{omp_col}' and/or '{vec_col}' missing for strict baseline comparison.")
        return

    # Baseline: No OpenMP and No Vectorization
    baseline_mask = (grouped_df[omp_col] == False) & (grouped_df[vec_col] == True)
    baseline_df = grouped_df[baseline_mask]

    if opt_col in baseline_df.columns:
        baseline_map = baseline_df.set_index([size_col, opt_col])[time_col].to_dict()
    else:
        baseline_map = baseline_df.set_index([size_col])[time_col].to_dict()

    # Parallel runs to evaluate
    parallel_mask = (grouped_df[omp_col] == True) & (grouped_df[vec_col] == False)
    if opt_col in grouped_df.columns and target_opt_level:
        parallel_mask &= (grouped_df[opt_col] == target_opt_level)

    parallel_df = grouped_df[parallel_mask].copy()

    def get_baseline(row):
        if opt_col in row:
            return baseline_map.get((row[size_col], row[opt_col]))
        return baseline_map.get(row[size_col])

    parallel_df['Speedup'] = parallel_df.apply(
        lambda row: get_baseline(row) / row[time_col] if get_baseline(row) else None,
        axis=1
    )

    parallel_df = parallel_df.dropna(subset=['Speedup'])

    if parallel_df.empty:
        print(f"No parallel data found for optimization level {target_opt_level} with matching baselines.")
        return

    plt.figure(figsize=(10, 6))
    unique_ns = sorted(parallel_df[size_col].unique())
    colors = cm.viridis(np.linspace(0, 1, len(unique_ns)))

    for i, n_val in enumerate(unique_ns):
        subset = parallel_df[parallel_df[size_col] == n_val].sort_values(by=threads_col)
        plt.plot(subset[threads_col], subset['Speedup'], marker='o', linewidth=2, color=colors[i], label=f"{n_val}")

    max_threads = parallel_df[threads_col].max()
    if pd.notna(max_threads):
        plt.plot([1, max_threads], [1, max_threads], color='red', linestyle='--', label='Ideal Speedup ($y=x$)')
        plt.xticks(range(1, int(max_threads) + 1))

    plt.title(f'Absolute Strong Scaling Speedup ({target_opt_level})', fontsize=14, pad=15)
    plt.xlabel('Number of Threads', fontsize=12, labelpad=10)
    plt.ylabel('Speedup ($T_{serial} / T_{parallel}$)', fontsize=12, labelpad=10)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title=size_col, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved strict speedup plot to '{output_file}'")
    plt.show()

def plot_gflops_scatter(df, size_col='N', time_col='Result', vec_col='-fno-vectorize', opt_col='Opt_Level', target_opt_level='-O2', output_file='gflops_scatter_plot.png'):
    """
    Plots GFLOPs vs Problem Size (N) separated by whether it is vectorized or not.
    Assumes standard matrix multiplication GFLOP calculation (2*N^3 + N^2).
    """
    df = df.copy()
    if opt_col in df.columns and target_opt_level:
        df = df[df[opt_col] == target_opt_level]

    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return

    # Compute GFLOPs
    df['gflops_per_sec'] = df.apply(lambda row: (row[size_col] ** 3 * 2 + row[size_col] ** 2) / 1e9 / row[time_col], axis=1)

    plt.figure(figsize=(10, 6))

    if vec_col in df.columns:
        no_vec = df[df[vec_col] == True]
        vec = df[df[vec_col] == False]
        plt.scatter(no_vec[size_col], no_vec['gflops_per_sec'], color='red', label='No Vectorize (True)', alpha=0.7)
        plt.scatter(vec[size_col], vec['gflops_per_sec'], color='blue', label='Vectorized (False)', alpha=0.7)
        plt.legend(title=vec_col)
    else:
        plt.scatter(df[size_col], df['gflops_per_sec'], color='blue', alpha=0.7)

    plt.title(f'Performance (GFLOPs) vs. Matrix Size (N) for {target_opt_level}')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('GFLOPs / sec')
    plt.grid(True)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved GFLOPs scatter plot to '{output_file}'")
    plt.show()

def plot_empirical_roofline(df, size_col='N', time_col='Result', threads_col='Threads', 
                            opt_col='Opt_Level', target_opt_level='-O3',
                            flop_func=None, theoretical_peak_gflops=None,
                            output_file='empirical_roofline.png'):
    """
    Plots an empirical roofline model representing Achieved GFLOP/s vs Problem Size (N).
    
    Since we can't reliably measure arithmetic intensity without source analysis tools, 
    this plots the performance ceiling bounded by either empirical peak or a given theoretical peak.
    
    If `theoretical_peak_gflops` is not provided, the maximum empirical GFLOP/s is used as the 'roof'.
    If `flop_func` is not provided, it defaults to standard Matrix Multiplication: 2 * N^3 + N^2.
    """
    df = df.copy()
    if opt_col in df.columns and target_opt_level:
        df = df[df[opt_col] == target_opt_level]
        
    if df.empty:
        print(f"No data found for optimization level {target_opt_level}.")
        return

    # Default FLOP calculator for typical O(N^3) algorithms like GEMM
    if flop_func is None:
        flop_func = lambda n: 2.0 * (n ** 3) + (n ** 2)

    # Compute Achieved GFLOP/s
    df['gflops_per_sec'] = df.apply(lambda row: flop_func(row[size_col]) / 1e9 / row[time_col], axis=1)
    
    # Compute trimmed mean for stability across runs
    grouped_df = compute_trimmed_mean(df, target_col='gflops_per_sec')
    if grouped_df.empty:
        return
        
    # Determine the absolute roofline bounds
    empirical_peak = df['gflops_per_sec'].max()
    peak_gflops = theoretical_peak_gflops if theoretical_peak_gflops else empirical_peak

    plt.figure(figsize=(10, 6))
    
    # Check if we have threading data to color distinct thread counts
    if threads_col in grouped_df.columns:
        unique_threads = sorted(grouped_df[threads_col].dropna().unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_threads)))
        
        for i, t_val in enumerate(unique_threads):
            # Sort by problem size to ensure lines draw cleanly
            subset = grouped_df[grouped_df[threads_col] == t_val].sort_values(by=size_col)
            if not subset.empty:
                plt.plot(subset[size_col], subset['gflops_per_sec'], marker='o', 
                         linewidth=2, color=colors[i], label=f"{int(t_val)} Threads")
    else:
        # Plot just N vs GFLOP/s without threading differentiation
        subset = grouped_df.sort_values(by=size_col)
        plt.plot(subset[size_col], subset['gflops_per_sec'], marker='o', linewidth=2, color='blue', label="Achieved")

    # Draw the Roofline
    plt.axhline(y=peak_gflops, color='red', linestyle='--', linewidth=2, 
                label=f'Peak Performance Roof ({peak_gflops:.1f} GFLOP/s)')
    
    # Formatting
    plt.title(f'Empirical Performance Roofline ({target_opt_level})', fontsize=14, pad=15)
    plt.xlabel('Problem Size (N)', fontsize=12, labelpad=10)
    plt.ylabel('Achieved Performance (GFLOP/s)', fontsize=12, labelpad=10)
    
    # Problem size is usually scaled logarithmically to clearly see asymptotic boundaries
    plt.xscale('log', base=2)
    
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Saved empirical roofline plot to '{output_file}'")
    plt.show()
