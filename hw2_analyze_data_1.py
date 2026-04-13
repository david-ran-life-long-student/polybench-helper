
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_3d_performance_surface(csv_filepath):
    # 1. Load the data
    df = pd.read_csv(csv_filepath)

    # 2. Clean and extract parameters
    # The column name is literally '-DN=512' based on the first run, so we parse it
    df['N'] = df['-DN=512'].apply(lambda x: int(str(x).split('=')[1]))
    df['Threads'] = df['OMP_NUM_THREADS']

    # Optional: If your full dataset contains multiple optimization levels or
    # vectorization settings, filter down to a specific baseline to avoid mixing data.
    # For example, filtering for -O3 and vectorized = True:
    df = df[(df['-O0'] == '-O3') & (df['-fno-vectorize'] == False)]
    # df = df[df['N'] < 600]

    # 3. Group by N and Threads to average out the multiple 'Run_Iteration's
    grouped_df = df.groupby(['N', 'Threads'])['Result'].mean().reset_index()

    # 4. Pivot the data to create a 2D grid for the Z-axis (Runtime)
    # Rows will be 'N' (Y-axis), Columns will be 'Threads' (X-axis)
    pivot_table = grouped_df.pivot(index='N', columns='Threads', values='Result')

    # Extract X, Y, and Z axes for matplotlib
    x_data = pivot_table.columns.values  # Threads
    y_data = pivot_table.index.values    # Problem Size N

    # Create a 2D meshgrid for X and Y coordinates
    X, Y = np.meshgrid(x_data, y_data)
    Z = pivot_table.values               # Runtime (Result)

    # 5. Plotting
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # You can use plot_surface for a solid colored surface,
    # or contour3D for topological contour lines.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)

    # Alternative: 3D Contour lines
    # ax.contour3D(X, Y, Z, 50, cmap=cm.viridis)

    # Labeling
    ax.set_title('3D Performance Surface: Runtime vs. Threads & Problem Size', fontsize=14)
    ax.set_xlabel('Number of Threads', fontsize=12, labelpad=10)
    ax.set_ylabel('Problem Size (N)', fontsize=12, labelpad=10)
    ax.set_zlabel('Runtime (Seconds)', fontsize=12, labelpad=10)

    # Add a color bar to map colors to runtime values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Runtime (s)')

    # Adjust the viewing angle (elevation, azimuth)
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.savefig('3d_performance_surface.png', dpi=300)
    print("Saved 3D surface plot to '3d_performance_surface.png'")
    plt.show()

if __name__ == "__main__":
    # Replace with your actual CSV file name
    plot_3d_performance_surface("data/gemm_omp_ikj.csv")