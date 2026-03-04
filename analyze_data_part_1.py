import pandas as pd
import matplotlib.pyplot as plt

def part_1(df:pd.DataFrame):
    """
    data sample head
    -fno-vectorize,-DN=512,-O0,Run_Iteration,Result
    True,-DN=512,-O0,1,1.872884
    True,-DN=512,-O1,1,0.392895
    True,-DN=512,-O2,1,0.147004
    True,-DN=512,-O3,1,0.147039

    """

    # filter for only -O2 runs
    df_filtered = df[df['-O0'] == '-O2']

    # convert n into a number from flag
    df_filtered['N'] = df_filtered.apply(lambda row : int(row["-DN=512"].split("=")[1]), axis=1)

    df_filtered['gflops_per_sec'] = df_filtered.apply(lambda row : (row['N'] ** 3 * 2 + row['N'] ** 2) / 1000000000 / row['Result'], axis=1)

    # plot values of flops and n on a 2d scatter plot
    # color based on whether '-fno-vectorize' column is true
    # 4. Plot values of flops and n on a 2d scatter plot
    plt.figure(figsize=(10, 6))

    # Separate the data based on the '-fno-vectorize' flag for coloring
    no_vec = df_filtered[df_filtered['-fno-vectorize'] == True]
    vec = df_filtered[df_filtered['-fno-vectorize'] == False]

    plt.scatter(no_vec['N'], no_vec['gflops_per_sec'], color='red', label='No Vectorize (True)', alpha=0.7)
    plt.scatter(vec['N'], vec['gflops_per_sec'], color='blue', label='Vectorized (False)', alpha=0.7)

    # Formatting the plot
    plt.title('Performance (GFLOPs) vs. Matrix Size (N) for -O2')
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('GFLOPs / sec')
    plt.legend(title='-fno-vectorize')
    plt.grid(True)

    # 5. Export plot to an image and data to CSV
    plt.savefig('./build/gflops_scatter_plot.png')
    plt.show() # Display the plot to the user

    #df_filtered.to_csv('./build/processed_o2_results.csv', index=False)
    print("Plot saved as PNG and data exported to CSV successfully.")
    # explot plot to csv

if __name__=="__main__":
    # load data
    df = pd.read_csv("./build/results_6c5c1a4c_4a2aa5f2.csv")

    part_1(df)