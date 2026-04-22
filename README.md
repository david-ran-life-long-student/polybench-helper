# Polybench-Helper

An automated experiment runner and analysis toolkit designed for evaluating and benchmarking C programs, specifically tailored around the PolyBench/C test suite.

## PolyBench/C Setup
Before using this tool, you must have the PolyBench/C test suite downloaded and accessible in your workspace (e.g., `polybench-c-4.2.1-beta/`).
- **Download & Documentation**: You can download the suite and read the official documentation at the [PolyBench/C Github](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1).

## Overview

Polybench-Helper streamlines the process of running extensive compiler and runtime parameter studies. It automatically generates all combinations of your experimental parameters, compiles the required binaries in parallel, executes the benchmarks systematically to avoid systemic biases, and provides analysis utilities for the collected results.

## Features

- **Parameter Definitions (`Mutable`):** Define parameters across three modes:
  - `compiler_flag`: Both boolean toggles (e.g., `-fno-vectorize` on/off) and value options (e.g., `-O0`, `-O2`, `-O3`).
  - `env_var`: Runtime environment variables (e.g., OpenMP thread counts).
  - `runtime_arg`: Command-line arguments passed to the executable.
- **Deterministic Builds & Caching (`Study`):**
  - All configurations are compiled in parallel.
  - Binaries are named using MD5 fingerprints of the compiler flags to cache and reuse builds across runs.
- **Robust Execution:**
  - Runs experiments in a round-robin "sweep" fashion to mitigate time-dependent systemic biases.
  - Automatically checkpoints data to CSV files after each full sweep. If interrupted, the suite can resume where it left off.
- **Data Analysis (`analysis_helper.py`):**
  - `compute_trimmed_mean`: Removes outliers (highest and lowest) for robust execution time aggregation.
  - `check_vectorization_impact`: Easily compare the execution time between vectorized and non-vectorized builds.
  - `print_optimal_configurations`: Displays the best performing setups for different problem sizes.
  - `plot_3d_performance_surface`: Generates 3D plots mapping variables like Threads and Problem Size to Runtime.

## Usage

The project centers around defining a `Study` in an experiment script.

### 1. Define the Base Environment
Set up your base compiler command (including your target C files and include directories).

### 2. Define Experimental Parameters
Use the `Mutable` class to specify the variables you want to test. 

```python
from experiment_helper import Study, Mutable

base_compile_command = "-I polybench-c-4.2.1-beta/utilities polybench-c-4.2.1-beta/utilities/polybench.c target_kernel.c -DPOLYBENCH_TIME"

study = Study(
    build_dir="build",
    experimental_params=[
        Mutable([f"-DN={i}" for i in [512, 1024, 2048]]), # Problem sizes
        Mutable(["-O0", "-O2", "-O3", "-Ofast"]),         # Optimization levels
        Mutable(["-fno-vectorize"])                       # Vectorization toggle (True/False automatically implied)
    ],
    base_compiler_command=base_compile_command,
    base_env_vars={},
    compiler="clang"
)
```

### 3. Build and Run
First, compile all the required binary permutations, then execute the benchmarks.

```python
# Compiles all configurations in parallel
study.ensure_all_builds_exist()

# Runs the benchmarks and saves results to a CSV in the build directory
study.run_experiments()
```

### 4. Analyze Results
Use the functions provided in `analysis_helper.py` to parse the resulting CSV files and generate insights or plots. Examples of usage can be found in the provided `hw*_analyze_data_*.py` files.

## Project Structure

- `experiment_helper.py`: Core logic for `Mutable` and `Study` classes, handling compilation and execution.
- `analysis_helper.py`: Functions for data manipulation, outlier removal, and plotting via Pandas and Matplotlib.
- `hw*_experiment.py`: Example scripts demonstrating how to define and run a study.
- `hw*_analyze_data_*.py`: Example scripts demonstrating how to load results and run analyses.
- `data/`: Directory containing generated plots and output CSVs.

## Requirements
- Python 3
- Pandas
- Numpy
- Matplotlib
- A C compiler (e.g., Clang or GCC)

## Note
The `hw*` scripts are intended as specific use-case examples of the framework and aren't meant to be reused directly, but they serve as an excellent starting point for learning how to use the `Study` API and `analysis_helper` tools.
