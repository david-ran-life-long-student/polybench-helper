"""
This file is created by Gemini under the direction of dran on 2/24/26
"""
import time

import pandas as pd
import itertools
import subprocess
import concurrent.futures
import os
import hashlib

RUNS_PER_EXPERIMENT = 10
HASH_LENGTH = 8
DEBUG = False

class Mutable:
    """
    This object defines a parameter of the experiments.
    Example:
    Mutable([10, 100, 1000], name="N", mode="runtime_arg") # Runtime arg
    Mutable([-ftree-vectorize]) # Boolean compiler flag
    Mutable(["-O0", "-O2", "-O3"]) # Non-boolean compiler flag
    """
    allowed_modes = ["env_var", "compiler_flag", "runtime_arg"]
    def __init__(self, value_range, name=None, mode="compiler_flag"):
        self.is_compiler_flag = mode == "compiler_flag"
        self.is_env_var = mode == "env_var"

        if mode not in self.allowed_modes:
            raise ValueError(f"mode: {mode} not in the expected list: {self.allowed_modes} ")

        if self.is_compiler_flag and name is None:
            self.name = str(value_range[0])
        else:
            self.name = name

        if len(value_range) == 1 and self.is_compiler_flag:
            self.value_range = [True, False]
        else:
            self.value_range = value_range

    def is_bool_flag(self):
        # Updated logic: A boolean flag is one whose range consists of True/False toggles
        if not self.is_compiler_flag or not self.value_range:
            return False
        return all(isinstance(x, bool) for x in self.value_range)


class Study:
    """
    This object models a group of experiments.
    """
    def __init__(self, build_dir, experimental_params, base_compiler_command, base_env_vars, compiler="clang", result_parser_func=None):
        self.build_dir = build_dir
        self.compiler = compiler
        self.base_compiler_flags = base_compiler_command
        self.base_env_vars = base_env_vars # this is a dict of string name to string value mapping
        self.result_parser_func = result_parser_func or self.default_result_parser

        # Sort input to guarantee deterministic order for execution and hashing
        experimental_params = sorted(experimental_params, key=lambda x: x.name)

        self.compiler_bool_flags = [p for p in experimental_params if p.is_bool_flag()]
        self.compiler_non_bool_flags = [p for p in experimental_params if p.is_compiler_flag and not p.is_bool_flag()]
        self.runtime_env_vars = [p for p in experimental_params if p.is_env_var]

        # Ensure runtime args don't accidentally capture env vars
        self.runtime_args = [p for p in experimental_params if not p.is_compiler_flag and not p.is_env_var]

        # Fixed typo: passing correct lists to their respective fingerprint functions
        self.compiler_bool_flags_fingerprint = self.get_compiler_bool_flag_fingerprint(self.compiler_bool_flags)
        self.compiler_non_bool_flags_fingerprint = self.get_compiler_num_flag_fingerprint(self.compiler_non_bool_flags)

        self.result_df = self.make_empty_dataframe()

        # Ensure build directory exists
        os.makedirs(self.build_dir, exist_ok=True)

    def make_empty_dataframe(self):
        """Returns an empty dataframe with columns mapped to the parameter names."""
        cols = [p.name for p in self.compiler_bool_flags + self.compiler_non_bool_flags + self.runtime_args + self.runtime_env_vars]
        cols += ["Run_Iteration", "Result"]
        return pd.DataFrame(columns=cols)

    def ensure_all_builds_exist(self):
        """Compiles all permutations of compiler flags in parallel if the executable is not already cached."""
        bool_ranges = [f.value_range for f in self.compiler_bool_flags]
        non_bool_ranges = [f.value_range for f in self.compiler_non_bool_flags]

        # Collect all the build commands we need to run
        build_tasks = []

        # Cartesian product of all compiler flag combinations
        for bool_combo in itertools.product(*bool_ranges):
            for non_bool_combo in itertools.product(*non_bool_ranges):

                # Construct the boolean bits string (e.g., '101' for True, False, True)
                try:
                    bool_bits = int("".join(["1" if b else "0" for b in bool_combo]), 2)
                except ValueError:
                    bool_bits = 'nobool'

                # Construct executable name
                non_bool_vals = "-".join([str(v).replace("-", "") for v in non_bool_combo])
                exe_name = f"{self.compiler_bool_flags_fingerprint}-{bool_bits}-{self.compiler_non_bool_flags_fingerprint}-{non_bool_vals}.exe"
                exe_path = os.path.join(self.build_dir, exe_name)

                # If binary doesn't exist, queue it for compilation
                if not os.path.exists(exe_path):
                    cmd = [self.compiler] + self.base_compiler_flags.split()

                    # Add active boolean flags
                    for flag_obj, is_active in zip(self.compiler_bool_flags, bool_combo):
                        if is_active:
                            cmd.append(flag_obj.name)

                    # Add non-boolean flags
                    cmd.extend(non_bool_combo)

                    # Add output flag
                    cmd.extend(["-o", exe_path])

                    build_tasks.append((cmd, exe_path))

        # Helper function to run a single compilation
        def compile_binary(task):
            cmd, path = task
            try:
                # capture_output prevents interleaved terminal spam and lets us grab stderr on failure
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"Successfully compiled: {path}")
            except subprocess.CalledProcessError as e:
                print(f"COMPILATION FAILED for {path}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Error Output:\n{e.stderr}")
                exit(1)

        # Execute builds in parallel
        if build_tasks:
            print(f"Starting {len(build_tasks)} parallel builds...")
            # ThreadPoolExecutor is perfect here since subprocess releases the GIL
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(compile_binary, build_tasks)
            print("Finished build phase.")
        else:
            print("All required binaries are already cached.")

    def run_experiments(self):
        """Runs benchmarks in a round-robin fashion to mitigate time-dependent systemic biases, saving after each full sweep."""
        bool_ranges = [f.value_range for f in self.compiler_bool_flags]
        non_bool_ranges = [f.value_range for f in self.compiler_non_bool_flags]
        runtime_ranges = [f.value_range for f in self.runtime_args]
        env_var_ranges = [f.value_range for f in self.runtime_env_vars]

        # Determine the save file name based on fingerprints
        results_filename = f"results_{self.compiler_bool_flags_fingerprint}_{self.compiler_non_bool_flags_fingerprint}.csv"
        results_filepath = os.path.join(self.build_dir, results_filename)

        config_cols = [p.name for p in self.compiler_bool_flags + self.compiler_non_bool_flags + self.runtime_args + self.runtime_env_vars] + ["Run_Iteration"]

        # Load existing DataFrame if it exists and build a lookup set to skip completed runs
        completed_keys = set()
        if os.path.exists(results_filepath):
            existing_df = pd.read_csv(results_filepath)
            print(f"Loaded {len(existing_df)} existing records from {results_filepath}")

            for _, row in existing_df.iterrows():
                key_dict = {col: str(row[col]) for col in config_cols if col in existing_df.columns}
                completed_keys.add(tuple(sorted(key_dict.items())))
        else:
            existing_df = pd.DataFrame()

        # Outermost loop: Complete one full sweep of all configurations before starting the next
        for run_iteration in range(1, RUNS_PER_EXPERIMENT + 1):
            print(f"\n{time.ctime()}\n--- Starting Sweep {run_iteration} of {RUNS_PER_EXPERIMENT} ---\n")
            sweep_start = time.time()
            results_list = []

            for bool_combo in itertools.product(*bool_ranges):
                for non_bool_combo in itertools.product(*non_bool_ranges):

                    # Construct the boolean bits string (e.g., '101' for True, False, True)
                    try:
                        bool_bits = int("".join(["1" if b else "0" for b in bool_combo]), 2)
                    except ValueError:
                        bool_bits = 'nobool'
                    non_bool_vals = "-".join([str(v).replace("-", "") for v in non_bool_combo])
                    exe_name = f"{self.compiler_bool_flags_fingerprint}-{bool_bits}-{self.compiler_non_bool_flags_fingerprint}-{non_bool_vals}.exe"
                    exe_path = os.path.join(self.build_dir, exe_name)

                    for runtime_combo in itertools.product(*runtime_ranges):
                        for env_var_combo in itertools.product(*env_var_ranges):
                            config_dict = {}
                            for i, flag in enumerate(self.compiler_bool_flags): config_dict[flag.name] = bool_combo[i]
                            for i, flag in enumerate(self.compiler_non_bool_flags): config_dict[flag.name] = non_bool_combo[i]
                            for i, flag in enumerate(self.runtime_args): config_dict[flag.name] = runtime_combo[i]
                            for i, flag in enumerate(self.runtime_env_vars): config_dict[flag.name] = env_var_combo[i]

                            # Check if this specific configuration and iteration is already completed
                            lookup_dict = {k: str(v) for k, v in config_dict.items()}
                            lookup_dict["Run_Iteration"] = str(run_iteration)
                            lookup_key = tuple(sorted(lookup_dict.items()))

                            if lookup_key in completed_keys:
                                continue  # Skip execution, we already have this data

                            # Setup the execution environment
                            run_env = os.environ.copy()
                            custom_env = {}
                            if self.base_env_vars:
                                run_env.update(self.base_env_vars)
                                custom_env.update(self.base_env_vars)
                            for i, flag in enumerate(self.runtime_env_vars):
                                run_env[flag.name] = str(env_var_combo[i])
                                custom_env[flag.name] = str(env_var_combo[i])

                            cmd = [f"./{exe_path}"]
                            for i, arg in enumerate(runtime_combo):
                                flag_name = self.runtime_args[i].name
                                if flag_name:
                                    cmd.extend([f"--{flag_name}", str(arg)])
                                else:
                                    cmd.append(str(arg))

                            try:
                                if DEBUG:
                                    print(f"test started: {' '.join(cmd)}\nenv:{custom_env}")
                                # Pass the custom env dict to the subprocess
                                output = subprocess.check_output(cmd, text=True, env=run_env).strip()
                                parsed_result = self.result_parser_func(output)
                            except subprocess.CalledProcessError as e:
                                print(f"Error running {' '.join(cmd)}: {e}")
                                parsed_result = None

                            # Record data
                            row_data = config_dict.copy()
                            row_data["Run_Iteration"] = run_iteration
                            
                            if isinstance(parsed_result, dict):
                                row_data.update(parsed_result)
                            else:
                                row_data["Result"] = parsed_result
                                
                            results_list.append(row_data)

            # Checkpoint: Save data at the end of every sweep
            new_df = pd.DataFrame(results_list)
            if not new_df.empty:
                existing_df = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
                existing_df.to_csv(results_filepath, index=False)

                # Update the class attribute to stay in sync
                self.result_df = existing_df

                # Update completed keys so we don't duplicate if we restart mid-sweep
                for _, row in new_df.iterrows():
                    key_dict = {col: str(row[col]) for col in config_cols if col in new_df.columns}
                    completed_keys.add(tuple(sorted(key_dict.items())))

                print(f"Sweep {run_iteration} complete. Checkpointed {len(new_df)} new records in {time.time() - sweep_start:.3f} seconds")

    @staticmethod
    def default_result_parser(input_str):
        try:
            return float(input_str)
        except ValueError:
            return None

    @staticmethod
    def get_compiler_bool_flag_fingerprint(experimental_params):
        """Returns a short MD5 hash of the names of boolean compile time params."""
        names = "".join([p.name for p in experimental_params])
        return hashlib.md5(names.encode()).hexdigest()[:HASH_LENGTH] if names else "nobool"

    @staticmethod
    def get_compiler_num_flag_fingerprint(experimental_params):
        """Returns a short MD5 hash of the names of non-boolean compile time params."""
        names = "".join([p.name for p in experimental_params])
        return hashlib.md5(names.encode()).hexdigest()[:HASH_LENGTH] if names else "nonum"


import inspect

class HWCounterMetric:
    """
    This is the abstraction of a counter metric
    This is defined by a set of counter strings and a function on how to compute the metric

    the set of counters is extracted automatically from the compute func kwargs
    when this is called we just unpack the kwargs and pass them to the compute func
        def compute_func_example(self, counter_1=0, counter_2=1):
        return counter_1 / counter_2

    """
    def __init__(self, name, compute_func):
        self.name = name
        # Exclude 'self' if the metric function is defined as a method
        self.counter_set = {
            param for param in inspect.signature(compute_func).parameters 
            if param != 'self'
        }
        self.compute_value = compute_func

    def __call__(self, *args, **kwargs):
        return self.compute_value(*args, **kwargs)



import json

class HWCounterStudy (Study):
    """
    This is a subclass of Study that will analyze the hardware counters collected by polybench instead of simple times
    polybench will read a file naming all the PAPI events to track and output a list of counter values

    the plan is to subclass the Study and do most of the analysis in a custom output parser
    in addition to the args of Study, this should take a list of HWCounterMetric
    """
    def __init__(self, build_dir, experimental_params, base_compiler_command, base_env_vars, hw_metrics, compiler="clang"):
        self.hw_metrics = hw_metrics
        
        # Extract unique counters required across all metrics
        self.counters = set()
        for metric in hw_metrics:
            self.counters.update(metric.counter_set)
        self.counter_list = sorted(list(self.counters))
        
        # Ensure build directory exists
        os.makedirs(build_dir, exist_ok=True)
        
        # Write papi_counters.list for Polybench to pick up during compilation
        papi_list_path = os.path.join(build_dir, "papi_counters.list")
        with open(papi_list_path, "w") as f:
            for counter in self.counter_list:
                f.write(f'"{counter}",\n')
                
        # Modify the compiler command to prioritize our papi_counters.list and link PAPI
        # We prepend -I {build_dir} so that it is found before the utilities folder
        modified_compiler_command = f"-I {build_dir} -DPOLYBENCH_PAPI {base_compiler_command} -lpapi"
        
        super().__init__(
            build_dir, 
            experimental_params, 
            modified_compiler_command, 
            base_env_vars, 
            compiler=compiler, 
            result_parser_func=self.parse_hw_counters
        )

    def parse_hw_counters(self, output):
        parts = output.strip().split()
        if not parts:
            return None
            
        try:
            values = [int(p) for p in parts]
        except ValueError:
            return None
            
        if len(values) != len(self.counter_list):
            return None
            
        # Map counter names to their values
        counter_values = dict(zip(self.counter_list, values))
        
        # Compute metrics based on HWCounterMetric functions
        results = {}
        for metric in self.hw_metrics:
            kwargs = {k: counter_values[k] for k in metric.counter_set}
            try:
                results[metric.name] = metric(**kwargs)
            except Exception:
                results[metric.name] = None
                
        return results




###############################################################################
# Utils
###############################################################################
def print_cpu_info():
    """
    this function gets the proc/cpuinfo, parses it and prints out these parameters:
    processor make and model
    processor clock frequency
    vectorization instructions supported (x86 specific)
    numa information if there is more than one numa node
    number of cores
    number of logical cpus
    the list of physical cpus (cpus that don't share a physical core together)
    :return: str list of logical cores in llvm libomp format
    """
    cpuinfo_path = '/proc/cpuinfo'

    if not os.path.exists(cpuinfo_path):
        print(f"Error: {cpuinfo_path} not found. This function requires a Linux environment.")
        return []

    processors = []
    current_cpu = {}

    # 1. Read and parse /proc/cpuinfo
    with open(cpuinfo_path, 'r') as f:
        for line in f:
            line = line.strip()
            # An empty line indicates the end of a single processor's block
            if not line:
                if current_cpu:
                    processors.append(current_cpu)
                    current_cpu = {}
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                current_cpu[key.strip()] = value.strip()

    # Catch the last block if the file doesn't end with a newline
    if current_cpu:
        processors.append(current_cpu)

    if not processors:
        return []

    # 2. Extract Make, Model, and Frequency
    model_name = processors[0].get('model name', 'Unknown Model')
    # Iterate through all processors to find the peak frequency
    clock_freq = 0.0
    for p in processors:
        freq_str = p.get('cpu MHz', '0')
        try:
            freq = float(freq_str)
            if freq > clock_freq:
                clock_freq = freq
        except ValueError:
            continue
    clock_freq = str(clock_freq)
    if clock_freq != 'Unknown':
        clock_freq += ' MHz'

    # 3. Determine Vectorization Instructions (x86 only)
    flags_string = processors[0].get('flags', '')
    cpu_flags = set(flags_string.lower().split())

    # Predefined set of common x86 vectorization and SIMD instruction sets
    known_vector_flags = {
        'mmx', 'sse', 'sse2', 'sse3', 'ssse3', 'sse4_1', 'sse4_2', 'sse4a',
        'avx', 'avx2', 'avx512f', 'avx512cd', 'avx512er', 'avx512pf',
        'avx512bw', 'avx512dq', 'avx512vl', 'avx512vnni', 'avx512bf16',
        'avx_vnni', 'fma', 'fma4'
    }

    supported_vectorization = sorted(list(cpu_flags.intersection(known_vector_flags)))
    vector_str = ', '.join(supported_vectorization) if supported_vectorization else "None detected"

    # 4. Determine NUMA Information
    numa_nodes = []
    numa_path = '/sys/devices/system/node'
    if os.path.exists(numa_path):
        numa_nodes = [d for d in os.listdir(numa_path) if d.startswith('node')]

    # 5. Calculate core topologies
    physical_cores_seen = set()
    physical_cpus_list = []
    logical_cores = []

    for p in processors:
        proc_id = p.get('processor')
        if proc_id is not None:
            logical_cores.append(proc_id)

        # Use defaults if running in a VM that obscures topology
        phys_id = p.get('physical id', '0')
        core_id = p.get('core id', proc_id)

        # A physical core is defined by a unique combination of socket ID and core ID
        core_tuple = (phys_id, core_id)

        if core_tuple not in physical_cores_seen:
            physical_cores_seen.add(core_tuple)
            if proc_id is not None:
                # Store one logical CPU per physical core
                physical_cpus_list.append(proc_id)

    num_cores = len(physical_cores_seen)
    num_logical_cpus = len(processors)

    # Print out the required parameters
    print(f"============================================================")
    print(f"Processor Make/Model     : {model_name}")
    print(f"Processor Clock Frequency: {clock_freq}")

    if len(numa_nodes) > 1:
        print(f"NUMA Information         : {len(numa_nodes)} NUMA nodes detected")

    print(f"Number of Cores          : {num_cores}")
    print(f"Number of Logical CPUs   : {num_logical_cpus}")
    print(f"List of Physical CPUs    : {', '.join(physical_cpus_list)}")
    print(f"Vector Instructions      : {vector_str}")
    print(f"============================================================")

    return f"{{{', '.join(physical_cpus_list)}}}", num_cores

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