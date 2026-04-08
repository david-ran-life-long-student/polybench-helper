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

class Mutable:
    """
    This object defines a parameter of the experiments.
    Example:
    Mutable([10, 100, 1000], name="N", is_compiler_flag=False) # Runtime arg
    Mutable([-ftree-vectorize]) # Boolean compiler flag
    Mutable(["-O0", "-O2", "-O3"]) # Non-boolean compiler flag
    """
    def __init__(self, value_range, name=None, mode="compiler_flag"):
        self.is_compiler_flag = mode == "compiler_flag"
        self.is_env_var = mode == "env_var"

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

        # Load existing DataFrame if it exists and build a lookup set to skip completed runs
        completed_keys = set()
        if os.path.exists(results_filepath):
            existing_df = pd.read_csv(results_filepath)
            print(f"Loaded {len(existing_df)} existing records from {results_filepath}")

            for _, row in existing_df.iterrows():
                key_dict = {col: str(row[col]) for col in existing_df.columns if col != "Result"}
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
                            if self.base_env_vars:
                                run_env.update(self.base_env_vars)
                            for i, flag in enumerate(self.runtime_env_vars):
                                run_env[flag.name] = str(env_var_combo[i])

                            cmd = [f"./{exe_path}"] + [str(arg) for arg in runtime_combo]

                            try:
                                print(f"test started: {' '.join(cmd)}")
                                # Pass the custom env dict to the subprocess
                                output = subprocess.check_output(cmd, text=True, env=run_env).strip()
                                parsed_result = self.result_parser_func(output)
                            except subprocess.CalledProcessError as e:
                                print(f"Error running {' '.join(cmd)}: {e}")
                                parsed_result = None

                            # Record data
                            row_data = config_dict.copy()
                            row_data["Run_Iteration"] = run_iteration
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
                    key_dict = {col: str(row[col]) for col in new_df.columns if col != "Result"}
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