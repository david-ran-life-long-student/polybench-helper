import sys
import os
import shutil

# Add parent directory to path so we can import experiment_helper
sys.path.append(os.path.abspath('..'))

import experiment_helper
from experiment_helper import Study, Mutable

def main():
    # Set RUNS_PER_EXPERIMENT = 1 for testing purposes
    experiment_helper.RUNS_PER_EXPERIMENT = 1

    params = [
        Mutable(["-DTEST_MACRO"], mode="compiler_flag"), # Boolean compiler flag
        Mutable(["-O0", "-O2"], name="opt_level", mode="compiler_flag"), # Non-bool compiler flag
        Mutable([10, 100], name="N", mode="runtime_arg") # Runtime argument
    ]
    
    # Clean up any existing results to ensure a fresh test run
    if os.path.exists("build"):
        shutil.rmtree("build")
    
    # Base command compiles our C file
    base_cmd = "test_c_args.c"
    
    study = Study("build", params, base_cmd, compiler="gcc", base_env_vars={})
    
    print("--- Ensuring builds exist ---")
    study.ensure_all_builds_exist()
    
    print("\n--- Running experiments ---")
    study.run_experiments()
    
    print("\n--- Results DataFrame ---")
    print(study.result_df.to_string())

    print("\n--- Verifying Results ---")
    # Expected combinations of parameters and results
    expected_data = [
        {"-DTEST_MACRO": True, "opt_level": "-O0", "N": 10, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": True, "opt_level": "-O0", "N": 100, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": True, "opt_level": "-O2", "N": 10, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": True, "opt_level": "-O2", "N": 100, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": False, "opt_level": "-O0", "N": 10, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": False, "opt_level": "-O0", "N": 100, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": False, "opt_level": "-O2", "N": 10, "Run_Iteration": 1, "Result": 42.0},
        {"-DTEST_MACRO": False, "opt_level": "-O2", "N": 100, "Run_Iteration": 1, "Result": 42.0},
    ]
    
    # Convert actual DataFrame to list of dicts for straightforward assertion
    actual_data = study.result_df.to_dict(orient="records")
    
    if len(actual_data) != len(expected_data):
        print(f"❌ TEST FAILED: Expected {len(expected_data)} rows, got {len(actual_data)}")
        sys.exit(1)
        
    for i, (expected_row, actual_row) in enumerate(zip(expected_data, actual_data)):
        for key in expected_row:
            if actual_row[key] != expected_row[key]:
                print(f"❌ TEST FAILED: Mismatch in row {i}.\nExpected: {expected_row}\nActual: {actual_row}")
                sys.exit(1)

    print("✅ TEST PASSED: DataFrame matches expected output exactly!")

if __name__ == "__main__":
    main()