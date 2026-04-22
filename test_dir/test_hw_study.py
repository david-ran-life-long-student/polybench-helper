import sys
import os
import shutil

# Add parent directory to path so we can import experiment_helper
sys.path.append(os.path.abspath('..'))

import experiment_helper
from experiment_helper import HWCounterStudy, HWCounterMetric, Mutable

# Define a couple of fake metrics
def l1_miss_rate(PAPI_L1_DCM, PAPI_TOT_CYC):
    # Just a fake calculation
    return PAPI_L1_DCM / PAPI_TOT_CYC

def double_inst(PAPI_TOT_INS):
    return PAPI_TOT_INS * 2

def main():
    # Run only 1 iteration for the test
    experiment_helper.RUNS_PER_EXPERIMENT = 1

    params = [
        Mutable(["-O0", "-O2"], name="opt_level", mode="compiler_flag"), 
        Mutable([10, 100], name="N", mode="runtime_arg") 
    ]
    
    metrics = [
        HWCounterMetric("L1_Miss_Rate", l1_miss_rate),
        HWCounterMetric("Double_Inst", double_inst)
    ]
    
    # Clean up any existing results to ensure a fresh test run
    if os.path.exists("build_hw"):
        shutil.rmtree("build_hw")
    
    # Base command compiles our C file
    base_cmd = "test_hw_counters.c"
    
    study = HWCounterStudy("build_hw", params, base_cmd, base_env_vars={}, hw_metrics=metrics, compiler="gcc")
    
    print("--- Ensuring builds exist ---")
    study.ensure_all_builds_exist()
    
    print("\n--- Running experiments ---")
    study.run_experiments()
    
    print("\n--- Results DataFrame ---")
    print(study.result_df.to_string())

if __name__ == "__main__":
    main()