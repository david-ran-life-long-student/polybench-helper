import sys
import os
import shutil

import experiment_helper
from experiment_helper import HWCounterStudy, HWCounterMetric, Mutable

# Define a couple of fake metrics
# def l1_miss_rate(PAPI_L1_DCM, PAPI_TOT_CYC):
#     # Just a fake calculation
#     return PAPI_L1_DCM / PAPI_TOT_CYC

def vec_to_total_instr_ratio(PAPI_VEC_INS, PAPI_TOT_INS):
    return PAPI_VEC_INS / PAPI_TOT_INS

def main():
    # Run only 1 iteration for the test
    experiment_helper.RUNS_PER_EXPERIMENT = 1

    base_compile_command = " ".join(["-I polybench-c-4.2.1-beta/utilities ",
                                     "-I polybench-c-4.2.1-beta/linear-algebra/kernels/2mm",
                                     "polybench-c-4.2.1-beta/utilities/polybench.c polybench-c-4.2.1-beta/linear-algebra/blas/gemm/gemm.c",
                                     "-DPOLYBENCH_USE_RESTRICT",
                                     ])

    study = HWCounterStudy("build", [
        Mutable([f"-DN={i}" for i in [32, 64, 128, 256, 512, 513, 1000, 1024]], name="N"),
        Mutable(["-O0", "-O2", "-O3"]),
        Mutable(["-fopenmp"]),
    ], base_compile_command, base_env_vars={}, hw_metrics=[
        vec_to_total_instr_ratio
    ], compiler="gcc")

    study.ensure_all_builds_exist()

    study.run_experiments()


if __name__ == "__main__":
    main()