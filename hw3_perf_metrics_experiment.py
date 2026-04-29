import sys
import os
import shutil

import experiment_helper
from experiment_helper import HWCounterStudy, HWCounterMetric, Mutable

def ipc(PAPI_TOT_INS, PAPI_TOT_CYC):
    return PAPI_TOT_INS / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0

def vipc(PAPI_VEC_DP, PAPI_TOT_CYC):
    return (PAPI_VEC_DP) / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0

def l1_miss_rate(PAPI_L1_DCM, PAPI_LST_INS):
    return (PAPI_L1_DCM / PAPI_LST_INS) * 100 if PAPI_LST_INS > 0 else 0

def l2_miss_rate(PAPI_L2_DCM, PAPI_L2_DCA):
    return (PAPI_L2_DCM / PAPI_L2_DCA) * 100 if PAPI_L2_DCA > 0 else 0

def l1_accesses_per_instruction(PAPI_LST_INS, PAPI_TOT_INS):
    return PAPI_LST_INS / PAPI_TOT_INS if PAPI_TOT_INS > 0 else 0

def l1_accesses_per_vector_instruction(PAPI_LST_INS, PAPI_VEC_DP):
    return PAPI_LST_INS / PAPI_VEC_DP if PAPI_VEC_DP > 0 else 0

def stall_rate(PAPI_RES_STL, PAPI_TOT_CYC):
    return (PAPI_RES_STL / PAPI_TOT_CYC) * 100 if PAPI_TOT_CYC > 0 else 0

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
        HWCounterMetric("IPC", ipc),
        HWCounterMetric("vIPC", vipc),
        HWCounterMetric("%L1m", l1_miss_rate),
        HWCounterMetric("%L2m", l2_miss_rate),
        HWCounterMetric("L1AC", l1_accesses_per_instruction),
        HWCounterMetric("vL1AC", l1_accesses_per_vector_instruction),
        HWCounterMetric("%Stall", stall_rate),
    ], compiler="gcc")

    study.ensure_all_builds_exist()

    study.run_experiments()


if __name__ == "__main__":
    main()
