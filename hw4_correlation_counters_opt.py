"""
hw4_correlation_counters_opt.py — Study B for the combined opt variant.

Points at correlation.opt.c (transpose + column-at-a-time stats fusion)
and writes to build/counters_opt/. Sweeps regions 4 (corr), 5 (transpose),
and 6 (fused stats+center). Whole-kernel (0) is skipped since it's the
sum of the parts.
"""
import experiment_helper
from experiment_helper import HWCounterStudy, HWCounterMetric, Mutable


def flops_per_cycle(PAPI_DP_OPS, PAPI_TOT_CYC):
    return PAPI_DP_OPS / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0


def arithmetic_intensity(PAPI_DP_OPS, PAPI_L3_TCM):
    return PAPI_DP_OPS / (PAPI_L3_TCM * 64) if PAPI_L3_TCM > 0 else 0


def ipc(PAPI_TOT_INS, PAPI_TOT_CYC):
    return PAPI_TOT_INS / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0


def vipc(PAPI_VEC_DP, PAPI_TOT_CYC):
    return PAPI_VEC_DP / PAPI_TOT_CYC if PAPI_TOT_CYC > 0 else 0


def dp_ops_per_vec_instr(PAPI_DP_OPS, PAPI_VEC_DP):
    return PAPI_DP_OPS / PAPI_VEC_DP if PAPI_VEC_DP > 0 else 0


def l1_miss_rate(PAPI_L1_DCM, PAPI_LST_INS):
    return (PAPI_L1_DCM / PAPI_LST_INS) * 100 if PAPI_LST_INS > 0 else 0


def l2_miss_rate(PAPI_L2_TCM, PAPI_L1_DCM):
    return (PAPI_L2_TCM / PAPI_L1_DCM) * 100 if PAPI_L1_DCM > 0 else 0


def l3_miss_rate(PAPI_L3_TCM, PAPI_L3_TCA):
    return (PAPI_L3_TCM / PAPI_L3_TCA) * 100 if PAPI_L3_TCA > 0 else 0


def stall_rate(PAPI_RES_STL, PAPI_TOT_CYC):
    return (PAPI_RES_STL / PAPI_TOT_CYC) * 100 if PAPI_TOT_CYC > 0 else 0


def main():
    experiment_helper.RUNS_PER_EXPERIMENT = 10

    base_compile_command = " ".join([
        "-I polybench-c-4.2.1-beta/utilities",
        "-I polybench-c-4.2.1-beta/datamining/correlation",
        "-I $HOME/papi-install/include",
        "-L $HOME/papi-install/lib",
        "-DPOLYBENCH_USE_RESTRICT",
        "polybench-c-4.2.1-beta/utilities/polybench.c",
        "polybench-c-4.2.1-beta/datamining/correlation/correlation.opt.c",
        "-lm",
    ])

    sizes = [128, 512, 2048]
    size_flags = [f"-DSIZE={s}" for s in sizes]
    # Skip whole-kernel (0); it's the sum of regions 4, 5, 6.
    region_flags = [f"-DTIME_REGION={r}" for r in [0, 4, 5, 6]]

    study = HWCounterStudy(
        build_dir="build/counters_opt",
        experimental_params=[
            Mutable(size_flags, name="Size"),
            Mutable(["-O0", "-O2", "-O3"], name="Opt_Level"),
            Mutable(region_flags, name="Region"),
        ],
        base_compiler_command=base_compile_command,
        base_env_vars={},
        hw_metrics=[
            HWCounterMetric("FLOPs_per_cycle", flops_per_cycle),
            HWCounterMetric("AI",              arithmetic_intensity),
            HWCounterMetric("IPC",             ipc),
            HWCounterMetric("vIPC",            vipc),
            HWCounterMetric("DP_per_vec",      dp_ops_per_vec_instr),
            HWCounterMetric("%L1m",            l1_miss_rate),
            HWCounterMetric("%L2m",            l2_miss_rate),
            HWCounterMetric("%L3m",            l3_miss_rate),
            HWCounterMetric("%Stall",          stall_rate),
        ],
        compiler="gcc",
    )

    study.ensure_all_builds_exist()
    study.run_experiments()


if __name__ == "__main__":
    main()
