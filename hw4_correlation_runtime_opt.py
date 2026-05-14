"""
hw4_correlation_runtime_opt.py — Study A for the combined opt variant.

Points at correlation.opt.c (transpose + column-at-a-time stats fusion)
and writes to build/runtime_opt/. Sweeps the timeable regions of the
opt variant: 0 (whole), 4 (corr), 5 (transpose), 6 (fused stats+center).

Compare to baseline:
    opt.region6 vs baseline.(region1 + region2 + region3)
    opt.region4 vs baseline.region4
    opt.region0 vs baseline.region0
"""
import experiment_helper
from experiment_helper import Study, Mutable


def main():
    experiment_helper.RUNS_PER_EXPERIMENT = 10

    base_compile_command = " ".join([
        "-I polybench-c-4.2.1-beta/utilities",
        "-I polybench-c-4.2.1-beta/datamining/correlation",
        "-DPOLYBENCH_USE_RESTRICT",
        "-DPOLYBENCH_TIME",
        "polybench-c-4.2.1-beta/utilities/polybench.c",
        "polybench-c-4.2.1-beta/datamining/correlation/correlation.opt.c",
        "-lm",
    ])

    sizes = [128, 256, 512, 1024, 2048]
    size_flags = [f"-DSIZE={s}" for s in sizes]
    region_flags = [f"-DTIME_REGION={r}" for r in [0, 4, 5, 6]]

    study = Study(
        build_dir="build/runtime_opt",
        experimental_params=[
            Mutable(size_flags, name="Size"),
            Mutable(["-O0", "-O2", "-O3"], name="Opt_Level"),
            Mutable(region_flags, name="Region"),
        ],
        base_compiler_command=base_compile_command,
        base_env_vars={},
        compiler="gcc",
    )

    study.ensure_all_builds_exist()
    study.run_experiments()


if __name__ == "__main__":
    main()
