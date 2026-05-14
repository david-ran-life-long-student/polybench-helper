"""
hw4_correlation_runtime.py — Study A: wall-clock runtime per region.

Sweeps TIME_REGION x Opt_Level x problem size and records POLYBENCH_TIME output.
The result column is wall-clock seconds spent in the selected region.
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
        "polybench-c-4.2.1-beta/datamining/correlation/correlation.localized.c",
        "-lm",
    ])

    # Square sizes (M == N) keep the sweep tractable; we can split if needed.
    sizes = [128, 512, 2048]
    size_flags = [f"-DSIZE={s}" for s in sizes]
    region_flags = [f"-DTIME_REGION={r}" for r in range(0, 5)]

    study = Study(
        build_dir="build/runtime",
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
