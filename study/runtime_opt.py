"""
runtime_opt.py — Study A for the combined opt variant.

Points at correlation.opt.c (transpose + column-at-a-time stats fusion)
and writes to build/runtime_opt/. Sweeps the timeable regions of the
opt variant: 0 (whole), 4 (corr), 5 (transpose), 6 (fused stats+center).

Compare to baseline:
    opt.region6 vs baseline.(region1 + region2 + region3)
    opt.region4 vs baseline.region4
    opt.region0 vs baseline.region0

Run from the repo root: python3 study/runtime_opt.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "framework"))

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
        "kernel/correlation.opt.c",
        "-lm",
    ])

    study = Study(
        build_dir="build/runtime_opt",
        experimental_params=[
            Mutable([f"-DSIZE={s}" for s in [128, 512, 2048]], name="Size"),
            Mutable(["-O2", "-O3"], name="Opt_Level"),
            Mutable([f"-DTIME_REGION={r}" for r in [0, 4, 5, 6]], name="Region"),
        ],
        base_compiler_command=base_compile_command,
        base_env_vars={},
        compiler="gcc",
    )

    study.ensure_all_builds_exist()
    study.run_experiments()


if __name__ == "__main__":
    main()
