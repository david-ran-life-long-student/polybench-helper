#! /usr/bin/python3
from experiment_helper import *


if __name__=="__main__":
    base_compile_command = " ".join(["-I polybench-c-4.2.1-beta/utilities ",
                          "-I polybench-c-4.2.1-beta/linear-algebra/blas/gemm",
                          "polybench-c-4.2.1-beta/utilities/polybench.c polybench-c-4.2.1-beta/linear-algebra/blas/gemm/gemm.c",
                          "-DPOLYBENCH_TIME"])
    build_dir = "build"

    s = Study(
        build_dir,
        [
            Mutable([f"-DN={i}" for i in range(100, 1000, 100)]),
            Mutable(["-O0", "-O2", "-O3"]),
        ], base_compile_command, compiler="clang"
    )

    s.ensure_all_builds_exist()

    s.run_experiments()
