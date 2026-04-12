#! /usr/bin/python3
from experiment_helper import *


if __name__=="__main__":

    physical_cores_list, cores  = print_cpu_info()

    base_compile_command = " ".join(["-I polybench-c-4.2.1-beta/utilities ",
                                     "-I polybench-c-4.2.1-beta/linear-algebra/blas/gemm",
                                     "polybench-c-4.2.1-beta/utilities/polybench.c polybench-c-4.2.1-beta/linear-algebra/blas/gemm/gemm.c",
                                     "-DPOLYBENCH_TIME"])
    base_env_vars = {
        "OMP_PLACES": physical_cores_list,  # this is targeting llvm libomp
        "OMP_SCHEDULE": "static",  # not sure why we need to say this, isn't static default?
    }

    build_dir = "build"

    s = Study(
        build_dir,
        [
            Mutable([f"-DN={i}" for i in [512, 513, 1000, 1024, 2000, 2048]]),
            Mutable(["-O0", "-O2", "-O3"]),
            Mutable(["-fopenmp"]),
            Mutable(["-fno-vectorize"]),
            Mutable([str(i + 1) for i in range(cores)], name="OMP_NUM_THREADS", mode="env_var")
        ], base_compile_command, base_env_vars, compiler="clang"
    )

    s.ensure_all_builds_exist()

    s.run_experiments()