#! /usr/bin/python3
from experiment_helper import *


if __name__=="__main__":
    base_compile_command = " ".join(["-I polybench-c-4.2.1-beta/utilities ",
                          "-I polybench-c-4.2.1-beta/linear-algebra/kernels/2mm",
                          "polybench-c-4.2.1-beta/utilities/polybench.c polybench-c-4.2.1-beta/linear-algebra/kernels/2mm/2mm.c",
                          "-DPOLYBENCH_TIME"])
    build_dir = "build"

    s = Study(
        build_dir,
        [
            Mutable([f"-DN={i}" for i in [512, 513, 1000, 1024, 2000, 2048]]),
            #Mutable(["-O0", "-O1", "-O2", "-O3", "-Ofast"]),
            #Mutable(["-fno-vectorize"]),
        ], base_compile_command, {}, compiler="clang"
    )

    s.ensure_all_builds_exist()

    s.run_experiments()
