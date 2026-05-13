"""
check_correctness.py — validate correlation.localized.c against upstream correlation.c

Builds both versions with -DPOLYBENCH_DUMP_ARRAYS, runs them, and compares the
dumped corr[M][M] matrix with floating-point tolerance.

Run manually before each round of optimization.
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(REPO, "build", "correctness")
POLY_UTIL = os.path.join(REPO, "polybench-c-4.2.1-beta/utilities")
POLY_C = os.path.join(POLY_UTIL, "polybench.c")
KERNEL_DIR = os.path.join(REPO, "polybench-c-4.2.1-beta/datamining/correlation")
UPSTREAM_SRC = os.path.join(KERNEL_DIR, "correlation.c")
LOCAL_SRC = os.path.join(KERNEL_DIR, "correlation.localized.c")

# Small enough to be fast, large enough to exercise all four regions meaningfully.
M = N = 128

TOLERANCE = 1e-6
COMPILER = "gcc"


def build(src_path, exe_path, extra_defs):
    cmd = [
        COMPILER, "-O2", "-DPOLYBENCH_USE_RESTRICT", "-DPOLYBENCH_DUMP_ARRAYS",
        f"-DM={M}", f"-DN={N}",
        "-I", POLY_UTIL, "-I", KERNEL_DIR,
        POLY_C, src_path,
        "-lm", "-o", exe_path,
    ]
    cmd.extend(extra_defs)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"BUILD FAILED for {src_path}\n{res.stderr}", file=sys.stderr)
        sys.exit(1)


def run_and_capture_dump(exe_path):
    """Polybench dumps arrays to stderr when -DPOLYBENCH_DUMP_ARRAYS is set."""
    res = subprocess.run([exe_path], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"RUN FAILED for {exe_path}\n{res.stderr}", file=sys.stderr)
        sys.exit(1)
    return res.stderr


def parse_corr_floats(dump_text):
    """Pull all floating-point numbers out of the dumped corr block."""
    # The dump looks like:
    #   ==BEGIN DUMP_ARRAYS==
    #   begin dump: corr
    #   <numbers>
    #   end   dump: corr
    #   ==END   DUMP_ARRAYS==
    match = re.search(
        r"begin dump: corr\s*(.+?)\s*end   dump: corr",
        dump_text, re.DOTALL,
    )
    if not match:
        print("Could not find corr dump in output:", file=sys.stderr)
        print(dump_text, file=sys.stderr)
        sys.exit(1)
    body = match.group(1)
    return [float(tok) for tok in body.split()]


def diff_arrays(a, b):
    if len(a) != len(b):
        return f"size mismatch: {len(a)} vs {len(b)}"
    worst = 0.0
    worst_idx = -1
    for i, (x, y) in enumerate(zip(a, b)):
        d = abs(x - y)
        if d > worst:
            worst = d
            worst_idx = i
    return worst, worst_idx


def main():
    os.makedirs(BUILD_DIR, exist_ok=True)

    upstream_exe = os.path.join(BUILD_DIR, "upstream.exe")
    local_exe = os.path.join(BUILD_DIR, "localized_r0.exe")

    print(f"Building upstream correlation.c (M=N={M})...")
    build(UPSTREAM_SRC, upstream_exe, [])

    print(f"Building correlation.localized.c with TIME_REGION=0 (M=N={M})...")
    build(LOCAL_SRC, local_exe, ["-DTIME_REGION=0"])

    print("Running upstream...")
    upstream_corr = parse_corr_floats(run_and_capture_dump(upstream_exe))

    print("Running localized (TIME_REGION=0)...")
    local_corr = parse_corr_floats(run_and_capture_dump(local_exe))

    print(f"Comparing {len(upstream_corr)} elements (tolerance={TOLERANCE})...")
    result = diff_arrays(upstream_corr, local_corr)
    if isinstance(result, str):
        print(f"FAIL: {result}")
        sys.exit(1)

    worst, idx = result
    if worst > TOLERANCE:
        print(f"FAIL: max abs diff {worst:.3e} at index {idx} exceeds tolerance {TOLERANCE:.0e}")
        i, j = divmod(idx, M)
        print(f"  corr[{i}][{j}]: upstream={upstream_corr[idx]} localized={local_corr[idx]}")
        sys.exit(1)

    print(f"OK: max abs diff {worst:.3e} within tolerance {TOLERANCE:.0e}")


if __name__ == "__main__":
    main()
