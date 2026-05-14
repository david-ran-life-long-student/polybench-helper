"""
check_correctness.py — validate a candidate correlation source against upstream.

Builds both upstream correlation.c and the candidate (default:
correlation.localized.c) with -DPOLYBENCH_DUMP_ARRAYS, runs both, and compares
the dumped corr[M][M] matrix with floating-point tolerance.

Usage:
    python3 check_correctness.py                       # localized
    python3 check_correctness.py phase1                # correlation.phase1.c
    python3 check_correctness.py phase1 --tol 1e-5     # custom tolerance

Run manually before measuring each phase.
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
POLY_UTIL = os.path.join(REPO, "polybench-c-4.2.1-beta/utilities")
POLY_C = os.path.join(POLY_UTIL, "polybench.c")
KERNEL_DIR = os.path.join(REPO, "polybench-c-4.2.1-beta/datamining/correlation")
UPSTREAM_SRC = os.path.join(KERNEL_DIR, "correlation.c")

# Small enough to be fast, large enough to exercise all four regions meaningfully.
M = N = 128

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "variant", nargs="?", default="localized",
        help="suffix of correlation.<variant>.c (default: localized)",
    )
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    candidate_src = os.path.join(KERNEL_DIR, f"correlation.{args.variant}.c")
    if not os.path.exists(candidate_src):
        print(f"No such file: {candidate_src}", file=sys.stderr)
        sys.exit(1)

    build_dir = os.path.join(REPO, "build", f"correctness_{args.variant}")
    os.makedirs(build_dir, exist_ok=True)

    upstream_exe = os.path.join(build_dir, "upstream.exe")
    candidate_exe = os.path.join(build_dir, f"{args.variant}_r0.exe")

    print(f"Building upstream correlation.c (M=N={M})...")
    build(UPSTREAM_SRC, upstream_exe, [])

    print(f"Building correlation.{args.variant}.c with TIME_REGION=0 (M=N={M})...")
    build(candidate_src, candidate_exe, ["-DTIME_REGION=0"])

    print("Running upstream...")
    upstream_corr = parse_corr_floats(run_and_capture_dump(upstream_exe))

    print(f"Running {args.variant} (TIME_REGION=0)...")
    candidate_corr = parse_corr_floats(run_and_capture_dump(candidate_exe))

    print(f"Comparing {len(upstream_corr)} elements (tolerance={args.tol})...")
    result = diff_arrays(upstream_corr, candidate_corr)
    if isinstance(result, str):
        print(f"FAIL: {result}")
        sys.exit(1)

    worst, idx = result
    if worst > args.tol:
        print(f"FAIL: max abs diff {worst:.3e} at index {idx} exceeds tolerance {args.tol:.0e}")
        i, j = divmod(idx, M)
        print(f"  corr[{i}][{j}]: upstream={upstream_corr[idx]} candidate={candidate_corr[idx]}")
        sys.exit(1)

    print(f"OK: max abs diff {worst:.3e} within tolerance {args.tol:.0e}")


if __name__ == "__main__":
    main()
