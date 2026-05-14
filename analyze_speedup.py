"""
analyze_speedup.py — whole-kernel speedup (baseline / opt) vs problem size.

Loads the runtime CSVs from build/runtime/ and build/runtime_opt/, computes a
trimmed mean of TIME_REGION=0 (whole kernel) per (size, opt_level), then
plots speedup as a scatter with one series per opt level.

Showing all three opt levels intentionally: the -O0 curve isolates pure
algorithmic/locality wins (no auto-vectorization either side), while -O3
includes the FMA throughput from explicit register-blocking.

Usage:
    python3 analyze_speedup.py
    python3 analyze_speedup.py --output data/speedup.png
"""
import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

from analysis_helper import compute_trimmed_mean


def _extract_int(value, prefix):
    m = re.search(rf"{re.escape(prefix)}=(-?\d+)", str(value))
    return int(m.group(1)) if m else None


def load_runtime(build_dir):
    matches = sorted(glob.glob(os.path.join(build_dir, "results_*.csv")))
    if not matches:
        raise FileNotFoundError(f"no CSV matching {build_dir}/results_*.csv")
    df = pd.read_csv(matches[-1])
    df["SizeN"] = df["Size"].apply(lambda v: _extract_int(v, "SIZE"))
    df["RegionN"] = df["Region"].apply(lambda v: _extract_int(v, "TIME_REGION"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="build/runtime")
    ap.add_argument("--opt",      default="build/runtime_opt")
    ap.add_argument("--output",   default="data/speedup_vs_size.png")
    args = ap.parse_args()

    base = load_runtime(args.baseline)
    opt  = load_runtime(args.opt)

    # Whole-kernel region only
    base_r0 = base[base["RegionN"] == 0]
    opt_r0  = opt [opt ["RegionN"] == 0]

    base_mean = compute_trimmed_mean(
        base_r0, target_col="Result",
        group_cols=["SizeN", "Opt_Level"],
    )
    opt_mean = compute_trimmed_mean(
        opt_r0, target_col="Result",
        group_cols=["SizeN", "Opt_Level"],
    )

    merged = base_mean.merge(opt_mean, on=["SizeN", "Opt_Level"],
                             suffixes=("_b", "_o"))
    merged["Speedup"] = merged["Result_b"] / merged["Result_o"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"-O0": "o", "-O2": "s", "-O3": "^"}
    for opt_level in sorted(merged["Opt_Level"].unique()):
        sub = merged[merged["Opt_Level"] == opt_level].sort_values("SizeN")
        ax.plot(sub["SizeN"], sub["Speedup"],
                marker=markers.get(opt_level, "o"), markersize=9,
                linewidth=1.5, linestyle="--", alpha=0.85,
                label=opt_level)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(merged["SizeN"].unique()))
    ax.set_xticklabels(sorted(merged["SizeN"].unique()))
    ax.set_xlabel("Problem size N (M=N)")
    ax.set_ylabel("Speedup  (baseline / opt)")
    ax.set_title("Whole-kernel speedup vs problem size")
    ax.axhline(1.0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Opt level")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")

    print("\nWhole-kernel speedup table:")
    pretty = merged.copy()
    pretty["Result_b"] = (pretty["Result_b"] * 1000).round(2)
    pretty["Result_o"] = (pretty["Result_o"] * 1000).round(2)
    pretty["Speedup"]  = pretty["Speedup"].round(2)
    pretty.columns = ["Size", "Opt_Level", "base_ms", "opt_ms", "Speedup"]
    print(pretty.sort_values(["Opt_Level", "Size"]).to_string(index=False))


if __name__ == "__main__":
    main()
