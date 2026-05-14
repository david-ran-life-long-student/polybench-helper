"""
analyze_breakdown.py — stacked-bar breakdown of region contributions.

For each problem size, shows two side-by-side stacked bars (baseline | opt)
at the chosen opt level. Baseline stack: R1 mean, R2 stddev, R3 center, R4 corr.
Opt stack: R5 transpose, R6 fused stats+center, R4 corr.

Uses linear y-axis even though R4 dominates at large sizes — that domination
is itself a finding, and the smaller components remain visible at small N.

Usage:
    python3 analyze_breakdown.py
    python3 analyze_breakdown.py --opt-level -O3 --output data/breakdown.png
"""
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
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
    df["SizeN"]   = df["Size"].apply(lambda v: _extract_int(v, "SIZE"))
    df["RegionN"] = df["Region"].apply(lambda v: _extract_int(v, "TIME_REGION"))
    return df


BASELINE_STACK = [
    (1, "mean (R1)",   "#cbe4f9"),
    (2, "stddev (R2)", "#9bcbe5"),
    (3, "center (R3)", "#5fa8d3"),
    (4, "corr (R4)",   "#1f4068"),
]

OPT_STACK = [
    (5, "transpose (R5)",          "#fde4cf"),
    (6, "stats+center fused (R6)", "#f7b267"),
    (4, "corr (R4)",               "#b1492a"),
]


def pivot_regions(df, opt_level):
    df = df[df["Opt_Level"] == opt_level]
    if df.empty:
        raise ValueError(f"no rows at Opt_Level={opt_level!r}")
    mean = compute_trimmed_mean(
        df, target_col="Result", group_cols=["SizeN", "RegionN"]
    )
    return mean.pivot(index="SizeN", columns="RegionN", values="Result") * 1000  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",  default="build/runtime")
    ap.add_argument("--opt",       default="build/runtime_opt")
    ap.add_argument("--opt-level", default="-O3")
    ap.add_argument("--output",    default="data/breakdown.png")
    args = ap.parse_args()

    base_pivot = pivot_regions(load_runtime(args.baseline), args.opt_level)
    opt_pivot  = pivot_regions(load_runtime(args.opt),      args.opt_level)

    sizes = sorted(set(base_pivot.index) & set(opt_pivot.index))
    n = len(sizes)
    x = np.arange(n)
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))

    bottom = np.zeros(n)
    for r, label, color in BASELINE_STACK:
        h = np.array([base_pivot.loc[s].get(r, 0.0) for s in sizes])
        ax.bar(x - width/2, h, width, bottom=bottom, color=color, label=f"base · {label}")
        bottom += h
    base_totals = bottom

    bottom = np.zeros(n)
    for r, label, color in OPT_STACK:
        h = np.array([opt_pivot.loc[s].get(r, 0.0) for s in sizes])
        ax.bar(x + width/2, h, width, bottom=bottom, color=color, label=f"opt · {label}")
        bottom += h
    opt_totals = bottom

    for i, s in enumerate(sizes):
        ax.text(x[i] - width/2, base_totals[i],
                f"{base_totals[i]:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i] + width/2, opt_totals[i],
                f"{opt_totals[i]:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Problem size N (M=N)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Region contribution breakdown — baseline (left) vs opt (right), {args.opt_level}")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"saved {args.output}")

    # Also print the underlying numbers for the report.
    print(f"\nBreakdown at {args.opt_level} (times in ms, trimmed mean):\n")
    rows = []
    for s in sizes:
        rows.append({
            "Size": s,
            "base_R1": base_pivot.loc[s].get(1, 0.0),
            "base_R2": base_pivot.loc[s].get(2, 0.0),
            "base_R3": base_pivot.loc[s].get(3, 0.0),
            "base_R4": base_pivot.loc[s].get(4, 0.0),
            "base_tot": base_totals[sizes.index(s)],
            "opt_R5":  opt_pivot.loc[s].get(5, 0.0),
            "opt_R6":  opt_pivot.loc[s].get(6, 0.0),
            "opt_R4":  opt_pivot.loc[s].get(4, 0.0),
            "opt_tot": opt_totals[sizes.index(s)],
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
