"""
analyze_hwmetrics.py — side-by-side HW counter metrics: baseline vs opt.

Loads the counter CSVs from build/counters/ and build/counters_opt/, picks a
single region (default R4 — the dominant cost and the most informative for
comparing vectorization & cache behavior), and prints a table of:
    FLOPs/cycle, AI, vIPC, DP_per_vec, %L1m, %L2m, %L3m, %Stall
per (size, baseline | opt) at the chosen opt level.

Defaults to -O3 because vectorization metrics (vIPC, DP_per_vec) are only
meaningful when the compiler actually emitted vector instructions; at -O0
nothing vectorizes so the comparison degenerates.

Usage:
    python3 analyze_hwmetrics.py
    python3 analyze_hwmetrics.py --region 6     # fused stats+center on opt side
    python3 analyze_hwmetrics.py --md
"""
import argparse
import glob
import os
import re

import pandas as pd

from analysis_helper import compute_trimmed_mean


def _extract_int(value, prefix):
    m = re.search(rf"{re.escape(prefix)}=(-?\d+)", str(value))
    return int(m.group(1)) if m else None


def load_counters(build_dir):
    matches = sorted(glob.glob(os.path.join(build_dir, "results_*.csv")))
    if not matches:
        raise FileNotFoundError(f"no CSV matching {build_dir}/results_*.csv")
    df = pd.read_csv(matches[-1])
    df["SizeN"]   = df["Size"].apply(lambda v: _extract_int(v, "SIZE"))
    df["RegionN"] = df["Region"].apply(lambda v: _extract_int(v, "TIME_REGION"))
    return df


METRICS = [
    "FLOPs_per_cycle",
    "AI",
    "vIPC",
    "DP_per_vec",
    "%L1m",
    "%L2m",
    "%L3m",
    "%Stall",
]


def reduce_metrics(df, opt_level, region):
    df = df[(df["Opt_Level"] == opt_level) & (df["RegionN"] == region)]
    if df.empty:
        return None
    present = [m for m in METRICS if m in df.columns]
    out = {}
    for m in present:
        per_size = compute_trimmed_mean(
            df, target_col=m, group_cols=["SizeN"]
        ).set_index("SizeN")[m]
        out[m] = per_size
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline",  default="build/counters")
    ap.add_argument("--opt",       default="build/counters_opt")
    ap.add_argument("--opt-level", default="-O3")
    ap.add_argument("--region",    type=int, default=4,
                    help="TIME_REGION value to compare. 4 (corr matrix) is the default.")
    ap.add_argument("--md", action="store_true",
                    help="emit a GitHub-flavored markdown table")
    args = ap.parse_args()

    base = reduce_metrics(load_counters(args.baseline), args.opt_level, args.region)
    opt  = reduce_metrics(load_counters(args.opt),      args.opt_level, args.region)

    if base is None or opt is None:
        raise SystemExit(
            f"no data at opt_level={args.opt_level!r} region={args.region}"
        )

    aligned = base.join(opt, lsuffix="_base", rsuffix="_opt", how="outer")
    # Reorder columns so each metric appears as (base, opt) pair
    ordered = []
    for m in METRICS:
        if f"{m}_base" in aligned.columns:
            ordered.append(f"{m}_base")
        if f"{m}_opt" in aligned.columns:
            ordered.append(f"{m}_opt")
    aligned = aligned[ordered].sort_index()

    formatted = aligned.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].apply(
            lambda v: f"{v:.3f}" if pd.notna(v) else "—"
        )

    header = (
        f"\nHW counter metrics at {args.opt_level}, TIME_REGION={args.region}\n"
        "  AI               = FLOPs / byte (DRAM-level via L3 misses * 64)\n"
        "  FLOPs_per_cycle  = DP FLOPs / cycle  (AVX2 peak = 16)\n"
        "  vIPC             = vector DP instructions / cycle\n"
        "  DP_per_vec       = effective lanes per vector instruction (AVX2 FMA ideal = 8)\n"
        "  %L1m, %L2m, %L3m = miss rates at each cache level\n"
        "  %Stall           = resource-stalled cycles\n"
    )
    print(header)
    if args.md:
        print(formatted.to_markdown())
    else:
        print(formatted.to_string())


if __name__ == "__main__":
    main()
