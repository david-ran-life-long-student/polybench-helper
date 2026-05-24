"""
compare_runtime.py — set up + run both runtime studies, then print the
profitability table comparing opt vs baseline.

Defines two Study objects (baseline pointing at correlation.localized.c,
opt at correlation.opt.c), builds and runs both via experiment_helper,
then loads the resulting CSVs into pandas and produces the speedup table.

The framework caches binaries and skips already-completed (config, run)
pairs, so re-running this is cheap once the data is in place.

Comparisons:
    stats+center:  baseline(R1 + R2 + R3)  vs  opt(R6)
    corr matrix:   baseline(R4)            vs  opt(R4)
    whole kernel:  baseline(R0)            vs  opt(R0)
    transpose:     -- new cost --              opt(R5)

Run from the repo root: python3 study/compare_runtime.py
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "framework"))

import pandas as pd

import experiment_helper
from experiment_helper import Study, Mutable
from analysis_helper import compute_trimmed_mean


SIZES = [128, 512, 2048]
OPT_LEVELS = ["-O0", "-O3"]
BASELINE_REGIONS = [0, 1, 2, 3, 4]
OPT_REGIONS = [0, 4, 5, 6]


def make_study(build_dir, source_path, region_values):
    base_compile_command = " ".join([
        "-I polybench-c-4.2.1-beta/utilities",
        "-I polybench-c-4.2.1-beta/datamining/correlation",
        "-DPOLYBENCH_USE_RESTRICT",
        "-DPOLYBENCH_TIME",
        "polybench-c-4.2.1-beta/utilities/polybench.c",
        source_path,
        "-lm",
    ])
    return Study(
        build_dir=build_dir,
        experimental_params=[
            Mutable([f"-DSIZE={s}" for s in SIZES], name="Size"),
            Mutable(OPT_LEVELS, name="Opt_Level"),
            Mutable([f"-DTIME_REGION={r}" for r in region_values], name="Region"),
        ],
        base_compiler_command=base_compile_command,
        base_env_vars={},
        compiler="gcc",
    )


def run_both_studies(runs):
    experiment_helper.RUNS_PER_EXPERIMENT = runs

    baseline = make_study(
        build_dir="build/runtime",
        source_path="kernel/correlation.localized.c",
        region_values=BASELINE_REGIONS,
    )
    opt = make_study(
        build_dir="build/runtime_opt",
        source_path="kernel/correlation.opt.c",
        region_values=OPT_REGIONS,
    )

    print("=== baseline build + run ===")
    baseline.ensure_all_builds_exist()
    baseline.run_experiments(runs=runs)

    print("\n=== opt build + run ===")
    opt.ensure_all_builds_exist()
    opt.run_experiments(runs=runs)

    return baseline.build_dir, opt.build_dir


def _extract_int(value, prefix):
    m = re.search(rf"{re.escape(prefix)}=(-?\d+)", str(value))
    return int(m.group(1)) if m else None


def load_runtime_csv(build_dir):
    pattern = os.path.join(build_dir, "results_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no CSV matching {pattern}")
    if len(matches) > 1:
        print(f"warning: multiple CSVs in {build_dir}, using {matches[-1]}", file=sys.stderr)
    df = pd.read_csv(matches[-1])
    df["SizeN"] = df["Size"].apply(lambda v: _extract_int(v, "SIZE"))
    df["RegionN"] = df["Region"].apply(lambda v: _extract_int(v, "TIME_REGION"))
    return df


def pivot_by_region(df):
    grouped = compute_trimmed_mean(
        df, target_col="Result",
        group_cols=["SizeN", "Opt_Level", "RegionN"],
    )
    return grouped.pivot_table(
        index=["SizeN", "Opt_Level"],
        columns="RegionN",
        values="Result",
    )


def build_comparison(base_pivot, opt_pivot):
    needed_base = set(BASELINE_REGIONS)
    needed_opt = set(OPT_REGIONS)
    missing_base = needed_base - set(base_pivot.columns)
    missing_opt = needed_opt - set(opt_pivot.columns)
    if missing_base:
        raise ValueError(f"baseline pivot missing regions {sorted(missing_base)}")
    if missing_opt:
        raise ValueError(f"opt pivot missing regions {sorted(missing_opt)}")

    aligned = base_pivot.join(opt_pivot, how="inner", lsuffix="_b", rsuffix="_o")
    dropped = len(base_pivot) + len(opt_pivot) - 2 * len(aligned)
    if dropped > 0:
        print(f"warning: {dropped} (Size, Opt) cells appear in only one study; dropped",
              file=sys.stderr)

    out = pd.DataFrame(index=aligned.index)
    base_sc = aligned["1_b"] + aligned["2_b"] + aligned["3_b"]
    opt_sc = aligned["6_o"]

    out["base_SC_ms"]  = base_sc * 1000
    out["opt_SC_ms"]   = opt_sc * 1000
    out["spd_SC"]      = base_sc / opt_sc
    out["base_R4_ms"]  = aligned["4_b"] * 1000
    out["opt_R4_ms"]   = aligned["4_o"] * 1000
    out["spd_R4"]      = aligned["4_b"] / aligned["4_o"]
    out["xpose_ms"]    = aligned["5_o"] * 1000
    out["base_tot_ms"] = aligned["0_b"] * 1000
    out["opt_tot_ms"]  = aligned["0_o"] * 1000
    out["spd_tot"]     = aligned["0_b"] / aligned["0_o"]
    return out.sort_index()


def format_table(df, markdown=False):
    fmts = {
        "base_SC_ms":  "{:.3f}", "opt_SC_ms":   "{:.3f}", "spd_SC":  "{:.2f}x",
        "base_R4_ms":  "{:.3f}", "opt_R4_ms":   "{:.3f}", "spd_R4":  "{:.2f}x",
        "xpose_ms":    "{:.3f}",
        "base_tot_ms": "{:.3f}", "opt_tot_ms":  "{:.3f}", "spd_tot": "{:.2f}x",
    }
    styled = df.copy()
    for col, fmt in fmts.items():
        styled[col] = styled[col].apply(lambda v: fmt.format(v) if pd.notna(v) else "—")
    return styled.to_markdown() if markdown else styled.to_string()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10,
                    help="runs per (config) — passed to RUNS_PER_EXPERIMENT")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip building/running; just analyze existing CSVs")
    ap.add_argument("--md", action="store_true",
                    help="emit a GitHub-flavored markdown table")
    args = ap.parse_args()

    if args.skip_run:
        base_dir, opt_dir = "build/runtime", "build/runtime_opt"
    else:
        base_dir, opt_dir = run_both_studies(args.runs)

    base = load_runtime_csv(base_dir)
    opt = load_runtime_csv(opt_dir)
    table = build_comparison(pivot_by_region(base), pivot_by_region(opt))

    header = (
        "\nRuntime profitability — opt vs baseline\n"
        "  SC    = stats + center (baseline R1+R2+R3 vs opt R6)\n"
        "  R4    = correlation matrix (baseline R4 vs opt R4)\n"
        "  xpose = transpose overhead introduced by opt (no baseline counterpart)\n"
        "  tot   = whole kernel (baseline R0 vs opt R0)\n"
        "  times in milliseconds; speedup = baseline / opt (higher is better)\n"
    )
    if not args.md:
        print(header)
    print(format_table(table, markdown=args.md))


if __name__ == "__main__":
    main()
