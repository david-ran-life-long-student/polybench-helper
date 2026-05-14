#!/usr/bin/env bash
#
# run_experiments.sh — drive the full baseline + opt experiment suite.
#
# Recommended invocation on the remote machine:
#
#     nohup ./run_experiments.sh > run.log 2>&1 &
#     disown
#
# Then later:
#
#     tail -f run.log              # watch progress
#     less runtime_table.txt       # final profitability table
#     ls build/runtime*/ build/counters*/   # CSVs for further analysis
#
# The experiment_helper framework caches binaries and skips
# already-completed (config, run-iteration) pairs, so rerunning this
# script after an interruption picks up where it left off.

set -euo pipefail

cd "$(dirname "$0")"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }


log "=== step 1/4: runtime study (baseline) ==="
# compare_runtime.py builds + runs both studies, then prints the table.
# Tee the table to a file so we can find it later without rerunning.
python3 hw4_correlation_runtime.py


log "=== step 2/4: runtime study (opt) ==="
# compare_runtime.py builds + runs both studies, then prints the table.
# Tee the table to a file so we can find it later without rerunning.
python3 hw4_correlation_runtime_opt.py


log "=== step 3/4: HW counter study — baseline ==="
python3 hw4_correlation_counters.py


log "=== step 4/4: HW counter study — opt ==="
python3 hw4_correlation_counters_opt.py


log "=== done ==="
echo
log "Artifacts:"
log "  runtime table:        runtime_table.txt"
log "  runtime CSVs:         build/runtime/results_*.csv  build/runtime_opt/results_*.csv"
log "  HW counter CSVs:      build/counters/results_*.csv build/counters_opt/results_*.csv"
