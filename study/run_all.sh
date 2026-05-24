#!/usr/bin/env bash
#
# run_all.sh — drive the full baseline + opt experiment suite.
#
# Run from anywhere; the script cd's to the repo root itself. Recommended
# invocation on the remote machine:
#
#     nohup ./study/run_all.sh > run.log 2>&1 &
#     disown
#
# Then later:
#
#     tail -f run.log                      # watch progress
#     ls build/runtime*/ build/counters*/  # raw CSVs (one per study)
#
# The experiment_helper framework caches binaries and skips already-completed
# (config, run-iteration) pairs, so rerunning after an interruption resumes
# where it left off. The curated CSVs analyzed by the analyze_*.py scripts live
# in results/ — copy the build/*/results_*.csv you want to keep into there.

set -euo pipefail

cd "$(dirname "$0")/.."          # repo root: scripts assume this is the CWD
export PYTHONUNBUFFERED=1        # so python output streams into the log live

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }


log "=== step 1/4: runtime study (baseline) ==="
python3 study/runtime_baseline.py

log "=== step 2/4: runtime study (opt) ==="
python3 study/runtime_opt.py

log "=== step 3/4: HW counter study (baseline) ==="
python3 study/counters_baseline.py

log "=== step 4/4: HW counter study (opt) ==="
python3 study/counters_opt.py

log "=== done ==="
echo
log "Raw CSVs:"
log "  runtime:     build/runtime/results_*.csv      build/runtime_opt/results_*.csv"
log "  HW counters: build/counters/results_*.csv     build/counters_opt/results_*.csv"
