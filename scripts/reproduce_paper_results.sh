#!/usr/bin/env bash
# ============================================================
# Reproduce all paper results (post-fix system, warmup=50)
# SIH 2026 PS 26055 — Smart Scan Strategy for EW
# Coding Saints, Team ID 120303
#
# Usage:
#   bash scripts/reproduce_paper_results.sh           # full run  (~25 min)
#   bash scripts/reproduce_paper_results.sh --quick   # 1-seed check (~2 min)
#
# Writes fresh CSVs to:  results/paper_results/
# Writes summary JSON:   results/paper_results/summary.json
# ============================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Parse --quick flag ────────────────────────────────────────────────────────
QUICK=0
for arg in "$@"; do
  if [[ "$arg" == "--quick" ]]; then QUICK=1; fi
done

if [[ $QUICK -eq 1 ]]; then
  N_SEEDS=1
  T_STEPS=500
  echo "============================================================"
  echo "  Smart Scan Strategy — PS 26055 — QUICK MODE"
  echo "  1 seed x 500 steps — pipeline check only, not paper numbers"
  echo "  Run without --quick for the full N=30, T=5000 paper results"
  echo "============================================================"
else
  N_SEEDS=30
  T_STEPS=5000
  echo "============================================================"
  echo "  Smart Scan Strategy — PS 26055 — Full Paper Reproduction"
  echo "  Coding Saints, Team ID 120303"
  echo "  Config: N=30, T=5000, n_bands=8, K=3, warmup=50, gamma=5"
  echo "  Estimated runtime: ~25 minutes"
  echo "  Add --quick for a 2-minute single-seed pipeline check"
  echo "============================================================"
fi
echo ""

# Verify dependencies
python3 -c "import numpy, scipy, matplotlib, pandas" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

python3 -c "
import sys; sys.path.insert(0,'.')
from src.evaluation.runner import WIQLPolicy, run_episode
from src.evaluation.env_generator import RenewalEnv
" || {
    echo "ERROR: Cannot import src/. Run from repo root."
    exit 1
}

mkdir -p results/paper_results

# ── Run the Python harness ────────────────────────────────────────────────────
python3 scripts/_repro_runner.py "$N_SEEDS" "$T_STEPS"

echo ""
echo "============================================================"
if [[ $QUICK -eq 1 ]]; then
  echo "  Quick check complete. Pipeline works end-to-end."
  echo "  Run without --quick for N=30 paper results."
else
  echo "  Full reproduction complete."
  echo "  Results: results/paper_results/"
  echo "  Summary: results/paper_results/summary.json"
fi
echo "============================================================"
