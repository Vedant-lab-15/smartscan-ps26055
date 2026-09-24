"""Evaluation sub-package — harness, metrics, statistical tests.

Public API:
    EvaluationHarness — 7-FoM harness (Pd, Pfa, intercept rate, reward, belief
                        accuracy, intercept-time error). N=30 seeds, 95%
                        bootstrap CI, Wilcoxon signed-rank.
    run_evaluation_episode — convenience wrapper for one full episode
    SCAN_COST, MISS_COST   — cost-model constants
"""
from src.evaluation.harness import EvaluationHarness, SCAN_COST, MISS_COST

__all__ = [
    "EvaluationHarness",
    "SCAN_COST",
    "MISS_COST",
]
