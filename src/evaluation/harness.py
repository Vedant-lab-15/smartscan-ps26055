"""Evaluation Harness — all figures of merit for the EW scheduler.

This module computes every metric the PS judges on:
  - Pd    : Probability of Detection
  - Pfa   : Probability of False Alarm
  - Sensitivity : true-positive rate (= Pd in this context — see note below)
  - Average intercept rate
  - Average reward / cost
  - % correct predictions (belief tracker accuracy vs ground truth)
  - Average intercept-time error (periodic emitters — placeholder until Module C)

NOTE on Sensitivity vs Pd:
  In radar/EW literature "sensitivity" sometimes refers to the minimum
  detectable signal (an RF hardware parameter), but in the PS scoring context
  it means the classifier's true-positive rate = Pd. We treat them as
  identical here and return both keys with the same value. If the PS scoring
  rubric later distinguishes them, update _compute_sensitivity() accordingly.

NOTE on Cost model:
  The multi-objective reward from the brief weights detections by threat
  likelihood. Currently the reward signal from ReceiverModel is binary
  (detected / not). We add an explicit cost term:
    cost_per_step = SCAN_COST * k_scan + MISS_COST * n_misses
  where SCAN_COST = 0.05 per scanned band (resource cost) and
  MISS_COST = 1.0 per missed occupied band (opportunity cost).
  net_reward = raw_reward - cost_per_step
  Both raw and net rewards are reported.

NOTE on Intercept-time error:
  Requires the PeriodicInterceptModule (Module C). Until that module lands,
  this metric returns None. The plumbing is in place — call
  harness.record_periodic_prediction(band_id, predicted_toa, actual_toa)
  from the main loop once Module C is active.

Usage:
    harness = EvaluationHarness(n_bands=8, k_scan=3)
    harness.reset()
    for t in range(T):
        # get true state from env (oracle — harness only)
        true_occ = env.get_true_state(t)           # dict {band_id: bool}
        obs_dict, reward, _ = receiver.step(action)
        belief = receiver.get_belief()
        harness.record_step(
            scanned=action,
            true_occupied={b for b, v in true_occ.items() if v},
            detected={b for b in action if len(obs_dict[b]) > 0},
            belief=belief,
            true_occupied_all={b for b, v in true_occ.items() if v},
            raw_reward=reward,
        )
    metrics = harness.summary()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Cost model constants (documented, easily tunable)
# ─────────────────────────────────────────────────────────────────────────────
SCAN_COST: float = 0.05   # resource cost per scanned band per step
MISS_COST: float = 1.0    # opportunity cost per missed occupied band


@dataclass
class _StepRecord:
    """Internal record for one time step."""
    scanned:           frozenset[int]
    true_occupied:     frozenset[int]   # bands that were truly occupied this step
    detected:          frozenset[int]   # bands where a pulse was observed
    belief:            np.ndarray       # full belief vector (n_bands,)
    raw_reward:        float


class EvaluationHarness:
    """Compute all PS figures of merit over an episode.

    The harness is the ONLY component that touches get_true_state() output —
    it is passed in via record_step(), never accessed directly here.

    Metrics returned by summary():
        pd                         – Probability of Detection ∈ [0,1] or None
        pfa                        – Probability of False Alarm ∈ [0,1] or None
        sensitivity                – Identical to pd (true-positive rate)
        avg_intercept_rate         – tp / total_steps
        avg_raw_reward             – mean per-step raw reward
        avg_net_reward             – mean per-step (raw_reward - cost)
        avg_cost                   – mean per-step cost
        pct_correct_predictions    – % steps where belief majority vote matches truth
        avg_intercept_time_error_us – mean |predicted - actual| ToA (None until Module C)
        n_steps                    – total steps recorded
        tp, fp, fn, tn             – cumulative confusion matrix counts
    """

    def __init__(self, n_bands: int, k_scan: int) -> None:
        self.n_bands = n_bands
        self.k_scan  = k_scan
        self._records: list[_StepRecord] = []
        # Periodic interception predictions: list of (predicted_toa, actual_toa)
        self._periodic_predictions: list[tuple[float, float]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Recording
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all accumulated data."""
        self._records.clear()
        self._periodic_predictions.clear()

    def record_step(
        self,
        scanned:        set[int],
        true_occupied:  set[int],
        detected:       set[int],
        belief:         np.ndarray,
        raw_reward:     float,
    ) -> None:
        """Record one time step.

        Args:
            scanned:       Band IDs that were scanned this step (the action).
            true_occupied: Band IDs that were truly occupied (from get_true_state).
                           Oracle access — only the harness receives this.
            detected:      Band IDs where a pulse was observed (subset of scanned).
            belief:        Full belief vector b_i ∈ [0,1] for all bands.
            raw_reward:    Per-step reward from ReceiverModel.step().
        """
        self._records.append(_StepRecord(
            scanned=frozenset(scanned),
            true_occupied=frozenset(true_occupied),
            detected=frozenset(detected),
            belief=np.array(belief, dtype=np.float64),
            raw_reward=float(raw_reward),
        ))

    def record_periodic_prediction(
        self, band_id: int, predicted_toa_us: float, actual_toa_us: float
    ) -> None:
        """Record a periodic-emitter interception prediction.

        Called by the main loop once Module C (PeriodicInterceptModule) is active.
        Until then, avg_intercept_time_error_us returns None.

        Args:
            band_id:          Which band the prediction is for (informational).
            predicted_toa_us: Module C's predicted next arrival time.
            actual_toa_us:    Ground-truth pulse ToA from the oracle.
        """
        self._periodic_predictions.append((predicted_toa_us, actual_toa_us))

    # ─────────────────────────────────────────────────────────────────────────
    # Confusion matrix accumulation
    # ─────────────────────────────────────────────────────────────────────────

    def _confusion(self) -> tuple[int, int, int, int]:
        """Return (tp, fp, fn, tn) accumulated over all steps.

        Definitions (per-band, per-step):
          tp: scanned ∧ occupied ∧ detected
          fp: scanned ∧ ¬occupied ∧ detected   (false alarm)
          fn: scanned ∧ occupied ∧ ¬detected   (missed detection)
          tn: scanned ∧ ¬occupied ∧ ¬detected
        Only scanned bands are counted — unscanned bands are neither
        detected nor missed (POMDP: we can't know what we didn't look at).
        """
        tp = fp = fn = tn = 0
        for r in self._records:
            for b in r.scanned:
                occ  = b in r.true_occupied
                det  = b in r.detected
                if occ and det:     tp += 1
                elif (not occ) and det:  fp += 1
                elif occ and (not det):  fn += 1
                else:               tn += 1
        return tp, fp, fn, tn

    # ─────────────────────────────────────────────────────────────────────────
    # Individual metric methods
    # ─────────────────────────────────────────────────────────────────────────

    def compute_pd(self) -> Optional[float]:
        """P(detected | scanned ∧ occupied). Returns None if no occupied bands scanned."""
        tp, _, fn, _ = self._confusion()
        if tp + fn == 0:
            return None
        return tp / (tp + fn)

    def compute_pfa(self) -> Optional[float]:
        """P(detected | scanned ∧ idle). Returns None if no idle bands scanned."""
        _, fp, _, tn = self._confusion()
        if fp + tn == 0:
            return None
        return fp / (fp + tn)

    def compute_sensitivity(self) -> Optional[float]:
        """True-positive rate = Pd in this context. See module docstring."""
        return self.compute_pd()

    def compute_avg_intercept_rate(self) -> float:
        """True detections per total steps (tp / n_steps)."""
        if not self._records:
            return 0.0
        tp, _, _, _ = self._confusion()
        return tp / len(self._records)

    def compute_avg_raw_reward(self) -> float:
        """Mean per-step raw reward from ReceiverModel."""
        if not self._records:
            return 0.0
        return float(np.mean([r.raw_reward for r in self._records]))

    def compute_avg_cost(self) -> float:
        """Mean per-step cost (scan resource cost + missed-detection penalty)."""
        if not self._records:
            return 0.0
        costs = []
        for r in self._records:
            n_misses = sum(
                1 for b in r.scanned
                if (b in r.true_occupied) and (b not in r.detected)
            )
            cost = SCAN_COST * len(r.scanned) + MISS_COST * n_misses
            costs.append(cost)
        return float(np.mean(costs))

    def compute_avg_net_reward(self) -> float:
        """Mean per-step net reward = raw_reward - cost."""
        if not self._records:
            return 0.0
        net = []
        for r in self._records:
            n_misses = sum(
                1 for b in r.scanned
                if (b in r.true_occupied) and (b not in r.detected)
            )
            cost = SCAN_COST * len(r.scanned) + MISS_COST * n_misses
            net.append(r.raw_reward - cost)
        return float(np.mean(net))

    def compute_pct_correct_predictions(self) -> float:
        """Belief tracker accuracy: fraction of (band, step) pairs where the
        belief majority vote (b_i > 0.5 → predict occupied) matches ground truth,
        considering ALL bands (not just scanned ones).

        Returns a value in [0, 1].
        """
        if not self._records:
            return 0.0
        correct = total = 0
        for r in self._records:
            for b in range(self.n_bands):
                predicted_occ = float(r.belief[b]) > 0.5
                actual_occ    = b in r.true_occupied
                if predicted_occ == actual_occ:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0.0

    def compute_avg_intercept_time_error_us(
        self, period_us: float | None = None
    ) -> float | None:
        """Mean intercept-time error (μs).

        For periodic emitters, raw |predicted - actual| is misleading when
        the prediction is off by exactly one period (phase-correct but
        cycle-off by 1). When period_us is provided, cyclic correction is
        applied:
            error = min(|raw_err|, period_us - |raw_err|)
        This gives the true phase error relative to the nearest emission.

        Returns None until Module C is active and record_periodic_prediction()
        has been called at least once.

        Args:
            period_us: If provided, apply cyclic (mod-T) correction. Recommended
                       for all periodic-emitter scenarios. None = raw absolute error.
        """
        if not self._periodic_predictions:
            return None
        errors = []
        for pred, actual in self._periodic_predictions:
            raw = abs(pred - actual)
            if period_us is not None and period_us > 0:
                # Cyclic correction: score against nearest emission modulo period
                err = min(raw, abs(period_us - raw))
            else:
                err = raw
            errors.append(err)
        return float(np.mean(errors))

    # ─────────────────────────────────────────────────────────────────────────
    # Full report
    # ─────────────────────────────────────────────────────────────────────────

    def summary(self, period_us: float | None = None) -> dict:
        """Return all metrics in a stable dict.

        Args:
            period_us: If provided, applies cyclic correction to
                       avg_intercept_time_error_us. Recommended for periodic-emitter
                       scenarios to avoid off-by-one-cycle inflation of this metric.

        Keys are fixed — downstream reporting/plotting can rely on these names.
        Metrics that require data not yet available return None (not omitted).
        """
        tp, fp, fn, tn = self._confusion()
        pd  = self.compute_pd()
        pfa = self.compute_pfa()
        return {
            "pd":                          pd,
            "pfa":                         pfa,
            "sensitivity":                 pd,
            "avg_intercept_rate":          self.compute_avg_intercept_rate(),
            "avg_raw_reward":              self.compute_avg_raw_reward(),
            "avg_cost":                    self.compute_avg_cost(),
            "avg_net_reward":              self.compute_avg_net_reward(),
            "pct_correct_predictions":     self.compute_pct_correct_predictions(),
            "avg_intercept_time_error_us": self.compute_avg_intercept_time_error_us(period_us),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_steps":  len(self._records),
            "n_bands":  self.n_bands,
            "k_scan":   self.k_scan,
        }

    def print_report(self, title: str = "Evaluation Report") -> None:
        """Print a formatted metrics report to stdout."""
        m = self.summary()
        w = 42
        print(f"\n{'─'*w}")
        print(f"  {title}")
        print(f"{'─'*w}")
        print(f"  Steps: {m['n_steps']}  |  Bands: {m['n_bands']}  |  K: {m['k_scan']}")
        print(f"{'─'*w}")

        def _fmt(v):
            if v is None: return "N/A (data pending)"
            if isinstance(v, float): return f"{v:.4f}"
            return str(v)

        rows = [
            ("Pd  (prob. detection)",     m["pd"]),
            ("Pfa (prob. false alarm)",   m["pfa"]),
            ("Sensitivity (= Pd)",        m["sensitivity"]),
            ("Avg intercept rate",        m["avg_intercept_rate"]),
            ("Avg raw reward",            m["avg_raw_reward"]),
            ("Avg cost",                  m["avg_cost"]),
            ("Avg net reward",            m["avg_net_reward"]),
            ("% correct predictions",     None if m["pct_correct_predictions"] is None
                                          else m["pct_correct_predictions"] * 100),
            ("Avg intercept-time err (μs)", m["avg_intercept_time_error_us"]),
            ("Confusion: TP/FP/FN/TN",   f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}"),
        ]
        for label, val in rows:
            # Format percentage separately
            if label.startswith("%"):
                display = f"{val:.2f}%" if val is not None else "N/A"
            elif label == "Confusion: TP/FP/FN/TN":
                display = str(val)
            else:
                display = _fmt(val)
            print(f"  {label:<36} {display}")
        print(f"{'─'*w}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runner — wires env + scheduler + harness for one episode
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_episode(
    rf_env,
    receiver,
    scheduler,                # any object with .select_arms(belief) -> set[int]
    t_steps: int,
    harness: EvaluationHarness | None = None,
    scheduler_update_fn=None, # optional callable(sched, obs, reward, b_prev, b_next)
) -> dict:
    """Run one full evaluation episode and return the metrics dict.

    Args:
        rf_env:            RFEnvironment — oracle ground truth source.
        receiver:          ReceiverModel — POMDP observation wrapper.
        scheduler:         Any scheduler with .select_arms(belief) -> set[int].
        t_steps:           Number of time steps to run.
        harness:           Optional pre-existing EvaluationHarness. If None,
                           a new one is created using receiver.n_bands / k_scan.
        scheduler_update_fn: Optional callable for online scheduler updates.
                           Signature: fn(scheduler, obs_dict, reward, belief_prev, belief_next)

    Returns:
        metrics dict from harness.summary()
    """
    if harness is None:
        harness = EvaluationHarness(
            n_bands=receiver.n_bands,
            k_scan=receiver.k_scan,
        )
    harness.reset()
    receiver.reset()

    for t in range(t_steps):
        belief_prev = receiver.get_belief().copy()
        action = scheduler.select_arms(belief_prev)

        obs_dict, raw_reward, done = receiver.step(action)
        belief_next = receiver.get_belief()

        # Oracle ground truth — harness access only
        true_state   = rf_env.get_true_state(t)
        true_occupied = {b for b, v in true_state.items() if v}
        detected      = {b for b in action if len(obs_dict[b]) > 0}

        harness.record_step(
            scanned=action,
            true_occupied=true_occupied,
            detected=detected,
            belief=belief_next,
            raw_reward=raw_reward,
        )

        if scheduler_update_fn is not None:
            scheduler_update_fn(scheduler, obs_dict, raw_reward, belief_prev, belief_next)

        if done:
            break

    return harness.summary()
