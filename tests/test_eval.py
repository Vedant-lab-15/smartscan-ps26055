"""Tests for EvaluationHarness — all PS figures of merit.

Every metric is verified against a hand-constructed scenario where the
correct answer can be computed by inspection. This is critical because
these numbers go in front of judges.

Scenarios:
  A) 4-step episode, 4 bands, K=2:
       Step 0: scanned={0,1}, true_occ={0,1}, detected={0,1}  → TP=2
       Step 1: scanned={0,1}, true_occ={0},   detected={0}    → TP=1 TN=1
       Step 2: scanned={0,1}, true_occ={1},   detected={}     → FN=1 TN=1
       Step 3: scanned={0,1}, true_occ={},    detected={1}    → FP=1 TN=1
     Expected:  tp=3, fp=1, fn=1, tn=3
                Pd  = 3/4 = 0.75
                Pfa = 1/4 = 0.25
                Pd + miss_rate = 0.75 + 0.25 = 1.0  ✓

  B) Belief accuracy check: 2 bands, 1 step
       belief=[0.9, 0.1], true_occ={0} (band 0 occupied, band 1 idle)
       Predicted: band 0 occupied (0.9>0.5) ✓, band 1 idle (0.1<0.5) ✓
       Expected: pct_correct = 2/2 = 1.0

  C) Pd + Miss Rate = 1 (Property 7) — confirmed algebraically.

  D) Intercept-time error: direct numerical check.

  E) run_evaluation_episode integration — wires full stack.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ew_smart_scan.eval.evaluation import EvaluationHarness, run_evaluation_episode


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_scenario_a() -> EvaluationHarness:
    """
    4-step episode, 4 bands, K=2.
    Manually verified confusion matrix: tp=3, fp=1, fn=1, tn=3.
    """
    h = EvaluationHarness(n_bands=4, k_scan=2)
    belief = np.full(4, 0.5)

    steps = [
        # (scanned, true_occupied, detected)
        ({0, 1}, {0, 1}, {0, 1}),   # step 0: both occupied, both detected → TP=2
        ({0, 1}, {0},   {0}),       # step 1: band 0 occ+det, band 1 idle+silent → TP=1 TN=1
        ({0, 1}, {1},   set()),     # step 2: band 1 occ+missed → FN=1; band 0 idle+silent → TN=1
        ({0, 1}, set(), {1}),       # step 3: band 1 idle+false alarm → FP=1; band 0 idle+silent → TN=1
    ]

    for scanned, true_occ, detected in steps:
        h.record_step(
            scanned=scanned,
            true_occupied=true_occ,
            detected=detected,
            belief=belief,
            raw_reward=1.0 if detected else 0.0,
        )
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_confusion_matrix_scenario_a():
    """Hand-verified confusion matrix matches expected tp/fp/fn/tn."""
    h = _build_scenario_a()
    m = h.summary()
    assert m["tp"] == 3, f"Expected tp=3, got {m['tp']}"
    assert m["fp"] == 1, f"Expected fp=1, got {m['fp']}"
    assert m["fn"] == 1, f"Expected fn=1, got {m['fn']}"
    assert m["tn"] == 3, f"Expected tn=3, got {m['tn']}"


def test_pd_scenario_a():
    """Pd = tp / (tp + fn) = 3 / 4 = 0.75."""
    h = _build_scenario_a()
    pd = h.compute_pd()
    assert pd is not None
    assert abs(pd - 0.75) < 1e-10, f"Expected Pd=0.75, got {pd}"


def test_pfa_scenario_a():
    """Pfa = fp / (fp + tn) = 1 / 4 = 0.25."""
    h = _build_scenario_a()
    pfa = h.compute_pfa()
    assert pfa is not None
    assert abs(pfa - 0.25) < 1e-10, f"Expected Pfa=0.25, got {pfa}"


def test_sensitivity_equals_pd():
    """Sensitivity == Pd (true-positive rate) by definition in this context."""
    h = _build_scenario_a()
    assert h.compute_sensitivity() == h.compute_pd()


# ─────────────────────────────────────────────────────────────────────────────
# Property 7: Pd + Miss Rate = 1.0
# ─────────────────────────────────────────────────────────────────────────────

def test_pd_plus_miss_rate_equals_one():
    """Pd + miss_rate = 1.0 exactly (Property 7 / Requirement 11.3)."""
    h = _build_scenario_a()
    tp, _, fn, _ = h._confusion()
    pd = h.compute_pd()
    miss_rate = fn / (tp + fn)
    assert abs(pd + miss_rate - 1.0) < 1e-9, (
        f"Pd ({pd}) + miss_rate ({miss_rate}) != 1.0"
    )


def test_pd_none_when_no_occupied_bands_scanned():
    """compute_pd() returns None when no occupied bands were scanned."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    belief = np.full(4, 0.2)
    # All scanned bands are idle
    h.record_step(
        scanned={0, 1},
        true_occupied=set(),    # nothing occupied
        detected=set(),
        belief=belief,
        raw_reward=0.0,
    )
    assert h.compute_pd() is None


def test_pfa_none_when_no_idle_bands_scanned():
    """compute_pfa() returns None when every scanned band was occupied."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    belief = np.full(4, 0.8)
    # All scanned bands are occupied
    h.record_step(
        scanned={0, 1},
        true_occupied={0, 1},
        detected={0, 1},
        belief=belief,
        raw_reward=1.0,
    )
    assert h.compute_pfa() is None


# ─────────────────────────────────────────────────────────────────────────────
# Property 7 (PBT): Pd + Miss Rate = 1 for any random episode
# ─────────────────────────────────────────────────────────────────────────────

@given(
    st.lists(
        st.tuples(
            # scanned (subset of 0..7)
            st.frozensets(st.integers(0, 7), min_size=1, max_size=3),
            # true_occupied (subset of 0..7)
            st.frozensets(st.integers(0, 7), min_size=0, max_size=8),
            # detected (subset of scanned)
            st.frozensets(st.integers(0, 7), min_size=0, max_size=3),
        ),
        min_size=5, max_size=50,
    )
)
def test_property_pd_plus_miss_rate_any_episode(steps):
    """Property 7: Pd + miss_rate == 1.0 for any episode with ≥1 scanned occupied band."""
    h = EvaluationHarness(n_bands=8, k_scan=3)
    belief = np.full(8, 0.5)
    for scanned, true_occ, detected in steps:
        # Only count detected pulses that were actually scanned
        detected_valid = detected & scanned
        h.record_step(
            scanned=set(scanned),
            true_occupied=set(true_occ),
            detected=set(detected_valid),
            belief=belief,
            raw_reward=float(len(detected_valid)) / max(len(scanned), 1),
        )
    tp, _, fn, _ = h._confusion()
    if tp + fn == 0:
        return  # no occupied bands scanned — skip
    pd = h.compute_pd()
    assert pd is not None
    miss_rate = fn / (tp + fn)
    assert abs(pd + miss_rate - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Intercept rate
# ─────────────────────────────────────────────────────────────────────────────

def test_avg_intercept_rate_scenario_a():
    """avg_intercept_rate = tp / n_steps = 3 / 4 = 0.75."""
    h = _build_scenario_a()
    rate = h.compute_avg_intercept_rate()
    assert abs(rate - 0.75) < 1e-10, f"Expected 0.75, got {rate}"


# ─────────────────────────────────────────────────────────────────────────────
# Reward / cost
# ─────────────────────────────────────────────────────────────────────────────

def test_avg_raw_reward():
    """avg_raw_reward is mean of the per-step rewards passed in."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    belief = np.full(4, 0.5)
    rewards = [0.5, 1.0, 0.0, 0.75]
    for r in rewards:
        h.record_step(
            scanned={0, 1},
            true_occupied={0},
            detected={0} if r > 0 else set(),
            belief=belief,
            raw_reward=r,
        )
    expected = sum(rewards) / len(rewards)
    assert abs(h.compute_avg_raw_reward() - expected) < 1e-10


def test_avg_net_reward_less_than_raw():
    """Net reward = raw - cost must be <= raw reward (cost ≥ 0)."""
    h = _build_scenario_a()
    assert h.compute_avg_net_reward() <= h.compute_avg_raw_reward() + 1e-10


def test_avg_cost_positive():
    """Cost is always >= 0."""
    h = _build_scenario_a()
    assert h.compute_avg_cost() >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# % Correct predictions (belief accuracy)
# ─────────────────────────────────────────────────────────────────────────────

def test_pct_correct_predictions_perfect():
    """Perfect belief → correct prediction for all bands."""
    h = EvaluationHarness(n_bands=2, k_scan=1)
    # band 0 occupied, band 1 idle
    belief = np.array([0.9, 0.1])  # predicts 0=occ, 1=idle
    h.record_step(
        scanned={0},
        true_occupied={0},
        detected={0},
        belief=belief,
        raw_reward=1.0,
    )
    assert abs(h.compute_pct_correct_predictions() - 1.0) < 1e-10


def test_pct_correct_predictions_inverted():
    """Inverted belief → 0% correct (worst case)."""
    h = EvaluationHarness(n_bands=2, k_scan=1)
    # band 0 idle, band 1 occupied — but belief says opposite
    belief = np.array([0.9, 0.1])  # predicts 0=occ (wrong), 1=idle (wrong)
    h.record_step(
        scanned={0},
        true_occupied={1},   # actual: band 1 is occupied
        detected=set(),
        belief=belief,
        raw_reward=0.0,
    )
    assert abs(h.compute_pct_correct_predictions() - 0.0) < 1e-10


def test_pct_correct_predictions_half():
    """One band right, one band wrong → 50% correct."""
    h = EvaluationHarness(n_bands=2, k_scan=1)
    belief = np.array([0.9, 0.9])  # predicts both occupied
    h.record_step(
        scanned={0},
        true_occupied={0},    # band 0: correct (occ predicted, occ actual)
                              # band 1: wrong (occ predicted, idle actual)
        detected={0},
        belief=belief,
        raw_reward=1.0,
    )
    assert abs(h.compute_pct_correct_predictions() - 0.5) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Intercept-time error (periodic module placeholder)
# ─────────────────────────────────────────────────────────────────────────────

def test_intercept_time_error_none_before_module_c():
    """Returns None before any periodic predictions are recorded."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    assert h.compute_avg_intercept_time_error_us() is None


def test_intercept_time_error_computes_correctly():
    """Mean absolute error computed correctly from recorded predictions."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    h.record_periodic_prediction(band_id=0, predicted_toa_us=1000.0, actual_toa_us=1050.0)
    h.record_periodic_prediction(band_id=0, predicted_toa_us=2000.0, actual_toa_us=1950.0)
    # Expected: (50 + 50) / 2 = 50.0 μs
    err = h.compute_avg_intercept_time_error_us()
    assert err is not None
    assert abs(err - 50.0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Summary dict keys
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_contains_all_required_keys():
    """summary() always returns all documented keys — stable output format."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    m = h.summary()
    required = {
        "pd", "pfa", "sensitivity",
        "avg_intercept_rate",
        "avg_raw_reward", "avg_cost", "avg_net_reward",
        "pct_correct_predictions",
        "avg_intercept_time_error_us",
        "tp", "fp", "fn", "tn",
        "n_steps", "n_bands", "k_scan",
    }
    missing = required - set(m.keys())
    assert not missing, f"summary() is missing keys: {missing}"


def test_summary_zero_steps():
    """summary() handles zero-step harness gracefully (no division by zero)."""
    h = EvaluationHarness(n_bands=4, k_scan=2)
    m = h.summary()
    assert m["pd"] is None
    assert m["pfa"] is None
    assert m["avg_raw_reward"] == 0.0
    assert m["n_steps"] == 0


def test_reset_clears_all_data():
    """reset() clears all records so subsequent summary is clean."""
    h = _build_scenario_a()
    assert h.summary()["n_steps"] == 4
    h.reset()
    m = h.summary()
    assert m["n_steps"] == 0
    assert m["pd"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Integration: run_evaluation_episode with real env + WIQL-UCB
# ─────────────────────────────────────────────────────────────────────────────

def test_run_evaluation_episode_integration():
    """run_evaluation_episode wires env+receiver+scheduler and returns valid metrics."""
    from ew_smart_scan.env.rf_environment import RFEnvironment
    from ew_smart_scan.env.belief_tracker import BeliefTracker
    from ew_smart_scan.env.receiver_model import ReceiverModel
    from ew_smart_scan.env.wiql_ucb import WIQLScheduler, ACTION_SCAN, ACTION_PASSIVE
    from ew_smart_scan.models.pdw_generator import PDWGenerator

    n_bands, k_scan, n_steps = 8, 3, 100
    band_cf    = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0,
                       p_emit=0.4, seed=42)
    env = RFEnvironment(source=gen, n_bands=n_bands,
                        band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt,
                              n_bands=n_bands, k_scan=k_scan)

    sched = WIQLScheduler(n_bands=n_bands, k_scan=k_scan)

    def update_fn(scheduler, obs_dict, reward, belief_prev, belief_next):
        for band_id in range(n_bands):
            was_scanned = band_id in scheduler.select_arms.__self__ if hasattr(
                scheduler.select_arms, '__self__') else False
        # Simple update: just call update for all bands
        for band_id in range(n_bands):
            obs = 1 if len(obs_dict.get(band_id, [])) > 0 else 0
            action = ACTION_SCAN  # approximate; real loop tracks action per band
            scheduler.update(
                band_id=band_id,
                action=action,
                reward=float(obs),
                belief_prev=float(belief_prev[band_id]),
                belief_next=float(belief_next[band_id]),
            )

    metrics = run_evaluation_episode(
        rf_env=env,
        receiver=receiver,
        scheduler=sched,
        t_steps=n_steps,
    )

    # Sanity checks — not value-specific, just well-formed
    assert metrics["n_steps"] == n_steps
    assert metrics["n_bands"] == n_bands
    assert metrics["k_scan"]  == k_scan
    assert 0 <= metrics["avg_raw_reward"] <= 1.0
    assert 0 <= metrics["pct_correct_predictions"] <= 1.0
    assert metrics["avg_intercept_time_error_us"] is None  # Module C not active

    pd = metrics["pd"]
    if pd is not None:
        assert 0.0 <= pd <= 1.0, f"Pd out of range: {pd}"
    pfa = metrics["pfa"]
    if pfa is not None:
        assert 0.0 <= pfa <= 1.0, f"Pfa out of range: {pfa}"

    print(f"\n  Integration metrics (100 steps, WIQL-UCB):")
    for k, v in metrics.items():
        if v is not None and isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
