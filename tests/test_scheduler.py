"""Tests for src.scheduler public API — imports through the clean src/ facade.

Verifies that all public symbols are accessible via:
    from src.scheduler import WIQLScheduler, BeliefTracker, PeriodicInterceptModule
    from src.scheduler import ACTION_SCAN, ACTION_PASSIVE
    from src.scheduler import combine_indices_rank, select_top_k_combined

All algorithmic correctness tests live in tests/test_env.py and tests/test_periodic.py.
These tests verify the public interface contract of the src/ layer.
"""
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Importability — all public symbols must be accessible through src.scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_import_wiql_scheduler():
    from src.scheduler import WIQLScheduler
    assert WIQLScheduler is not None


def test_import_belief_tracker():
    from src.scheduler import BeliefTracker
    assert BeliefTracker is not None


def test_import_periodic_intercept_module():
    from src.scheduler import PeriodicInterceptModule
    assert PeriodicInterceptModule is not None


def test_import_action_constants():
    from src.scheduler import ACTION_SCAN, ACTION_PASSIVE
    assert ACTION_SCAN == 1
    assert ACTION_PASSIVE == 0


def test_import_index_helpers():
    from src.scheduler import combine_indices_rank, select_top_k_combined
    assert callable(combine_indices_rank)
    assert callable(select_top_k_combined)


# ─────────────────────────────────────────────────────────────────────────────
# WIQLScheduler interface through src.scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_wiql_scheduler_select_arms_returns_k():
    from src.scheduler import WIQLScheduler
    sched = WIQLScheduler(n_bands=8, k_scan=3)
    belief = np.full(8, 0.5)
    arms = sched.select_arms(belief)
    assert len(arms) == 3
    assert isinstance(arms, set)
    assert all(0 <= b < 8 for b in arms)


def test_wiql_scheduler_first_visit_all_inf():
    from src.scheduler import WIQLScheduler
    sched = WIQLScheduler(n_bands=6, k_scan=2)
    belief = np.full(6, 0.5)
    indices = sched.compute_indices(belief)
    assert np.all(np.isinf(indices))


def test_wiql_scheduler_k_ge_n_raises():
    from src.scheduler import WIQLScheduler
    with pytest.raises(ValueError):
        WIQLScheduler(n_bands=4, k_scan=4)


# ─────────────────────────────────────────────────────────────────────────────
# BeliefTracker interface through src.scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_belief_tracker_beliefs_in_unit_interval():
    from src.scheduler import BeliefTracker
    bt = BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.85)
    for obs in [1, 0, None, 1, 0]:
        bt.update(0, obs)
    beliefs = bt.get_all()
    assert np.all(beliefs >= 0.0)
    assert np.all(beliefs <= 1.0)


def test_belief_tracker_invalid_params_raise():
    from src.scheduler import BeliefTracker
    with pytest.raises(ValueError):
        BeliefTracker(n_bands=4, p_stay_occ=1.1, p_stay_idle=0.9)


# ─────────────────────────────────────────────────────────────────────────────
# PeriodicInterceptModule interface through src.scheduler
# ─────────────────────────────────────────────────────────────────────────────

def test_periodic_module_cold_start_returns_zero():
    from src.scheduler import PeriodicInterceptModule
    module = PeriodicInterceptModule(n_bands=4)
    idx = module.compute_all_indices(t_now_us=1000.0)
    assert np.all(idx >= 0.0)
    # Cold start: no samples yet → all indices should be 0
    assert np.all(idx == 0.0)


def test_periodic_module_no_state_leakage_to_belief():
    """Property 4: PeriodicInterceptModule has no shared state with BeliefTracker."""
    from src.scheduler import BeliefTracker, PeriodicInterceptModule
    bt = BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.85)
    beliefs_before = bt.get_all().copy()
    module = PeriodicInterceptModule(n_bands=4)
    # Ingest some pulses into the module
    for t in [100.0, 600.0, 1100.0, 1600.0]:
        module.ingest_pulse(0, t)
    module.compute_all_indices(t_now_us=2000.0)
    # BeliefTracker state must be completely unchanged
    beliefs_after = bt.get_all()
    assert np.allclose(beliefs_before, beliefs_after), \
        "PeriodicInterceptModule must not modify BeliefTracker state (Property 4)"
