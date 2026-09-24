"""Tests for belief state update correctness — via src.scheduler.BeliefTracker.

Tests the HMM belief update equations directly:
  - Chapman-Kolmogorov (unscanned bands)
  - Bayesian posterior update (scanned bands with hit or miss)
  - Boundary conditions and numerical stability
"""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.scheduler import BeliefTracker


# ─────────────────────────────────────────────────────────────────────────────
# Chapman-Kolmogorov (unscanned step)
# ─────────────────────────────────────────────────────────────────────────────

def test_ck_update_from_known_prior():
    """update(obs=None) produces exactly the Chapman-Kolmogorov prediction.

    Manual calculation:
      b_pred = p_stay_occ · b + (1 - p_stay_idle) · (1 - b)
             = 0.8 · 0.5 + (1 - 0.9) · 0.5
             = 0.40 + 0.05 = 0.45
    """
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.8, p_stay_idle=0.9, prior=0.5)
    bt.update(0, None)
    beliefs = bt.get_all()
    assert abs(beliefs[0] - 0.45) < 1e-10, f"Expected 0.45, got {beliefs[0]}"


def test_ck_only_affects_updated_band():
    """update(0, None) updates only band 0; band 1 stays at prior."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.4)
    bt.update(0, None)
    beliefs = bt.get_all()
    assert abs(beliefs[1] - 0.4) < 1e-10, f"Band 1 should remain at prior 0.4, got {beliefs[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian update (scanned bands)
# ─────────────────────────────────────────────────────────────────────────────

def test_hit_increases_belief():
    """Observing a pulse (obs=1) raises belief above the CK prediction."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.5)
    b_pred = bt.predict(0)
    bt.update(0, 1)
    assert bt.get_all()[0] > b_pred, "Hit should increase belief above CK prediction"


def test_miss_decreases_belief():
    """No pulse (obs=0) lowers belief below the CK prediction."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.5)
    b_pred = bt.predict(0)
    bt.update(0, 0)
    assert bt.get_all()[0] < b_pred, "Miss should decrease belief below CK prediction"


def test_repeated_hits_drive_belief_high():
    """Repeated hits drive belief toward 1.0."""
    bt = BeliefTracker(n_bands=1, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.5)
    for _ in range(50):
        bt.update(0, 1)
    assert bt.get_all()[0] > 0.85, f"Repeated hits should drive belief high, got {bt.get_all()[0]}"


def test_repeated_misses_drive_belief_low():
    """Repeated misses drive belief toward 0.0."""
    bt = BeliefTracker(n_bands=1, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.5)
    for _ in range(50):
        bt.update(0, 0)
    assert bt.get_all()[0] < 0.15, f"Repeated misses should drive belief low, got {bt.get_all()[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_belief_never_exactly_zero_or_one():
    """Bayesian update with non-degenerate p_detect/p_fa can never produce exactly 0 or 1."""
    bt = BeliefTracker(n_bands=1, p_stay_occ=0.9, p_stay_idle=0.85,
                       prior=0.5, p_detect=0.9, p_fa=0.01)
    for obs in [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]:
        bt.update(0, obs)
        b = float(bt.get_all()[0])
        assert 0.0 < b < 1.0, f"Belief should be strictly in (0,1), got {b}"


def test_get_all_returns_copy():
    """Mutating the returned array from get_all() does not affect internal state."""
    bt = BeliefTracker(n_bands=3, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.3)
    arr = bt.get_all()
    arr[:] = 0.999
    internal = bt.get_all()
    assert np.allclose(internal, 0.3), "get_all() must return a copy, not a reference"


# ─────────────────────────────────────────────────────────────────────────────
# Property-based: belief always stays in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

@given(
    st.lists(
        st.tuples(
            st.integers(0, 7),
            st.one_of(st.just(None), st.integers(0, 1)),
        ),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=100)
def test_property_belief_always_in_unit_interval(updates):
    """Property 1: belief[i](t) ∈ [0, 1] for any sequence of observations."""
    bt = BeliefTracker(n_bands=8, p_stay_occ=0.9, p_stay_idle=0.85)
    for band_id, obs in updates:
        bt.update(band_id, obs)
    beliefs = bt.get_all()
    assert np.all(beliefs >= 0.0), f"Belief below 0: {beliefs}"
    assert np.all(beliefs <= 1.0), f"Belief above 1: {beliefs}"


# ─────────────────────────────────────────────────────────────────────────────
# Constructor validation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,match", [
    ({"p_stay_occ": 0.0, "p_stay_idle": 0.9}, "p_stay_occ"),
    ({"p_stay_occ": 0.9, "p_stay_idle": 1.0}, "p_stay_idle"),
    ({"p_stay_occ": 0.9, "p_stay_idle": 0.9, "prior": -0.1}, "prior"),
    ({"p_stay_occ": 0.9, "p_stay_idle": 0.9, "p_detect": 0.01, "p_fa": 0.1}, "p_detect"),
])
def test_constructor_validation(kwargs, match):
    base = {"n_bands": 4, "p_stay_occ": 0.9, "p_stay_idle": 0.85}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        BeliefTracker(**base)
