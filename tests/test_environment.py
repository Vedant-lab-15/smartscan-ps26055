"""Tests for src.environment public API — RFEnvironment, ReceiverModel, PDWGenerator.

Verifies the public interface contract of the src/environment layer and
key correctness properties:
  - Property 2: No oracle leakage (unscanned bands always return [])
  - POMDP gating: only scanned bands receive observations
  - Environment reproducibility: same seed → same occupancy sequence
"""
import numpy as np
import pytest

from src.environment import (
    RFEnvironment,
    ReceiverModel,
    PDWGenerator,
    Pulse,
    validate_pulse,
)
from src.scheduler import BeliefTracker


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stack(n_bands: int = 4, k_scan: int = 2,
                p_emit: float = 0.5, seed: int = 42):
    """Create a fresh (env, receiver) stack for testing."""
    band_cf    = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf,
                       dt_us=10.0, p_emit=p_emit, seed=seed)
    env = RFEnvironment(source=gen, n_bands=n_bands,
                        band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt  = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    rec = ReceiverModel(rf_env=env, belief_tracker=bt,
                         n_bands=n_bands, k_scan=k_scan)
    rec.reset()
    return env, rec, bt


# ─────────────────────────────────────────────────────────────────────────────
# RFEnvironment
# ─────────────────────────────────────────────────────────────────────────────

def test_rf_env_true_state_n_bands_keys():
    """get_true_state(t) returns exactly n_bands boolean-valued keys."""
    env, _, _ = _make_stack(n_bands=6)
    for t in range(5):
        state = env.get_true_state(t)
        assert len(state) == 6
        assert set(state.keys()) == set(range(6))
        assert all(isinstance(v, bool) for v in state.values())


def test_rf_env_reproducibility():
    """Same seed → identical occupancy sequence across two instances."""
    env1, _, _ = _make_stack(n_bands=4, seed=7)
    env2, _, _ = _make_stack(n_bands=4, seed=7)
    for t in range(20):
        assert env1.get_true_state(t) == env2.get_true_state(t), \
            f"Occupancy mismatch at t={t} — check seeding"


def test_rf_env_sample_pulses_occupied_band_non_empty():
    """sample_pulses returns non-empty list for an occupied band (p_emit=1.0)."""
    env, _, _ = _make_stack(n_bands=4, p_emit=1.0, seed=0)
    found_occupied = False
    for t in range(10):
        state = env.get_true_state(t)
        for band_id, occupied in state.items():
            pulses = env.sample_pulses(band_id, t)
            if occupied:
                assert len(pulses) > 0, \
                    f"Occupied band {band_id} returned no pulses at t={t}"
                found_occupied = True
            else:
                assert len(pulses) == 0, \
                    f"Idle band {band_id} returned pulses at t={t}"
    assert found_occupied, "Expected some occupied bands with p_emit=1.0"


# ─────────────────────────────────────────────────────────────────────────────
# ReceiverModel — POMDP gating (Property 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_receiver_obs_has_n_bands_keys():
    """step(action) returns obs_dict with exactly n_bands keys."""
    _, rec, _ = _make_stack(n_bands=4, k_scan=1)
    obs, _, _ = rec.step({0})
    assert len(obs) == 4


def test_receiver_unscanned_bands_always_empty():
    """Unscanned bands always return empty pulse lists — Property 2 (no oracle leakage)."""
    _, rec, _ = _make_stack(n_bands=4, k_scan=1, p_emit=1.0, seed=0)
    for _ in range(10):
        obs, _, _ = rec.step({0})
        for band_id in [1, 2, 3]:
            assert obs[band_id] == [], \
                f"Oracle leakage: band {band_id} returned pulses without being scanned"


def test_receiver_k_ge_n_raises():
    """ReceiverModel raises ValueError when k_scan >= n_bands."""
    band_edges = [(i * 100.0, (i+1)*100.0) for i in range(4)]
    gen = PDWGenerator(n_bands=4, band_cf_mhz=[100.0+i*100 for i in range(4)], dt_us=10.0)
    env = RFEnvironment(source=gen, n_bands=4, band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt = BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.85)
    with pytest.raises(ValueError, match="k_scan"):
        ReceiverModel(rf_env=env, belief_tracker=bt, n_bands=4, k_scan=4)


def test_receiver_reward_in_range():
    """Reward returned by step() is always in [0, 1]."""
    _, rec, _ = _make_stack(n_bands=4, k_scan=2)
    for _ in range(20):
        _, reward, _ = rec.step({0, 1})
        assert 0.0 <= reward <= 1.0, f"Reward out of [0,1]: {reward}"


def test_receiver_beliefs_stay_in_unit_interval():
    """Belief vector stays in [0, 1] after many steps with mixed actions."""
    _, rec, _ = _make_stack(n_bands=8, k_scan=3, p_emit=0.4, seed=5)
    for t in range(100):
        action = {t % 8, (t+1) % 8, (t+2) % 8}
        rec.step(action)
    beliefs = rec.get_belief()
    assert np.all(beliefs >= 0.0), f"Belief below 0: {beliefs}"
    assert np.all(beliefs <= 1.0), f"Belief above 1: {beliefs}"


def test_receiver_unscanned_bands_get_ck_update():
    """Unscanned bands receive Chapman-Kolmogorov update (restless property)."""
    _, rec, bt = _make_stack(n_bands=4, k_scan=2, p_emit=1.0, seed=0)
    # CK from prior 0.5: b_pred = 0.9*0.5 + 0.15*(1-0.5) = 0.45 + 0.075 = 0.525
    expected_ck = 0.9 * 0.5 + (1.0 - 0.85) * 0.5
    rec.step({0, 1})
    beliefs = bt.get_all()
    for band_id in [2, 3]:
        assert abs(beliefs[band_id] - expected_ck) < 1e-10, \
            f"Band {band_id}: expected CK={expected_ck:.6f}, got {beliefs[band_id]:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# PDWGenerator
# ─────────────────────────────────────────────────────────────────────────────

def test_pdw_generator_reproducibility():
    """Same seed produces identical pulse sequences."""
    band_cf = [900.0 + i * 100.0 for i in range(4)]
    gen1 = PDWGenerator(n_bands=4, band_cf_mhz=band_cf, dt_us=10.0, p_emit=0.5, seed=42)
    gen2 = PDWGenerator(n_bands=4, band_cf_mhz=band_cf, dt_us=10.0, p_emit=0.5, seed=42)
    for t in range(10):
        p1 = gen1.generate(t)
        p2 = gen2.generate(t)
        assert len(p1) == len(p2), f"Different pulse counts at t={t}"


def test_pdw_generator_per_band_p_emit():
    """Per-band p_emit=[1.0, 0.001, ...] makes band 0 always emit, band 1 almost never."""
    band_cf = [900.0 + i * 100.0 for i in range(4)]
    # PDWGenerator requires p_emit in (0, 1] — use 1.0 for always-on, 0.001 for near-silent
    p_emit = [1.0, 0.001, 0.5, 0.5]
    gen = PDWGenerator(n_bands=4, band_cf_mhz=band_cf, dt_us=10.0,
                       p_emit=p_emit, seed=0)
    band0_total = 0
    for t in range(50):
        pulses = gen.generate(t)
        band0_pulses = [p for p in pulses if 900 <= p.cf_mhz < 1000]
        band0_total += len(band0_pulses)
    # Band 0 with p_emit=1.0 should emit in every step
    assert band0_total == 50, f"Band 0 (p_emit=1.0) should emit in all 50 steps, got {band0_total}"


# ─────────────────────────────────────────────────────────────────────────────
# Pulse data model
# ─────────────────────────────────────────────────────────────────────────────

def test_pulse_validate_accepts_valid():
    """validate_pulse accepts a well-formed Pulse without raising."""
    p = Pulse(toa_us=1000.0, cf_mhz=900.0, pw_us=1.0,
              aoa_deg=45.0, amp_db=-30.0, emitter=0)
    validate_pulse(p)  # should not raise


def test_pulse_validate_rejects_negative_toa():
    """validate_pulse rejects toa_us <= 0."""
    p = Pulse(toa_us=-1.0, cf_mhz=900.0, pw_us=1.0,
              aoa_deg=0.0, amp_db=-30.0, emitter=0)
    with pytest.raises(ValueError):
        validate_pulse(p)


def test_pulse_validate_rejects_out_of_range_cf():
    """validate_pulse rejects CF outside [0, 18000] MHz."""
    p = Pulse(toa_us=100.0, cf_mhz=20000.0, pw_us=1.0,
              aoa_deg=0.0, amp_db=-30.0, emitter=0)
    with pytest.raises(ValueError):
        validate_pulse(p)
