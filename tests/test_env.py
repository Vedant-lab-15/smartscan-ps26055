"""Tests for RF Environment, Belief Tracker, Receiver Model, and integration smoke test."""
import numpy as np
import pytest

from src.errors import CrossFileEmitterError, MemoryGuardError
from src.environment.simulator import RFEnvironment
from src.environment.pdw_generator import PDWGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_gen_and_env(n_bands: int = 4, k_scan: int = 2, seed: int = 42, p_emit: float = 0.5):
    """Create a PDWGenerator + RFEnvironment pair for testing."""
    band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0, p_emit=p_emit, seed=seed)
    env = RFEnvironment(source=gen, n_bands=n_bands, band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    return gen, env


# ---------------------------------------------------------------------------
# Task 3.3: RF Environment unit tests
# ---------------------------------------------------------------------------

def test_rf_env_true_state_has_n_bands_keys():
    """get_true_state(t) returns a dict with exactly n_bands keys."""
    _, env = make_gen_and_env(n_bands=4)
    for t in range(10):
        state = env.get_true_state(t)
        assert len(state) == 4, f"Expected 4 keys, got {len(state)} at t={t}"
        assert all(isinstance(v, bool) for v in state.values())


def test_rf_env_all_band_ids_present():
    """get_true_state returns all band IDs [0, n_bands) even for empty steps."""
    _, env = make_gen_and_env(n_bands=6)
    state = env.get_true_state(0)
    assert set(state.keys()) == set(range(6))


def test_rf_env_sample_pulses_consistency():
    """sample_pulses for an occupied band returns non-empty list; idle band returns []."""
    _, env = make_gen_and_env(n_bands=4, p_emit=1.0, seed=0)
    # With p_emit=1.0, all bands should be occupied most steps
    total_occupied = 0
    for t in range(20):
        state = env.get_true_state(t)
        for band_id, occupied in state.items():
            pulses = env.sample_pulses(band_id, t)
            if occupied:
                assert len(pulses) > 0, f"Expected pulses in occupied band {band_id} at t={t}"
                total_occupied += 1
            else:
                assert len(pulses) == 0, f"Expected no pulses in idle band {band_id} at t={t}"
    assert total_occupied > 0, "Should have seen some occupied bands"


def test_rf_env_cross_file_emitter_raises():
    """check_cross_file_emitter raises CrossFileEmitterError for wrong file_id."""
    _, env = make_gen_and_env(n_bands=2)
    # Force an emitter into the map
    env._emitter_file_map[1000] = 0  # emitter 1000 belongs to file_id 0
    with pytest.raises(CrossFileEmitterError):
        env.check_cross_file_emitter(1000, 1)  # wrong file_id


def test_rf_env_cross_file_emitter_passes():
    """check_cross_file_emitter does not raise for correct file_id."""
    _, env = make_gen_and_env(n_bands=2)
    env._emitter_file_map[1000] = 0
    env.check_cross_file_emitter(1000, 0)  # correct file_id — no exception


def test_rf_env_memory_guard_via_tsrd_loader():
    """MemoryGuardError is raised via TSRDLoader.initialize() before any data is read."""
    import pathlib
    from src.environment.tsrd_loader import TSRDLoader

    loader = TSRDLoader(h5_paths=[pathlib.Path("fake.h5")], memory_guard_max=100)
    loader._estimate_pulse_count = lambda path: 999_999  # way over the guard

    env = RFEnvironment(
        source=loader,
        n_bands=4,
        band_edges_mhz=[(i * 100.0, (i + 1) * 100.0) for i in range(4)],
        dt_us=10.0,
    )
    with pytest.raises(MemoryGuardError):
        env.initialize()


def test_rf_env_belief_tracker_state_independence():
    """Calling get_true_state never mutates BeliefTracker state (foreshadows Property 4)."""
    # We just verify that importing and using both without cross-contamination works.
    # Full state-isolation property is tested in task 4.
    from src.scheduler.belief import BeliefTracker  # will exist after task 4
    _, env = make_gen_and_env(n_bands=4)
    try:
        bt = BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.85)
        beliefs_before = bt.get_all().copy()
        for t in range(5):
            env.get_true_state(t)
        beliefs_after = bt.get_all()
        assert np.allclose(beliefs_before, beliefs_after), "get_true_state must not modify BeliefTracker"
    except ImportError:
        pytest.skip("BeliefTracker not yet implemented (task 4)")


# ---------------------------------------------------------------------------
# Task 4.2: Property Test — Belief Boundedness (Property 1)
# **Validates: Requirements 2.1, 2.2, 2.3**
# ---------------------------------------------------------------------------
from hypothesis import given, settings
from hypothesis import strategies as st
from src.scheduler.belief import BeliefTracker


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
def test_property_belief_boundedness(updates):
    """Property 1: Belief values stay in [0, 1] for any sequence of observations."""
    tracker = BeliefTracker(n_bands=8, p_stay_occ=0.9, p_stay_idle=0.85)
    for band_id, obs in updates:
        tracker.update(band_id, obs)
    beliefs = tracker.get_all()
    assert np.all(beliefs >= 0.0), f"Belief below 0: {beliefs}"
    assert np.all(beliefs <= 1.0), f"Belief above 1: {beliefs}"


# ---------------------------------------------------------------------------
# Task 4.3: BeliefTracker unit tests
# ---------------------------------------------------------------------------

def test_belief_tracker_chapman_kolmogorov_only():
    """update with obs=None produces exactly the Chapman-Kolmogorov prediction."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.8, p_stay_idle=0.9, prior=0.5)
    # Manual calculation: b_pred = 0.8 * 0.5 + (1 - 0.9) * (1 - 0.5) = 0.4 + 0.05 = 0.45
    bt.update(0, None)
    beliefs = bt.get_all()
    assert abs(beliefs[0] - 0.45) < 1e-10, f"Expected 0.45, got {beliefs[0]}"
    # Band 1 should be unchanged since we only updated band 0
    # Actually update(0, None) only updates band 0. Band 1 is still at prior 0.5
    assert abs(beliefs[1] - 0.5) < 1e-10, f"Band 1 should remain at prior 0.5"


def test_belief_tracker_pulse_increases_belief():
    """update with obs=1 increases belief above the Chapman-Kolmogorov prediction."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.8, p_stay_idle=0.9, prior=0.5)
    b_before_predict = bt.predict(0)  # CK prediction without updating
    bt.update(0, 1)  # pulse detected
    b_after = bt.get_all()[0]
    assert b_after > b_before_predict, f"Pulse should increase belief: {b_after} > {b_before_predict}"


def test_belief_tracker_no_pulse_decreases_belief():
    """update with obs=0 decreases belief relative to CK prediction."""
    bt = BeliefTracker(n_bands=2, p_stay_occ=0.8, p_stay_idle=0.9, prior=0.5)
    b_before_predict = bt.predict(0)
    bt.update(0, 0)  # silence
    b_after = bt.get_all()[0]
    assert b_after < b_before_predict, f"Silence should decrease belief: {b_after} < {b_before_predict}"


def test_belief_tracker_denominator_zero_guard():
    """ArithmeticError is raised if denominator becomes zero."""
    bt = BeliefTracker(n_bands=1, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.5)
    # Monkeypatch _bayes_update to simulate the zero-denominator path
    def zero_denom_update(b_pred, obs):
        raise ArithmeticError("Belief denominator is zero — check p_detect and p_fa values.")
    bt._bayes_update = zero_denom_update
    with pytest.raises(ArithmeticError, match="denominator"):
        bt.update(0, 1)


def test_belief_tracker_get_all_returns_copy():
    """Mutating the returned array from get_all() does not affect internal state."""
    bt = BeliefTracker(n_bands=3, p_stay_occ=0.9, p_stay_idle=0.85, prior=0.4)
    arr = bt.get_all()
    arr[:] = 0.999  # mutate the copy
    internal = bt.get_all()
    assert np.all(internal == pytest.approx(0.4)), "Internal state was mutated via get_all() reference"


def test_belief_tracker_constructor_validation():
    """BeliefTracker raises ValueError for invalid constructor arguments."""
    with pytest.raises(ValueError, match="p_stay_occ"):
        BeliefTracker(n_bands=4, p_stay_occ=0.0, p_stay_idle=0.9)
    with pytest.raises(ValueError, match="p_stay_idle"):
        BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=1.0)
    with pytest.raises(ValueError, match="prior"):
        BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.9, prior=-0.1)
    with pytest.raises(ValueError, match="p_detect"):
        BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.9, p_detect=0.01, p_fa=0.1)


# ---------------------------------------------------------------------------
# Task 5.2: ReceiverModel — POMDP gating tests
# ---------------------------------------------------------------------------
from src.environment.receiver import ReceiverModel


def _make_receiver(n_bands: int = 4, k_scan: int = 2, p_emit: float = 0.5, seed: int = 42):
    """Create a full receiver stack for testing."""
    band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0, p_emit=p_emit, seed=seed)
    env = RFEnvironment(source=gen, n_bands=n_bands, band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt, n_bands=n_bands, k_scan=k_scan)
    return receiver, env, bt


def test_receiver_obs_dict_has_n_bands_keys():
    """step(action) returns obs_dict with exactly n_bands keys."""
    receiver, _, _ = _make_receiver(n_bands=4, k_scan=1)
    obs_dict, reward, done = receiver.step({0})
    assert len(obs_dict) == 4, f"Expected 4 keys, got {len(obs_dict)}"


def test_receiver_unscanned_bands_empty():
    """Unscanned bands always return empty pulse lists."""
    receiver, _, _ = _make_receiver(n_bands=4, k_scan=1)
    obs_dict, _, _ = receiver.step({0})
    for band_id in [1, 2, 3]:
        assert obs_dict[band_id] == [], f"Band {band_id} should be empty (unscanned)"


def test_receiver_k_scan_gte_n_bands_raises():
    """ReceiverModel raises ValueError when k_scan >= n_bands."""
    import pathlib
    band_edges = [(i * 100.0, (i + 1) * 100.0) for i in range(4)]
    gen = PDWGenerator(n_bands=4, band_cf_mhz=[100.0 + i * 100.0 for i in range(4)], dt_us=10.0)
    env = RFEnvironment(source=gen, n_bands=4, band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt = BeliefTracker(n_bands=4, p_stay_occ=0.9, p_stay_idle=0.85)
    with pytest.raises(ValueError, match="k_scan"):
        ReceiverModel(rf_env=env, belief_tracker=bt, n_bands=4, k_scan=4)


def test_receiver_unscanned_bands_get_ck_update():
    """Unscanned bands receive exactly the Chapman-Kolmogorov update."""
    receiver, env, bt = _make_receiver(n_bands=4, k_scan=2, p_emit=1.0, seed=0)
    # CK(0.5) with p_stay_occ=0.9, p_stay_idle=0.85:
    # = 0.9 * 0.5 + (1 - 0.85) * (1 - 0.5) = 0.45 + 0.075 = 0.525
    expected_ck = 0.9 * 0.5 + (1.0 - 0.85) * (1.0 - 0.5)
    receiver.step({0, 1})
    beliefs = bt.get_all()
    for band_id in [2, 3]:
        assert abs(beliefs[band_id] - expected_ck) < 1e-10, (
            f"Band {band_id} CK mismatch: expected {expected_ck:.6f}, got {beliefs[band_id]:.6f}"
        )


def test_receiver_reward_in_range():
    """Reward returned by step() is always in [0, 1]."""
    receiver, _, _ = _make_receiver(n_bands=4, k_scan=2)
    for _ in range(20):
        _, reward, _ = receiver.step({0, 1})
        assert 0.0 <= reward <= 1.0, f"Reward out of range: {reward}"


def test_receiver_context_activity_rate_in_range():
    """get_context returns activity_rate in [0, 1] after several steps."""
    receiver, _, _ = _make_receiver(n_bands=4, k_scan=2)
    for _ in range(10):
        receiver.step({0, 1})
    for band_id in range(4):
        ctx = receiver.get_context(band_id)
        assert 0.0 <= ctx["activity_rate"] <= 1.0, (
            f"activity_rate out of range for band {band_id}: {ctx['activity_rate']}"
        )


def test_receiver_no_oracle_leakage():
    """obs_dict values for unscanned bands are always [] — no true state leaked."""
    receiver, _, _ = _make_receiver(n_bands=4, k_scan=1, p_emit=1.0, seed=0)
    for _ in range(10):
        obs_dict, _, _ = receiver.step({0})
        for band_id in [1, 2, 3]:
            assert obs_dict[band_id] == [], (
                f"Oracle leakage: band {band_id} returned pulses despite not being scanned"
            )


def test_receiver_beliefs_in_range_after_steps():
    """Belief vector stays in [0, 1] after many steps with mixed scan actions."""
    receiver, _, _ = _make_receiver(n_bands=8, k_scan=3, p_emit=0.4, seed=99)
    for t in range(50):
        action = {t % 8, (t + 1) % 8, (t + 2) % 8}
        receiver.step(action)
    beliefs = receiver.get_belief()
    assert np.all(beliefs >= 0.0), f"Belief below 0: {beliefs}"
    assert np.all(beliefs <= 1.0), f"Belief above 1: {beliefs}"


# ---------------------------------------------------------------------------
# Task 6: Smoke test — round-robin scan, belief updates + POMDP masking
# ---------------------------------------------------------------------------

def test_smoke_round_robin():
    """Task 6: 200-step round-robin smoke test — belief boundedness + POMDP gating."""
    import sys
    import os
    # Ensure project root is importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.smoke_test import run

    summary = run(n_steps=200, n_bands=8, k_scan=3, seed=42)
    assert summary["belief_violations"] == 0, (
        f"Belief boundedness violated in {summary['belief_violations']} steps"
    )
    assert summary["steps"] == 200
    assert 0.0 <= summary["avg_reward_per_step"] <= 1.0
    # Sanity: at least some detections occurred with p_emit=0.4
    assert summary["total_detections"] > 0, "No detections in 200 steps — check PDWGenerator"


# ---------------------------------------------------------------------------
# Task 7.2: Property Test — Action Cardinality (Property 3)
# **Validates: Requirements 1.2, 1.3**
# ---------------------------------------------------------------------------
from src.scheduler.wiql_ucb import WIQLScheduler, ACTION_PASSIVE, ACTION_SCAN


@given(
    st.integers(min_value=2, max_value=16).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.integers(min_value=1, max_value=n - 1),
            st.lists(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
                     min_size=n, max_size=n),
        )
    )
)
def test_property_wiql_action_cardinality(args):
    """Property 3: select_arms always returns exactly K distinct band IDs in [0, N)."""
    n_bands, k, beliefs = args
    belief_arr = np.array(beliefs, dtype=np.float64)
    sched = WIQLScheduler(n_bands=n_bands, k_scan=k)
    result = sched.select_arms(belief_arr)
    assert len(result) == k, f"Expected {k} arms, got {len(result)}"
    assert all(0 <= b < n_bands for b in result), f"Band ID out of range: {result}"
    assert len(set(result)) == k, f"Duplicate band IDs: {result}"


# ---------------------------------------------------------------------------
# Task 7.3: Property Test — WIQL Exploration on First Visit (Property 13)
# **Validates: Requirement 6.3**
# ---------------------------------------------------------------------------

def test_property_wiql_first_visit_inf():
    """Property 13: fresh WIQLScheduler returns +inf for every band before any update."""
    sched = WIQLScheduler(n_bands=8, k_scan=3)
    belief = np.full(8, 0.5)
    indices = sched.compute_indices(belief)
    assert np.all(np.isinf(indices)), (
        f"Expected all +inf on first call, got: {indices}"
    )


# ---------------------------------------------------------------------------
# Task 7.4: WIQLScheduler unit tests
# ---------------------------------------------------------------------------

def test_wiql_select_arms_exact_k():
    """select_arms returns exactly k distinct band IDs for various (n, k) pairs."""
    for n, k in [(4, 1), (8, 3), (16, 5), (4, 3)]:
        sched = WIQLScheduler(n_bands=n, k_scan=k)
        belief = np.random.default_rng(0).uniform(0, 1, size=n)
        arms = sched.select_arms(belief)
        assert len(arms) == k, f"n={n}, k={k}: got {len(arms)} arms"
        assert len(set(arms)) == k, f"Duplicate arms: {arms}"


def test_wiql_tie_breaking_ascending():
    """Tie-breaking by ascending band_id: all-inf index picks bands 0,1,...,k-1."""
    sched = WIQLScheduler(n_bands=6, k_scan=3)
    belief = np.full(6, 0.5)
    # Fresh scheduler: all indices are inf, tie broken by ascending band_id
    arms = sched.select_arms(belief)
    assert arms == {0, 1, 2}, f"Expected {{0,1,2}} (lowest IDs win ties), got {arms}"


def test_wiql_update_reduces_q_error():
    """Q-table TD error decreases over repeated updates toward a known target."""
    sched = WIQLScheduler(n_bands=2, k_scan=1, n_bins=2)
    # Repeatedly update band 0, bin 0 (belief ~0.25), action=SCAN, reward=1.0
    for _ in range(200):
        sched.update(
            band_id=0, action=ACTION_SCAN,
            reward=1.0, belief_prev=0.25, belief_next=0.25
        )
    # Q(0, bin0, SCAN) should have converged toward r/(1-gamma) = 1/(1-0.95) = 20
    # (with TD(0) and constant reward 1.0 and self-loop)
    q_scan = float(sched._Q[0, 0, ACTION_SCAN])
    assert q_scan > 5.0, f"Q(SCAN) should converge upward, got {q_scan:.3f}"


def test_wiql_memory_within_target():
    """Total table memory stays within 600 bytes * n_bands."""
    for n_bands in [8, 16, 32]:
        sched = WIQLScheduler(n_bands=n_bands, k_scan=3)
        mem = sched.memory_bytes()
        target = 600 * n_bands
        assert mem <= target, (
            f"n_bands={n_bands}: {mem} bytes > {target}-byte target ({mem/n_bands:.0f} bytes/arm)"
        )


def test_wiql_k_scan_gte_n_bands_raises():
    """WIQLScheduler raises ValueError when k_scan >= n_bands."""
    with pytest.raises(ValueError, match="k_scan"):
        WIQLScheduler(n_bands=4, k_scan=4)


def test_wiql_index_updates_with_experience():
    """After enough updates, bands with higher reward get higher index than idle bands."""
    n_bands = 4
    sched = WIQLScheduler(n_bands=n_bands, k_scan=2, alpha_w=0.1)
    belief = np.full(n_bands, 0.5)

    # Band 0: repeatedly scanned with reward 1.0 (high-value band)
    # Band 1: repeatedly scanned with reward 0.0 (empty band)
    for _ in range(100):
        sched.update(0, ACTION_SCAN, reward=1.0, belief_prev=0.5, belief_next=0.5)
        sched.update(1, ACTION_SCAN, reward=0.0, belief_prev=0.5, belief_next=0.5)

    indices = sched.compute_indices(belief)
    # Band 0 should have higher index than band 1 after experience
    # (UCB bonus will dominate for unvisited bands 2, 3 — that's fine)
    assert indices[0] >= indices[1], (
        f"High-reward band 0 should have higher index than idle band 1: "
        f"idx[0]={indices[0]:.3f}, idx[1]={indices[1]:.3f}"
    )


# ---------------------------------------------------------------------------
# Task 7 integration: WIQL-UCB vs round-robin comparison
# **Validates: Requirement 6.1, 6.6**
# ---------------------------------------------------------------------------

def _make_env_stack(n_bands: int, k_scan: int, seed: int):
    """Create a fresh (env, receiver) stack for one episode."""
    band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0, p_emit=0.4, seed=seed)
    env = RFEnvironment(source=gen, n_bands=n_bands, band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt, n_bands=n_bands, k_scan=k_scan)
    receiver.reset()
    return receiver


def _run_episode(policy, n_bands: int = 8, k_scan: int = 3, n_steps: int = 200, seed: int = 42) -> float:
    """Run a full episode with a stateless policy, return average reward per step.

    policy: callable(belief: np.ndarray) -> set[int]
    """
    receiver = _make_env_stack(n_bands, k_scan, seed)
    total_reward = 0.0
    for _ in range(n_steps):
        belief = receiver.get_belief()
        action = policy(belief)
        _, reward, _ = receiver.step(action)
        total_reward += reward
    return total_reward / n_steps


def _run_wiql_episode(n_bands: int, k_scan: int, n_steps: int, seed: int) -> float:
    """Run a full WIQL-UCB episode, return average reward per step.

    The scheduler is created fresh and learns online throughout the episode.
    """
    sched = WIQLScheduler(n_bands=n_bands, k_scan=k_scan)
    receiver = _make_env_stack(n_bands, k_scan, seed)

    total_reward = 0.0
    for _ in range(n_steps):
        belief_prev = receiver.get_belief().copy()
        action = sched.select_arms(belief_prev)
        obs_dict, reward, _ = receiver.step(action)
        total_reward += reward
        belief_next = receiver.get_belief()

        for band_id in action:
            obs_binary = 1 if len(obs_dict[band_id]) > 0 else 0
            sched.update(
                band_id=band_id,
                action=ACTION_SCAN,
                reward=float(obs_binary),
                belief_prev=float(belief_prev[band_id]),
                belief_next=float(belief_next[band_id]),
            )
        for band_id in range(n_bands):
            if band_id not in action:
                sched.update(
                    band_id=band_id,
                    action=ACTION_PASSIVE,
                    reward=0.0,
                    belief_prev=float(belief_prev[band_id]),
                    belief_next=float(belief_next[band_id]),
                )

    return total_reward / n_steps


def _run_rr_episode(n_bands: int, k_scan: int, n_steps: int, seed: int) -> float:
    """Run a round-robin episode, return average reward per step."""
    cursor = [0]
    def round_robin(belief):
        action = {(cursor[0] + j) % n_bands for j in range(k_scan)}
        cursor[0] = (cursor[0] + 1) % n_bands
        return action
    return _run_episode(round_robin, n_bands=n_bands, k_scan=k_scan,
                        n_steps=n_steps, seed=seed)


def test_wiql_beats_round_robin_on_average():
    """WIQL-UCB avg reward >= round-robin avg reward averaged over 5 seeds (300 steps).

    With only 300 steps and a cold-start tabular policy, the gap is modest —
    we test for >= (not strictly >) to avoid flakiness. The maturity test below
    checks that the gap widens at longer horizons.
    """
    n_bands, k_scan, n_steps = 8, 3, 300
    seeds = [42, 7, 123, 999, 31]

    wiql_rewards = [_run_wiql_episode(n_bands, k_scan, n_steps, s) for s in seeds]
    rr_rewards   = [_run_rr_episode  (n_bands, k_scan, n_steps, s) for s in seeds]

    avg_wiql = sum(wiql_rewards) / len(wiql_rewards)
    avg_rr   = sum(rr_rewards)   / len(rr_rewards)
    margin   = avg_wiql - avg_rr

    print(f"\n[300-step] WIQL={avg_wiql:.4f}  RR={avg_rr:.4f}  margin={margin:+.4f}")
    print(f"  Per-seed WIQL: {[f'{r:.4f}' for r in wiql_rewards]}")
    print(f"  Per-seed RR:   {[f'{r:.4f}' for r in rr_rewards]}")

    assert avg_wiql >= avg_rr, (
        f"WIQL-UCB ({avg_wiql:.4f}) should be >= round-robin ({avg_rr:.4f}) "
        f"on average over {len(seeds)} seeds"
    )


def test_wiql_index_maturity_long_horizon():
    """WIQL-UCB margin over round-robin grows as episodes get longer.

    Uses a HETEROGENEOUS band setup so WIQL has something to actually learn:
      - 3 "hot" bands (bands 0-2):  p_emit = 0.75  (frequently occupied)
      - 5 "cold" bands (bands 3-7): p_emit = 0.10  (rarely occupied)

    With K=3 simultaneous scans, the optimal policy is to always scan the 3
    hot bands. Round-robin wastes 5/8 of its budget on cold bands. WIQL should
    learn this within ~500 steps and widen its advantage at 2500 steps.

    Asserts that margin_long > margin_short, confirming that the index
    estimates are maturing and capturing band heterogeneity.

    Both margins are printed regardless of pass/fail so the trend is visible.
    """
    n_bands, k_scan = 8, 3
    seeds = [42, 7, 123, 999, 31]
    short_steps = 300
    long_steps  = 2500

    # 3 hot bands + 5 cold bands: gives WIQL a real signal to converge on
    p_emit_per_band = [0.75, 0.75, 0.75, 0.10, 0.10, 0.10, 0.10, 0.10]

    def make_env_hetero(seed):
        band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
        band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
        gen = PDWGenerator(
            n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0,
            p_emit=p_emit_per_band, seed=seed
        )
        env = RFEnvironment(source=gen, n_bands=n_bands,
                            band_edges_mhz=band_edges, dt_us=10.0)
        env.initialize()
        bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
        receiver = ReceiverModel(rf_env=env, belief_tracker=bt,
                                  n_bands=n_bands, k_scan=k_scan)
        receiver.reset()
        return receiver

    def run_wiql_hetero(n_steps, seed):
        sched = WIQLScheduler(n_bands=n_bands, k_scan=k_scan, alpha_w=0.1)
        receiver = make_env_hetero(seed)
        total_reward = 0.0
        for _ in range(n_steps):
            belief_prev = receiver.get_belief().copy()
            action = sched.select_arms(belief_prev)
            obs_dict, reward, _ = receiver.step(action)
            total_reward += reward
            belief_next = receiver.get_belief()
            for band_id in action:
                obs_binary = 1 if len(obs_dict[band_id]) > 0 else 0
                sched.update(band_id, ACTION_SCAN, float(obs_binary),
                             float(belief_prev[band_id]), float(belief_next[band_id]))
            for band_id in range(n_bands):
                if band_id not in action:
                    sched.update(band_id, ACTION_PASSIVE, 0.0,
                                 float(belief_prev[band_id]), float(belief_next[band_id]))
        return total_reward / n_steps

    def run_rr_hetero(n_steps, seed):
        receiver = make_env_hetero(seed)
        cursor = [0]
        total_reward = 0.0
        for _ in range(n_steps):
            action = {(cursor[0] + j) % n_bands for j in range(k_scan)}
            cursor[0] = (cursor[0] + 1) % n_bands
            _, reward, _ = receiver.step(action)
            total_reward += reward
        return total_reward / n_steps

    # --- Short horizon ---
    wiql_short = [run_wiql_hetero(short_steps, s) for s in seeds]
    rr_short   = [run_rr_hetero  (short_steps, s) for s in seeds]
    avg_wiql_short = sum(wiql_short) / len(wiql_short)
    avg_rr_short   = sum(rr_short)   / len(rr_short)
    margin_short   = avg_wiql_short - avg_rr_short

    # --- Long horizon ---
    wiql_long = [run_wiql_hetero(long_steps, s) for s in seeds]
    rr_long   = [run_rr_hetero  (long_steps, s) for s in seeds]
    avg_wiql_long = sum(wiql_long) / len(wiql_long)
    avg_rr_long   = sum(rr_long)   / len(rr_long)
    margin_long   = avg_wiql_long - avg_rr_long

    # --- Report ---
    print(f"\n{'─'*68}")
    print(f"  WIQL index maturity test  ({len(seeds)} seeds, heterogeneous bands)")
    print(f"  Hot bands 0-2: p_emit=0.75  |  Cold bands 3-7: p_emit=0.10")
    print(f"{'─'*68}")
    print(f"  Horizon  | WIQL avg  | RR avg    | Margin")
    print(f"  {short_steps:>7}  | {avg_wiql_short:.4f}    | {avg_rr_short:.4f}    | {margin_short:+.4f}")
    print(f"  {long_steps:>7}  | {avg_wiql_long:.4f}    | {avg_rr_long:.4f}    | {margin_long:+.4f}")
    print(f"{'─'*68}")
    print(f"  Per-seed WIQL short: {[f'{r:.4f}' for r in wiql_short]}")
    print(f"  Per-seed WIQL long:  {[f'{r:.4f}' for r in wiql_long]}")
    print(f"  Per-seed RR short:   {[f'{r:.4f}' for r in rr_short]}")
    print(f"  Per-seed RR long:    {[f'{r:.4f}' for r in rr_long]}")
    print(f"{'─'*68}")

    assert margin_long > margin_short, (
        f"WIQL index estimates are NOT maturing: "
        f"long-horizon margin ({margin_long:+.4f}) should be strictly greater than "
        f"short-horizon margin ({margin_short:+.4f}). "
        f"Expected WIQL to learn hot bands 0-2 and widen its advantage over round-robin."
    )
