"""Round-robin smoke test — proves the RF environment produces sane belief updates
and observation masking end-to-end with no scheduler logic.

Usage:
    python scripts/smoke_test.py

This wires PDWGenerator → RFEnvironment → BeliefTracker → ReceiverModel and runs
200 time steps with a simple round-robin scan policy (3 of 8 bands at a time).

Assertions:
  - All beliefs remain in [0, 1] at every step (Property 1)
  - Unscanned bands always return empty observation lists (POMDP gating)
  - Final summary printed to stdout

No scheduler logic here — that is the next module (WIQL-UCB).
"""
from __future__ import annotations

import sys

import numpy as np

from ew_smart_scan.env.belief_tracker import BeliefTracker
from ew_smart_scan.env.receiver_model import ReceiverModel
from ew_smart_scan.env.rf_environment import RFEnvironment
from ew_smart_scan.models.pdw_generator import PDWGenerator


def run(n_steps: int = 200, n_bands: int = 8, k_scan: int = 3, seed: int = 42) -> dict:
    """Run the smoke test and return a summary dict."""
    # --- Setup ---
    band_cf_mhz = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges_mhz = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]

    gen = PDWGenerator(
        n_bands=n_bands,
        band_cf_mhz=band_cf_mhz,
        dt_us=10.0,
        p_emit=0.4,
        seed=seed,
    )
    env = RFEnvironment(
        source=gen,
        n_bands=n_bands,
        band_edges_mhz=band_edges_mhz,
        dt_us=10.0,
    )
    env.initialize()

    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt, n_bands=n_bands, k_scan=k_scan)
    receiver.reset()

    # --- Round-robin cursor ---
    cursor = 0
    total_reward = 0.0
    total_detections = 0
    step_violations = 0

    for t in range(n_steps):
        # Round-robin: scan k_scan consecutive bands (wrapping around)
        action = {(cursor + j) % n_bands for j in range(k_scan)}
        cursor = (cursor + 1) % n_bands

        obs_dict, reward, _ = receiver.step(action)
        total_reward += reward

        # --- Assertion 1: Belief boundedness (Property 1) ---
        beliefs = receiver.get_belief()
        if not (np.all(beliefs >= 0.0) and np.all(beliefs <= 1.0)):
            step_violations += 1
            print(f"  [FAIL] t={t}: belief out of [0,1]: {beliefs}", file=sys.stderr)

        # --- Assertion 2: POMDP gating — unscanned bands must return [] ---
        for band_id, pulses in obs_dict.items():
            if band_id not in action:
                assert pulses == [], (
                    f"Oracle leakage at t={t}: band {band_id} returned {len(pulses)} pulses "
                    f"despite not being in action {action}"
                )

        # Count detections
        total_detections += sum(1 for b in action if len(obs_dict[b]) > 0)

    if step_violations > 0:
        raise AssertionError(
            f"Smoke test FAILED: {step_violations} belief-boundedness violations in {n_steps} steps."
        )

    final_beliefs = receiver.get_belief()
    summary = {
        "steps": n_steps,
        "n_bands": n_bands,
        "k_scan": k_scan,
        "total_reward": round(total_reward, 4),
        "avg_reward_per_step": round(total_reward / n_steps, 4),
        "total_detections": total_detections,
        "final_beliefs": np.round(final_beliefs, 3).tolist(),
        "belief_violations": step_violations,
    }
    return summary


if __name__ == "__main__":
    print("Running round-robin smoke test (200 steps, 8 bands, K=3)...")
    summary = run()
    print(
        f"Smoke test passed: {summary['steps']} steps, "
        f"avg_reward={summary['avg_reward_per_step']}, "
        f"detections={summary['total_detections']}, "
        f"final beliefs: {summary['final_beliefs']}"
    )
