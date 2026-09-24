"""Receiver Model — POMDP observation wrapper.

Wraps RFEnvironment and exposes ONLY observations from currently-scanned bands.
Unscanned bands return empty pulse lists — the scheduler never learns the true
occupancy state of bands it is not currently scanning. This is what makes the
problem a POMDP.

Critical invariant (Property 2 / Requirement 7.5):
    RF_Environment.get_true_state() output is NEVER included in the return value
    of ReceiverModel.step(). Ground truth is only used INTERNALLY to compute reward
    and is discarded after that computation.
"""
from __future__ import annotations

import numpy as np

from src.scheduler.belief import BeliefTracker
from src.environment.simulator import RFEnvironment
from src.environment.pulse import Pulse

# Maximum sliding window size for activity_rate computation
_ACTIVITY_WINDOW = 50


class ReceiverModel:
    """POMDP observation wrapper around RFEnvironment.

    The receiver can scan K bands simultaneously out of N total bands (K < N).
    Only scanned bands return pulse observations; all others return empty lists.
    Belief states are updated through BeliefTracker.
    """

    def __init__(
        self,
        rf_env: RFEnvironment,
        belief_tracker: BeliefTracker,
        n_bands: int,
        k_scan: int,
    ) -> None:
        """
        Args:
            rf_env:         The RF environment (ground truth source).
            belief_tracker: HMM belief state tracker for all bands.
            n_bands:        Total number of frequency bands.
            k_scan:         Number of bands to scan simultaneously. Must be < n_bands.

        Raises:
            ValueError: if k_scan >= n_bands (Requirement 1.4).
        """
        if k_scan >= n_bands:
            raise ValueError(
                f"k_scan ({k_scan}) must be < n_bands ({n_bands}). "
                "The receiver cannot scan all bands simultaneously."
            )

        self._rf_env = rf_env          # private — ground truth never forwarded to caller
        self._belief_tracker = belief_tracker
        self.n_bands = n_bands
        self.k_scan = k_scan
        self._t: int = 0

        # Sliding window context history: band_id -> list of binary obs (0 or 1)
        # Used to compute activity_rate for DQWIC context features
        self._context_history: dict[int, list[int]] = {i: [] for i in range(n_bands)}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the receiver to time step 0 and return the initial belief vector."""
        self._t = 0
        self._belief_tracker.reset()
        self._context_history = {i: [] for i in range(self.n_bands)}
        return self._belief_tracker.get_all()

    # ------------------------------------------------------------------
    # Core POMDP step
    # ------------------------------------------------------------------

    def step(
        self, action: set[int]
    ) -> tuple[dict[int, list[Pulse]], float, bool]:
        """Advance one time step, returning observations only for scanned bands.

        Critical POMDP gate: only bands in `action` receive pulse observations.
        All other bands return empty lists — no leakage of ground-truth state.

        Args:
            action: Set of band IDs to scan this time step. Must be a subset of
                    [0, n_bands) with len == k_scan (enforced by scheduler; not
                    re-enforced here to keep the model agnostic to scheduler type).

        Returns:
            obs_dict: {band_id: list[Pulse]} — scanned bands return actual pulses
                      (possibly empty); unscanned bands return [].
            reward:   Fraction of scanned bands that had a successful detection
                      (pulse observed AND band truly occupied).
            done:     Always False for this environment (no terminal state).
        """
        t = self._t

        # --- Ground truth access (INTERNAL ONLY — never propagated to caller) ---
        # We call rf_env.step() once to compute reward; the dict is not returned.
        true_state = self._rf_env.step(t)  # {band_id: bool} — oracle, discarded below

        # --- Build observation dict ---
        obs_dict: dict[int, list[Pulse]] = {}

        for band_id in range(self.n_bands):
            if band_id in action:
                # Scanned band: get real pulses (ground truth accessed internally)
                pulses = self._rf_env.sample_pulses(band_id, t)
                obs_dict[band_id] = pulses

                # Binary observation for belief update: 1 if any pulse detected
                obs_binary = 1 if len(pulses) > 0 else 0
                self._belief_tracker.update(band_id, obs_binary)

                # Update sliding window context history
                self._context_history[band_id].append(obs_binary)
                if len(self._context_history[band_id]) > _ACTIVITY_WINDOW:
                    self._context_history[band_id].pop(0)
            else:
                # Unscanned band: empty observation list (POMDP gating)
                obs_dict[band_id] = []
                # Chapman-Kolmogorov prediction only (restless property)
                self._belief_tracker.update(band_id, None)

        # --- Compute reward (uses true_state internally, NOT returned to caller) ---
        # Reward = fraction of scanned bands where a pulse was detected AND band was occupied
        n_true_detections = sum(
            1
            for band_id in action
            if true_state.get(band_id, False) and len(obs_dict[band_id]) > 0
        )
        reward = n_true_detections / max(len(action), 1)

        # Discard true_state — it must NOT be propagated further
        del true_state

        self._t += 1
        return obs_dict, reward, False

    # ------------------------------------------------------------------
    # Belief and context access
    # ------------------------------------------------------------------

    def get_belief(self) -> np.ndarray:
        """Return the current belief vector (copy from BeliefTracker)."""
        return self._belief_tracker.get_all()

    def get_context(self, band_id: int) -> dict:
        """Return ContextFeatures-compatible dict for a single band.

        Fields:
            band_id:       The band index.
            freq_agility:  Placeholder (0.0) until DQWIC task is implemented.
            snr_db:        Placeholder (-60.0) until SNR estimation is implemented.
            activity_rate: Fraction of recent steps the band was found occupied,
                           computed over a sliding window of size 50.
        """
        history = self._context_history.get(band_id, [])
        activity_rate = float(np.mean(history[-_ACTIVITY_WINDOW:])) if history else 0.0
        return {
            "band_id": band_id,
            "freq_agility": 0.0,    # placeholder — populated in DQWIC task
            "snr_db": -60.0,        # placeholder — populated in DQWIC task
            "activity_rate": activity_rate,
        }

    def get_context_all(self) -> list[dict]:
        """Return context features for all bands."""
        return [self.get_context(i) for i in range(self.n_bands)]
