"""Belief State Tracker for the EW POMDP.

Maintains b_i(t) = P(band i is occupied | observation history) for all bands
using a two-state Hidden Markov Model. Unobserved bands are propagated forward
via the Chapman-Kolmogorov equation (the "restless" property — states evolve
even when not observed).

This module has NO dependency on PeriodicInterceptModule and shares NO mutable
state with it. (Property 4 / Requirement 8.1)
"""
from __future__ import annotations

import numpy as np


class BeliefTracker:
    """HMM belief state tracker for N frequency bands.

    The belief b_i(t) is updated at each time step:
      - For scanned bands: Bayesian posterior using likelihood of observation
      - For unscanned bands: Chapman-Kolmogorov prediction only

    Correctness property P1: ∀ t, ∀ i: belief[i](t) ∈ [0, 1]
    """

    def __init__(
        self,
        n_bands: int,
        p_stay_occ: float,
        p_stay_idle: float,
        prior: float = 0.5,
        p_detect: float = 0.9,
        p_fa: float = 0.01,
    ) -> None:
        """
        Args:
            n_bands:     Number of frequency bands to track.
            p_stay_occ:  P(occ→occ) — probability of remaining occupied. Must be in (0, 1).
            p_stay_idle: P(idle→idle) — probability of remaining idle. Must be in (0, 1).
            prior:       Initial belief for all bands. Must be in [0, 1].
            p_detect:    P(pulse detected | band occupied). Must be > p_fa.
            p_fa:        P(pulse detected | band idle) — false alarm rate.
        """
        if not (0.0 < p_stay_occ < 1.0):
            raise ValueError(f"p_stay_occ must be in (0, 1), got {p_stay_occ}")
        if not (0.0 < p_stay_idle < 1.0):
            raise ValueError(f"p_stay_idle must be in (0, 1), got {p_stay_idle}")
        if not (0.0 <= prior <= 1.0):
            raise ValueError(f"prior must be in [0, 1], got {prior}")
        if p_detect <= p_fa:
            raise ValueError(f"p_detect ({p_detect}) must be > p_fa ({p_fa})")

        self.n_bands = n_bands
        self.p_stay_occ = p_stay_occ
        self.p_stay_idle = p_stay_idle
        self._prior = prior
        self.p_detect = p_detect
        self.p_fa = p_fa
        self._t: int = 0

        # Core belief state — shape (n_bands,), values in [0, 1]
        self._beliefs: np.ndarray = np.full(n_bands, prior, dtype=np.float64)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chapman_kolmogorov(self, b_prev: float) -> float:
        """Predictive update (Chapman-Kolmogorov): always applied, even for unscanned bands."""
        return self.p_stay_occ * b_prev + (1.0 - self.p_stay_idle) * (1.0 - b_prev)

    def _bayes_update(self, b_pred: float, obs: int) -> float:
        """Bayesian posterior update for a scanned band.

        Args:
            b_pred: Predictive belief P(occ at t+1 | history to t).
            obs:    1 if pulse detected, 0 if silent.

        Raises:
            ArithmeticError: if the denominator is zero (degenerate case).
        """
        if obs == 1:
            likelihood_occ = self.p_detect
            likelihood_idle = self.p_fa
        else:
            likelihood_occ = 1.0 - self.p_detect
            likelihood_idle = 1.0 - self.p_fa

        numerator = likelihood_occ * b_pred
        denominator = numerator + likelihood_idle * (1.0 - b_pred)

        if denominator == 0.0:
            raise ArithmeticError("Belief denominator is zero — check p_detect and p_fa values.")

        return numerator / denominator

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, band_id: int, observation: int | None) -> None:
        """Update belief for band_id given an observation.

        Args:
            band_id:     Index of the band to update. Must be in [0, n_bands).
            observation: None (not scanned), 0 (scanned, no pulse), 1 (scanned, pulse detected).

        After update, belief[band_id] is guaranteed to be in [0, 1] (clipped for float safety).
        """
        if not (0 <= band_id < self.n_bands):
            raise IndexError(f"band_id {band_id} out of range [0, {self.n_bands})")

        # Step 1: Chapman-Kolmogorov prediction (always applied)
        b_pred = self._chapman_kolmogorov(float(self._beliefs[band_id]))

        # Step 2: if unscanned, prediction is the new belief
        if observation is None:
            self._beliefs[band_id] = np.clip(b_pred, 0.0, 1.0)
            return

        # Step 3: Bayesian update
        b_next = self._bayes_update(b_pred, observation)

        # Step 4: clip to guard against floating-point drift
        self._beliefs[band_id] = np.clip(b_next, 0.0, 1.0)

    def predict(self, band_id: int) -> float:
        """Return the predicted next belief for band_id WITHOUT updating stored state.

        Used to peek at what the next-step belief would be.
        """
        if not (0 <= band_id < self.n_bands):
            raise IndexError(f"band_id {band_id} out of range [0, {self.n_bands})")
        return float(self._chapman_kolmogorov(float(self._beliefs[band_id])))

    def get_all(self) -> np.ndarray:
        """Return a COPY of the current belief vector.

        Returns a copy to prevent callers from accidentally mutating internal state.
        Invariant: all values in [0, 1].
        """
        return self._beliefs.copy()

    def reset(self, prior: float | None = None) -> None:
        """Reset all beliefs to prior (or the original prior if None)."""
        p = prior if prior is not None else self._prior
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"prior must be in [0, 1], got {p}")
        self._beliefs[:] = p
        self._t = 0
