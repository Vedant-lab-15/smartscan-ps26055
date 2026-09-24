"""WIQL-UCB Scheduler — Whittle Index Q-Learning with UCB exploration.

Tabular Whittle-index scheduler for the RMAB/POMDP scheduling problem.
This is the baseline fallback before Neural-Q-Whittle; its selling points are:

  - Tabular: ~600 bytes per arm (float32, 16 bins × 2 actions Q-table +
             16-entry visit count + 1 index estimate per arm).
  - Sub-millisecond decision latency on any hardware.
  - No prior knowledge of channel dynamics required.
  - Proven UCB exploration with shrinking bonus as belief states are visited.

Design choices (documented here, not buried in code):

  Belief discretization:
    Belief b_i ∈ [0, 1] is discretized into N_BINS=16 uniform bins of width
    1/16 each. Bin k covers [k/16, (k+1)/16). This gives enough resolution to
    distinguish high-occupancy from low-occupancy bands without explosion in
    table size. 16 bins × 2 actions × float32 = 128 bytes Q-table per arm;
    total per arm with visit counts and index ≈ 196 bytes. Well within the
    600-byte/arm target from the spec.

  Q-learning update:
    TD(0) with learning rate 1/(N[band, bin, action] + 1) — harmonic decay,
    self-tuning, no separate learning-rate hyperparameter needed.

  Whittle index update:
    W(b) is updated by a small step toward the subsidy that equalises
    Q(b, active) and Q(b, passive). Step size alpha_w=0.05 (slow timescale
    relative to Q — per the two-timescale requirement). This is the "WIQL"
    (Whittle Index Q-Learning) update from the WIQL-UCB paper.

  UCB bonus:
    bonus(band, bin) = c_ucb * sqrt(log(total_visits + 1) / (N[band, bin] + 1))
    where N[band, bin] = sum of visits across both actions for that (band, bin)
    pair. c_ucb=1.0 (standard UCB1). Unvisited (band, bin) pairs return +inf
    to force exploration first (Property 13).

  Tie-breaking: ascending band_id, stable argsort.

References:
  - Whittle, P. (1988). Restless bandits: Activity allocation in a changing world.
  - Neural-Q-Whittle (2023): theoretical foundation for two-timescale Whittle learning.
  - WIQL-UCB (2025/26): tabular Whittle-UCB, our fallback baseline.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Constants — documented design choices
# ------------------------------------------------------------------

#: Number of uniform belief bins. 16 bins keeps the Q-table to ~128 bytes/arm
#: (float32 Q-values, 2 actions) — well within the 600-byte/arm target.
N_BINS: int = 16

#: UCB exploration constant. 1.0 = standard UCB1. Higher values favour
#: exploration; lower favour exploitation. No strong prior here; 1.0 is the
#: well-studied default.
C_UCB: float = 1.0

#: Whittle index update step size (slow timescale). Must be << Q learning rate
#: to satisfy the two-timescale convergence requirement. 0.05 is aggressive
#: enough for 200–2000 step episodes but slow enough to be stable.
ALPHA_W: float = 0.05

#: Actions
ACTION_PASSIVE: int = 0
ACTION_SCAN: int = 1


class WIQLScheduler:
    """Tabular Whittle Index Q-Learning with UCB exploration.

    State space:  N_BINS discrete belief levels per band
    Action space: {passive=0, scan=1} per band
    Index:        Whittle index W_i(b) per (band, bin), updated toward
                  Q(b, active) - Q(b, passive) = 0 equilibrium
    Selection:    Top-K bands by (W_i(b) + UCB_bonus_i(b)) at each step

    Property 13: unvisited (band, bin) pairs return +inf → explored first.
    Property 3:  select_arms always returns exactly K distinct band IDs.
    """

    def __init__(
        self,
        n_bands: int,
        k_scan: int,
        n_bins: int = N_BINS,
        c_ucb: float = C_UCB,
        alpha_w: float = ALPHA_W,
        gamma: float = 0.95,
        epsilon_explore: float = 0.01,
    ) -> None:
        """
        Args:
            n_bands:         Number of frequency bands (arms).
            k_scan:          Number of bands to scan simultaneously (K < n_bands).
            n_bins:          Number of uniform belief bins. Default 16.
            c_ucb:           UCB exploration constant. Default 1.0.
            alpha_w:         Whittle index update step size (slow timescale). Default 0.05.
            gamma:           Discount factor for Q-learning. Default 0.95.
            epsilon_explore: Minimum UCB floor — prevents permanent lockout of bands in
                             non-stationary environments. Default 0.01.
                             U_i(t) = max(c·sqrt(ln(t)/N_i(t)), epsilon_explore)

        Memory per arm (float32):
            Q-table:      n_bins × 2 actions × 4 bytes = 128 bytes  (n_bins=16)
            Visit counts: n_bins × 2 actions × 4 bytes = 128 bytes
            Whittle index: n_bins × 4 bytes             =  64 bytes
            Total per arm: 320 bytes  (well within 600-byte target)
        """
        if k_scan >= n_bands:
            raise ValueError(f"k_scan ({k_scan}) must be < n_bands ({n_bands})")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")

        self.n_bands = n_bands
        self.k_scan = k_scan
        self.n_bins = n_bins
        self.c_ucb = c_ucb
        self.alpha_w = alpha_w
        self.gamma = gamma
        self.epsilon_explore = epsilon_explore

        # Q-table: shape (n_bands, n_bins, 2) — Q(band, bin, action)
        # float32 to keep memory tight.
        self._Q: np.ndarray = np.zeros((n_bands, n_bins, 2), dtype=np.float32)

        # Visit counts: shape (n_bands, n_bins, 2) — N(band, bin, action)
        self._N: np.ndarray = np.zeros((n_bands, n_bins, 2), dtype=np.int32)

        # Whittle index estimates: shape (n_bands, n_bins)
        # Initialised to 0.5 — neutral starting point.
        self._W: np.ndarray = np.full((n_bands, n_bins), 0.5, dtype=np.float32)

        # Total visit counter for UCB log term
        self._total_visits: int = 0

        # Log memory usage
        q_bytes = self._Q.nbytes + self._N.nbytes + self._W.nbytes
        target = 600 * n_bands
        if q_bytes > target:
            logger.warning(
                "WIQL-UCB table size %d bytes exceeds %d-byte target for %d bands "
                "(%.0f bytes/arm vs 600-byte target). Consider reducing n_bins.",
                q_bytes, target, n_bands, q_bytes / n_bands,
            )
        else:
            logger.debug(
                "WIQL-UCB tables: %d bytes total (%.0f bytes/arm) for %d bands.",
                q_bytes, q_bytes / n_bands, n_bands,
            )

    # ------------------------------------------------------------------
    # Belief discretization
    # ------------------------------------------------------------------

    def _belief_to_bin(self, belief: float) -> int:
        """Map belief ∈ [0, 1] to bin index ∈ [0, n_bins-1].

        Uniform bins: bin k covers [k/n_bins, (k+1)/n_bins).
        belief=1.0 maps to the last bin (n_bins-1) by clipping.
        """
        return min(int(belief * self.n_bins), self.n_bins - 1)

    # ------------------------------------------------------------------
    # Index computation
    # ------------------------------------------------------------------

    def compute_indices(self, belief: np.ndarray) -> np.ndarray:
        """Compute Whittle index + UCB bonus for each band given current beliefs.

        UCB term includes a minimum exploration floor ε to prevent permanent
        lockout of bands in non-stationary environments (e.g. a periodic emitter
        that was silent may become active again):
            U_i(t) = max(c · sqrt(ln(t) / N_i(t)), ε)
        This ensures every band is re-explored at a minimum rate regardless of
        how many observations it has accumulated.

        For unvisited (band, bin) pairs (N=0 for both actions), returns +inf
        to force exploration (Property 13 / Requirement 6.3).

        Args:
            belief: shape (n_bands,), values in [0, 1].

        Returns:
            indices: shape (n_bands,), each value is W_i(b_i) + UCB_bonus.
        """
        indices = np.empty(self.n_bands, dtype=np.float64)
        log_total = math.log(self._total_visits + 1)  # +1 avoids log(0)

        for i in range(self.n_bands):
            b = float(np.clip(belief[i], 0.0, 1.0))
            bin_idx = self._belief_to_bin(b)

            # Total visits to this (band, bin) across both actions
            n_passive = int(self._N[i, bin_idx, ACTION_PASSIVE])
            n_scan = int(self._N[i, bin_idx, ACTION_SCAN])
            n_total = n_passive + n_scan

            if n_total == 0:
                # Unvisited — force exploration with +inf
                indices[i] = np.inf
            else:
                # UCB bonus with minimum exploration floor (non-stationarity fix)
                ucb_raw = self.c_ucb * math.sqrt(log_total / n_total)
                ucb_bonus = max(ucb_raw, self.epsilon_explore)
                indices[i] = float(self._W[i, bin_idx]) + ucb_bonus

        return indices

    # ------------------------------------------------------------------
    # Arm selection
    # ------------------------------------------------------------------

    def select_arms(self, belief: np.ndarray) -> set[int]:
        """Select top-K bands to scan given current belief vector.

        Ties broken by ascending band_id (stable argsort).
        Always returns exactly k_scan distinct band IDs (Property 3).

        Args:
            belief: shape (n_bands,), values in [0, 1].

        Returns:
            Set of exactly k_scan distinct band IDs.
        """
        indices = self.compute_indices(belief)

        # Stable argsort descending: negate finite values; inf stays at top.
        # Strategy: sort by (-index, band_id) for stable tie-breaking.
        order = np.argsort(-indices, kind="stable")
        selected = set(int(order[j]) for j in range(self.k_scan))

        assert len(selected) == self.k_scan, (
            f"select_arms returned {len(selected)} bands, expected {self.k_scan}"
        )
        return selected

    # ------------------------------------------------------------------
    # Q-table and index update
    # ------------------------------------------------------------------

    def update(
        self,
        band_id: int,
        action: int,
        reward: float,
        belief_prev: float,
        belief_next: float,
    ) -> None:
        """Update Q-table and Whittle index for one (band, action, transition).

        Called after each scan step for bands that were ACTUALLY scanned.
        For passive bands, the caller may call this with action=ACTION_PASSIVE
        and reward=0 to update their Q-estimates from predicted transitions.

        Args:
            band_id:     The band that was acted upon.
            action:      ACTION_PASSIVE (0) or ACTION_SCAN (1).
            reward:      Observed reward (0.0 or 1.0 for binary detection).
            belief_prev: Belief b_i BEFORE the step (used to find the bin).
            belief_next: Belief b_i AFTER the step (used for Q bootstrap).
        """
        bin_prev = self._belief_to_bin(float(np.clip(belief_prev, 0.0, 1.0)))
        bin_next = self._belief_to_bin(float(np.clip(belief_next, 0.0, 1.0)))

        # Harmonic learning rate — self-tuning, no hyperparameter
        n = int(self._N[band_id, bin_prev, action])
        lr = 1.0 / (n + 1)

        # TD(0) Q-update: Q(b, a) ← Q(b, a) + lr * (r + γ * max_a' Q(b', a') - Q(b, a))
        q_next_max = float(np.max(self._Q[band_id, bin_next, :]))
        td_target = reward + self.gamma * q_next_max
        self._Q[band_id, bin_prev, action] += lr * (
            td_target - float(self._Q[band_id, bin_prev, action])
        )

        # Visit count update
        self._N[band_id, bin_prev, action] += 1
        self._total_visits += 1

        # Whittle index update (slow timescale):
        # W(b) moves toward the value that equalises Q(b, active) ≈ Q(b, passive).
        # Specifically: W ← W + alpha_w * (Q(b, active) - Q(b, passive))
        # This nudges the index upward when active is better (under-explored bands
        # that look good), and downward when passive is better.
        q_active = float(self._Q[band_id, bin_prev, ACTION_SCAN])
        q_passive = float(self._Q[band_id, bin_prev, ACTION_PASSIVE])
        self._W[band_id, bin_prev] += self.alpha_w * (q_active - q_passive)
        # Clip index to reasonable range to prevent runaway values
        self._W[band_id, bin_prev] = float(
            np.clip(self._W[band_id, bin_prev], -10.0, 10.0)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def memory_bytes(self) -> int:
        """Return total memory used by all tables in bytes."""
        return int(self._Q.nbytes + self._N.nbytes + self._W.nbytes)

    def get_index_table(self, band_id: int) -> np.ndarray:
        """Return the Whittle index vector for a band (shape: n_bins). Diagnostic only."""
        return self._W[band_id].copy()
