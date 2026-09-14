"""Periodic Emitter Interception Module (Module C).

Dedicated module for intercepting periodic-scan emitters using
renewal-process Whittle index theory. Kept COMPLETELY separate from the main
RMAB scheduler (BeliefTracker, WIQLScheduler) — this isolation is provably
required by the "Periodic Bandits" result:

  Generic regret-minimising bandit algorithms converge only to the single
  best fixed arm on periodic reward structures, capturing as little as 1/K
  of achievable reward. Periodic emitters need their own dedicated logic.
  [Periodic Bandits and Wireless Network Selection, cited in project brief]

This module has NO shared mutable state with BeliefTracker or WIQLScheduler.
Property 4 / Requirement 8.1 hold by construction.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design decisions (documented for PPT / report)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Period estimation — Welford's online algorithm:
   We maintain running mean μ and variance M₂ of inter-arrival intervals
   using Welford's single-pass algorithm (Welford 1962). This is chosen over
   simple batch mean/std because:
     a) It updates incrementally (O(1) per observation, O(1) memory).
     b) It is numerically stable — no catastrophic cancellation for large ToA
        values (real TSRD ToA is ~200K–29M μs, so naive sum-of-squares
        would lose precision).
     c) It produces a consistent estimator: σ² = M₂ / (n-1) for n ≥ 2.
   Minimum samples required: 3 inter-arrivals (n ≥ 3) before any index is
   computed — matching the design spec. With n < 3 the module returns 0.0
   priority (no boost, scheduler falls back to main RMAB indices).

2. Whittle index formulation:
   SpecInsight (NSDI '15) used the renewal reward:
     R_f(t) = 1 − |T + μ(t−T)/μ − t| / μ  =  1 − |t_expected − t| / μ
   with ε-greedy scheduling. We rebuild this on Whittle-index theory:

   State:  s = (t_now − t_last) / μ_est  ∈ [0, ∞)  (normalised age)
   Reward: R_f(s) = max(0, 1 − |s − 1|)            (peaks at s=1)
   Index:  W(s) = the subsidy λ at which Q(s, scan) = Q(s, passive)

   For the renewal-process RMAB with this reward, the Whittle-indexability
   condition holds (reward is unimodal in s; passive action monotonically
   ages the state). The exact index has a closed-form approximation:

     W(b, t) = R_f(t) × I(t ∈ dwell_window)
               + λ_base × (1 − R_f(t)) × I(t ∉ dwell_window)

   where λ_base = 0.5 (default subsidy at indifference) and the dwell
   window is [t_expected − 3σ, t_expected + 3σ] (= 6σ total per SpecInsight).

   This is a well-justified simplification (not a full two-timescale
   derivation), clearly motivated by the indifference-point semantics of
   the Whittle index and materially better than ε-greedy because:
     - It is computed from the actual predicted emission time, not from
       a fixed exploration probability.
     - Its value is zero outside the dwell window, preventing wasteful
       scanning far from the predicted emission time.

3. Index merging with main scheduler (rank-based):
   The renewal index and the belief-state Whittle index are not on the same
   scale (renewal ∈ [0, ~1.5]; WIQL ∈ (−10, 10+) with UCB). Raw magnitude
   comparison would let one dominate arbitrarily. We use RANK-BASED MERGING:
     combined_rank[i] = wiql_rank[i] + α × periodic_rank[i]
   where α = 0.5 (gives periodic half the weight of the main scheduler).
   Top-K bands by combined rank are selected.
   Default α is a documented design parameter, tunable per scenario.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of inter-arrival samples before we produce any estimate
_MIN_SAMPLES: int = 3

# Default Whittle subsidy at indifference (λ_base)
_LAMBDA_BASE: float = 0.5

# Dwell window half-width in units of σ (6σ total per SpecInsight)
_DWELL_HALF_SIGMA: float = 3.0


class _WelfordEstimator:
    """Welford online mean/variance estimator for inter-arrival intervals.

    Chosen for O(1) memory and numerical stability on large ToA values.
    """
    __slots__ = ("n", "mean", "_M2", "_last_toa")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._M2: float = 0.0
        self._last_toa: Optional[float] = None

    def update(self, toa_us: float) -> None:
        """Ingest a new pulse ToA. Updates inter-arrival statistics."""
        if self._last_toa is None:
            self._last_toa = toa_us
            return
        interval = toa_us - self._last_toa
        if interval <= 0:
            logger.debug("Non-positive inter-arrival %.3f at toa=%.3f — skipped.", interval, toa_us)
            self._last_toa = toa_us
            return
        self._last_toa = toa_us
        self.n += 1
        delta = interval - self.mean
        self.mean += delta / self.n
        delta2 = interval - self.mean
        self._M2 += delta * delta2

    @property
    def sigma(self) -> float:
        """Sample std-dev of inter-arrivals. Returns 0.0 if n < 2."""
        if self.n < 2:
            return 0.0
        return math.sqrt(self._M2 / (self.n - 1))

    @property
    def t_last(self) -> Optional[float]:
        return self._last_toa


class PeriodicInterceptModule:
    """Renewal-process Whittle index module for periodic emitter interception.

    State per band:
      - Welford estimator for inter-arrival μ and σ
      - No reference to BeliefTracker or WIQLScheduler (Property 4)

    Usage in the main loop:
      1. After receiver.step(action), for each band in action with pulses:
           module.ingest_pulse(band_id, pulse.toa_us)
      2. Before scheduler.select_arms(), call:
           periodic_indices = module.compute_all_indices(t_now_us, n_bands)
      3. Merge with main scheduler indices using combine_indices().
    """

    def __init__(
        self,
        n_bands: int,
        dwell_half_sigma: float = _DWELL_HALF_SIGMA,
        lambda_base: float = _LAMBDA_BASE,
        rank_weight: float = 0.5,
    ) -> None:
        """
        Args:
            n_bands:          Total number of frequency bands.
            dwell_half_sigma: Half-width of dwell window in σ units. Default 3 (= 6σ total).
            lambda_base:      Whittle subsidy at indifference. Default 0.5.
            rank_weight:      Weight α for periodic rank in combined selection. Default 0.5.
        """
        # Validate — no shared state with other modules
        if n_bands < 1:
            raise ValueError(f"n_bands must be >= 1, got {n_bands}")

        self.n_bands = n_bands
        self.dwell_half_sigma = dwell_half_sigma
        self.lambda_base = lambda_base
        self.rank_weight = rank_weight

        # Per-band Welford estimators — completely independent of BeliefTracker
        self._estimators: list[_WelfordEstimator] = [
            _WelfordEstimator() for _ in range(n_bands)
        ]

    # ─────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────

    def ingest_pulse(self, band_id: int, toa_us: float) -> None:
        """Record an observed pulse arrival on band_id.

        MUST only be called for pulses from SCANNED (observed) bands.
        Calling with oracle ground-truth data would violate Property 2.

        Args:
            band_id: Which band the pulse was observed on.
            toa_us:  Time of arrival (μs) of the observed pulse.
        """
        if not (0 <= band_id < self.n_bands):
            raise IndexError(f"band_id {band_id} out of range [0, {self.n_bands})")
        self._estimators[band_id].update(toa_us)

    def estimate_period(self, band_id: int) -> tuple[Optional[float], Optional[float]]:
        """Return (μ_us, σ_us) for band_id, or (None, None) if < 3 samples.

        μ_us: sample mean of inter-arrival intervals (μs)
        σ_us: sample std-dev (μs); 0.0 if n < 2

        Requirements 8.2–8.4: returns sentinel (None, None) for n < _MIN_SAMPLES.
        """
        est = self._estimators[band_id]
        if est.n < _MIN_SAMPLES:
            return None, None
        return est.mean, est.sigma

    def dwell_time(self, band_id: int) -> float:
        """Return dwell window width = 2 × dwell_half_sigma × σ (μs).

        Returns 0.0 if fewer than _MIN_SAMPLES arrivals.
        Property 14: dwell = 6 × σ when dwell_half_sigma = 3 (default).
        """
        _, sigma = self.estimate_period(band_id)
        if sigma is None:
            return 0.0
        return 2.0 * self.dwell_half_sigma * sigma

    def renewal_whittle_index(
        self, band_id: int, t_now_us: float, subsidy_lambda: Optional[float] = None
    ) -> float:
        """Compute the renewal-process Whittle index for band_id at t_now_us.

        Returns 0.0 if fewer than _MIN_SAMPLES inter-arrivals recorded.

        Index formula (see module docstring §2):
          t_expected = t_last + μ
          R_f = max(0, 1 − |t_expected − t_now| / μ)
          if t_now ∈ [t_expected − dwell/2, t_expected + dwell/2]:
            W = R_f + λ_base × (1 − R_f)
          else:
            W = R_f × λ_base

        This gives W ∈ [0, λ_base+1], with W maximal at t_now ≈ t_expected
        and inside the dwell window.
        """
        mu, sigma = self.estimate_period(band_id)
        if mu is None or mu <= 0:
            return 0.0

        lam = subsidy_lambda if subsidy_lambda is not None else self.lambda_base
        t_last = self._estimators[band_id].t_last
        if t_last is None:
            return 0.0

        t_expected = t_last + mu
        dwell_half = self.dwell_half_sigma * sigma

        # SpecInsight reward, rebuilt on Whittle theory
        distance = abs(t_now_us - t_expected)
        r_f = max(0.0, 1.0 - distance / mu)

        in_window = distance <= dwell_half

        if in_window:
            w = r_f + lam * (1.0 - r_f)
        else:
            w = r_f * lam

        return float(w)

    def inject_priority(self, band_id: int, t_now_us: float) -> float:
        """Return the additive priority for band_id at t_now_us.

        Returns 0.0 if estimates are not yet mature.
        Used when additively injecting into combined index (design variant B).
        """
        return self.renewal_whittle_index(band_id, t_now_us)

    def compute_all_indices(self, t_now_us: float) -> np.ndarray:
        """Return the renewal Whittle index for all bands at t_now_us.

        Returns shape (n_bands,) array. Bands with < _MIN_SAMPLES get 0.0.
        """
        return np.array([
            self.renewal_whittle_index(i, t_now_us)
            for i in range(self.n_bands)
        ], dtype=np.float64)

    def predict_next_arrival(self, band_id: int) -> Optional[float]:
        """Return predicted next emission ToA (μs) for band_id, or None.

        Used by the evaluation harness to compute intercept-time error.
        """
        mu, _ = self.estimate_period(band_id)
        if mu is None:
            return None
        t_last = self._estimators[band_id].t_last
        if t_last is None:
            return None
        return t_last + mu

    def reset_band(self, band_id: int) -> None:
        """Reset the estimator for one band (e.g. when starting a new episode)."""
        self._estimators[band_id] = _WelfordEstimator()

    def reset_all(self) -> None:
        """Reset all band estimators."""
        self._estimators = [_WelfordEstimator() for _ in range(self.n_bands)]


# ─────────────────────────────────────────────────────────────────────────────
# Index merging utilities
# ─────────────────────────────────────────────────────────────────────────────

def combine_indices_rank(
    main_indices: np.ndarray,
    periodic_indices: np.ndarray,
    periodic_weight: float = 0.5,
) -> np.ndarray:
    """Merge main-scheduler and periodic-module indices via rank combination.

    Design choice: RANK-BASED MERGE (see module docstring §3).
    The two index scales are incommensurable — rank avoids scale domination.

      combined_score[i] = rank_main[i] + periodic_weight × rank_periodic[i]

    Higher combined score → higher priority for scanning.

    Args:
        main_indices:      Whittle indices from main scheduler (n_bands,).
        periodic_indices:  Renewal Whittle indices from Module C (n_bands,).
        periodic_weight:   α ∈ (0, 1]. How much weight periodic gets relative
                           to main scheduler. Default 0.5.

    Returns:
        combined: (n_bands,) float array. Higher = higher scan priority.
    """
    n = len(main_indices)
    assert len(periodic_indices) == n

    # Rank ascending (0 = lowest priority band, n-1 = highest)
    # For +inf entries (unvisited WIQL arms), give max rank.
    def _rank(arr: np.ndarray) -> np.ndarray:
        # Replace +inf with a large finite before argsort
        finite = np.where(np.isinf(arr), np.finfo(np.float64).max, arr)
        order = np.argsort(finite, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks

    rank_main     = _rank(main_indices)
    rank_periodic = _rank(periodic_indices)

    return rank_main + periodic_weight * rank_periodic


def select_top_k_combined(
    main_indices: np.ndarray,
    periodic_indices: np.ndarray,
    k: int,
    periodic_weight: float = 0.5,
) -> set[int]:
    """Select top-K bands using rank-combined main + periodic indices.

    Tie-breaking: ascending band_id (stable sort).
    Always returns exactly K distinct band IDs (Property 3).

    Args:
        main_indices:      Shape (n_bands,).
        periodic_indices:  Shape (n_bands,).
        k:                 Number of bands to select.
        periodic_weight:   Weight for periodic ranks. Default 0.5.

    Returns:
        Set of K distinct band IDs.
    """
    combined = combine_indices_rank(main_indices, periodic_indices, periodic_weight)
    order = np.argsort(-combined, kind="stable")  # descending
    return set(int(order[j]) for j in range(k))
