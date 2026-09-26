"""Clarkson Periodic Sensor Scheduling Baseline.

Implements the analytical baseline from:
  Clarkson, I. V. L. (2003). "The Arithmetic of Receiver Scheduling for
  Electronic Support." Proc. IEEE Aerospace Conference.

  Clarkson, I. V. L. (2005). "Optimal Periodic Sensor Scheduling in
  Electronic Support." Defense Applications of Signal Processing (DASP).

─────────────────────────────────────────────────────────────────────────────
Background — what Clarkson's method solves
─────────────────────────────────────────────────────────────────────────────
In classical EW scheduling, the receiver has M simultaneous channels. The
problem is: given a set of emitters with known scan periods T_i and duty
cycles d_i, choose a receiver sweep period T_r and a fixed dwell sequence
so that the worst-case intercept probability is maximised.

Clarkson (2003) shows that when an emitter's period T_i and the receiver's
sweep period T_r are rationally related (T_i / T_r = p/q for small integers),
systematic misses occur: the receiver always arrives at the band when the
emitter is off. To avoid this, T_r should be chosen as an irrational multiple
of all emitter periods — equivalently, a period whose ratio to every T_i has
a Farey approximation with a large denominator q (the "arithmetic" of the
title refers to Farey sequences and Diophantine approximation).

Clarkson (2005) extends this to the multi-band case with K simultaneous
scans, showing that the optimal dwell sequence visits each emitter's band
at intervals that are incommensurable with the emitter's period.

─────────────────────────────────────────────────────────────────────────────
Implementation — what we do here
─────────────────────────────────────────────────────────────────────────────
This is an ANALYTICAL REFERENCE BASELINE, not a fair comparison:
  - Clarkson's method assumes full knowledge of emitter periods and bands.
  - Our WIQL-UCB scheduler operates with ZERO prior intelligence.
  - We use this as an upper-bound reference, not as a competing system.

The implementation:

  1. Accept the emitter set (ground-truth periods and bands) at construction
     time. This is the "known parameters" assumption from the paper.

  2. Compute a receiver sweep period T_r using the "incommensurability"
     heuristic: choose T_r = LCM(T_1, ..., T_m) / φ where φ = golden ratio
     ≈ 1.618. This gives T_r an irrational relationship to each T_i (since
     LCM/φ is irrational), avoiding harmonic lockout. This is the key insight
     of Clarkson (2003) expressed as a simple approximation.

     Formally, we choose T_r such that for each emitter i:
         |T_r / T_i - p/q| > 1/(2q²)    for all integers p, q ≤ Q_MAX
     (i.e., T_r/T_i is a "badly approximable" number). The golden-ratio
     construction achieves this by Hurwitz's theorem.

  3. Build a fixed cyclic dwell schedule: at each step t, scan the K bands
     whose "phase deficit" |t·K - scheduled_band_phase| is smallest. This
     gives each band equal coverage at rate K/N while rotating through the
     bands at the irrational rate.

  4. For non-periodic (background/agile) scenarios, fall back to round-robin
     since Clarkson's method is only defined for known-period emitters.

─────────────────────────────────────────────────────────────────────────────
Documented approximation
─────────────────────────────────────────────────────────────────────────────
This is NOT a complete implementation of Farey-series Diophantine scheduling.
The full construction from Clarkson (2003) requires:
  - Computing best rational approximations p_i/q_i to each T_i/T_r ratio
  - Choosing T_r to maximise min_i q_i (the "depth of approximation")
  - Building the dwell sequence via a Stern-Brocot tree traversal

That construction is O(N_emitters × Q_MAX²) and requires exact period
knowledge with high precision. We use the φ-based approximation instead,
which achieves the same "avoid harmonic lockout" property with O(1) cost
and is documented as an approximation in this file.

The approximation is conservative: it may not achieve the optimal worst-case
intercept probability, but it avoids the systematic misses that round-robin
suffers for harmonically-related periods.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_PHI = (1 + math.sqrt(5)) / 2  # golden ratio ≈ 1.618033...


class ClarksonPolicy:
    """Analytical periodic scheduling baseline (Clarkson 2003/2005).

    This is a KNOWN-PARAMETERS baseline — it takes ground-truth emitter
    periods and bands at construction time. It is used as an analytical
    reference bound, not a fair no-prior comparison.

    Interface matches RoundRobinPolicy and RandomPolicy so it plugs into
    the existing harness without modification.

    Args:
        n_bands:        Total number of frequency bands.
        k_scan:         Number of bands to scan simultaneously.
        emitter_periods: List of emitter periods in slots. May be empty for
                         non-periodic scenarios (falls back to round-robin).
        emitter_bands:   List of emitter band indices (parallel to periods).
                         If empty or None, uses all bands.
        seed:           Unused (for API compatibility with RandomPolicy).
    """

    def __init__(
        self,
        n_bands: int,
        k_scan: int,
        emitter_periods: Sequence[float] | None = None,
        emitter_bands: Sequence[int] | None = None,
        seed: int | None = None,   # unused, API compatibility
    ) -> None:
        if k_scan >= n_bands:
            raise ValueError(f"k_scan ({k_scan}) must be < n_bands ({n_bands})")

        self.n_bands = n_bands
        self.k_scan  = k_scan
        self._t: int = 0

        # Precompute dwell phase offsets using φ-based incommensurability
        if emitter_periods and len(emitter_periods) > 0:
            self._schedule = self._build_schedule(
                n_bands, k_scan,
                list(emitter_periods),
                list(emitter_bands) if emitter_bands else list(range(n_bands)),
            )
        else:
            # No period information → fall back to round-robin
            self._schedule = None  # signals round-robin mode

        # Precompute the full cyclic schedule as a list of sets for fast lookup
        # We precompute one full LCM cycle (capped at 10000 steps for memory)
        if self._schedule is not None:
            cycle_len = min(10_000, self._schedule["cycle_len"])
            self._precomputed: list[set[int]] = [
                self._schedule_at(t) for t in range(cycle_len)
            ]
            self._cycle_len = cycle_len
        else:
            self._precomputed = []
            self._cycle_len = n_bands  # round-robin period

    # ─────────────────────────────────────────────────────────────────────────
    # Schedule construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_schedule(
        self,
        n_bands: int,
        k_scan: int,
        periods: list[float],
        bands: list[int],
    ) -> dict:
        """Build the φ-incommensurable sweep schedule.

        Strategy:
          1. Compute T_r = LCM(periods) / φ — ensures T_r/T_i is irrational
             for each T_i.
          2. Assign each band a phase offset proportional to its index in the
             priority order (periodic-emitter bands first, then others).
          3. At each step t, select the K bands with smallest phase error
             |t - phase_i / T_r| mod 1.

        Returns a dict with:
          - sweep_period: T_r in slots
          - phases:       per-band phase offsets in [0, 1)
          - cycle_len:    integer number of steps in one full cycle
        """
        # Step 1: compute LCM of integer-rounded periods (Clarkson uses integer periods)
        int_periods = [max(1, round(p)) for p in periods]
        lcm_val = int_periods[0]
        for p in int_periods[1:]:
            lcm_val = lcm_val * p // math.gcd(lcm_val, p)
        lcm_val = min(lcm_val, 10_000)  # cap to avoid huge precomputation

        # Step 2: T_r = LCM / φ — irrational relative to each period
        T_r = lcm_val / _PHI

        # Step 3: assign phase offsets
        # Priority order: periodic-emitter bands get phases clustered near 0
        # (they're most important to catch); other bands fill in uniformly.
        priority_bands = list(dict.fromkeys(bands))  # unique, preserving order
        other_bands    = [b for b in range(n_bands) if b not in priority_bands]
        ordered_bands  = priority_bands + other_bands

        # Phase for band i = i / n_bands, but rotated so priority bands come first
        phases = np.zeros(n_bands)
        for rank, band in enumerate(ordered_bands):
            # Use φ-spaced phases to maximise irrational separation
            phases[band] = (rank * _PHI) % 1.0

        return {
            "sweep_period": T_r,
            "phases":       phases,
            "ordered_bands": ordered_bands,
            "cycle_len":    int(round(T_r)) * 2 + 1,  # approximate cycle length
        }

    def _schedule_at(self, t: int) -> set[int]:
        """Return the K bands to scan at time step t (φ-incommensurable schedule)."""
        if self._schedule is None:
            # Round-robin fallback
            start = (t * self.k_scan) % self.n_bands
            return {(start + j) % self.n_bands for j in range(self.k_scan)}

        phases    = self._schedule["phases"]
        T_r       = self._schedule["sweep_period"]

        # Current receiver phase: advances at rate 1/T_r steps/slot
        t_phase = (t / T_r) % 1.0

        # Score each band: how close is its assigned phase to the current receiver phase?
        # Circular distance on [0, 1)
        dists = np.array([
            min(abs(t_phase - phases[b]), 1.0 - abs(t_phase - phases[b]))
            for b in range(self.n_bands)
        ])

        # Select K bands with smallest circular distance (closest to receiver phase)
        order = np.argsort(dists, kind="stable")
        return set(int(order[j]) for j in range(self.k_scan))

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def select_arms(self, belief=None) -> set[int]:
        """Return K bands to scan at the current time step.

        Args:
            belief: Ignored — Clarkson's schedule is precomputed, not adaptive.

        Returns:
            Set of K distinct band indices.
        """
        if self._precomputed:
            action = self._precomputed[self._t % self._cycle_len]
        else:
            # Round-robin fallback
            start = (self._t * self.k_scan) % self.n_bands
            action = {(start + j) % self.n_bands for j in range(self.k_scan)}
        self._t += 1
        return action

    def reset(self) -> None:
        """Reset the step counter."""
        self._t = 0

    @classmethod
    def from_renewal_env(cls, env, k_scan: int) -> "ClarksonPolicy":
        """Construct a ClarksonPolicy from a RenewalEnv's emitter set.

        Extracts period (in slots) and band for each periodic/jittered emitter.
        Background and freq_agile emitters are ignored (no known period).

        Args:
            env:    A RenewalEnv instance (must have _emitters set).
            k_scan: Number of simultaneous scans.

        Returns:
            ClarksonPolicy configured with the env's periodic emitter parameters.
        """
        periods = []
        bands   = []
        for em in getattr(env, "_emitters", []):
            if em.emitter_type in ("periodic", "jittered") and em.T_slots > 0:
                periods.append(em.T_slots)
                bands.append(em.band)
        return cls(
            n_bands=env.n_bands,
            k_scan=k_scan,
            emitter_periods=periods if periods else None,
            emitter_bands=bands   if bands   else None,
        )
