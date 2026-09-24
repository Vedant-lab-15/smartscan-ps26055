"""
Statistical Environment Generator — PS 26055 Harness
=====================================================
TSRD is a statistics source only — never the environment.
Emitter parameters are extracted from TSRD characterisation; occupancy is
generated fresh from those distributions each episode.

TSRD-grounded parameters (from Phase 1 real-data analysis):
  Periodic emitters — two characterised instances:
    Class P1: T=1511.5μs, σ_T=133.2μs, CF≈359MHz, σ/T=8.8%
    Class P2: T=502.5μs,  σ_T=35.9μs,  CF≈27MHz,  σ/T=7.1%
  Background/random emitters: CoV≥1.0 (irregular renewal, Poisson-like)
  Frequency-agile emitters: hop across 3–5 bands, modelled as uniform random
    hop with mean dwell ≈ 10 slots (no TSRD-specific hop rate available;
    documented as assumption).

NOTE: Two periodic emitter instances is a thin statistical basis.
This is noted here and in summary outputs rather than suppressed.

Emitter types:
  'background'   — Bernoulli(p) per slot; p drawn from Beta(1,3) per emitter
  'periodic'     — renewal process, inter-arrival ~ TruncNormal(T, σ_T)
  'freq_agile'   — hops bands every ~dwell_slots slots, Bernoulli active
  'jittered'     — periodic + jitter drawn from Uniform(-δ, +δ) where δ=ratio*T
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# TSRD-grounded constants
# ─────────────────────────────────────────────────────────────────────────────

# Time slot size in microseconds (1 slot = DT_US μs)
DT_US: float = 1_000.0  # 1 ms default

# Periodic emitter classes from TSRD characterisation (2 instances)
TSRD_PERIODIC_CLASSES = [
    {"T_us": 1511.5, "sigma_T_us": 133.2, "label": "P1-TSRD"},
    {"T_us": 502.5,  "sigma_T_us": 35.9,  "label": "P2-TSRD"},
]

# Background occupancy: Bernoulli(p), p ~ Beta(1, 3) → mean p ≈ 0.25, heavy tail toward 0
BACKGROUND_P_ALPHA = 1.0
BACKGROUND_P_BETA  = 3.0

# Freq-agile: dwell on one band for Poisson(lambda_dwell) slots
FREQ_AGILE_MEAN_DWELL_SLOTS = 10
FREQ_AGILE_N_BANDS_HOPPED = 4  # number of bands in the hop set


# ─────────────────────────────────────────────────────────────────────────────
# Detection model
# ─────────────────────────────────────────────────────────────────────────────

def p_detect_logistic(snr_db: float, snr_threshold_db: float = 10.0,
                      slope: float = 0.5) -> float:
    """P(detection | SNR, occupied) via logistic function.

    snr_threshold_db: SNR at which P_detect = 0.5
    slope: steepness (higher = harder threshold)
    """
    return 1.0 / (1.0 + math.exp(-slope * (snr_db - snr_threshold_db)))


# ─────────────────────────────────────────────────────────────────────────────
# Emitter data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Emitter:
    """One emitter placed in the environment."""
    emitter_type: Literal["background", "periodic", "freq_agile", "jittered"]
    band: int                      # primary band (or first band in hop set)
    hop_bands: list[int] = field(default_factory=list)  # for freq_agile
    T_slots: float = 0.0           # nominal period in slots
    sigma_T_slots: float = 0.0     # period jitter in slots
    p_active: float = 0.3          # for background type
    snr_db: float = 15.0           # mean SNR when active
    jitter_ratio: float = 0.0      # δ/T for jittered type
    label: str = ""                # human-readable label

    # Internal state (set by reset())
    _next_arrival_slot: float = field(default=0.0, init=False, repr=False)
    _current_hop_band: int = field(default=0, init=False, repr=False)
    _slots_on_current_band: int = field(default=0, init=False, repr=False)
    _next_hop_slot: int = field(default=0, init=False, repr=False)

    def reset(self, rng: np.random.Generator, t0_slot: int = 0) -> None:
        """Initialise emitter state for a new episode."""
        if self.emitter_type in ("periodic", "jittered"):
            # First arrival uniformly within the first period (phase randomisation)
            phase_offset = rng.uniform(0, max(self.T_slots, 1))
            self._next_arrival_slot = t0_slot + phase_offset
        elif self.emitter_type == "freq_agile":
            if self.hop_bands:
                self._current_hop_band = rng.integers(0, len(self.hop_bands))
            self._next_hop_slot = t0_slot + rng.poisson(FREQ_AGILE_MEAN_DWELL_SLOTS)

    def is_active_at(self, t_slot: int, rng: np.random.Generator) -> tuple[int, float]:
        """Return (band, snr_db) if active at t_slot, else (-1, 0).

        Advances internal state. Must be called for every slot in order.
        """
        if self.emitter_type == "background":
            active = rng.random() < self.p_active
            return (self.band, self.snr_db) if active else (-1, 0.0)

        elif self.emitter_type in ("periodic", "jittered"):
            if t_slot < self._next_arrival_slot:
                return (-1, 0.0)
            # Pulse arrived — schedule next
            if self.emitter_type == "jittered" and self.jitter_ratio > 0:
                delta = self.jitter_ratio * self.T_slots
                jitter = rng.uniform(-delta, delta)
            else:
                jitter = 0.0
            base_interval = max(
                1.0,
                rng.normal(self.T_slots, self.sigma_T_slots) if self.sigma_T_slots > 0
                else self.T_slots
            )
            self._next_arrival_slot = self._next_arrival_slot + base_interval + jitter
            return (self.band, self.snr_db)

        elif self.emitter_type == "freq_agile":
            # Advance hop if needed
            while t_slot >= self._next_hop_slot:
                if self.hop_bands:
                    self._current_hop_band = (
                        (self._current_hop_band + 1) % len(self.hop_bands)
                    )
                dwell = max(1, rng.poisson(FREQ_AGILE_MEAN_DWELL_SLOTS))
                self._next_hop_slot += dwell
            active_band = self.hop_bands[self._current_hop_band] if self.hop_bands else self.band
            active = rng.random() < self.p_active
            return (active_band, self.snr_db) if active else (-1, 0.0)

        return (-1, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class RenewalEnv:
    """Renewal-process environment for the statistical harness.

    One episode = T slots, K bands.
    Ground truth is hidden; the receiver observes only via scans.
    """

    def __init__(
        self,
        n_bands: int = 16,
        dt_us: float = DT_US,
        episode_len: int = 5000,
        k_scan: int = 3,
        p_false_alarm: float = 0.01,
        snr_threshold_db: float = 10.0,
        seed: int = 0,
    ) -> None:
        self.n_bands      = n_bands
        self.dt_us        = dt_us
        self.episode_len  = episode_len
        self.k_scan       = k_scan
        self.p_fa         = p_false_alarm
        self.snr_thresh   = snr_threshold_db
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)
        self._emitters: list[Emitter] = []
        self._t: int = 0

        # Ground truth cache: slot -> {band -> (active:bool, snr:float)}
        self._gt_cache: dict[int, dict[int, tuple[bool, float]]] = {}

    def set_emitters(self, emitters: list[Emitter]) -> None:
        """Assign emitters and reset to slot 0."""
        self._emitters = emitters
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._t = 0
        self._gt_cache.clear()
        for em in self._emitters:
            em.reset(self.rng)

    def _get_ground_truth(self, t: int) -> dict[int, tuple[bool, float]]:
        """Compute (active, snr) for every band at slot t."""
        if t in self._gt_cache:
            return self._gt_cache[t]
        gt: dict[int, tuple[bool, float]] = {b: (False, 0.0) for b in range(self.n_bands)}
        for em in self._emitters:
            band, snr = em.is_active_at(t, self.rng)
            if band >= 0 and 0 <= band < self.n_bands:
                # Multiple emitters on same band: take highest SNR
                if not gt[band][0] or snr > gt[band][1]:
                    gt[band] = (True, snr)
        self._gt_cache[t] = gt
        return gt

    def get_true_occupancy(self, t: int) -> dict[int, bool]:
        """Oracle: true occupancy per band at slot t. Used by eval harness only."""
        return {b: v[0] for b, v in self._get_ground_truth(t).items()}

    def step(self, action: set[int]) -> dict[int, dict]:
        """Advance one slot; return observations for scanned bands only.

        Returns:
          obs[band] = {
            "scanned": bool,
            "detected": bool,   # True if pulse seen (hit or false alarm)
            "true_active": bool # ground truth — never feed to scheduler
          }
        """
        t = self._t
        gt = self._get_ground_truth(t)
        obs: dict[int, dict] = {}

        for band in range(self.n_bands):
            scanned = band in action
            true_active, snr = gt[band]

            if scanned:
                if true_active:
                    p_det = p_detect_logistic(snr, self.snr_thresh)
                    detected = self.rng.random() < p_det
                else:
                    detected = self.rng.random() < self.p_fa
            else:
                detected = False

            obs[band] = {
                "scanned":     scanned,
                "detected":    detected,
                "true_active": true_active,
            }

        self._t += 1
        return obs

    @property
    def t(self) -> int:
        return self._t

    @property
    def done(self) -> bool:
        return self._t >= self.episode_len


# ─────────────────────────────────────────────────────────────────────────────
# Scenario factories
# ─────────────────────────────────────────────────────────────────────────────

def make_background_scenario(
    rng: np.random.Generator,
    n_bands: int,
    n_background: int = 6,
) -> list[Emitter]:
    """Random/background occupancy only — no structured emitters."""
    emitters = []
    for i in range(n_background):
        band = int(rng.integers(0, n_bands))
        p = rng.beta(BACKGROUND_P_ALPHA, BACKGROUND_P_BETA)
        snr = rng.uniform(5.0, 25.0)
        emitters.append(Emitter(
            emitter_type="background",
            band=band, p_active=float(p), snr_db=float(snr),
            label=f"bg_{i}",
        ))
    return emitters


def make_periodic_scenario(
    rng: np.random.Generator,
    n_bands: int,
    dt_us: float,
    n_background: int = 4,
    n_periodic: int = 2,
    jitter_ratio: float = 0.0,
) -> list[Emitter]:
    """Spatially-scanning periodic emitters + background."""
    emitters = make_background_scenario(rng, n_bands, n_background)

    # Pick from TSRD-characterised classes (round-robin)
    for i in range(n_periodic):
        cls = TSRD_PERIODIC_CLASSES[i % len(TSRD_PERIODIC_CLASSES)]
        T_slots = cls["T_us"] / dt_us
        sigma_T_slots = cls["sigma_T_us"] / dt_us
        band = int(rng.integers(0, n_bands))
        snr = rng.uniform(12.0, 22.0)
        em_type = "jittered" if jitter_ratio > 0 else "periodic"
        emitters.append(Emitter(
            emitter_type=em_type,
            band=band,
            T_slots=T_slots,
            sigma_T_slots=sigma_T_slots,
            snr_db=float(snr),
            jitter_ratio=jitter_ratio,
            label=f"{cls['label']}_j{jitter_ratio:.2f}",
        ))
    return emitters


def make_freq_agile_scenario(
    rng: np.random.Generator,
    n_bands: int,
    n_background: int = 3,
    n_agile: int = 3,
) -> list[Emitter]:
    """Frequency-agile emitters + background."""
    emitters = make_background_scenario(rng, n_bands, n_background)

    for i in range(n_agile):
        # Hop set: FREQ_AGILE_N_BANDS_HOPPED contiguous or random bands
        start = int(rng.integers(0, max(1, n_bands - FREQ_AGILE_N_BANDS_HOPPED)))
        hop_bands = list(range(start, min(start + FREQ_AGILE_N_BANDS_HOPPED, n_bands)))
        snr = rng.uniform(8.0, 20.0)
        emitters.append(Emitter(
            emitter_type="freq_agile",
            band=hop_bands[0],
            hop_bands=hop_bands,
            p_active=0.6,
            snr_db=float(snr),
            label=f"agile_{i}",
        ))
    return emitters
