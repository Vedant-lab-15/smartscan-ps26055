"""Synthetic PDW Generator.

Produces Pulse records conforming to the same schema and label semantics as the
TSRD HDF5 loader. Labels reset per 'file' (file_id parameter), enforcing
Requirement 4 (no cross-file emitter label joins).

Emission modes (per band, backward-compatible):
  - Bernoulli (default): each band emits independently with probability p_emit
    per time step. This is the original mode and remains unchanged.
  - Periodic: band emits with period T_us ± Gaussian jitter sigma_us around
    each expected emission time. Realistic model for radar emitters with
    a known Pulse Repetition Interval (PRI). Enabled by passing
    periodic_config={band_idx: (T_us, sigma_us, t_start_us)} at construction.

A single environment can mix both modes — periodic_config specifies only the
periodic bands; the rest remain Bernoulli. This reflects reality: not all
emitters in the environment are periodic.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.environment.pulse import Pulse, validate_pulse


@dataclass
class PeriodicBandConfig:
    """Configuration for one periodic-emission band.

    Attributes:
        period_us:   Nominal emission period (μs). Must be > 0.
        sigma_us:    Gaussian jitter std-dev on each inter-arrival (μs). ≥ 0.
        t_start_us:  Time of first emission (μs). Default 0.
    """
    period_us:  float
    sigma_us:   float
    t_start_us: float = 0.0


class PDWGenerator:
    """Synthetic Pulse Descriptor Word generator.

    Drop-in replacement for TSRDLoader — the rest of the system must not
    distinguish between the two sources.

    Emission modes:
      Bernoulli bands: emit independently per step with probability p_emit.
      Periodic bands:  emit at intervals T_us ± N(0, σ_us) regardless of
                       time-step boundaries. Whether a pulse falls in a given
                       time step [t*dt_us, (t+1)*dt_us) is determined by the
                       continuous emission schedule.

    Per-band periodic config is passed as a dict:
      periodic_config = {0: PeriodicBandConfig(period_us=500.0, sigma_us=10.0)}
    Bands not in the dict use Bernoulli emission. This is backward-compatible —
    existing code that doesn't pass periodic_config is unaffected.
    """

    def __init__(
        self,
        n_bands: int,
        band_cf_mhz: list[float],
        dt_us: float,
        p_emit: float | list[float] = 0.3,
        pw_us_range: tuple[float, float] = (1.0, 50.0),
        aoa_deg_range: tuple[float, float] = (-90.0, 90.0),
        amp_db_range: tuple[float, float] = (-80.0, -20.0),
        seed: int | None = None,
        periodic_config: dict[int, PeriodicBandConfig] | None = None,
    ) -> None:
        if len(band_cf_mhz) != n_bands:
            raise ValueError(f"band_cf_mhz length ({len(band_cf_mhz)}) must equal n_bands ({n_bands})")
        if dt_us <= 0:
            raise ValueError(f"dt_us must be > 0, got {dt_us}")

        # p_emit can be a scalar (broadcast to all bands) or a per-band list
        if isinstance(p_emit, (int, float)):
            if not (0.0 < float(p_emit) <= 1.0):
                raise ValueError(f"p_emit must be in (0, 1], got {p_emit}")
            p_emit_list = [float(p_emit)] * n_bands
        else:
            p_emit_list = list(p_emit)
            if len(p_emit_list) != n_bands:
                raise ValueError(
                    f"p_emit list length ({len(p_emit_list)}) must equal n_bands ({n_bands})"
                )
            for i, p in enumerate(p_emit_list):
                if not (0.0 < p <= 1.0):
                    raise ValueError(f"p_emit[{i}] must be in (0, 1], got {p}")

        # Validate periodic configs
        self._periodic: dict[int, PeriodicBandConfig] = {}
        if periodic_config:
            for band_idx, cfg in periodic_config.items():
                if not (0 <= band_idx < n_bands):
                    raise ValueError(f"periodic_config band_idx {band_idx} out of range [0, {n_bands})")
                if cfg.period_us <= 0:
                    raise ValueError(f"period_us must be > 0 for band {band_idx}, got {cfg.period_us}")
                if cfg.sigma_us < 0:
                    raise ValueError(f"sigma_us must be >= 0 for band {band_idx}, got {cfg.sigma_us}")
                self._periodic[band_idx] = cfg

        self.n_bands = n_bands
        self.band_cf_mhz = list(band_cf_mhz)
        self.dt_us = dt_us
        self._p_emit_list: list[float] = p_emit_list
        self.p_emit: float | list[float] = p_emit
        self.pw_us_range = pw_us_range
        self.aoa_deg_range = aoa_deg_range
        self.amp_db_range = amp_db_range
        self._rng = np.random.default_rng(seed)
        self._seed = seed

        # Cache: t_step -> list[Pulse]
        self._cache: dict[int, list[Pulse]] = {}

        # For periodic bands: pre-compute the full emission schedule up to some
        # horizon lazily as steps are requested. Store list of scheduled ToAs.
        # _periodic_schedule[band_idx] = sorted list of true emission times (μs)
        self._periodic_schedule: dict[int, list[float]] = {}
        self._periodic_schedule_horizon_us: dict[int, float] = {}
        for band_idx in self._periodic:
            self._periodic_schedule[band_idx] = []
            self._periodic_schedule_horizon_us[band_idx] = -1.0

    def _extend_periodic_schedule(self, band_idx: int, up_to_us: float) -> None:
        """Extend the emission schedule for band_idx up to up_to_us μs."""
        cfg = self._periodic[band_idx]
        schedule = self._periodic_schedule[band_idx]
        horizon  = self._periodic_schedule_horizon_us[band_idx]

        if horizon >= up_to_us:
            return  # already covers the requested window

        # Start from t_start_us if empty, or from last scheduled emission
        if not schedule:
            next_toa = cfg.t_start_us
        else:
            next_toa = schedule[-1] + cfg.period_us + self._rng.normal(0, cfg.sigma_us)

        # Generate emissions until we pass up_to_us (with some buffer)
        while next_toa <= up_to_us + cfg.period_us * 2:
            if next_toa >= 0:
                # Clamp to a small positive value if exactly zero — validate_pulse
                # requires toa_us > 0, and t_start_us=0.0 would produce toa=0.0.
                toa_to_append = max(next_toa, 1e-6)
                schedule.append(toa_to_append)
            jitter = self._rng.normal(0, cfg.sigma_us) if cfg.sigma_us > 0 else 0.0
            next_toa = next_toa + cfg.period_us + jitter

        self._periodic_schedule_horizon_us[band_idx] = up_to_us + cfg.period_us * 2

    def _get_periodic_pulses_in_step(
        self, band_idx: int, t_step: int, file_id: int
    ) -> list[Pulse]:
        """Return periodic-mode pulses that fall in [t_step*dt_us, (t_step+1)*dt_us)."""
        t_lo = t_step * self.dt_us
        t_hi = (t_step + 1) * self.dt_us
        self._extend_periodic_schedule(band_idx, t_hi)

        pulses = []
        for toa in self._periodic_schedule[band_idx]:
            if t_lo <= toa < t_hi:
                try:
                    p = Pulse(
                        toa_us=toa,
                        cf_mhz=self.band_cf_mhz[band_idx],
                        pw_us=float(self._rng.uniform(*self.pw_us_range)),
                        aoa_deg=float(self._rng.uniform(*self.aoa_deg_range)),
                        amp_db=float(self._rng.uniform(*self.amp_db_range)),
                        emitter=band_idx * 1000 + file_id,
                    )
                    validate_pulse(p)
                    pulses.append(p)
                except ValueError as e:
                    warnings.warn(f"Skipping invalid periodic pulse at t={t_step}, band={band_idx}: {e}")
            elif toa >= t_hi:
                break  # schedule is ordered; no more pulses in this window
        return pulses

    def get_periodic_schedule(self, band_idx: int) -> list[float]:
        """Return a copy of the pre-computed emission schedule for a periodic band.

        Oracle access — used only by the evaluation harness and tests.
        Raises ValueError if band_idx is not a periodic band.
        """
        if band_idx not in self._periodic:
            raise ValueError(f"Band {band_idx} is not a periodic band.")
        return list(self._periodic_schedule[band_idx])

    def generate(self, t_step: int, file_id: int = 0) -> list[Pulse]:
        """Generate pulses for time step t_step.

        emitter = band_index * 1000 + file_id — unique per file, never overlaps
        across file_id values (Requirement 4).

        Periodic bands emit from their pre-computed schedule; Bernoulli bands
        emit with independent probability p_emit per step.
        """
        if t_step in self._cache:
            return self._cache[t_step]

        pulses: list[Pulse] = []
        base_toa = t_step * self.dt_us

        for band_idx in range(self.n_bands):
            if band_idx in self._periodic:
                # Periodic mode: emit if a scheduled emission falls in this step
                band_pulses = self._get_periodic_pulses_in_step(band_idx, t_step, file_id)
                pulses.extend(band_pulses)
            else:
                # Bernoulli mode (original behaviour — unchanged)
                if self._rng.random() < self._p_emit_list[band_idx]:
                    jitter = self._rng.uniform(0.0, self.dt_us * 0.1)
                    toa = base_toa + jitter
                    if toa <= 0:
                        toa = self.dt_us * 0.001
                    try:
                        p = Pulse(
                            toa_us=toa,
                            cf_mhz=self.band_cf_mhz[band_idx],
                            pw_us=float(self._rng.uniform(*self.pw_us_range)),
                            aoa_deg=float(self._rng.uniform(*self.aoa_deg_range)),
                            amp_db=float(self._rng.uniform(*self.amp_db_range)),
                            emitter=band_idx * 1000 + file_id,
                        )
                        validate_pulse(p)
                        pulses.append(p)
                    except ValueError as e:
                        warnings.warn(f"Skipping invalid synthetic pulse at t={t_step}, band={band_idx}: {e}")

        pulses.sort(key=lambda pulse: pulse.toa_us)
        self._cache[t_step] = pulses
        return pulses

    def reset(self, seed: int | None = None) -> None:
        """Reset the internal RNG, cache, and periodic schedules."""
        self._rng = np.random.default_rng(seed if seed is not None else self._seed)
        self._cache.clear()
        for band_idx in self._periodic:
            self._periodic_schedule[band_idx] = []
            self._periodic_schedule_horizon_us[band_idx] = -1.0
