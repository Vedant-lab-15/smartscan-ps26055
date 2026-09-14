"""RF Environment — simulated bands × time occupancy grid.

Wraps a data source (PDWGenerator or TSRDLoader) and manages ground-truth
emitter state that is NEVER directly exposed to the scheduler. The scheduler
only learns about band state through the ReceiverModel POMDP wrapper.

Ground-truth access is restricted to:
  - `get_true_state(t)` — oracle-only, used exclusively by EvaluationHarness
  - `step(t)` — same as get_true_state (internal use + oracle)

Neither method's output is ever passed to a Scheduler or BeliefTracker.
"""
from __future__ import annotations

import logging
from typing import Union

from ew_smart_scan.env.errors import CrossFileEmitterError, MemoryGuardError
from ew_smart_scan.models.pdw_generator import PDWGenerator
from ew_smart_scan.models.pulse import Pulse
from ew_smart_scan.models.tsrd_loader import TSRDLoader

logger = logging.getLogger(__name__)

DataSource = Union[TSRDLoader, PDWGenerator]


class RFEnvironment:
    """Simulated RF environment: N frequency bands × discrete time steps.

    The environment maintains hidden ground-truth occupancy. Only the
    EvaluationHarness may call get_true_state(); the scheduler/belief tracker
    must never receive that information directly.
    """

    def __init__(
        self,
        source: DataSource,
        n_bands: int,
        band_edges_mhz: list[tuple[float, float]],
        dt_us: float,
        memory_guard_max: int = 500_000_000,
    ) -> None:
        if len(band_edges_mhz) != n_bands:
            raise ValueError(
                f"band_edges_mhz length ({len(band_edges_mhz)}) must equal n_bands ({n_bands})"
            )
        if dt_us <= 0:
            raise ValueError(f"dt_us must be > 0, got {dt_us}")

        self._source = source  # private — never exposed to external callers
        self.n_bands = n_bands
        self.band_edges_mhz = band_edges_mhz
        self.dt_us = dt_us
        self.memory_guard_max = memory_guard_max

        # Occupancy index: t_step -> set of occupied band indices
        # Populated lazily for PDWGenerator; eagerly for TSRDLoader
        self._occupancy: dict[int, set[int]] = {}

        # Pulse cache: (band_id, t_step) -> list[Pulse]
        self._pulse_cache: dict[tuple[int, int], list[Pulse]] = {}

        # Per-file emitter provenance: emitter_label -> file_id
        # Used to detect and reject cross-file emitter label joins
        self._emitter_file_map: dict[int, int] = {}

        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load or index pulse data.

        For TSRDLoader: performs memory guard check and builds full occupancy
        index eagerly (file-by-file, chunked — never all in memory at once).
        For PDWGenerator: no-op (lazy on first step() call).
        """
        if isinstance(self._source, TSRDLoader):
            self._source.initialize()  # raises MemoryGuardError if needed
            self._build_occupancy_from_tsrd()
        # PDWGenerator: lazy — nothing to do yet
        self._initialized = True

    def _build_occupancy_from_tsrd(self) -> None:
        """Iterate TSRD files and populate _occupancy and _pulse_cache."""
        # Track per-file per-band last toa for monotonicity check
        last_toa: dict[tuple[int, int], float] = {}  # (file_id, band_id) -> last toa_us

        for file_id, path in enumerate(self._source.h5_paths):
            for pulse in self._source.iter_pulses(path, file_id):
                band_id = self._bin_pulse_to_band(pulse)
                if band_id is None:
                    continue

                # Monotonicity check within (file_id, band_id)
                # Real TSRD has ties (interleaved pulses) — warn but don't raise.
                # Strictly backward ToA (toa < last_toa) is flagged as an error.
                key = (file_id, band_id)
                if key in last_toa:
                    if pulse.toa_us < last_toa[key]:
                        logger.warning(
                            "Backward ToA in file %s, band %d: %.2f < %.2f — skipping pulse.",
                            path.name, band_id, pulse.toa_us, last_toa[key]
                        )
                        continue  # skip this pulse, don't crash
                    # Equal ToA (tie) is fine — multiple emitters at same time step
                last_toa[key] = pulse.toa_us

                # Track emitter provenance
                if pulse.emitter in self._emitter_file_map:
                    if self._emitter_file_map[pulse.emitter] != file_id:
                        raise CrossFileEmitterError(
                            f"Emitter label {pulse.emitter} seen in file_id "
                            f"{self._emitter_file_map[pulse.emitter]} and {file_id}"
                        )
                else:
                    self._emitter_file_map[pulse.emitter] = file_id

                # Bin to time step
                t_step = int(pulse.toa_us / self.dt_us)

                # Update occupancy
                if t_step not in self._occupancy:
                    self._occupancy[t_step] = set()
                self._occupancy[t_step].add(band_id)

                # Update pulse cache
                cache_key = (band_id, t_step)
                if cache_key not in self._pulse_cache:
                    self._pulse_cache[cache_key] = []
                self._pulse_cache[cache_key].append(pulse)

    # ------------------------------------------------------------------
    # Band assignment
    # ------------------------------------------------------------------

    def _bin_pulse_to_band(self, pulse: Pulse) -> int | None:
        """Return the band index for a pulse's centre frequency, or None if out of range."""
        for i, (lo, hi) in enumerate(self.band_edges_mhz):
            if lo <= pulse.cf_mhz < hi:
                return i
        logger.warning(
            "Pulse CF %.2f MHz falls outside all %d bands — skipping.", pulse.cf_mhz, self.n_bands
        )
        return None

    # ------------------------------------------------------------------
    # Oracle access (evaluation harness ONLY)
    # ------------------------------------------------------------------

    def step(self, t: int) -> dict[int, bool]:
        """Return ground-truth band occupancy at time step t.

        WARNING: This is oracle access. Return value must NEVER be passed to
        a Scheduler or BeliefTracker directly.
        """
        if isinstance(self._source, PDWGenerator):
            self._ensure_step_generated(t)

        occupied = self._occupancy.get(t, set())
        return {i: (i in occupied) for i in range(self.n_bands)}

    def get_true_state(self, t: int) -> dict[int, bool]:
        """Oracle access only — never pass return value to Scheduler or BeliefTracker.

        Used only by EvaluationHarness to compute Pd/Pfa against ground truth.
        """
        return self.step(t)

    # ------------------------------------------------------------------
    # Pulse sampling (used by ReceiverModel for scanned bands only)
    # ------------------------------------------------------------------

    def sample_pulses(self, band_id: int, t: int) -> list[Pulse]:
        """Return pulses that arrived in band_id during time step t.

        For PDWGenerator: generates step t lazily and caches result.
        For TSRDLoader: reads from pre-built index.
        """
        if isinstance(self._source, PDWGenerator):
            self._ensure_step_generated(t)
        return self._pulse_cache.get((band_id, t), [])

    # ------------------------------------------------------------------
    # PDWGenerator lazy generation
    # ------------------------------------------------------------------

    def _ensure_step_generated(self, t: int) -> None:
        """Generate and cache pulses for step t if not already done."""
        if t in self._occupancy:
            return  # already generated

        pulses = self._source.generate(t_step=t, file_id=0)
        occupied_bands: set[int] = set()

        for pulse in pulses:
            band_id = self._bin_pulse_to_band(pulse)
            if band_id is None:
                continue
            occupied_bands.add(band_id)
            cache_key = (band_id, t)
            if cache_key not in self._pulse_cache:
                self._pulse_cache[cache_key] = []
            self._pulse_cache[cache_key].append(pulse)

        self._occupancy[t] = occupied_bands

    # ------------------------------------------------------------------
    # Cross-file emitter guard
    # ------------------------------------------------------------------

    def check_cross_file_emitter(self, emitter: int, file_id: int) -> None:
        """Raise CrossFileEmitterError if emitter label belongs to a different file_id."""
        if emitter in self._emitter_file_map:
            if self._emitter_file_map[emitter] != file_id:
                raise CrossFileEmitterError(
                    f"Emitter label {emitter} is from file_id "
                    f"{self._emitter_file_map[emitter]}, not {file_id}."
                )
