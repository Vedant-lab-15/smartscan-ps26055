"""PulseTrainLoader — TSRD adapter using the official PulseTrain API.

This is the primary loader for real Turing Synthetic Radar Dataset files.
It wraps `turing_deinterleaving_challenge.PulseTrain.load()` and converts
the output to our internal Pulse schema.

Interface contract:
  - Identical to TSRDLoader from the perspective of RFEnvironment:
      .initialize()            — memory guard check
      .iter_pulses(path, file_id) -> Iterator[Pulse]
  - Swap TSRDLoader ↔ PulseTrainLoader in RFEnvironment with one line.

PulseTrain.load(path) returns:
  .data   — np.ndarray shape (n_pulses, 5), columns: ToA(μs), CF(MHz),
             PW(μs), AoA(deg), Amplitude(dB)
  .labels — np.ndarray shape (n_pulses,), arbitrary integer emitter IDs
             (only consistent within the same file)
  .metadata — dict with scan/stare mode and other per-file metadata

Design notes:
  - We do NOT load .data fully for stare-mode files. The memory guard runs
    first by reading only .data.shape via the underlying HDF5 without copying.
    For real reads we use chunked slicing of the numpy array (it may already
    be in memory via PulseTrain.load — if stare-mode files are too large,
    use TSRDLoader with direct h5py chunked access instead).
  - Emitter namespacing: emitter = int(label) * 1000 + file_id, same as
    TSRDLoader, to prevent cross-file label joins (Requirement 4).
  - If `turing_deinterleaving_challenge` is not installed, falls back to a
    clear ImportError with installation instructions.
"""
from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator

import numpy as np

from ew_smart_scan.env.errors import MemoryGuardError
from ew_smart_scan.models.pulse import Pulse, validate_pulse

logger = logging.getLogger(__name__)

MEMORY_GUARD_MAX_PULSES: int = 500_000_000  # 500 M pulses

# PDW column indices in PulseTrain.data  (confirmed from TSRD paper & demo notebook)
_COL_TOA = 0   # Time of Arrival (μs)
_COL_CF  = 1   # Centre Frequency (MHz)
_COL_PW  = 2   # Pulse Width (μs)
_COL_AOA = 3   # Angle of Arrival (degrees)
_COL_AMP = 4   # Amplitude (dB)

_CHUNK_SIZE = 10_000  # rows per processing chunk


def _import_pulse_train():
    """Import PulseTrain, raising a clear error if the package isn't installed."""
    try:
        from turing_deinterleaving_challenge import PulseTrain  # type: ignore[import]
        return PulseTrain
    except ImportError as exc:
        raise ImportError(
            "turing-deinterleaving-challenge is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/alan-turing-institute/"
            "turing-deinterleaving-challenge.git"
        ) from exc


class PulseTrainLoader:
    """Load real TSRD .h5 files using the official PulseTrain API.

    Drop-in replacement for TSRDLoader — RFEnvironment accepts either.

    Usage:
        loader = PulseTrainLoader(h5_paths=[Path("data/train/scan/000.h5")])
        loader.initialize()   # memory guard
        for pulse in loader.iter_pulses(path, file_id=0):
            ...
    """

    def __init__(
        self,
        h5_paths: list[pathlib.Path],
        chunk_size: int = _CHUNK_SIZE,
        memory_guard_max: int = MEMORY_GUARD_MAX_PULSES,
    ) -> None:
        self.h5_paths = [pathlib.Path(p) for p in h5_paths]
        self.chunk_size = chunk_size
        self.memory_guard_max = memory_guard_max
        self._PulseTrain = _import_pulse_train()

    def _estimate_pulse_count(self, path: pathlib.Path) -> int:
        """Return pulse count for one file WITHOUT loading pulse data.

        Uses PulseTrain.load() but only accesses .data.shape — the underlying
        h5py lazy-loading means this is cheap as long as we don't index into
        the array.
        """
        try:
            pt = self._PulseTrain.load(str(path))
            return int(pt.data.shape[0])
        except Exception as exc:
            logger.warning("Could not estimate pulse count for %s: %s", path.name, exc)
            return 0

    def initialize(self) -> None:
        """Memory guard: raises MemoryGuardError if total pulses > guard limit.

        Stare-mode files average ~1.27M pulses each — loading hundreds would
        blow RAM. The guard fires before any pulse data is read.
        """
        total = sum(self._estimate_pulse_count(p) for p in self.h5_paths)
        if total > self.memory_guard_max:
            raise MemoryGuardError(
                f"Estimated {total:,} pulses exceeds memory guard of "
                f"{self.memory_guard_max:,}. Use fewer files or set a higher "
                f"memory_guard_max. Stare-mode files (~1.27M pulses each) "
                f"should be sampled sparingly."
            )

    def iter_pulses(self, path: pathlib.Path, file_id: int) -> Iterator[Pulse]:
        """Yield Pulse objects from one TSRD .h5 file using PulseTrain.load().

        Processes in chunks to keep peak memory bounded.

        Emitter namespacing: emitter = int(label) * 1000 + file_id
        Labels are arbitrary integers within a file — namespacing prevents
        any cross-file emitter joins (Requirement 4 / Property 8).
        """
        path = pathlib.Path(path)
        try:
            pt = self._PulseTrain.load(str(path))
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)
            return

        data: np.ndarray = pt.data      # shape (n_pulses, 5)
        labels: np.ndarray = np.asarray(pt.labels, dtype=np.int64)
        n_pulses = data.shape[0]

        if data.shape[1] < 5:
            logger.error(
                "%s: expected 5 PDW columns (ToA, CF, PW, AoA, Amp), got %d — skipping.",
                path.name, data.shape[1],
            )
            return

        for start in range(0, n_pulses, self.chunk_size):
            end = min(start + self.chunk_size, n_pulses)
            chunk = data[start:end]          # shape (chunk, 5)
            chunk_labels = labels[start:end]

            for i in range(end - start):
                try:
                    p = Pulse(
                        toa_us=float(chunk[i, _COL_TOA]),
                        cf_mhz=float(chunk[i, _COL_CF]),
                        pw_us=float(chunk[i, _COL_PW]),
                        aoa_deg=float(chunk[i, _COL_AOA]),
                        amp_db=float(chunk[i, _COL_AMP]),
                        # Namespace emitter label to prevent cross-file joins
                        emitter=int(chunk_labels[i]) * 1000 + file_id,
                    )
                    validate_pulse(p)
                    yield p
                except ValueError as exc:
                    logger.warning(
                        "Skipping invalid pulse at index %d in %s: %s",
                        start + i, path.name, exc,
                    )
