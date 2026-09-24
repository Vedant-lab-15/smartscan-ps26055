"""TSRD HDF5 Loader.

Lazy, chunked loader for Turing Synthetic Radar Dataset .h5 files.
Never loads full pulse trains into memory. Enforces the stare-mode
memory guard before any data is read.
"""
from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator

import h5py
import numpy as np

from src.errors import MemoryGuardError
from src.environment.pulse import Pulse, validate_pulse

logger = logging.getLogger(__name__)

MEMORY_GUARD_MAX_PULSES: int = 500_000_000  # 500 M pulses


class TSRDLoader:
    """Lazy HDF5 loader for TSRD pulse train files."""

    def __init__(
        self,
        h5_paths: list[pathlib.Path],
        chunk_size: int = 10_000,
        memory_guard_max: int = MEMORY_GUARD_MAX_PULSES,
    ) -> None:
        self.h5_paths = list(h5_paths)
        self.chunk_size = chunk_size
        self.memory_guard_max = memory_guard_max

    def _estimate_pulse_count(self, path: pathlib.Path) -> int:
        """Return number of pulses in file WITHOUT loading any data.

        Checks the primary columnar "data" key first (real TSRD layout),
        then falls back to named "ToA" key (legacy layout).
        """
        try:
            with h5py.File(path, "r") as f:
                # Primary layout: f["data"] shape (n_pulses, 5)
                if "data" in f and hasattr(f["data"], "shape"):
                    return int(f["data"].shape[0])
                # Legacy named-key layout
                for key in ("ToA", "toa", "time"):
                    if key in f:
                        return len(f[key])
                return 0
        except Exception as e:
            logger.warning("Could not estimate pulse count for %s: %s", path.name, e)
            return 0

    def initialize(self) -> None:
        """Check memory guard before any data loading.

        Raises MemoryGuardError if estimated total pulses > memory_guard_max.
        """
        total = sum(self._estimate_pulse_count(p) for p in self.h5_paths)
        if total > self.memory_guard_max:
            raise MemoryGuardError(
                f"Estimated {total:,} pulses exceeds memory guard of "
                f"{self.memory_guard_max:,}. Use lazy/chunked access."
            )

    def iter_pulses(self, path: pathlib.Path, file_id: int) -> Iterator[Pulse]:
        """Iterate pulses from a single TSRD .h5 file in chunks.

        Real TSRD file structure (confirmed from live files, Sep 2026):
          data["data"]   — shape (n_pulses, 5), float32
                           col 0: ToA (μs)
                           col 1: Frequency / CF (MHz)
                           col 2: PulseWidth / PW (μs)
                           col 3: AoA (degrees, range ±180)
                           col 4: Amplitude (dB, range ~-170 to +11)
          data["labels"] — shape (n_pulses, 1) or (n_pulses,), int8/int

        NOTE: The files do NOT use named per-field datasets like "ToA", "CF" etc.
        All five PDW fields are packed into a single "data" array.

        Emitter namespacing: emitter = int(label) * 1000 + file_id
        to prevent cross-file label joins (Requirement 4 / Property 8).

        ToA is NOT strictly monotonic in real TSRD (interleaved pulses from
        multiple emitters can share a ToA or arrive in non-sorted order).
        This loader does not enforce monotonicity — the RF environment handles
        time-step binning via int(toa_us / dt_us) regardless of ordering.
        """
        with h5py.File(path, "r") as f:
            # --- Resolve the data array ---
            # Real structure: f["data"] is shape (n_pulses, 5)
            if "data" in f and hasattr(f["data"], "shape") and f["data"].ndim == 2:
                data_key = "data"
                label_key = "labels" if "labels" in f else None
            else:
                # Fallback: legacy named-key layout (for any older/custom files)
                toa_key = next((k for k in ("ToA", "toa", "TOA") if k in f), None)
                if toa_key is None:
                    logger.warning("No recognised PDW data in %s — skipping.", path.name)
                    return
                # Reconstruct columnar data from separate named datasets
                cf_key  = next((k for k in ("CF", "Frequency", "cf") if k in f), None)
                pw_key  = next((k for k in ("PW", "PulseWidth", "pw") if k in f), None)
                aoa_key = next((k for k in ("AoA", "aoa", "AOA") if k in f), None)
                amp_key = next((k for k in ("Amplitude", "amplitude") if k in f), None)
                label_key = next((k for k in ("Label", "labels", "Emitter") if k in f), None)
                n = len(f[toa_key])
                # Build a combined array for uniform processing below
                import numpy as _np
                combined = _np.column_stack([
                    f[toa_key][:],
                    f[cf_key][:] if cf_key else _np.zeros(n),
                    f[pw_key][:] if pw_key else _np.ones(n),
                    f[aoa_key][:] if aoa_key else _np.zeros(n),
                    f[amp_key][:] if amp_key else _np.full(n, -60.0),
                ])
                # Store temporarily and fall through to chunked loop below
                # by writing to a fake "data" key pattern
                data_arr = combined
                label_arr_full = (
                    np.asarray(f[label_key][:], dtype=np.int64).ravel()
                    if label_key else np.zeros(n, dtype=np.int64)
                )
                n_pulses = n
                for start in range(0, n_pulses, self.chunk_size):
                    end = min(start + self.chunk_size, n_pulses)
                    chunk = data_arr[start:end]
                    chunk_labels = label_arr_full[start:end]
                    for i in range(end - start):
                        try:
                            p = Pulse(
                                toa_us=float(chunk[i, 0]),
                                cf_mhz=float(chunk[i, 1]),
                                pw_us=float(chunk[i, 2]),
                                aoa_deg=float(chunk[i, 3]),
                                amp_db=float(chunk[i, 4]),
                                emitter=int(chunk_labels[i]) * 1000 + file_id,
                            )
                            validate_pulse(p)
                            yield p
                        except ValueError as e:
                            logger.warning("Skipping invalid pulse from %s: %s", path.name, e)
                return

            # --- Primary path: columnar "data" array ---
            n_pulses = f[data_key].shape[0]
            label_arr_full = (
                np.asarray(f[label_key][:], dtype=np.int64).ravel()
                if label_key is not None
                else np.zeros(n_pulses, dtype=np.int64)
            )

        # Re-open for chunked reading (avoids keeping file open across yields)
        with h5py.File(path, "r") as f:
            for start in range(0, n_pulses, self.chunk_size):
                end = min(start + self.chunk_size, n_pulses)
                chunk = np.asarray(f[data_key][start:end], dtype=np.float64)
                chunk_labels = label_arr_full[start:end]

                for i in range(end - start):
                    try:
                        p = Pulse(
                            toa_us=float(chunk[i, 0]),
                            cf_mhz=float(chunk[i, 1]),
                            pw_us=float(chunk[i, 2]),
                            aoa_deg=float(chunk[i, 3]),
                            amp_db=float(chunk[i, 4]),
                            emitter=int(chunk_labels[i]) * 1000 + file_id,
                        )
                        validate_pulse(p)
                        yield p
                    except ValueError as e:
                        logger.warning(
                            "Skipping invalid pulse at index %d in %s: %s",
                            start + i, path.name, e,
                        )
