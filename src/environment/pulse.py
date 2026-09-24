"""Pulse Descriptor Word (PDW) data model.

A Pulse represents a single radar pulse record with 5 measured fields plus
a per-file emitter label. Labels are meaningful ONLY within a single .h5 file
and must never be joined across files.
"""
import dataclasses
import math
import re


@dataclasses.dataclass(frozen=True)
class Pulse:
    toa_us: float    # Time of Arrival (microseconds), must be > 0
    cf_mhz: float    # Centre Frequency (MHz), must be in [0, 18000]
    pw_us: float     # Pulse Width (microseconds), must be > 0
    aoa_deg: float   # Angle of Arrival (degrees), must be in [-180, 360]
    amp_db: float    # Amplitude (dBm), must be finite
    emitter: int     # Emitter label — arbitrary integer, consistent within one file only


def validate_pulse(p: Pulse) -> None:
    """Raise ValueError with a descriptive message if any field is out of range.

    Validation ranges are based on real TSRD data (confirmed Sep 2026):
      - toa_us:  must be finite and > 0 (real range: ~180K–29M μs)
      - cf_mhz:  [0, 18000] MHz covering 0–18 GHz receiver range
      - pw_us:   > 0 (real range: 0.007–368 μs)
      - aoa_deg: [-180, 360] — real TSRD uses full ±180° range
      - amp_db:  finite, no hard clamp (real range: -171 to +11 dB)
      - emitter: >= 0 (arbitrary per-file integer)
    """
    if not math.isfinite(p.toa_us) or p.toa_us <= 0:
        raise ValueError(f"toa_us must be finite and > 0, got {p.toa_us}")
    if not (0.0 <= p.cf_mhz <= 18000.0):
        raise ValueError(f"cf_mhz must be in [0, 18000], got {p.cf_mhz}")
    if p.pw_us <= 0:
        raise ValueError(f"pw_us must be > 0, got {p.pw_us}")
    if not (-180.0 <= p.aoa_deg <= 360.0):
        raise ValueError(f"aoa_deg must be in [-180, 360], got {p.aoa_deg}")
    if not math.isfinite(p.amp_db):
        raise ValueError(f"amp_db must be finite, got {p.amp_db}")
    if p.emitter < 0:
        raise ValueError(f"emitter must be >= 0, got {p.emitter}")


def pulse_to_str(p: Pulse) -> str:
    """Serialize a Pulse to a deterministic human-readable string.

    Uses 17-significant-digit scientific notation for all float fields so
    the string is parseable back to float without precision loss (IEEE 754
    double requires at most 17 significant digits for exact round-trip).
    """
    return (
        f"Pulse("
        f"toa_us={p.toa_us:.17e}, "
        f"cf_mhz={p.cf_mhz:.17e}, "
        f"pw_us={p.pw_us:.17e}, "
        f"aoa_deg={p.aoa_deg:.17e}, "
        f"amp_db={p.amp_db:.17e}, "
        f"emitter={p.emitter})"
    )


def pulse_from_str(s: str) -> Pulse:
    """Parse a Pulse from the string produced by pulse_to_str."""
    pattern = (
        r"Pulse\("
        r"toa_us=([^,]+), "
        r"cf_mhz=([^,]+), "
        r"pw_us=([^,]+), "
        r"aoa_deg=([^,]+), "
        r"amp_db=([^,]+), "
        r"emitter=(\d+)\)"
    )
    m = re.fullmatch(pattern, s.strip())
    if m is None:
        raise ValueError(f"Cannot parse Pulse from string: {s!r}")
    return Pulse(
        toa_us=float(m.group(1)),
        cf_mhz=float(m.group(2)),
        pw_us=float(m.group(3)),
        aoa_deg=float(m.group(4)),
        amp_db=float(m.group(5)),
        emitter=int(m.group(6)),
    )
