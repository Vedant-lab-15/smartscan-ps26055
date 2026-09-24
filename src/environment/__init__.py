"""Environment sub-package — RF simulator, POMDP receiver, TSRD stats.

Public API:
    RFEnvironment   — simulated N-band RF environment (ground-truth source)
    ReceiverModel   — POMDP gating wrapper (only scanned bands return observations)
    PDWGenerator    — synthetic PDW emitter (Bernoulli or periodic renewal)
    TSRDLoader      — lazy h5py reader for TSRD .h5 files
    Pulse           — PDW data record (toa_us, cf_mhz, pw_us, aoa_deg, amp_db)
"""
from src.environment.simulator import RFEnvironment
from src.environment.receiver import ReceiverModel
from src.environment.pdw_generator import PDWGenerator, PeriodicBandConfig
from src.environment.tsrd_loader import TSRDLoader
from src.environment.pulse import Pulse, validate_pulse

__all__ = [
    "RFEnvironment",
    "ReceiverModel",
    "PDWGenerator",
    "PeriodicBandConfig",
    "TSRDLoader",
    "Pulse",
    "validate_pulse",
]
