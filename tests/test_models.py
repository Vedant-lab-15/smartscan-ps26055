"""Tests for Pulse data model, PDWGenerator, and TSRDLoader."""
import math
import pathlib
import warnings
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ew_smart_scan.models.pulse import Pulse, pulse_from_str, pulse_to_str, validate_pulse


# ---------------------------------------------------------------------------
# Pulse strategy — produces valid Pulses within spec bounds
# ---------------------------------------------------------------------------
pulse_strategy = st.builds(
    Pulse,
    toa_us=st.floats(min_value=1e-3, max_value=1e9, allow_nan=False, allow_infinity=False),
    cf_mhz=st.floats(min_value=0.0, max_value=18000.0, allow_nan=False, allow_infinity=False),
    pw_us=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    aoa_deg=st.floats(min_value=-180.0, max_value=360.0, allow_nan=False, allow_infinity=False),
    amp_db=st.floats(min_value=-200.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    emitter=st.integers(min_value=0, max_value=10000),
)


# ---------------------------------------------------------------------------
# Property 9: Pulse Field Validity
# **Validates: Requirements 3.4, 3.5, 3.6, 3.7**
# ---------------------------------------------------------------------------
@given(pulse_strategy)
def test_property_pulse_field_validity(p: Pulse) -> None:
    """Property 9: Any Pulse built within valid ranges passes validate_pulse."""
    validate_pulse(p)  # must not raise


# ---------------------------------------------------------------------------
# Property 11: PDW Round-Trip Fidelity
# **Validates: Requirements 13.2, 13.3**
# ---------------------------------------------------------------------------
@given(pulse_strategy)
def test_property_pdw_roundtrip(p: Pulse) -> None:
    """Property 11: pulse_from_str(pulse_to_str(p)) reconstructs p within float precision."""
    s = pulse_to_str(p)
    p2 = pulse_from_str(s)
    assert math.isclose(p.toa_us, p2.toa_us, rel_tol=1e-9), f"toa_us mismatch: {p.toa_us} vs {p2.toa_us}"
    assert math.isclose(p.cf_mhz, p2.cf_mhz, rel_tol=1e-9)
    assert math.isclose(p.pw_us, p2.pw_us, rel_tol=1e-9)
    assert math.isclose(p.aoa_deg, p2.aoa_deg, rel_tol=1e-9)
    assert math.isclose(p.amp_db, p2.amp_db, rel_tol=1e-9)
    assert p.emitter == p2.emitter


# ---------------------------------------------------------------------------
# validate_pulse — unit tests
# ---------------------------------------------------------------------------
def test_validate_pulse_invalid_toa():
    with pytest.raises(ValueError, match="toa_us"):
        validate_pulse(Pulse(toa_us=-1.0, cf_mhz=900.0, pw_us=5.0, aoa_deg=0.0, amp_db=-60.0, emitter=0))

def test_validate_pulse_invalid_cf():
    with pytest.raises(ValueError, match="cf_mhz"):
        validate_pulse(Pulse(toa_us=100.0, cf_mhz=20000.0, pw_us=5.0, aoa_deg=0.0, amp_db=-60.0, emitter=0))

def test_validate_pulse_invalid_pw():
    with pytest.raises(ValueError, match="pw_us"):
        validate_pulse(Pulse(toa_us=100.0, cf_mhz=900.0, pw_us=0.0, aoa_deg=0.0, amp_db=-60.0, emitter=0))

def test_validate_pulse_invalid_emitter():
    with pytest.raises(ValueError, match="emitter"):
        validate_pulse(Pulse(toa_us=100.0, cf_mhz=900.0, pw_us=5.0, aoa_deg=0.0, amp_db=-60.0, emitter=-1))


# ---------------------------------------------------------------------------
# PDWGenerator unit tests (task 2.6)
# ---------------------------------------------------------------------------
from ew_smart_scan.models.pdw_generator import PDWGenerator


def test_pdw_generator_pulses_valid():
    """All pulses from PDWGenerator with a fixed seed pass validate_pulse."""
    gen = PDWGenerator(
        n_bands=4,
        band_cf_mhz=[900.0, 1000.0, 1100.0, 1200.0],
        dt_us=10.0,
        seed=42,
    )
    for t in range(50):
        for p in gen.generate(t_step=t, file_id=0):
            validate_pulse(p)  # must not raise


def test_pdw_generator_no_cross_file_label_overlap():
    """Emitter labels from file_id=0 and file_id=1 must never overlap."""
    gen = PDWGenerator(
        n_bands=4,
        band_cf_mhz=[900.0, 1000.0, 1100.0, 1200.0],
        dt_us=10.0,
        p_emit=1.0,  # always emit to guarantee pulses
        seed=0,
    )
    labels_0 = {p.emitter for t in range(10) for p in gen.generate(t_step=t, file_id=0)}
    gen.reset(seed=0)
    labels_1 = {p.emitter for t in range(10) for p in gen.generate(t_step=t, file_id=1)}
    assert labels_0.isdisjoint(labels_1), "Emitter labels overlap across file IDs!"


def test_pdw_generator_toa_monotonic():
    """toa_us values must be monotonically increasing across sequential t_step calls."""
    gen = PDWGenerator(
        n_bands=2,
        band_cf_mhz=[900.0, 1000.0],
        dt_us=100.0,
        p_emit=1.0,
        seed=7,
    )
    prev_toa = -1.0
    for t in range(20):
        pulses = gen.generate(t_step=t, file_id=0)
        for p in pulses:
            assert p.toa_us > prev_toa, f"toa_us not monotonic at t={t}: {p.toa_us} <= {prev_toa}"
            prev_toa = p.toa_us


# ---------------------------------------------------------------------------
# TSRDLoader unit tests (task 2.6)
# ---------------------------------------------------------------------------
from ew_smart_scan.models.tsrd_loader import TSRDLoader
from ew_smart_scan.env.errors import MemoryGuardError


def test_tsrd_loader_memory_guard_raises():
    """TSRDLoader.initialize() raises MemoryGuardError when estimated pulses > guard."""
    loader = TSRDLoader(
        h5_paths=[pathlib.Path("fake_file.h5")],
        memory_guard_max=100,
    )
    # Patch _estimate_pulse_count to return a value above the guard
    loader._estimate_pulse_count = lambda path: 200
    with pytest.raises(MemoryGuardError, match="exceeds memory guard"):
        loader.initialize()


def test_tsrd_loader_memory_guard_passes():
    """TSRDLoader.initialize() does not raise when total is within guard."""
    loader = TSRDLoader(
        h5_paths=[pathlib.Path("fake_file.h5")],
        memory_guard_max=1000,
    )
    loader._estimate_pulse_count = lambda path: 500
    loader.initialize()  # should not raise


# ---------------------------------------------------------------------------
# PulseTrainLoader tests (task 16 / TSRD integration)
# ---------------------------------------------------------------------------

class _MockPulseTrain:
    """Minimal mock of turing_deinterleaving_challenge.PulseTrain."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.metadata = {}

    @classmethod
    def load(cls, path):
        # Build a small synthetic pulse train: 25 pulses, 5 PDW columns
        rng = pathlib.Path(path).stem  # use filename as seed proxy
        n = 25
        toa   = (1.0 + pathlib.Path(path).stat().st_size % 100) * 10.0  # deterministic
        toas  = [toa + i * 50.0 for i in range(n)]
        data  = pathlib.Path  # will be replaced below
        import numpy as np
        data = np.column_stack([
            np.array(toas),                              # ToA (μs) — increasing
            np.full(n, 950.0),                           # CF (MHz)
            np.full(n, 5.0),                             # PW (μs)
            np.full(n, 45.0),                            # AoA (deg)
            np.full(n, -60.0),                           # Amplitude (dB)
        ])
        labels = np.zeros(n, dtype=np.int64)
        return cls(data, labels)


def _make_mock_h5(tmp_path, filename="test.h5"):
    """Create a minimal real .h5 file for integration testing."""
    import numpy as np
    import h5py
    path = tmp_path / filename
    n = 30
    with h5py.File(path, "w") as f:
        # TSRD uses a single dataset named "pdws" with shape (n, 5)
        # but PulseTrain.load() abstracts this — we test the loader with mocks instead
        # For direct h5py testing, write named columns as TSRDLoader expects
        f.create_dataset("ToA", data=np.linspace(100.0, 1600.0, n))
        f.create_dataset("CF",  data=np.full(n, 950.0))
        f.create_dataset("PW",  data=np.full(n, 5.0))
        f.create_dataset("AoA", data=np.full(n, 30.0))
        f.create_dataset("Amplitude", data=np.full(n, -55.0))
        f.create_dataset("Label", data=np.zeros(n, dtype=np.int64))
    return path


def test_pulse_train_loader_iter_pulses_with_mock(tmp_path):
    """PulseTrainLoader.iter_pulses yields valid Pulses from a mocked PulseTrain."""
    import sys, types, numpy as np
    from ew_smart_scan.models.pulse import validate_pulse

    # Build a minimal fake turing_deinterleaving_challenge module
    fake_module = types.ModuleType("turing_deinterleaving_challenge")

    n = 20
    mock_data = np.column_stack([
        np.linspace(10.0, 1000.0, n),   # ToA
        np.full(n, 900.0),              # CF
        np.full(n, 3.0),               # PW
        np.full(n, 0.0),               # AoA
        np.full(n, -50.0),             # Amplitude
    ])
    mock_labels = np.arange(n, dtype=np.int64) % 3

    class FakePulseTrain:
        @staticmethod
        def load(path):
            class PT:
                data = mock_data
                labels = mock_labels
                metadata = {}
            return PT()

    fake_module.PulseTrain = FakePulseTrain

    # Inject mock module
    sys.modules["turing_deinterleaving_challenge"] = fake_module

    try:
        from importlib import reload
        import ew_smart_scan.models.pulse_train_loader as ptl
        reload(ptl)
        from ew_smart_scan.models.pulse_train_loader import PulseTrainLoader

        loader = PulseTrainLoader(
            h5_paths=[tmp_path / "fake.h5"],
            memory_guard_max=10_000_000,
        )
        # Patch _estimate_pulse_count to avoid real file I/O
        loader._estimate_pulse_count = lambda path: n

        pulses = list(loader.iter_pulses(tmp_path / "fake.h5", file_id=0))
        assert len(pulses) == n, f"Expected {n} pulses, got {len(pulses)}"
        for p in pulses:
            validate_pulse(p)   # must not raise
    finally:
        del sys.modules["turing_deinterleaving_challenge"]


def test_pulse_train_loader_emitter_namespacing(tmp_path):
    """Emitter labels from file_id=0 and file_id=1 never overlap."""
    import sys, types, numpy as np

    n = 5
    fake_module = types.ModuleType("turing_deinterleaving_challenge")

    class FakePulseTrain:
        @staticmethod
        def load(path):
            class PT:
                data = np.column_stack([
                    np.linspace(10.0, 500.0, n),
                    np.full(n, 900.0), np.full(n, 3.0),
                    np.full(n, 0.0),   np.full(n, -50.0),
                ])
                labels = np.zeros(n, dtype=np.int64)  # all label=0
                metadata = {}
            return PT()

    fake_module.PulseTrain = FakePulseTrain
    sys.modules["turing_deinterleaving_challenge"] = fake_module

    try:
        from importlib import reload
        import ew_smart_scan.models.pulse_train_loader as ptl
        reload(ptl)
        from ew_smart_scan.models.pulse_train_loader import PulseTrainLoader

        loader = PulseTrainLoader([tmp_path / "a.h5", tmp_path / "b.h5"])
        loader._estimate_pulse_count = lambda path: n

        pulses_0 = list(loader.iter_pulses(tmp_path / "a.h5", file_id=0))
        pulses_1 = list(loader.iter_pulses(tmp_path / "b.h5", file_id=1))

        labels_0 = {p.emitter for p in pulses_0}
        labels_1 = {p.emitter for p in pulses_1}
        assert labels_0.isdisjoint(labels_1), (
            f"Emitter labels overlap across file IDs: {labels_0 & labels_1}"
        )
    finally:
        del sys.modules["turing_deinterleaving_challenge"]


def test_pulse_train_loader_memory_guard(tmp_path):
    """PulseTrainLoader.initialize() raises MemoryGuardError when over limit."""
    import sys, types
    from ew_smart_scan.env.errors import MemoryGuardError

    fake_module = types.ModuleType("turing_deinterleaving_challenge")

    class FakePulseTrain:
        @staticmethod
        def load(path):
            import numpy as np
            class PT:
                data = np.zeros((600_000_000, 5))  # 600M pulses — over the guard
                labels = np.zeros(600_000_000, dtype=np.int64)
                metadata = {}
            return PT()

    fake_module.PulseTrain = FakePulseTrain
    sys.modules["turing_deinterleaving_challenge"] = fake_module

    try:
        from importlib import reload
        import ew_smart_scan.models.pulse_train_loader as ptl
        reload(ptl)
        from ew_smart_scan.models.pulse_train_loader import PulseTrainLoader

        loader = PulseTrainLoader(
            [tmp_path / "big.h5"],
            memory_guard_max=500_000_000,
        )
        with pytest.raises(MemoryGuardError, match="exceeds memory guard"):
            loader.initialize()
    finally:
        del sys.modules["turing_deinterleaving_challenge"]


def test_credential_error_no_hf_token(monkeypatch):
    """download_tsrd raises CredentialError when HF_TOKEN is not set."""
    from ew_smart_scan.env.errors import CredentialError
    from ew_smart_scan.data_download import download_tsrd

    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(CredentialError, match="HF_TOKEN"):
        download_tsrd(target_dir=pathlib.Path("/tmp/tsrd_test"))


def test_pulse_train_loader_plugs_into_rf_env(tmp_path):
    """PulseTrainLoader plugs into RFEnvironment as a drop-in for TSRDLoader."""
    import sys, types, numpy as np
    from ew_smart_scan.env.rf_environment import RFEnvironment
    from ew_smart_scan.env.belief_tracker import BeliefTracker
    from ew_smart_scan.env.receiver_model import ReceiverModel

    n_pulses = 30
    n_bands = 4
    dt_us = 10.0

    # Synthetic pulses spread across 4 bands, monotonically increasing ToA
    band_cfs = [950.0, 1050.0, 1150.0, 1250.0]
    # 30 pulses cycling through 4 bands
    toas   = [float((i + 1) * dt_us * 0.5) for i in range(n_pulses)]
    cfs    = [band_cfs[i % n_bands] for i in range(n_pulses)]
    pws    = [5.0] * n_pulses
    aoas   = [30.0] * n_pulses
    amps   = [-50.0] * n_pulses
    labels = [i % 4 for i in range(n_pulses)]

    mock_data = np.column_stack([toas, cfs, pws, aoas, amps])
    mock_labels = np.array(labels, dtype=np.int64)

    fake_module = types.ModuleType("turing_deinterleaving_challenge")

    class FakePulseTrain:
        @staticmethod
        def load(path):
            class PT:
                data = mock_data
                labels_ = mock_labels
                labels = mock_labels
                metadata = {}
            return PT()

    fake_module.PulseTrain = FakePulseTrain
    sys.modules["turing_deinterleaving_challenge"] = fake_module

    try:
        from importlib import reload
        import ew_smart_scan.models.pulse_train_loader as ptl
        reload(ptl)
        from ew_smart_scan.models.pulse_train_loader import PulseTrainLoader

        loader = PulseTrainLoader(
            [tmp_path / "scan_000.h5"],
            memory_guard_max=10_000_000,
        )
        loader._estimate_pulse_count = lambda path: n_pulses

        band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
        env = RFEnvironment(
            source=loader,
            n_bands=n_bands,
            band_edges_mhz=band_edges,
            dt_us=dt_us,
        )
        env.initialize()

        # Step a few time steps and verify get_true_state returns n_bands keys
        for t in range(5):
            state = env.get_true_state(t)
            assert len(state) == n_bands, f"t={t}: expected {n_bands} keys, got {len(state)}"
            assert all(isinstance(v, bool) for v in state.values())

        # Wire up receiver and run a few steps — no crashes
        bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
        receiver = ReceiverModel(rf_env=env, belief_tracker=bt,
                                  n_bands=n_bands, k_scan=2)
        receiver.reset()
        for _ in range(5):
            obs_dict, reward, _ = receiver.step({0, 1})
            beliefs = receiver.get_belief()
            assert np.all(beliefs >= 0.0) and np.all(beliefs <= 1.0)

    finally:
        del sys.modules["turing_deinterleaving_challenge"]


# ---------------------------------------------------------------------------
# Real-data smoke test — skipped automatically if data/ is not populated
# ---------------------------------------------------------------------------

def test_real_tsrd_scan_file_if_available():
    """If a real scan-mode .h5 file exists in data/tsrd_sample/, load it and verify schema.

    This test is skipped automatically when no real data is present.
    To run: python scripts/fetch_tsrd_sample.py  then re-run pytest.

    NOTE: Real TSRD data has interleaved pulses — ToA is non-decreasing but
    NOT strictly increasing (multiple emitters can share the same ToA value).
    The loader handles this correctly; we do not assert strict monotonicity here.
    """
    from ew_smart_scan.models.tsrd_loader import TSRDLoader
    from ew_smart_scan.models.pulse import validate_pulse

    data_dir = pathlib.Path("data/tsrd_sample")
    scan_files = sorted(data_dir.rglob("*.h5"))
    if not scan_files:
        pytest.skip("No real TSRD .h5 files found in data/tsrd_sample/ — skipping.")

    path = scan_files[0]
    loader = TSRDLoader([path], memory_guard_max=10_000_000)
    loader.initialize()

    pulses = []
    for i, p in enumerate(loader.iter_pulses(path, file_id=0)):
        validate_pulse(p)
        pulses.append(p)
        if i >= 999:
            break

    assert len(pulses) > 0, "No pulses loaded from real file"
    toas = [p.toa_us for p in pulses]
    # Non-decreasing (not strictly monotonic) — TSRD has ties from interleaving
    assert all(toas[i] <= toas[i+1] for i in range(len(toas)-1)), \
        "ToA values are not even non-decreasing — something is wrong with ordering"
    print(f"\nReal TSRD file: {path.name}, loaded {len(pulses)} pulses, "
          f"ToA range [{toas[0]:.1f}, {toas[-1]:.1f}] μs")
