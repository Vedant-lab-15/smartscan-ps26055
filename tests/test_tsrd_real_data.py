"""TSRD real-data loader validation tests.

These tests are SKIPPED automatically when data/tsrd_sample/ is absent or
empty. They run when real TSRD scan-mode files have been downloaded with:

    python scripts/fetch_tsrd_sample.py

Tests validate:
  1. TSRDLoader (h5py direct) — or PulseTrainLoader if TSRDLoader can't read
     the real structure — yields valid Pulse objects.
  2. PDW field schema matches PDWGenerator output (same Pulse dataclass).
  3. ToA is monotonically increasing per file.
  4. Emitter labels reset per file (no cross-file collisions via namespacing).
  5. Explicit report if TSRDLoader needed changes vs the original assumption.

Run with:
    python3 -m pytest tests/test_tsrd_real_data.py -v -s
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

SAMPLE_DIR = pathlib.Path("data/tsrd_sample")


def _find_h5_files() -> list[pathlib.Path]:
    """Return sorted list of .h5 files in SAMPLE_DIR, recursively."""
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.rglob("*.h5"))


def _skip_if_no_data():
    files = _find_h5_files()
    if not files:
        pytest.skip(
            f"No .h5 files found in {SAMPLE_DIR}. "
            "Download sample data first:\n"
            "  python scripts/fetch_tsrd_sample.py\n"
            "then re-run this test."
        )
    return files


# ---------------------------------------------------------------------------
# Fixture: load files once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_h5_files():
    return _skip_if_no_data()


# ---------------------------------------------------------------------------
# Test 1: TSRDLoader can read real files (direct h5py path)
# ---------------------------------------------------------------------------

def test_tsrd_loader_reads_real_files(sample_h5_files, tmp_path):
    """TSRDLoader yields Pulse objects from real TSRD .h5 files.

    This test reveals whether the original key-name assumptions in TSRDLoader
    match the actual HDF5 structure. Pass/fail + warnings are reported clearly.
    """
    from ew_smart_scan.models.tsrd_loader import TSRDLoader
    from ew_smart_scan.models.pulse import validate_pulse
    import h5py

    # First: inspect the raw HDF5 structure of one file so we can report it
    path = sample_h5_files[0]
    print(f"\nInspecting HDF5 structure of: {path.name}")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        print(f"  Top-level keys: {keys}")
        for k in keys:
            item = f[k]
            if hasattr(item, "shape"):
                print(f"    {k}: shape={item.shape}, dtype={item.dtype}")
            else:
                print(f"    {k}: group — sub-keys: {list(item.keys())[:10]}")

    # Now try loading via TSRDLoader
    loader = TSRDLoader(
        h5_paths=sample_h5_files,
        memory_guard_max=10_000_000,  # 10M pulses max for sample
    )
    loader.initialize()

    all_results: dict[str, dict] = {}
    tsrd_loader_worked = True
    tsrd_loader_issue = None

    for file_id, path in enumerate(sample_h5_files):
        pulses = []
        try:
            for p in loader.iter_pulses(path, file_id=file_id):
                validate_pulse(p)
                pulses.append(p)
        except Exception as exc:
            tsrd_loader_worked = False
            tsrd_loader_issue = str(exc)
            print(f"\n  TSRDLoader failed on {path.name}: {exc}")
            break

        all_results[path.name] = {
            "n_pulses": len(pulses),
            "pulses": pulses,
        }
        print(f"  {path.name}: {len(pulses)} pulses loaded via TSRDLoader")

    if tsrd_loader_worked and all(r["n_pulses"] > 0 for r in all_results.values()):
        print("\n  RESULT: TSRDLoader read real files WITHOUT changes needed.")
    else:
        print(
            f"\n  RESULT: TSRDLoader could not read real files.\n"
            f"  Issue: {tsrd_loader_issue}\n"
            f"  FIX NEEDED: The real .h5 files likely use a different internal\n"
            f"  structure than the named-key assumption (ToA/CF/PW/AoA/Amplitude).\n"
            f"  Use PulseTrainLoader (ew_smart_scan/models/pulse_train_loader.py)\n"
            f"  which uses the official PulseTrain.load() API instead."
        )
        pytest.fail(
            f"TSRDLoader could not read real files: {tsrd_loader_issue}\n"
            "See test output above for the actual HDF5 structure and fix guidance."
        )


# ---------------------------------------------------------------------------
# Test 2: PDW schema matches PDWGenerator output
# ---------------------------------------------------------------------------

def test_pdw_schema_matches_pdwgenerator(sample_h5_files):
    """Pulses from real files use the same Pulse dataclass as PDWGenerator."""
    from ew_smart_scan.models.tsrd_loader import TSRDLoader
    from ew_smart_scan.models.pdw_generator import PDWGenerator
    from ew_smart_scan.models.pulse import Pulse

    # Load a few pulses from a real file
    path = sample_h5_files[0]
    loader = TSRDLoader([path], memory_guard_max=10_000_000)
    loader.initialize()

    real_pulses = []
    for i, p in enumerate(loader.iter_pulses(path, file_id=0)):
        real_pulses.append(p)
        if i >= 99:
            break

    if not real_pulses:
        pytest.skip("TSRDLoader returned no pulses — schema comparison not possible.")

    # Get a few pulses from PDWGenerator
    gen = PDWGenerator(
        n_bands=4,
        band_cf_mhz=[900.0, 1000.0, 1100.0, 1200.0],
        dt_us=10.0,
        p_emit=1.0,
        seed=42,
    )
    gen_pulses = []
    for t in range(10):
        gen_pulses.extend(gen.generate(t_step=t, file_id=0))

    assert gen_pulses, "PDWGenerator returned no pulses"

    # Both should be instances of the same Pulse class
    assert isinstance(real_pulses[0], Pulse), "Real pulse is not a Pulse instance"
    assert isinstance(gen_pulses[0], Pulse), "Generated pulse is not a Pulse instance"

    # Both should have the same fields
    real_fields = set(real_pulses[0].__dataclass_fields__.keys())
    gen_fields  = set(gen_pulses[0].__dataclass_fields__.keys())
    assert real_fields == gen_fields, (
        f"Field mismatch: real={real_fields}, generated={gen_fields}"
    )

    print(f"\n  Real pulse sample:      {real_pulses[0]}")
    print(f"  Generated pulse sample: {gen_pulses[0]}")
    print(f"  Schema match: ✓  Fields: {sorted(real_fields)}")


# ---------------------------------------------------------------------------
# Test 3: ToA is monotonically increasing per file
# ---------------------------------------------------------------------------

def test_toa_monotonic_per_file(sample_h5_files):
    """Report ToA ordering behaviour per file.

    Real TSRD has interleaved pulses from multiple emitters. ToA is
    NON-DECREASING (sorted ascending) but NOT strictly increasing — ties
    occur when multiple emitters transmit at the same microsecond. This is
    by design in the dataset and is not a loader bug.

    We assert non-decreasing (no backward jumps), not strictly increasing.
    Tie counts are printed for information.
    """
    from ew_smart_scan.models.tsrd_loader import TSRDLoader

    loader = TSRDLoader(sample_h5_files, memory_guard_max=50_000_000)
    loader.initialize()

    print()
    for file_id, path in enumerate(sample_h5_files):
        pulses = list(loader.iter_pulses(path, file_id=file_id))
        if not pulses:
            print(f"  {path.name}: 0 pulses — skipping")
            continue

        toas = [p.toa_us for p in pulses]

        # Count ties (equal consecutive ToA) vs backward jumps
        ties      = sum(1 for i in range(len(toas)-1) if toas[i] == toas[i+1])
        backwards = [(i, toas[i], toas[i+1]) for i in range(len(toas)-1)
                     if toas[i] > toas[i+1]]

        status = "✓ non-decreasing" if not backwards else f"✗ {len(backwards)} backward jumps"
        print(f"  {path.name}: {len(pulses)} pulses, "
              f"ties={ties}, {status}, "
              f"range=[{toas[0]:.0f}, {toas[-1]:.0f}] μs")

        if backwards:
            print(f"    First backward jump at idx {backwards[0][0]}: "
                  f"{backwards[0][1]:.2f} → {backwards[0][2]:.2f}")
            pytest.fail(
                f"{path.name}: ToA has {len(backwards)} backward jumps "
                f"(not just ties). This is unexpected — check the loader."
            )


# ---------------------------------------------------------------------------
# Test 4: Emitter label isolation across files
# ---------------------------------------------------------------------------

def test_emitter_labels_isolated_across_files(sample_h5_files):
    """Emitter labels from different files never collide after namespacing.

    Checks both:
    - Raw labels: expected to potentially collide (TDC README: labels are
      arbitrary per file)
    - Namespaced labels (label * 1000 + file_id): must NOT collide
    """
    from ew_smart_scan.models.tsrd_loader import TSRDLoader

    loader = TSRDLoader(sample_h5_files, memory_guard_max=10_000_000)
    loader.initialize()

    raw_labels_per_file: dict[str, set[int]] = {}
    namespaced_labels_per_file: dict[str, set[int]] = {}

    print()
    for file_id, path in enumerate(sample_h5_files):
        pulses = list(loader.iter_pulses(path, file_id=file_id))
        if not pulses:
            continue

        # The namespaced emitter = raw_label * 1000 + file_id
        # To recover raw: raw = (emitter - file_id) // 1000
        namespaced = {p.emitter for p in pulses}
        raw = {(p.emitter - file_id) // 1000 for p in pulses}

        raw_labels_per_file[path.name] = raw
        namespaced_labels_per_file[path.name] = namespaced

        print(
            f"  {path.name} (file_id={file_id}): "
            f"{len(raw)} distinct raw labels, "
            f"{len(namespaced)} distinct namespaced labels"
        )
        print(f"    Raw label sample: {sorted(raw)[:10]}")

    # Check 1: raw labels MAY collide across files (expected per TDC README)
    file_names = list(raw_labels_per_file.keys())
    if len(file_names) >= 2:
        raw_overlap = raw_labels_per_file[file_names[0]] & raw_labels_per_file[file_names[1]]
        print(f"\n  Raw label overlap between file 0 and file 1: {len(raw_overlap)} labels")
        print(f"  (Expected: may be non-zero — TDC README says labels are arbitrary per file)")

    # Check 2: namespaced labels must NOT collide across files
    ns_names = list(namespaced_labels_per_file.keys())
    for i in range(len(ns_names)):
        for j in range(i + 1, len(ns_names)):
            overlap = (
                namespaced_labels_per_file[ns_names[i]]
                & namespaced_labels_per_file[ns_names[j]]
            )
            assert not overlap, (
                f"Namespaced emitter labels collide between {ns_names[i]} and "
                f"{ns_names[j]}: {overlap}. "
                "The label * 1000 + file_id namespacing should prevent this — "
                "check that file_id values are unique."
            )

    print("\n  Namespaced label isolation: ✓  No cross-file collisions")


# ---------------------------------------------------------------------------
# Test 5: Full integration — real files feed into RFEnvironment
# ---------------------------------------------------------------------------

def test_real_files_feed_into_rf_environment(sample_h5_files):
    """Real TSRD files load through TSRDLoader → RFEnvironment without error.

    Uses 18 bands of 1 GHz each (covering 0–18 GHz) matching real TSRD CF range.
    Real TSRD CF range confirmed: ~5–12000 MHz across the 3 sample files.
    """
    from ew_smart_scan.models.tsrd_loader import TSRDLoader
    from ew_smart_scan.env.rf_environment import RFEnvironment
    from ew_smart_scan.env.belief_tracker import BeliefTracker
    from ew_smart_scan.env.receiver_model import ReceiverModel

    # Use only 1 file for speed
    path = sample_h5_files[0]
    loader = TSRDLoader([path], memory_guard_max=50_000_000)

    # Real TSRD covers 0–18 GHz. 18 bands × 1000 MHz each.
    n_bands = 18
    band_edges = [(float(i * 1000), float((i + 1) * 1000)) for i in range(n_bands)]
    # dt_us chosen so file spans ~100 steps (file ToA range ~29M μs / 300K = ~100 steps)
    dt_us = 300_000.0

    env = RFEnvironment(
        source=loader,
        n_bands=n_bands,
        band_edges_mhz=band_edges,
        dt_us=dt_us,
    )
    env.initialize()

    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt,
                              n_bands=n_bands, k_scan=3)
    receiver.reset()

    occupied_steps = 0
    for _ in range(20):
        obs_dict, reward, _ = receiver.step({0, 5, 10})
        beliefs = receiver.get_belief()
        assert np.all(beliefs >= 0.0) and np.all(beliefs <= 1.0)
        if any(len(v) > 0 for v in obs_dict.values()):
            occupied_steps += 1

    print(f"\n  Real data integration: ✓  file={path.name}, "
          f"dt_us={dt_us:.0f}, 20 steps, "
          f"steps_with_pulses={occupied_steps}/20")
    print(f"  Final beliefs (bands 0-5): {np.round(beliefs[:6], 3)}")
