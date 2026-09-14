"""Ablation: Module C real-data failure mode analysis.

Three conditions per emitter, controlled single-variable comparison:
  A) Baseline: standard K=3-of-18 competitive scan access
  B) Guaranteed dedicated: periodic band always scanned (removes coverage-ratio bottleneck)
  C) Active-window only (emitter 31 only): restrict evaluation to the emitter's
     actual active window under guaranteed access (removes active-window dilution)

Warmup is emitter-aware: mature phase = second half of the ACTIVE WINDOW,
not the second half of the full file. The previous evaluation used second half
of the full file, which placed the entire active window inside the warmup for
both emitters.

Usage: python3 -m scripts.ablation_module_c
"""
from __future__ import annotations

import pathlib
import collections
import math

import h5py
import numpy as np

from ew_smart_scan.models.tsrd_loader import TSRDLoader
from ew_smart_scan.env.rf_environment import RFEnvironment
from ew_smart_scan.env.belief_tracker import BeliefTracker
from ew_smart_scan.env.receiver_model import ReceiverModel
from ew_smart_scan.env.periodic_intercept import PeriodicInterceptModule

SAMPLE_DIR = pathlib.Path("data/tsrd_sample/scan/train_scan")
N_BANDS    = 18
BAND_EDGES = [(float(i * 1000), float((i + 1) * 1000)) for i in range(N_BANDS)]
K_SCAN     = 3

# Synthetic reference (from validated synthetic tests)
SYNTH_C_RATE    = 0.9838
SYNTH_C_ERR_US  = 16.6


# ─────────────────────────────────────────────────────────────────────────────
# Core runner — single condition
# ─────────────────────────────────────────────────────────────────────────────

def run_condition(
    h5_path: pathlib.Path,
    emitter_label: int,
    T_us: float,
    dt_us: float,
    guaranteed_access: bool,        # Condition B/C: always scan the periodic band
    t_start_us_override: float | None = None,  # Condition C: restrict start of eval window
    t_end_us_override: float | None = None,    # Condition C: restrict end of eval window
    seed: int = 42,
) -> dict:
    """Run one ablation condition. Returns per-condition metrics dict."""

    with h5py.File(h5_path, "r") as f:
        data   = f["data"][:].astype(np.float64)
        labels = f["labels"][:].ravel()

    # Identify band for this emitter
    cf_emitter = float(np.mean([
        data[i, 1] for i in range(len(data)) if int(labels[i]) == emitter_label
    ]))
    emitter_band = next(
        (b for b, (lo, hi) in enumerate(BAND_EDGES) if lo <= cf_emitter < hi), None
    )
    if emitter_band is None:
        return {"error": f"CF {cf_emitter:.0f} MHz outside all {N_BANDS} bands"}

    toa_all = data[:, 0]
    file_toa_min = float(toa_all.min())
    file_toa_max = float(toa_all.max())

    # Determine evaluation window in absolute ToA terms
    eval_start_us = t_start_us_override if t_start_us_override is not None else file_toa_min
    eval_end_us   = t_end_us_override   if t_end_us_override   is not None else file_toa_max

    # The RFEnvironment occupancy dict is keyed by int(toa_us / dt_us)
    # (raw ToA, NOT offset from file start). Step range is therefore:
    step_start = int(eval_start_us / dt_us)
    step_end   = int(eval_end_us   / dt_us)
    n_steps    = max(1, step_end - step_start)

    # Emitter pulses within the eval window
    em_toas_all = np.sort(np.array([
        data[i, 0] for i in range(len(data)) if int(labels[i]) == emitter_label
    ]))
    em_toas = em_toas_all[
        (em_toas_all >= eval_start_us) & (em_toas_all < eval_end_us)
    ]

    if len(em_toas) < 4:
        return {
            "error": f"Only {len(em_toas)} emitter pulses in eval window "
                     f"[{eval_start_us:.0f}, {eval_end_us:.0f}] us"
        }

    # Mature phase = second half of the eval window steps
    warmup_steps = n_steps // 2

    # Build environment over eval window only (offset ToA by eval_start_us)
    # We rebuild a fresh loader each condition for isolation
    loader = TSRDLoader([h5_path], memory_guard_max=50_000_000)
    env = RFEnvironment(
        source=loader, n_bands=N_BANDS,
        band_edges_mhz=BAND_EDGES, dt_us=dt_us,
    )
    env.initialize()

    bt       = BeliefTracker(n_bands=N_BANDS, p_stay_occ=0.9, p_stay_idle=0.85)
    receiver = ReceiverModel(rf_env=env, belief_tracker=bt,
                              n_bands=N_BANDS, k_scan=K_SCAN)
    receiver.reset()
    module   = PeriodicInterceptModule(n_bands=N_BANDS)
    rng      = np.random.default_rng(seed)

    intercept_count   = 0
    actual_emissions  = 0
    cyc_late          = []
    all_cyc           = []

    for t_local, t_abs in enumerate(range(step_start, step_end)):
        t_now_us = t_abs * dt_us   # raw-ToA-based time for Module C

        periodic_idx = module.compute_all_indices(t_now_us)

        if guaranteed_access:
            # Condition B/C: periodic band always gets a scan slot
            action = {emitter_band}
            others = [b for b in range(N_BANDS) if b != emitter_band]
            for j in range(K_SCAN - 1):
                action.add(others[j % len(others)])
        else:
            # Condition A: standard competitive access
            in_dwell = periodic_idx[emitter_band] > 0.4
            if in_dwell:
                action = {emitter_band}
                others = [b for b in range(N_BANDS) if b != emitter_band]
                for j in range(K_SCAN - 1):
                    action.add(others[j % len(others)])
            else:
                c = (t_local * K_SCAN) % N_BANDS
                action = {(c + j) % N_BANDS for j in range(K_SCAN)}

        # Step the environment using the absolute time step
        obs_dict = {}
        true_state = env.get_true_state(t_abs)
        for band_id in range(N_BANDS):
            if band_id in action:
                pulses = env.sample_pulses(band_id, t_abs)
                obs_dict[band_id] = pulses
                # Update belief tracker (scan → Bayesian update)
                obs_binary = 1 if len(pulses) > 0 else 0
                bt.update(band_id, obs_binary)
            else:
                obs_dict[band_id] = []
                bt.update(band_id, None)  # Chapman-Kolmogorov

        t_occ = {b for b, v in true_state.items() if v}

        # Ingest pulses into Module C
        for b in action:
            for pulse in obs_dict[b]:
                module.ingest_pulse(b, pulse.toa_us)

        if emitter_band in t_occ:
            actual_emissions += 1
            predicted = module.predict_next_arrival(emitter_band)
            if emitter_band in action and len(obs_dict[emitter_band]) > 0:
                intercept_count += 1
                if predicted is not None:
                    act_toa = obs_dict[emitter_band][0].toa_us
                    raw_e   = abs(act_toa - predicted)
                    cyc_e   = min(raw_e, abs(T_us - raw_e))
                    all_cyc.append(cyc_e)
                    if t_local >= warmup_steps:
                        cyc_late.append(cyc_e)

    intercept_rate = intercept_count / max(actual_emissions, 1)
    avg_late  = float(np.mean(cyc_late)) if cyc_late else float("nan")
    avg_all   = float(np.mean(all_cyc))  if all_cyc  else float("nan")

    return {
        "intercept_rate":      intercept_rate,
        "avg_cyclic_err_late": avg_late,   # mature phase only (primary metric)
        "avg_cyclic_err_all":  avg_all,    # all steps (secondary)
        "n_mature_errors":     len(cyc_late),
        "n_all_errors":        len(all_cyc),
        "actual_emissions":    actual_emissions,
        "intercept_count":     intercept_count,
        "n_steps_local":       n_steps,
        "warmup_steps":        warmup_steps,
        "emitter_band":        emitter_band,
        "T_us":                T_us,
        "sigma_us":            133.2 if emitter_label == 10 else 35.9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 72)
    out("  Module C Ablation — Real-Data Failure Mode Analysis")
    out("  SIH 2026 — PS 26055 (DRDO)")
    out("=" * 72)
    out()
    out("  Three conditions, one variable changed per condition:")
    out("    A) Baseline: standard K=3-of-18 competitive access")
    out("    B) Guaranteed dedicated: periodic band always scanned")
    out("    C) Active-window only (emitter 31): eval restricted to emitter's")
    out("       actual active period under guaranteed access")
    out()
    out("  Warmup = second half of EACH CONDITION'S evaluation window")
    out("  (not the full file — the previous run used the wrong denominator)")
    out()

    # ── Emitter 10 config ────────────────────────────────────────────────────
    h5_0 = SAMPLE_DIR / "config_0.h5"
    em10_toa_min = 17_200_518.0   # active window start (from analysis)
    em10_toa_max = 17_398_528.0   # active window end

    out("─" * 72)
    out("  EMITTER 10  (config_0.h5)  T≈1512μs  σ≈133μs  CoV=0.088  DT=750μs")
    out("  Active window: 0.198s (0.68% of 29s file)  |  131 periods  |  132 pulses")
    out("─" * 72)
    out()
    out("  NOTE: Conditions A/B below are evaluated over the ACTIVE WINDOW only")
    out("  (17.2M–17.4M μs). At DT=750μs this gives 264 steps.")
    out("  Warmup = first 132 steps; mature = last 132 steps.")
    out()

    em10_results = {}

    # Condition A — active-window eval, standard access
    out("  Running Condition A (competitive access, active window)...")
    em10_results["A"] = run_condition(
        h5_0, emitter_label=10, T_us=1511.5, dt_us=750.0,
        guaranteed_access=False,
        t_start_us_override=em10_toa_min,
        t_end_us_override=em10_toa_max,
    )
    # Condition B — active-window eval, guaranteed access
    out("  Running Condition B (guaranteed access, active window)...")
    em10_results["B"] = run_condition(
        h5_0, emitter_label=10, T_us=1511.5, dt_us=750.0,
        guaranteed_access=True,
        t_start_us_override=em10_toa_min,
        t_end_us_override=em10_toa_max,
    )

    out()
    out(f"  {'Cond':<5} {'Intercept rate':>16} {'CycErr late (μs)':>18} {'n_mature':>9} {'Emissions':>10}")
    out(f"  {'─'*5} {'─'*16} {'─'*18} {'─'*9} {'─'*10}")
    for cond, res in em10_results.items():
        if "error" in res:
            out(f"  {cond:<5}  ERROR: {res['error']}")
            continue
        err = res["avg_cyclic_err_late"]
        es  = f"{err:.1f}" if not math.isnan(err) else "N/A"
        out(f"  {cond:<5} {res['intercept_rate']:>16.4f} {es:>18} "
            f"{res['n_mature_errors']:>9} {res['actual_emissions']:>10}")

    # Compute A→B lift
    if "A" in em10_results and "B" in em10_results:
        a, b = em10_results["A"], em10_results["B"]
        if "error" not in a and "error" not in b:
            lift = b["intercept_rate"] - a["intercept_rate"]
            out(f"\n  A→B lift: {a['intercept_rate']:.4f} → {b['intercept_rate']:.4f} ({lift:+.4f})")
            out(f"  dwell window = 6σ = {6*133.2:.0f}μs = {6*133.2/1511.5*100:.0f}% of T"
                f"  (tight: leaves ~{50 - 6*133.2/1511.5*50:.0f}% of period outside dwell)")

    # ── Emitter 31 config ────────────────────────────────────────────────────
    h5_2 = SAMPLE_DIR / "config_2.h5"
    em31_toa_start = 6_450_078.0   # emitter active start
    em31_toa_end   = 6_549_574.0   # emitter active end
    # Active window at DT=250μs: 397 steps

    out()
    out("─" * 72)
    out("  EMITTER 31  (config_2.h5)  T≈503μs  σ≈36μs  CoV=0.071  DT=250μs")
    out("  Active window: 0.099s (0.36% of 27.9s file)  |  198 periods  |  199 pulses")
    out("─" * 72)
    out()

    em31_results = {}

    # Condition A — FULL FILE eval, standard access (original condition)
    out("  Running Condition A (competitive access, full file — original)...")
    em31_results["A_full"] = run_condition(
        h5_2, emitter_label=31, T_us=502.5, dt_us=250.0,
        guaranteed_access=False,
    )

    # Condition B — FULL FILE eval, guaranteed access
    out("  Running Condition B (guaranteed access, full file)...")
    em31_results["B_full"] = run_condition(
        h5_2, emitter_label=31, T_us=502.5, dt_us=250.0,
        guaranteed_access=True,
    )

    # Condition C — ACTIVE WINDOW only, guaranteed access
    out("  Running Condition C (guaranteed access, active window only)...")
    em31_results["C_active"] = run_condition(
        h5_2, emitter_label=31, T_us=502.5, dt_us=250.0,
        guaranteed_access=True,
        t_start_us_override=em31_toa_start,
        t_end_us_override=em31_toa_end,
    )

    out()
    out(f"  {'Cond':<10} {'Window':<14} {'Intercept rate':>16} {'CycErr late (μs)':>18} {'n_mature':>9} {'Emissions':>10}")
    out(f"  {'─'*10} {'─'*14} {'─'*16} {'─'*18} {'─'*9} {'─'*10}")
    display = [
        ("A_full",   "Full file",    "A"),
        ("B_full",   "Full file",    "B"),
        ("C_active", "Active only",  "C"),
    ]
    for key, window, cond_label in display:
        res = em31_results[key]
        if "error" in res:
            out(f"  {cond_label:<10} {window:<14}  ERROR: {res['error']}")
            continue
        err = res["avg_cyclic_err_late"]
        es  = f"{err:.1f}" if not math.isnan(err) else "N/A"
        out(f"  {cond_label:<10} {window:<14} {res['intercept_rate']:>16.4f} {es:>18} "
            f"{res['n_mature_errors']:>9} {res['actual_emissions']:>10}")

    # ── Diagnosis ────────────────────────────────────────────────────────────
    out()
    out("─" * 72)
    out("  DIAGNOSIS")
    out("─" * 72)
    out()

    # Emitter 10 diagnosis
    a10 = em10_results.get("A", {})
    b10 = em10_results.get("B", {})
    if "error" not in a10 and "error" not in b10:
        lift10 = b10["intercept_rate"] - a10["intercept_rate"]
        a10_err = a10["avg_cyclic_err_late"]
        b10_err = b10["avg_cyclic_err_late"]
        out("  EMITTER 10:")
        if lift10 > 0.3:
            out(f"  → COVERAGE-RATIO-DOMINATED: A={a10['intercept_rate']:.3f} → B={b10['intercept_rate']:.3f} (+{lift10:.3f})")
            out(f"    Guaranteed access provides large improvement.")
            out(f"    Design conclusion: smarter scan-budget allocation (flag-then-dedicate)")
            out(f"    is the real fix, not an algorithmic one.")
        elif lift10 > 0.1:
            out(f"  → MIXED (coverage + jitter): A={a10['intercept_rate']:.3f} → B={b10['intercept_rate']:.3f} (+{lift10:.3f})")
            out(f"    Coverage is a factor but jitter limits further improvement.")
            if not math.isnan(b10_err):
                out(f"    Cyclic err under full access: {b10_err:.1f}μs  (σ=133μs, dwell=6σ=798μs=53% of T)")
                out(f"    Large real jitter leaves limited timing precision even with full access.")
        else:
            out(f"  → JITTER-DOMINATED: A={a10['intercept_rate']:.3f} → B={b10['intercept_rate']:.3f} (+{lift10:.3f})")
            out(f"    Even guaranteed access does not materially improve intercept rate.")
            out(f"    σ=133μs means 6σ dwell window = 798μs = 53% of T — nearly half the period")
            out(f"    is always in-window, so the dwell-window selection provides little benefit.")

    out()

    # Emitter 31 diagnosis
    b31 = em31_results.get("B_full", {})
    c31 = em31_results.get("C_active", {})
    a31 = em31_results.get("A_full", {})
    if "error" not in b31 and "error" not in c31:
        b31_rate = b31["intercept_rate"]
        c31_rate = c31["intercept_rate"]
        c31_err  = c31["avg_cyclic_err_late"]
        out("  EMITTER 31:")
        if b31_rate < 0.05 and c31_rate > 0.4:
            out(f"  → ACTIVE-WINDOW-DOMINATED: B(full)={b31_rate:.3f} → C(active window)={c31_rate:.3f}")
            out(f"    Under full-file eval, the emitter is active for only 0.36% of steps.")
            out(f"    Even with guaranteed access, most steps have no target — warmup never")
            out(f"    completes over the active window. This is a dataset coverage property,")
            out(f"    not an algorithm limitation.")
            if not math.isnan(c31_err):
                out(f"    Under active-window eval: intercept_rate={c31_rate:.3f}, CycErr={c31_err:.1f}μs")
                out(f"    (σ=36μs noise floor; converged error should approach ~18μs = 0.5σ)")
        elif b31_rate > 0.4:
            out(f"  → COVERAGE-RATIO-DOMINATED: B(full)={b31_rate:.3f}")
            out(f"    Guaranteed access already gives good intercept rate on full file.")
        else:
            out(f"  → ACTIVE-WINDOW-DOMINATED (or emitter inactive during eval window)")
            out(f"    B(full)={b31_rate:.3f}, C(active)={c31_rate:.3f}")

    # ── Comparison vs synthetic ───────────────────────────────────────────────
    out()
    out("─" * 72)
    out("  COMPARISON vs SYNTHETIC (reference: 0.9838 rate, 16.6μs error)")
    out("─" * 72)
    out()

    rows = [
        ("Synthetic (σ=20μs, designed)", SYNTH_C_RATE, SYNTH_C_ERR_US),
    ]
    if "error" not in b10:
        rows.append(("Em10 Cond-B (full-access, σ=133μs)", b10["intercept_rate"], b10["avg_cyclic_err_late"]))
    if "error" not in c31:
        rows.append(("Em31 Cond-C (active-win, σ=36μs)", c31["intercept_rate"], c31["avg_cyclic_err_late"]))

    out(f"  {'Case':<42} {'Rate':>7} {'CycErr(μs)':>12}")
    out(f"  {'─'*42} {'─'*7} {'─'*12}")
    for label, rate, err in rows:
        es = f"{err:.1f}" if not (math.isnan(err) if isinstance(err, float) else False) else "N/A"
        out(f"  {label:<42} {rate:>7.4f} {es:>12}")

    # Save
    out_path = pathlib.Path("data/ablation_module_c_results.md")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
