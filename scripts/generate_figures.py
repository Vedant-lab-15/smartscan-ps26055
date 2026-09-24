#!/usr/bin/env python3
"""Generate all publication-quality figures from existing results CSVs.

Usage:
    python scripts/generate_figures.py

Reads:
    results/test1_raw.csv        — core scheduler multi-scenario comparison
    results/test2_raw.csv        — periodic module jitter sweep
    results/ask3_gamma_sweep.csv — periodic module γ-sweep

Writes (to results/figures/):
    test1_barchart.png     — mean ± 95% CI, worst-case tick per algorithm/scenario
    test2_jitter_curve.png — intercept rate vs jitter ratio with CI band
    gamma_sweep.png        — periodic multi-band gain vs γ
    demo_visualization.png — 3-panel single-band trace (belief, priority, scan decisions)

Requires: matplotlib >= 3.7
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np

# ── ensure project root is importable ────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── matplotlib setup ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping figure generation.")
    sys.exit(0)

# Dark-theme style matching the Streamlit dashboard
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
BORDER    = "#30363d"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"
ORANGE    = "#f0883e"
RED       = "#f85149"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "text.color":        TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        BORDER,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": TEXT,
    "font.size":         10,
})


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI helper
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 ci: float = 0.95) -> tuple[float, float]:
    """Return (ci_lo, ci_hi) using percentile bootstrap."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    alpha = (1.0 - ci) / 2.0
    boot_means = [np.mean(arr[np.random.randint(0, len(arr), size=len(arr))])
                  for _ in range(n_boot)]
    return (float(np.percentile(boot_means, 100 * alpha)),
            float(np.percentile(boot_means, 100 * (1.0 - alpha))))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Core scheduler bar chart (Test 1)
# ─────────────────────────────────────────────────────────────────────────────

def figure_test1():
    csv_path = RESULTS_DIR / "test1_raw.csv"
    if not csv_path.exists():
        print(f"  SKIP test1_barchart.png — {csv_path} not found")
        return

    rows = list(csv.DictReader(open(csv_path)))
    scenarios  = ["background", "periodic", "freq_agile"]
    algos      = ["round_robin", "random", "wiql_ucb", "wiql_ucb_no_bias"]
    algo_labels = {
        "round_robin":       "Round-Robin",
        "random":            "Random",
        "wiql_ucb":          "WIQL-UCB",
        "wiql_ucb_no_bias":  "WIQL (no periodic bias)",
    }
    colors = {
        "round_robin":       MUTED,
        "random":            "#6e7681",
        "wiql_ucb":          GREEN,
        "wiql_ucb_no_bias":  BLUE,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Intercept Rate by Scenario  (error bars = 95% CI, red tick = worst case)",
        color=TEXT, fontsize=11,
    )

    for ax, scen in zip(axes, scenarios):
        ax.set_title(scen.replace("_", " ").title(), color=TEXT, fontsize=11)
        ax.set_ylabel("Intercept Rate", color=TEXT)
        ax.set_ylim(0, 1.0)

        x = np.arange(len(algos))
        for i, algo in enumerate(algos):
            vals = [float(r["intercept_rate"])
                    for r in rows
                    if r["scenario"] == scen and r["algorithm"] == algo]
            if not vals:
                continue
            mean = float(np.mean(vals))
            lo, hi = bootstrap_ci(vals)
            worst  = float(np.min(vals))
            ax.bar(i, mean, color=colors.get(algo, TEXT), alpha=0.85,
                   yerr=[[mean - lo], [hi - mean]], capsize=4,
                   error_kw={"ecolor": TEXT, "linewidth": 1.5})
            ax.plot([i - 0.35, i + 0.35], [worst, worst],
                    color=RED, linewidth=2, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([algo_labels.get(a, a) for a in algos],
                           rotation=18, ha="right", fontsize=8)

    plt.tight_layout()
    out = FIGURES_DIR / "test1_barchart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Jitter sweep (Test 2)
# ─────────────────────────────────────────────────────────────────────────────

def figure_test2():
    csv_path = RESULTS_DIR / "test2_raw.csv"
    if not csv_path.exists():
        print(f"  SKIP test2_jitter_curve.png — {csv_path} not found")
        return

    rows = list(csv.DictReader(open(csv_path)))
    jitter_ratios = sorted({float(r["jitter_ratio"]) for r in rows})

    wiql_means, wiql_los, wiql_his, wiql_worsts = [], [], [], []
    rr_means = []

    for j in jitter_ratios:
        wiql_vals = [float(r["intercept_rate"])
                     for r in rows
                     if float(r["jitter_ratio"]) == j and r["algorithm"] == "wiql_ucb"]
        rr_vals = [float(r["intercept_rate"])
                   for r in rows
                   if float(r["jitter_ratio"]) == j and r["algorithm"] == "round_robin"]
        if not wiql_vals:
            continue
        m = float(np.mean(wiql_vals))
        lo, hi = bootstrap_ci(wiql_vals)
        wiql_means.append(m)
        wiql_los.append(lo)
        wiql_his.append(hi)
        wiql_worsts.append(float(np.min(wiql_vals)))
        rr_means.append(float(np.mean(rr_vals)) if rr_vals else 0.0)

    pct = [j * 100 for j in jitter_ratios[:len(wiql_means)]]
    rr_baseline = float(np.mean(rr_means)) if rr_means else 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(pct, wiql_los, wiql_his, alpha=0.25, color=GREEN,
                    label="95% CI (WIQL-UCB)")
    ax.plot(pct, wiql_means, color=GREEN, linewidth=2.5, marker="o",
            label="WIQL-UCB mean")
    ax.plot(pct, wiql_worsts, color=RED, linewidth=1.5, linestyle="--",
            marker="x", label="Worst case (min seed)")
    ax.axhline(rr_baseline, color=MUTED, linewidth=1.5, linestyle=":",
               label=f"Round-Robin baseline ({rr_baseline:.3f})")

    ax.set_xlabel("Jitter / Period (%)", color=TEXT)
    ax.set_ylabel("Intercept Rate", color=TEXT)
    ax.set_title("Periodic Module: Intercept Rate vs Jitter", color=TEXT, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = FIGURES_DIR / "test2_jitter_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: γ-sweep (periodic multi-band integration)
# ─────────────────────────────────────────────────────────────────────────────

def figure_gamma_sweep():
    csv_path = RESULTS_DIR / "ask3_gamma_sweep.csv"
    if not csv_path.exists():
        print(f"  SKIP gamma_sweep.png — {csv_path} not found")
        return

    rows = list(csv.DictReader(open(csv_path)))
    gammas = sorted({float(r["gamma"]) for r in rows})

    means, los, his = [], [], []
    for g in gammas:
        vals = [float(r["intercept_rate"])
                for r in rows if float(r["gamma"]) == g]
        if not vals:
            continue
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        means.append(m)
        los.append(lo)
        his.append(hi)

    gammas_plot = gammas[:len(means)]
    baseline = means[0] if means else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(gammas_plot, los, his, alpha=0.25, color=BLUE, label="95% CI")
    ax.plot(gammas_plot, means, color=BLUE, linewidth=2.5, marker="o",
            label="Mean intercept rate")
    ax.axhline(baseline, color=MUTED, linewidth=1.5, linestyle=":",
               label=f"γ=0 baseline ({baseline:.3f})")

    ax.set_xlabel("γ (periodic recurrence weight)", color=TEXT)
    ax.set_ylabel("Intercept Rate", color=TEXT)
    ax.set_title("Periodic Module γ-Sweep (multi-band integration)", color=TEXT, fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "gamma_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Demo visualization (generated from a fresh short run)
# ─────────────────────────────────────────────────────────────────────────────

def figure_demo():
    """Generate a 200-step single-band demo visualization without a running harness."""
    try:
        from src.scheduler.belief import BeliefTracker
        from src.scheduler.wiql_ucb import WIQLScheduler, ACTION_SCAN, ACTION_PASSIVE
        from src.scheduler.periodic import PeriodicInterceptModule
        from src.environment.pdw_generator import PDWGenerator, PeriodicBandConfig
        from src.environment.simulator import RFEnvironment
        from src.environment.receiver import ReceiverModel
    except ImportError as e:
        print(f"  SKIP demo_visualization.png — import error: {e}")
        return

    n_bands, k_scan = 4, 2
    periodic_band   = 0
    T_us, sigma_us  = 500.0, 20.0
    dt_us           = 50.0
    n_steps         = 200

    band_cf    = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(
        n_bands=n_bands, band_cf_mhz=band_cf, dt_us=dt_us,
        p_emit=[0.1] * n_bands, seed=42,
        periodic_config={
            periodic_band: PeriodicBandConfig(period_us=T_us, sigma_us=sigma_us,
                                              t_start_us=T_us)
        },
    )
    env = RFEnvironment(source=gen, n_bands=n_bands,
                        band_edges_mhz=band_edges, dt_us=dt_us)
    env.initialize()
    bt  = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    rec = ReceiverModel(rf_env=env, belief_tracker=bt,
                         n_bands=n_bands, k_scan=k_scan)
    rec.reset()
    sched  = WIQLScheduler(n_bands=n_bands, k_scan=k_scan)
    module = PeriodicInterceptModule(n_bands=n_bands)

    trace = {
        "belief": [], "priority": [], "true_active": [],
        "hit": [], "miss": [], "scanned": [], "predicted_toa": [],
    }

    for t in range(n_steps):
        t_us   = t * dt_us
        b_prev = rec.get_belief().copy()

        periodic_idx = module.compute_all_indices(t_us)
        wiql_idx     = sched.compute_indices(b_prev)
        combined_idx = wiql_idx + periodic_idx
        action = set(sorted(np.argsort(-combined_idx)[:k_scan].tolist()))

        obs_dict, _, _ = rec.step(action)
        true_state = env.get_true_state(t)
        b_next = rec.get_belief()

        for b in action:
            for pulse in obs_dict[b]:
                module.ingest_pulse(b, pulse.toa_us)

        for b in action:
            obs = 1 if len(obs_dict[b]) > 0 else 0
            sched.update(b, ACTION_SCAN, float(obs),
                         float(b_prev[b]), float(b_next[b]))
        for b in range(n_bands):
            if b not in action:
                sched.update(b, ACTION_PASSIVE, 0.0,
                             float(b_prev[b]), float(b_next[b]))

        true_occ = true_state.get(periodic_band, False)
        scanned  = periodic_band in action
        detected = scanned and len(obs_dict.get(periodic_band, [])) > 0

        trace["belief"].append(float(b_next[periodic_band]))
        idx_val = float(combined_idx[periodic_band])
        trace["priority"].append(min(idx_val, 20.0) if np.isfinite(idx_val) else 20.0)
        trace["true_active"].append(true_occ)
        trace["hit"].append(detected and true_occ)
        trace["miss"].append(scanned and not detected and true_occ)
        trace["scanned"].append(scanned)
        trace["predicted_toa"].append(module.predict_next_arrival(periodic_band))

    # Plot
    from harness.outputs import plot_demo
    out = plot_demo(trace, dt_us=dt_us, filename="demo_visualization.png")
    # Copy to figures dir as well
    import shutil
    shutil.copy(out, FIGURES_DIR / "demo_visualization.png")
    print(f"  → {out}")
    print(f"  → {FIGURES_DIR / 'demo_visualization.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    print()
    figure_test1()
    figure_test2()
    figure_gamma_sweep()
    figure_demo()
    print()
    print(f"All figures written to: {FIGURES_DIR}/")
