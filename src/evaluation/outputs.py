"""
Output generation: CSVs, summary tables, plots, demo figure.
"""
from __future__ import annotations

import pathlib
import csv
import numpy as np

OUTPUT_DIR = pathlib.Path("results")


def ensure_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], filename: str) -> pathlib.Path:
    ensure_dir()
    path = OUTPUT_DIR / filename
    if not rows:
        path.write_text("")
        return path
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary markdown tables
# ─────────────────────────────────────────────────────────────────────────────

def write_test1_summary(summary: dict, filename: str = "test1_summary.md") -> pathlib.Path:
    ensure_dir()
    path = OUTPUT_DIR / filename
    lines = ["# Test 1 Summary — Core Scheduler vs Baselines\n",
             "CI method: bootstrap 2000 resamples, 95%\n\n"]

    for scen, algo_dict in summary.items():
        lines.append(f"## Scenario: {scen}\n\n")
        lines.append("| Algorithm | Intercept Rate (mean±CI) | Worst | Avg Reward (mean±CI) | Worst | Wilcoxon p vs WIQL |\n")
        lines.append("|---|---|---|---|---|---|\n")

        wilcoxon = algo_dict.get("_wilcoxon", {})
        wiql_rate = algo_dict.get("wiql_ucb", {}).get("intercept_rate", {})

        for algo, metrics in algo_dict.items():
            if algo.startswith("_"):
                continue
            ir = metrics.get("intercept_rate", {})
            rw = metrics.get("avg_reward", {})
            ir_str = (f"{ir.get('mean',0):.3f} [{ir.get('ci_lo',0):.3f}, {ir.get('ci_hi',0):.3f}]"
                      if ir else "N/A")
            rw_str = (f"{rw.get('mean',0):.3f} [{rw.get('ci_lo',0):.3f}, {rw.get('ci_hi',0):.3f}]"
                      if rw else "N/A")
            w_best = f"{ir.get('worst',0):.3f}" if ir else "N/A"
            rw_worst = f"{rw.get('worst',0):.3f}" if rw else "N/A"

            if algo == "wiql_ucb":
                p_str = "—"
            elif f"wiql_vs_{algo.replace('wiql_ucb_no_bias','wiql_ucb_no_bias')}" in wilcoxon:
                p_str = "—"
            else:
                key = f"wiql_vs_{algo}"
                p_val = wilcoxon.get(key, float("nan"))
                p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"

            lines.append(f"| {algo} | {ir_str} | {w_best} | {rw_str} | {rw_worst} | {p_str} |\n")
        lines.append("\n")

    path.write_text("".join(lines))
    return path


def write_test2_summary(summary: dict, filename: str = "test2_summary.md") -> pathlib.Path:
    ensure_dir()
    path = OUTPUT_DIR / filename
    lines = ["# Test 2 Summary — Periodic Module vs Jitter\n",
             "CI method: bootstrap 2000 resamples, 95%\n\n",
             "| Jitter ratio | WIQL mean | WIQL CI lo | WIQL CI hi | WIQL worst | RR mean | RR worst |\n",
             "|---|---|---|---|---|---|---|\n"]

    for j_ratio, algo_dict in sorted(summary.items()):
        wiql = algo_dict.get("wiql_ucb", {}).get("intercept_rate", {})
        rr   = algo_dict.get("round_robin", {}).get("intercept_rate", {})
        lines.append(
            f"| {j_ratio:.2f} "
            f"| {wiql.get('mean',0):.4f} "
            f"| {wiql.get('ci_lo',0):.4f} "
            f"| {wiql.get('ci_hi',0):.4f} "
            f"| {wiql.get('worst',0):.4f} "
            f"| {rr.get('mean',0):.4f} "
            f"| {rr.get('worst',0):.4f} |\n"
        )

    path.write_text("".join(lines))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_test1(summary: dict, filename: str = "test1_barchart.png") -> pathlib.Path:
    ensure_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return OUTPUT_DIR / filename

    scenarios = [s for s in summary if not s.startswith("_")]
    algos = ["round_robin", "random", "wiql_ucb", "wiql_ucb_no_bias"]
    algo_labels = {"round_robin": "Round-Robin", "random": "Random",
                   "wiql_ucb": "WIQL-UCB", "wiql_ucb_no_bias": "WIQL (no periodic bias)"}
    colors = {"round_robin": "#8b949e", "random": "#6e7681",
              "wiql_ucb": "#3fb950", "wiql_ucb_no_bias": "#58a6ff"}

    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5),
                             facecolor="#0d1117")
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        ax.set_facecolor("#161b22")
        ax.set_title(scen.replace("_", " ").title(), color="#e6edf3", fontsize=12)
        ax.set_ylabel("Intercept Rate", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        x = np.arange(len(algos))
        for i, algo in enumerate(algos):
            ir = summary[scen].get(algo, {}).get("intercept_rate", {})
            if not ir:
                continue
            mean  = ir.get("mean",  0)
            ci_lo = ir.get("ci_lo", mean)
            ci_hi = ir.get("ci_hi", mean)
            worst = ir.get("worst", mean)
            err_lo = mean - ci_lo
            err_hi = ci_hi - mean
            ax.bar(i, mean, color=colors.get(algo, "#fff"), alpha=0.85,
                   yerr=[[err_lo], [err_hi]], capsize=4,
                   error_kw={"ecolor": "#e6edf3", "linewidth": 1.5})
            # worst-case tick
            ax.plot([i - 0.3, i + 0.3], [worst, worst], color="#f85149",
                    linewidth=2, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([algo_labels.get(a, a) for a in algos],
                           rotation=15, ha="right", color="#8b949e", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.axhline(0, color="#30363d", linewidth=0.5)

    fig.suptitle("Test 1 — Intercept Rate by Scenario (error bars = 95% CI, red tick = worst case)",
                 color="#e6edf3", fontsize=10)
    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out


def plot_test2(summary: dict, rr_mean_baseline: float,
               filename: str = "test2_jitter_curve.png") -> pathlib.Path:
    ensure_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return OUTPUT_DIR / filename

    ratios = sorted(summary.keys())
    wiql_means  = [summary[r]["wiql_ucb"]["intercept_rate"]["mean"]   for r in ratios]
    wiql_lo     = [summary[r]["wiql_ucb"]["intercept_rate"]["ci_lo"]  for r in ratios]
    wiql_hi     = [summary[r]["wiql_ucb"]["intercept_rate"]["ci_hi"]  for r in ratios]
    wiql_worst  = [summary[r]["wiql_ucb"]["intercept_rate"]["worst"]  for r in ratios]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")

    pct = [r * 100 for r in ratios]
    ax.fill_between(pct, wiql_lo, wiql_hi, alpha=0.25, color="#3fb950", label="95% CI")
    ax.plot(pct, wiql_means,  color="#3fb950", linewidth=2.5, marker="o", label="WIQL-UCB mean")
    ax.plot(pct, wiql_worst,  color="#f85149", linewidth=1.5, linestyle="--",
            marker="x", label="Worst case (min)")
    ax.axhline(rr_mean_baseline, color="#8b949e", linewidth=1.5, linestyle=":",
               label=f"Round-Robin baseline ({rr_mean_baseline:.3f})")

    ax.set_xlabel("Jitter / Period (%)", color="#e6edf3")
    ax.set_ylabel("Intercept Rate", color="#e6edf3")
    ax.set_title("Test 2 — Periodic Module: Intercept Rate vs Jitter",
                 color="#e6edf3", fontsize=12)
    ax.legend(facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max(pct) * 1.05)

    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Demo visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_demo(trace: dict, dt_us: float, filename: str = "demo_visualization.png") -> pathlib.Path:
    ensure_dir()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return OUTPUT_DIR / filename

    T = len(trace["belief"])
    t_ms = np.arange(T) * dt_us / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, facecolor="#0d1117")
    fig.suptitle(
        "Scheduler anticipates a periodic threat and raises priority ahead of predicted arrival",
        color="#e6edf3", fontsize=11, y=0.98,
    )

    # ── Panel 1: Belief state ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")
    ax1.set_ylabel("P(occupied)", color="#e6edf3", fontsize=9)
    ax1.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#30363d")

    ax1.plot(t_ms, trace["belief"], color="#58a6ff", linewidth=1.2, label="Belief b(t)")
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="#30363d", linewidth=0.5, linestyle=":")

    # Ground-truth pulse arrivals as vertical ticks
    for i, ta in enumerate(trace["true_active"]):
        if ta:
            ax1.axvline(t_ms[i], color="#f0883e", alpha=0.6, linewidth=1.0)

    legend_patches = [
        mpatches.Patch(color="#58a6ff", label="Belief b(t)"),
        mpatches.Patch(color="#f0883e", alpha=0.6, label="True pulse arrival"),
    ]
    ax1.legend(handles=legend_patches, facecolor="#161b22",
               labelcolor="#e6edf3", edgecolor="#30363d", fontsize=8, loc="upper right")

    # ── Panel 2: Priority index ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    ax2.set_ylabel("Priority index", color="#e6edf3", fontsize=9)
    ax2.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor("#30363d")

    prios = np.array(trace["priority"], dtype=float)
    prios_clipped = np.clip(prios, 0, np.percentile(prios[np.isfinite(prios)], 98) if prios[np.isfinite(prios)].size else 1)
    ax2.fill_between(t_ms, 0, prios_clipped, color="#388bfd", alpha=0.4)
    ax2.plot(t_ms, prios_clipped, color="#388bfd", linewidth=1.0)

    # Mark predicted arrival times
    for i, pred in enumerate(trace["predicted_toa"]):
        if pred is not None:
            pred_ms = pred / 1000.0
            if 0 <= pred_ms <= t_ms[-1] + 1:
                ax2.axvline(pred_ms, color="#58a6ff", alpha=0.5, linewidth=0.8,
                            linestyle="--")

    ax2.set_ylabel("Priority index\n(spikes = predicted\narrival window)",
                   color="#e6edf3", fontsize=8)
    ax2.legend(
        handles=[mpatches.Patch(color="#388bfd", alpha=0.5, label="Priority (WIQL+periodic)"),
                 mpatches.Patch(color="#58a6ff", alpha=0.5, label="Predicted next arrival")],
        facecolor="#161b22", labelcolor="#e6edf3", edgecolor="#30363d", fontsize=8, loc="upper right"
    )

    # ── Panel 3: Scan decisions ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#161b22")
    ax3.set_ylabel("Scan decision", color="#e6edf3", fontsize=9)
    ax3.set_xlabel("Time (ms)", color="#e6edf3", fontsize=9)
    ax3.tick_params(colors="#8b949e", labelsize=8)
    for sp in ax3.spines.values(): sp.set_edgecolor("#30363d")
    ax3.set_ylim(-0.3, 1.3)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Not scanned", "Scanned"], color="#8b949e", fontsize=7)

    for i in range(T):
        if trace["hit"][i]:
            ax3.scatter(t_ms[i], 1.0, color="#3fb950", s=40, zorder=5, marker="*")
        elif trace["miss"][i]:
            ax3.scatter(t_ms[i], 1.0, color="#f85149", s=20, zorder=5, marker="x")
        elif trace["scanned"][i]:
            ax3.scatter(t_ms[i], 1.0, color="#8b949e", s=10, zorder=3, alpha=0.5)

    legend_scan = [
        mpatches.Patch(color="#3fb950", label="Hit (scanned + detected)"),
        mpatches.Patch(color="#f85149", label="Miss (scanned, not detected)"),
        mpatches.Patch(color="#8b949e", alpha=0.5, label="Scanned (idle)"),
    ]
    ax3.legend(handles=legend_scan, facecolor="#161b22",
               labelcolor="#e6edf3", edgecolor="#30363d", fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out
