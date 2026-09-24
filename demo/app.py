"""
Smart Scan Strategy — PS 26055 · SIH 2026 (DRDO)
Whittle-inspired priority scheduler for EW spectrum scanning
Coding Saints, Team ID 120303

Run locally:  streamlit run demo/app.py
Deployed at:  https://huggingface.co/spaces/Vedant-lab-15/smartscan-demo
"""
from __future__ import annotations

import sys
import math
import pathlib

import numpy as np
import streamlit as st

# ── ensure project root on path (works locally, on HF Spaces, and from demo/) ─
_HERE = pathlib.Path(__file__).resolve().parent
# Support three layouts:
#   1. Local: demo/app.py  → parent = demo/, parent.parent = repo root
#   2. HF Space: app.py at root → parent = root (src/ is sibling)
#   3. HF Space: demo/app.py   → parent = demo/, parent.parent = root
for candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (candidate / "src").is_dir():
        _ROOT = candidate
        break
else:
    _ROOT = _HERE

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.env_generator import (
    RenewalEnv,
    make_background_scenario,
    make_periodic_scenario,
    make_freq_agile_scenario,
    TSRD_PERIODIC_CLASSES,
)
from src.evaluation.runner import WIQLPolicy, run_episode, policy_round_robin

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Scan Strategy — PS 26055",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.metric-card {
    background: #161b22; border-radius: 10px; padding: 16px 20px;
    text-align: center; border: 1px solid #30363d;
}
.metric-label { font-size: 0.72rem; color: #8b949e; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 1.9rem; font-weight: 700; }
.green  { color: #3fb950; }
.blue   { color: #58a6ff; }
.orange { color: #f0883e; }
.red    { color: #f85149; }
.delta-pos { color: #3fb950; font-size: 1rem; font-weight: 600; }
.delta-neg { color: #f85149; font-size: 1rem; font-weight: 600; }
.section-rule { border: none; border-top: 1px solid #30363d; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("# 📡 Smart Scan Strategy — PS 26055")
st.caption(
    "Whittle-inspired priority scheduler for EW spectrum scanning · "
    "Coding Saints (ID 120303) · SIH 2026 DRDO"
)
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Simulation Settings")

    scenario = st.selectbox(
        "Scenario",
        ["Background (random occupancy)", "Periodic (radar emitter)", "Frequency-Agile (hopping)"],
        index=0,
        help="Background: random emitters. Periodic: TSRD-grounded radar PRI. Freq-Agile: hopping emitter.",
    )

    n_bands = st.slider("Frequency bands (N)", min_value=4, max_value=16, value=8, step=2)
    k_scan  = st.slider("Simultaneous scans (K)", min_value=1, max_value=4, value=3)
    t_steps = st.slider("Time steps", min_value=200, max_value=2000, value=500, step=100)
    seed    = st.slider("Random seed", min_value=0, max_value=99, value=42)

    st.markdown("---")
    run_clicked = st.button("▶ Run Episode", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "The **WIQL-UCB scheduler** maintains a per-band belief state "
        "and computes a priority index:\n\n"
        "> `I_i = W_i·b_i + γ·bonus_i + UCB_i`\n\n"
        "It focuses on active bands while round-robin sweeps blindly."
    )
    st.markdown(
        "**Code & paper:** [github.com/Vedant-lab-15/smartscan-ps26055]"
        "(https://github.com/Vedant-lab-15/smartscan-ps26055)"
    )

# ── helpers ───────────────────────────────────────────────────────────────────

def run_full_episode(scenario_name: str, n_bands: int, k_scan: int,
                     t_steps: int, seed: int) -> dict:
    """Run one full episode, returning per-step logs for both schedulers."""
    rng = np.random.default_rng(seed)
    DT_PERIODIC = 100.0
    DT_DEFAULT  = 1_000.0

    if scenario_name.startswith("Background"):
        emitters = make_background_scenario(rng, n_bands)
        dt_us = DT_DEFAULT
    elif scenario_name.startswith("Periodic"):
        emitters = make_periodic_scenario(rng, n_bands, dt_us=DT_PERIODIC)
        dt_us = DT_PERIODIC
    else:
        emitters = make_freq_agile_scenario(rng, n_bands)
        dt_us = DT_DEFAULT

    # ── WIQL-UCB episode ──────────────────────────────────────────────────────
    env_w = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed)
    env_w.set_emitters(emitters)

    policy_w = WIQLPolicy(n_bands=n_bands, k=k_scan, use_periodic_bias=True)
    policy_w.reset()

    wiql_rewards: list[float] = []
    wiql_intercepts: list[int] = []
    wiql_occupied: list[int] = []
    wiql_actions: list[set] = []
    wiql_beliefs: list[np.ndarray] = []  # per step, all bands
    wiql_indices: list[np.ndarray] = []  # per step, all bands
    occ_grid: list[list[bool]] = []      # [step][band]

    env_w.reset()
    for t in range(t_steps):
        action = policy_w.select(t, dt_us)
        obs    = env_w.step(action)
        policy_w.update(t, action, obs, dt_us)

        slot_int = 0; slot_occ = 0; slot_rew = 0.0
        occ_row = []
        for b in range(n_bands):
            ta = obs[b]["true_active"]; sc = obs[b]["scanned"]; de = obs[b]["detected"]
            occ_row.append(ta)
            if ta: slot_occ += 1
            if ta and sc and de:
                slot_int += 1; slot_rew += 1.0
            elif sc and de and not ta:
                slot_rew -= 0.1

        wiql_rewards.append(slot_rew / max(k_scan, 1))
        wiql_intercepts.append(slot_int)
        wiql_occupied.append(slot_occ)
        wiql_actions.append(action)
        wiql_beliefs.append(policy_w.bt.get_all().copy())
        # Compute current index vector (may have inf for unvisited)
        raw_idx = policy_w.sched.compute_indices(policy_w.bt.get_all())
        finite_idx = np.where(np.isinf(raw_idx), 5.0, raw_idx)
        wiql_indices.append(finite_idx.copy())
        occ_grid.append(occ_row)

    # ── Round-Robin episode (same emitters, same seed) ─────────────────────
    env_r = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed)
    env_r.set_emitters(emitters)
    env_r.reset()

    rr_rewards: list[float] = []
    rr_intercepts: list[int] = []
    rr_occupied: list[int] = []

    for t in range(t_steps):
        action = policy_round_robin(t, n_bands, k_scan)
        obs    = env_r.step(action)

        slot_int = 0; slot_occ = 0; slot_rew = 0.0
        for b in range(n_bands):
            ta = obs[b]["true_active"]; sc = obs[b]["scanned"]; de = obs[b]["detected"]
            if ta: slot_occ += 1
            if ta and sc and de:
                slot_int += 1; slot_rew += 1.0
            elif sc and de and not ta:
                slot_rew -= 0.1

        rr_rewards.append(slot_rew / max(k_scan, 1))
        rr_intercepts.append(slot_int)
        rr_occupied.append(slot_occ)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_occ_w = max(sum(wiql_occupied), 1)
    total_occ_r = max(sum(rr_occupied),   1)

    return {
        "wiql_rate":  sum(wiql_intercepts) / total_occ_w,
        "rr_rate":    sum(rr_intercepts)   / total_occ_r,
        "wiql_rewards":     wiql_rewards,
        "rr_rewards":       rr_rewards,
        "wiql_intercepts":  wiql_intercepts,
        "rr_intercepts":    rr_intercepts,
        "wiql_occupied":    wiql_occupied,
        "wiql_actions":     wiql_actions,
        "wiql_beliefs":     wiql_beliefs,   # list[np.ndarray(n_bands)]
        "wiql_indices":     wiql_indices,   # list[np.ndarray(n_bands)]
        "occ_grid":         occ_grid,       # list[list[bool]]
        "n_bands":          n_bands,
        "k_scan":           k_scan,
        "t_steps":          t_steps,
        "scenario":         scenario_name,
    }


def cumulative_intercept_rate(intercepts: list[int], occupied: list[int]) -> list[float]:
    """Running intercept rate: cum_int / max(cum_occ, 1) at each step."""
    rates = []
    ci = 0; co = 0
    for i, o in zip(intercepts, occupied):
        ci += i; co += o
        rates.append(ci / max(co, 1))
    return rates


# ── main panel ────────────────────────────────────────────────────────────────

if not run_clicked:
    st.info(
        "Configure the scenario in the sidebar, then press **▶ Run Episode** to simulate. "
        "Results appear instantly — no waiting for a live loop."
    )
    st.markdown("""
    **What this demo shows:**
    - 📊 **Band occupancy heatmap** — which bands are active at each step (ground truth)
    - 🔵 **Scan overlay** — which bands WIQL-UCB chose to scan vs round-robin
    - 📈 **Belief state** — the scheduler's per-band probability estimate P(occupied)
    - 🎯 **Priority index** — the combined Whittle + UCB score driving scan decisions
    - 🏆 **Intercept counter** — WIQL-UCB vs round-robin head-to-head
    - 📉 **Cumulative intercept rate** — who pulls ahead over time
    """)
else:
    with st.spinner("Running episode..."):
        results = run_full_episode(scenario, n_bands, k_scan, t_steps, seed)

    n  = results["n_bands"]
    T  = results["t_steps"]
    wr = results["wiql_rate"]
    rr = results["rr_rate"]
    delta = wr - rr

    # ── Headline metrics ──────────────────────────────────────────────────────
    st.markdown("### 📊 Episode Results")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">WIQL-UCB Intercept Rate</div>
          <div class="metric-value green">{wr:.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Round-Robin Intercept Rate</div>
          <div class="metric-value orange">{rr:.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        cls   = "delta-pos" if delta >= 0 else "delta-neg"
        arrow = "▲" if delta >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">WIQL Advantage</div>
          <div class="metric-value {cls}">{arrow} {abs(delta):.3f}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Scenario / Bands / K</div>
          <div class="metric-value blue" style="font-size:1.1rem">
            {scenario.split('(')[0].strip()}<br>{n} bands · K={k_scan}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Cumulative intercept rate chart ───────────────────────────────────────
    st.markdown("### 📈 Cumulative Intercept Rate — WIQL-UCB vs Round-Robin")

    import pandas as pd

    wiql_cum = cumulative_intercept_rate(results["wiql_intercepts"], results["wiql_occupied"])
    rr_cum   = cumulative_intercept_rate(results["rr_intercepts"],   results["wiql_occupied"])

    # Downsample to 200 points max for clean rendering
    step_every = max(1, T // 200)
    xs = list(range(0, T, step_every))
    chart_df = pd.DataFrame({
        "Step": xs,
        "WIQL-UCB": [wiql_cum[i] for i in xs],
        "Round-Robin": [rr_cum[i]  for i in xs],
    }).set_index("Step")

    st.line_chart(chart_df, use_container_width=True, height=220,
                  color=["#3fb950", "#8b949e"])
    st.caption(
        "Higher = better. WIQL-UCB focuses on active bands; "
        "round-robin sweeps blindly regardless of occupancy."
    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Band occupancy heatmap ────────────────────────────────────────────────
    st.markdown("### 🗺️ Band Occupancy — Ground Truth + WIQL-UCB Scan Decisions")
    col_heat, col_rr_heat = st.columns(2)

    # Build heatmap arrays: rows=bands, cols=time (downsampled)
    sample_steps = min(150, T)
    step_every_h = max(1, T // sample_steps)
    sampled = list(range(0, T, step_every_h))[:sample_steps]

    # Occupancy: 1.0 = active, 0.0 = idle
    occ_arr = np.array([[1.0 if results["occ_grid"][t][b] else 0.0
                         for t in sampled] for b in range(n)])

    # Scan overlay: 2.0 = scanned+active, 1.5 = scanned+idle, rest = occ_arr value
    wiql_scan_arr = occ_arr.copy()
    rr_scan_arr   = occ_arr.copy()
    for ti, t in enumerate(sampled):
        w_action = results["wiql_actions"][t]
        r_action = {(t * k_scan + j) % n for j in range(k_scan)}
        for b in range(n):
            if b in w_action:
                wiql_scan_arr[b, ti] = 2.0 if occ_arr[b, ti] > 0.5 else 1.5
            if b in r_action:
                rr_scan_arr[b, ti]   = 2.0 if occ_arr[b, ti] > 0.5 else 1.5

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#0d1117", "#21262d", "#f0883e", "#3fb950"])

    def make_heatmap(arr, title):
        fig, ax = plt.subplots(figsize=(7, max(2, n * 0.35)),
                               facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        # Map: 0=idle, 0.5=(unused), 1.0→orange(active), 1.5→dim-blue(scanned-idle), 2.0→green(scanned-hit)
        # Remap to 4 discrete levels: 0, 0.67, 1.33, 2.0
        remapped = np.clip(arr, 0, 2.0)
        ax.imshow(remapped, aspect="auto", cmap=cmap, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"B{b}" for b in range(n)],
                           color="#8b949e", fontsize=7)
        ax.set_xticks([])
        ax.set_title(title, color="#e6edf3", fontsize=9, pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout(pad=0.4)
        return fig

    with col_heat:
        fig_w = make_heatmap(wiql_scan_arr, "WIQL-UCB  (🟢 scanned+hit  🟠 active  ░ scanned-idle)")
        st.pyplot(fig_w, use_container_width=True)
        plt.close(fig_w)

    with col_rr_heat:
        fig_r = make_heatmap(rr_scan_arr, "Round-Robin  (🟢 scanned+hit  🟠 active  ░ scanned-idle)")
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

    st.caption(
        "Each row = one frequency band. Each column = one time step (downsampled). "
        "WIQL-UCB concentrates scans on active bands; round-robin distributes evenly."
    )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Belief state and priority index at final step ─────────────────────────
    st.markdown("### 🧠 Final Belief State & Priority Index (last step)")
    col_belief, col_priority = st.columns(2)

    final_belief = results["wiql_beliefs"][-1]
    final_index  = results["wiql_indices"][-1]
    band_labels  = [f"B{b}" for b in range(n)]

    def bar_chart(values, labels, title, color):
        fig, ax = plt.subplots(figsize=(max(4, n * 0.5), 2.8),
                               facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        bars = ax.bar(labels, values, color=color, alpha=0.85, width=0.6)
        ax.set_ylim(0, max(float(np.max(values)) * 1.15, 0.1))
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_title(title, color="#e6edf3", fontsize=9, pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.yaxis.label.set_color("#8b949e")
        fig.tight_layout(pad=0.5)
        return fig

    with col_belief:
        fig_b = bar_chart(final_belief, band_labels,
                          "Belief b_i = P(band occupied)", "#58a6ff")
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)
        st.caption("High belief → scheduler visits more often. Converges to true occupancy pattern.")

    with col_priority:
        fig_p = bar_chart(final_index, band_labels,
                          "Priority index I_i = W·b + UCB", "#3fb950")
        st.pyplot(fig_p, use_container_width=True)
        plt.close(fig_p)
        st.caption("Highest-priority bands are scanned next step. Combines belief, Whittle index, and UCB.")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ── Rolling reward comparison ─────────────────────────────────────────────
    st.markdown("### 🏆 Rolling Reward — WIQL-UCB vs Round-Robin (20-step window)")

    import pandas as pd
    w_roll = pd.Series(results["wiql_rewards"]).rolling(20, min_periods=1).mean()
    r_roll = pd.Series(results["rr_rewards"]).rolling(20, min_periods=1).mean()

    roll_df = pd.DataFrame({
        "Step": list(range(T)),
        "WIQL-UCB": w_roll.tolist(),
        "Round-Robin": r_roll.tolist(),
    }).set_index("Step")

    # Downsample
    roll_df = roll_df.iloc[::step_every]
    st.line_chart(roll_df, use_container_width=True, height=200,
                  color=["#3fb950", "#8b949e"])
    st.caption("20-step rolling average reward per step. Positive spikes = successful intercepts.")

    # ── Periodic emitter callout ──────────────────────────────────────────────
    if scenario.startswith("Periodic"):
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown("### 🎯 Periodic Emitter Info")
        st.info(
            f"Scenario uses TSRD-grounded periodic emitters:\n"
            f"- **P1**: T = {TSRD_PERIODIC_CLASSES[0]['T_us']} μs, "
            f"σ = {TSRD_PERIODIC_CLASSES[0]['sigma_T_us']} μs "
            f"(CoV = {TSRD_PERIODIC_CLASSES[0]['sigma_T_us']/TSRD_PERIODIC_CLASSES[0]['T_us']*100:.1f}%)\n"
            f"- **P2**: T = {TSRD_PERIODIC_CLASSES[1]['T_us']} μs, "
            f"σ = {TSRD_PERIODIC_CLASSES[1]['sigma_T_us']} μs "
            f"(CoV = {TSRD_PERIODIC_CLASSES[1]['sigma_T_us']/TSRD_PERIODIC_CLASSES[1]['T_us']*100:.1f}%)\n\n"
            "The periodic module uses a Welford online estimator to learn the PRI from "
            "observed inter-arrival times, then opens a targeted dwell window (±3σ) "
            "around the predicted next arrival. Convergence: 30/30 seeds with 50-step "
            "round-robin pre-phase (worst-case intercept rate 0.419)."
        )

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
st.markdown(
    "Full research paper, code, and methodology: "
    "[github.com/Vedant-lab-15/smartscan-ps26055]"
    "(https://github.com/Vedant-lab-15/smartscan-ps26055) · "
    "SIH 2026 PS 26055 (DRDO) · MIT License"
)
