"""
Smart Scan Strategy — Interactive Prototype
PS 26055 · SIH 2026 (DRDO) · Coding Saints, Team ID 120303

Run locally:  streamlit run demo/app.py
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
import streamlit as st

# ── path resolution ───────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
for candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (candidate / "src").is_dir():
        _ROOT = candidate
        break
else:
    _ROOT = _HERE
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.env_generator import (
    RenewalEnv, Emitter,
    make_background_scenario, make_periodic_scenario, make_freq_agile_scenario,
    TSRD_PERIODIC_CLASSES as _TPC,
)
from src.evaluation.runner import WIQLPolicy, policy_round_robin
from src.baselines.random_policy import RandomPolicy
from src.baselines.round_robin import RoundRobinPolicy

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Smart Scan · PS 26055",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",  # always open — never auto-collapse
)

# ── matplotlib global style (applied before any figure is created) ────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.size":            10,
    "axes.edgecolor":       "#bdc3c7",
    "axes.labelcolor":      "#2c3e50",
    "axes.titleweight":     "600",
    "axes.titlesize":       11,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.color":          "#7f8c8d",
    "ytick.color":          "#7f8c8d",
    "grid.color":           "#ecf0f1",
    "grid.linewidth":       0.6,
    "figure.facecolor":     "#f8f9fa",
    "axes.facecolor":       "#f8f9fa",
})

# ── CSS + fonts ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #2d3436;
}
code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Hide default Streamlit chrome */
footer                              { visibility: hidden; }
#MainMenu                           { visibility: hidden; }
header [data-testid="stToolbar"]    { display: none; }

/* Force sidebar always visible — never auto-collapse on narrow viewports.
   The collapse button is still present but hidden by toolbarMode=minimal;
   this ensures the sidebar panel itself is always shown. */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    min-width: 18rem !important;
    width: 22rem !important;
    transform: none !important;
    opacity: 1 !important;
}
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
section[data-testid="stSidebarContent"]   { display: block !important; }

/* Tighten top padding */
.block-container { padding-top: 0 !important; }

/* App header bar */
.app-header {
    background: #1a2332;
    color: #f8f9fa;
    padding: 12px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    border-bottom: 3px solid #e67e22;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.app-header .title {
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.app-header .status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    color: #b2bec3;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #27ae60;
    box-shadow: 0 0 6px #27ae60;
}

/* Metric cards */
.metric-card {
    background: white;
    border-left: 4px solid #e67e22;
    padding: 14px 18px;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    margin-bottom: 10px;
}
.metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7f8c8d;
    font-weight: 600;
    margin-bottom: 3px;
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1a2332;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta {
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 3px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-delta.positive { color: #27ae60; }
.metric-delta.negative { color: #c0392b; }

/* Sidebar section labels */
.sidebar-section {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #7f8c8d;
    font-weight: 700;
    margin: 14px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #dfe6e9;
}

/* Warn box */
.warn-box {
    background: #fef9f0;
    border: 1px solid #f39c12;
    border-radius: 4px;
    padding: 8px 12px;
    font-size: 0.8rem;
    color: #856404;
    margin: 4px 0 8px 0;
}

/* Section divider */
hr.rule { border: none; border-top: 1px solid #dfe6e9; margin: 1.2rem 0; }

/* App footer */
.app-footer {
    margin-top: 40px;
    padding-top: 14px;
    border-top: 1px solid #dfe6e9;
    font-size: 0.78rem;
    color: #7f8c8d;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.app-footer a {
    color: #e67e22;
    text-decoration: none;
    font-weight: 600;
}
.run-note {
    font-size: 0.73rem;
    color: #95a5a6;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="title">📡 Smart Scan Strategy &nbsp;·&nbsp; PS 26055</div>
  <div class="status">
    <span class="status-dot"></span>LIVE PROTOTYPE
    &nbsp;|&nbsp; Coding Saints · Team 120303 · SIH 2026
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem;
         color:#7f8c8d; text-align:center; padding:6px 0 14px 0;
         border-bottom:1px solid #dfe6e9; margin-bottom:8px;">
    CODING SAINTS · TEAM 120303
    </div>
    """, unsafe_allow_html=True)

    # ── Layer 1: Scenario ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Scenario</div>', unsafe_allow_html=True)
    with st.expander("Layer 1 — Scenario Editor", expanded=True):
        emitter_type = st.radio(
            "Emitter type",
            ["random", "periodic", "frequency-agile", "mixed"],
            index=0,
            horizontal=True,
            help="random: Bernoulli. periodic: TSRD-grounded radar PRI. "
                 "freq-agile: hopping. mixed: all three.",
        )
        n_emitters   = st.slider("Number of emitters", 1, 5, 3)
        duty_cycle   = st.slider("Duty cycle", 0.1, 0.9, 0.5, 0.05,
                                 help="P(active per slot) for random/mixed emitters.")
        period_slots = st.slider("Period (slots, periodic/mixed)", 5, 50, 10,
                                 help="Nominal PRI in slots. dt=100μs periodic, 1ms others.")
        hop_size     = st.slider("Hop set size (freq-agile/mixed)", 2, 8, 4)
        n_bands      = st.slider("Bands (N)", 4, 32, 8, 2)
        k_scan       = st.slider("Simultaneous scans (K)", 1, min(5, n_bands - 1), 3)
        t_steps      = st.slider("Time steps", 200, 2000, 500, 100)
        seed         = st.slider("Random seed", 0, 99, 42)

    # ── Layer 2: Scheduler ────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Scheduler</div>', unsafe_allow_html=True)
    with st.expander("Layer 2 — Scheduler Configuration", expanded=False):
        c_ucb        = st.slider("UCB constant c", 0.1, 5.0, 1.0, 0.1,
                                 help="Higher = more exploration.")
        epsilon      = st.slider("UCB floor ε", 0.00, 0.10, 0.01, 0.005,
                                 help="Minimum UCB bonus — prevents band lockout.")
        gamma        = st.slider("Periodic-bias γ", 0.0, 10.0, 5.0, 0.5,
                                 help="γ=0 disables periodic module. Paper default: 5.")
        warmup_steps = st.slider("RR warmup steps", 0, 200, 50, 10,
                                 help="Round-robin pre-phase. Set 0 → convergence failure.")
        prior        = st.slider("Belief prior b₀", 0.1, 0.9, 0.5, 0.05)
        p_detect     = st.slider("Pd", 0.5, 1.0, 0.9, 0.05,
                                 help="P(detect | scanned ∧ occupied).")
        p_fa         = st.slider("Pfa", 0.00, 0.10, 0.01, 0.005,
                                 help="P(detect | scanned ∧ idle).")

    # ── Layer 3: Baseline ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">Baseline</div>', unsafe_allow_html=True)
    with st.expander("Layer 3 — Baseline Selector", expanded=False):
        baseline_choice = st.selectbox(
            "Compare WIQL-UCB against",
            ["Round-Robin", "Random", "Clarkson (2003)", "Teissier (2026)"],
            index=0,
            help="Clarkson/Teissier are analytical bounds — falls back to Round-Robin.",
        )
        if baseline_choice in ("Clarkson (2003)", "Teissier (2026)"):
            st.markdown(
                f'<div class="warn-box">⚠ {baseline_choice} is an analytical bound, '
                'not a runnable scheduler. Showing Round-Robin instead.</div>',
                unsafe_allow_html=True,
            )
            _effective_baseline = "Round-Robin"
            _baseline_note = f"{baseline_choice} → Round-Robin"
        else:
            _effective_baseline = baseline_choice
            _baseline_note = None

    st.markdown("---")
    run_clicked = st.button("▶  Run Episode", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.74rem; color:#7f8c8d; line-height:1.7;">
    <strong>Try these:</strong><br>
    γ=0 → periodic module off<br>
    warmup=0 → convergence failure<br>
    c=0.1 → under-exploration<br>
    N=32, K=3 → sparse coverage<br>
    Pd=0.6 → noisy receiver
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (logic unchanged from prototype — only plt styles updated)
# ═══════════════════════════════════════════════════════════════════════════════

def build_custom_emitters(emitter_type, n_emitters, n_bands, duty_cycle,
                          period_slots, hop_size, dt_us, rng):
    emitters: list[Emitter] = []
    if emitter_type == "random":
        for i in range(n_emitters):
            emitters.append(Emitter(emitter_type="background",
                band=int(rng.integers(0, n_bands)), p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)), label=f"bg_{i}"))
    elif emitter_type == "periodic":
        T = period_slots; sig = max(0.5, T * 0.08)
        for i in range(n_emitters):
            emitters.append(Emitter(emitter_type="periodic",
                band=int(rng.integers(0, n_bands)), T_slots=float(T),
                sigma_T_slots=float(sig), snr_db=float(rng.uniform(12.0, 22.0)),
                label=f"periodic_{i}"))
    elif emitter_type == "frequency-agile":
        for i in range(n_emitters):
            s = int(rng.integers(0, max(1, n_bands - hop_size)))
            hops = list(range(s, min(s + hop_size, n_bands)))
            emitters.append(Emitter(emitter_type="freq_agile", band=hops[0],
                hop_bands=hops, p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)), label=f"agile_{i}"))
    else:  # mixed
        per = max(1, n_emitters // 3); rem = n_emitters - per * 3
        for i in range(per):
            emitters.append(Emitter(emitter_type="background",
                band=int(rng.integers(0, n_bands)), p_active=duty_cycle,
                snr_db=float(rng.uniform(5.0, 18.0)), label=f"bg_{i}"))
        for i in range(per):
            T = period_slots
            emitters.append(Emitter(emitter_type="periodic",
                band=int(rng.integers(0, n_bands)), T_slots=float(T),
                sigma_T_slots=max(0.5, T * 0.08),
                snr_db=float(rng.uniform(12.0, 22.0)), label=f"periodic_{i}"))
        for i in range(per + rem):
            s = int(rng.integers(0, max(1, n_bands - hop_size)))
            hops = list(range(s, min(s + hop_size, n_bands)))
            emitters.append(Emitter(emitter_type="freq_agile", band=hops[0],
                hop_bands=hops, p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)), label=f"agile_{i}"))
    return emitters


def run_episode_core(emitters, n_bands, k_scan, t_steps, seed, dt_us,
                     c_ucb, epsilon, gamma, warmup_steps, prior,
                     p_detect, p_fa, baseline):
    env_w = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed, p_false_alarm=p_fa)
    env_w.set_emitters(emitters)
    policy_w = WIQLPolicy(n_bands=n_bands, k=k_scan, use_periodic_bias=(gamma > 0),
                          c_ucb=c_ucb, epsilon_explore=epsilon,
                          periodic_gamma=gamma, rr_warmup_steps=warmup_steps,
                          prior=prior, p_detect=p_detect, p_fa=p_fa)
    policy_w.reset()
    wiql_rewards, wiql_intercepts, wiql_occupied = [], [], []
    wiql_actions, wiql_beliefs, wiql_indices, occ_grid = [], [], [], []
    env_w.reset()
    for t in range(t_steps):
        action = policy_w.select(t, dt_us)
        obs    = env_w.step(action)
        policy_w.update(t, action, obs, dt_us)
        si = so = 0; sr = 0.0; occ_row = []
        for b in range(n_bands):
            ta = obs[b]["true_active"]; sc = obs[b]["scanned"]; de = obs[b]["detected"]
            occ_row.append(ta)
            if ta: so += 1
            if ta and sc and de:        si += 1; sr += 1.0
            elif sc and de and not ta:  sr -= 0.1
        wiql_rewards.append(sr / max(k_scan, 1))
        wiql_intercepts.append(si); wiql_occupied.append(so)
        wiql_actions.append(action)
        wiql_beliefs.append(policy_w.bt.get_all().copy())
        raw = policy_w.sched.compute_indices(policy_w.bt.get_all())
        wiql_indices.append(np.where(np.isinf(raw), 5.0, raw).copy())
        occ_grid.append(occ_row)

    env_b = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed, p_false_alarm=p_fa)
    env_b.set_emitters(emitters); env_b.reset()
    base_pol = (RandomPolicy(n_bands=n_bands, k_scan=k_scan, seed=seed + 1)
                if baseline == "Random"
                else RoundRobinPolicy(n_bands=n_bands, k_scan=k_scan))
    base_rewards, base_intercepts, base_occupied = [], [], []
    for t in range(t_steps):
        action = base_pol.select_arms(); obs = env_b.step(action)
        si = so = 0; sr = 0.0
        for b in range(n_bands):
            ta = obs[b]["true_active"]; sc = obs[b]["scanned"]; de = obs[b]["detected"]
            if ta: so += 1
            if ta and sc and de:        si += 1; sr += 1.0
            elif sc and de and not ta:  sr -= 0.1
        base_rewards.append(sr / max(k_scan, 1))
        base_intercepts.append(si); base_occupied.append(so)

    return {
        "wiql_rate":       sum(wiql_intercepts) / max(sum(wiql_occupied), 1),
        "base_rate":       sum(base_intercepts) / max(sum(base_occupied),  1),
        "wiql_rewards":    wiql_rewards, "base_rewards":    base_rewards,
        "wiql_intercepts": wiql_intercepts, "base_intercepts": base_intercepts,
        "wiql_occupied":   wiql_occupied, "wiql_actions":    wiql_actions,
        "wiql_beliefs":    wiql_beliefs,  "wiql_indices":    wiql_indices,
        "occ_grid":        occ_grid,
    }


def cum_rate(intercepts, occupied):
    rates = []; ci = co = 0
    for i, o in zip(intercepts, occupied):
        ci += i; co += o; rates.append(ci / max(co, 1))
    return rates


def gamma_sweep(emitters, n_bands, k_scan, t_steps, seed, dt_us,
                c_ucb, epsilon, warmup_steps, prior, p_detect, p_fa):
    gammas = [0.0, 1.0, 3.0, 5.0, 10.0]
    rates = [run_episode_core(emitters, n_bands, k_scan, t_steps, seed, dt_us,
                              c_ucb, epsilon, g, warmup_steps, prior, p_detect,
                              p_fa, "Round-Robin")["wiql_rate"]
             for g in gammas]
    return gammas, rates


def metric_card(label, value_str, delta=None, delta_label=""):
    delta_html = ""
    if delta is not None:
        cls = "positive" if delta >= 0 else "negative"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = (f'<div class="metric-delta {cls}">'
                      f'{sign} {abs(delta):.3f} {delta_label}</div>')
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value_str}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ── Heatmap colormap: off-white → light-grey → orange → green ─────────────────
_CMAP4 = ListedColormap(["#f8f9fa", "#dfe6e9", "#e67e22", "#27ae60"])


def make_heatmap(arr, title):
    fh = max(1.6, n_bands * 0.27)
    fig, ax = plt.subplots(figsize=(6.5, fh))
    ax.imshow(np.clip(arr, 0, 2.0), aspect="auto", cmap=_CMAP4,
              vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(n_bands))
    ax.set_yticklabels([f"B{b}" for b in range(n_bands)], fontsize=6)
    ax.set_xticks([])
    ax.set_title(title, fontsize=9, pad=4)
    for sp in ax.spines.values(): sp.set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def bar_fig(values, labels, title, color, accent="#e67e22"):
    fw = max(3.5, len(labels) * 0.44)
    fig, ax = plt.subplots(figsize=(fw, 2.5))
    bars = ax.bar(labels, values, color=color, alpha=0.82, width=0.65,
                  edgecolor="white", linewidth=0.4)
    # Highlight the highest bar in accent colour
    if len(values):
        mx = int(np.argmax(values))
        bars[mx].set_color(accent)
        bars[mx].set_alpha(1.0)
    ax.set_ylim(0, max(float(np.max(values)) * 1.18, 0.1))
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=9, pad=4)
    ax.yaxis.grid(True, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# IDLE STATE
# ═══════════════════════════════════════════════════════════════════════════════
if not run_clicked:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### What you can do")
        st.markdown("""
1. **Pick a scenario** — random occupancy, periodic radar, frequency-agile hopping, or mixed
2. **Tune the scheduler** — UCB exploration, periodic-bias γ, warmup steps
3. **Pick a baseline** — Round-Robin, Random, or Clarkson / Teissier (analytical bounds)
4. **Press ▶ Run** — intercept rates, occupancy heatmap, belief state, priority index, and γ sensitivity

**Experiments:**
- **γ = 0** on Periodic → periodic module off → rate drops
- **warmup = 0** on Periodic → convergence failure (9/30 seeds in paper)
- **N = 32, K = 3** → harder coverage problem
- **Pd = 0.6** → noisy receiver → wider belief uncertainty
- **Mixed** → heterogeneous traffic
        """)
    with col_b:
        st.markdown("#### Visualizations")
        st.markdown("""
| Panel | Description |
|---|---|
| Intercept counter | WIQL-UCB vs baseline, head-to-head |
| Cumulative rate | Who pulls ahead, and when |
| Occupancy heatmap | Ground-truth + scan decisions (WIQL vs baseline) |
| Belief & priority | Scheduler internal state at episode end |
| γ sensitivity | Intercept rate across 5 γ values |
        """)
        st.info(
            "Default parameters match the paper's values. "
            "Change them to explore the design space.",
            icon="ℹ️",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# RUN STATE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    DT_PERIODIC = 100.0
    DT_DEFAULT  = 1_000.0
    dt_us = DT_PERIODIC if emitter_type == "periodic" else DT_DEFAULT

    with st.spinner("Running episode..."):
        rng = np.random.default_rng(seed)
        emitters = build_custom_emitters(
            emitter_type, n_emitters, n_bands, duty_cycle,
            period_slots, hop_size, dt_us, rng)
        results = run_episode_core(
            emitters, n_bands, k_scan, t_steps, seed, dt_us,
            c_ucb, epsilon, gamma, warmup_steps, prior, p_detect, p_fa,
            baseline=_effective_baseline)

    T   = t_steps
    n   = n_bands
    wr  = results["wiql_rate"]
    br  = results["base_rate"]
    delta = wr - br
    base_label = baseline_choice

    if _baseline_note:
        st.markdown(f'<div class="warn-box">ℹ {_baseline_note}</div>',
                    unsafe_allow_html=True)

    # ── Headline metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("WIQL-UCB Intercept Rate", f"{wr:.3f}")
    with m2:
        metric_card(f"{base_label} Rate", f"{br:.3f}",
                    delta=delta, delta_label=f"vs {base_label}")
    with m3:
        metric_card("WIQL Advantage", f"{delta:+.3f}")
    with m4:
        scan_frac = k_scan / n_bands * 100
        metric_card("Scan Fraction K/N", f"{scan_frac:.0f}%")

    st.markdown(
        f'<div class="run-note">'
        f'Prototype run · single seed · {t_steps} steps · {emitter_type} emitters · '
        f'N={n_bands} bands · K={k_scan} · γ={gamma} · warmup={warmup_steps} — '
        f'<a href="https://github.com/Vedant-lab-15/smartscan-ps26055/blob/main/results/headline_numbers.json">'
        f'full N=30 results</a>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Cumulative intercept rate ─────────────────────────────────────────────
    import pandas as pd
    st.markdown(f"#### Cumulative Intercept Rate — WIQL-UCB vs {base_label}")

    wiql_cum = cum_rate(results["wiql_intercepts"], results["wiql_occupied"])
    base_cum = cum_rate(results["base_intercepts"], results["wiql_occupied"])
    step_every = max(1, T // 200)
    xs = list(range(0, T, step_every))
    cum_df = pd.DataFrame({
        "Step": xs,
        "WIQL-UCB": [wiql_cum[i] for i in xs],
        base_label: [base_cum[i]  for i in xs],
    }).set_index("Step")
    st.line_chart(cum_df, use_container_width=True, height=200,
                  color=["#e67e22", "#b2bec3"])
    st.caption("Higher = better. The gap opens as WIQL-UCB learns which bands are active.")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Band occupancy heatmaps ───────────────────────────────────────────────
    st.markdown("#### Band Occupancy — Ground Truth + Scan Decisions")
    sample_steps = min(120, T)
    step_h = max(1, T // sample_steps)
    sampled = list(range(0, T, step_h))[:sample_steps]

    occ_arr = np.array([[1.0 if results["occ_grid"][t][b] else 0.0
                         for t in sampled] for b in range(n)])
    wiql_scan_arr = occ_arr.copy()
    base_scan_arr = occ_arr.copy()
    for ti, t in enumerate(sampled):
        w_act = results["wiql_actions"][t]
        if _effective_baseline == "Random":
            rng_b = np.random.default_rng(t + seed * 1000)
            b_act = set(rng_b.choice(n, size=k_scan, replace=False).tolist())
        else:
            cursor = (t * k_scan) % n
            b_act = {(cursor + j) % n for j in range(k_scan)}
        for b in range(n):
            if b in w_act:
                wiql_scan_arr[b, ti] = 2.0 if occ_arr[b, ti] > 0.5 else 1.5
            if b in b_act:
                base_scan_arr[b, ti] = 2.0 if occ_arr[b, ti] > 0.5 else 1.5

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_w = make_heatmap(wiql_scan_arr, f"WIQL-UCB  ■ hit  ■ active  ░ scanned-idle")
        st.pyplot(fig_w, use_container_width=True); plt.close(fig_w)
    with col_h2:
        fig_b = make_heatmap(base_scan_arr, f"{base_label}  ■ hit  ■ active  ░ scanned-idle")
        st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
    st.caption(
        "Row = band · Column = time step (downsampled) · "
        "Orange = active band · Green = scanned + intercepted · "
        "WIQL-UCB concentrates on active bands."
    )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Belief & priority ─────────────────────────────────────────────────────
    st.markdown("#### Final Belief State & Priority Index")
    final_belief = results["wiql_beliefs"][-1]
    final_index  = results["wiql_indices"][-1]
    band_labels  = [f"B{b}" for b in range(n)]

    col_bl, col_pr = st.columns(2)
    with col_bl:
        fig_b2 = bar_fig(final_belief, band_labels,
                         "Belief  b_i = P(band occupied)", "#74b9ff")
        st.pyplot(fig_b2, use_container_width=True); plt.close(fig_b2)
        st.caption(
            "Converges toward true occupancy. "
            "Highest bar (orange) = most likely active band."
        )
    with col_pr:
        fig_p2 = bar_fig(final_index, band_labels,
                         "Priority index  I_i = W·b + UCB", "#74b9ff")
        st.pyplot(fig_p2, use_container_width=True); plt.close(fig_p2)
        st.caption(
            "Highest bar (orange) = band scanned next. "
            "Combines Whittle index, belief, and UCB exploration bonus."
        )
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Rolling reward ────────────────────────────────────────────────────────
    st.markdown(f"#### Rolling Reward — WIQL-UCB vs {base_label} (20-step window)")
    w_roll = pd.Series(results["wiql_rewards"]).rolling(20, min_periods=1).mean()
    b_roll = pd.Series(results["base_rewards"]).rolling(20, min_periods=1).mean()
    roll_df = pd.DataFrame({
        "Step": list(range(T)),
        "WIQL-UCB": w_roll.tolist(),
        base_label: b_roll.tolist(),
    }).set_index("Step").iloc[::step_every]
    st.line_chart(roll_df, use_container_width=True, height=180,
                  color=["#e67e22", "#b2bec3"])
    st.caption("Positive spikes = successful intercepts.")
    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── γ sensitivity panel ───────────────────────────────────────────────────
    st.markdown("#### Parameter Sensitivity — Intercept Rate vs Periodic-Bias Weight γ")
    st.caption(
        "5-point sweep, all other parameters fixed. "
        "γ=0 = periodic module off; γ=5 = paper's recommended value."
    )
    with st.spinner("Computing γ sweep..."):
        gs, gr = gamma_sweep(emitters, n_bands, k_scan,
                             min(t_steps, 400), seed, dt_us,
                             c_ucb, epsilon, warmup_steps, prior, p_detect, p_fa)
    sens_df = pd.DataFrame({"γ": gs, "Intercept Rate": gr}).set_index("γ")
    st.line_chart(sens_df, use_container_width=True, height=170,
                  color=["#e67e22"])
    idx_nearest = int(np.argmin([abs(gv - gamma) for gv in gs]))
    st.caption(
        f"Current γ = {gamma}  →  nearest sweep point γ={gs[idx_nearest]:.0f}"
        f"  →  rate = {gr[idx_nearest]:.3f}"
    )

    # ── Periodic emitter info ─────────────────────────────────────────────────
    if emitter_type in ("periodic", "mixed"):
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown("#### Periodic Emitter Configuration")
        T_us = period_slots * dt_us
        sigma_us = T_us * 0.08
        st.info(
            f"**User-defined:** T = {T_us:.0f} μs ({period_slots} slots), "
            f"σ = {sigma_us:.0f} μs (CoV = 8%)\n\n"
            f"**TSRD-grounded reference:** "
            f"P1 T={_TPC[0]['T_us']} μs / σ={_TPC[0]['sigma_T_us']} μs · "
            f"P2 T={_TPC[1]['T_us']} μs / σ={_TPC[1]['sigma_T_us']} μs\n\n"
            "Welford online estimator learns PRI from inter-arrivals. "
            "Convergence: 30/30 seeds with warmup ≥ 50 (worst-case rate 0.419).",
            icon="🎯",
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
  <div>Coding Saints · Team ID 120303 · SIH 2026 · PS 26055 (DRDO)</div>
  <div>
    <a href="https://github.com/Vedant-lab-15/smartscan-ps26055" target="_blank">
      GitHub — smartscan-ps26055
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
