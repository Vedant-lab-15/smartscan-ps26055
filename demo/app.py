"""
Smart Scan Strategy — Interactive Prototype
PS 26055 · SIH 2026 (DRDO) · Coding Saints, Team ID 120303

Whittle-inspired priority scheduler for EW spectrum scanning.
A judge can drive the scenario, tune scheduler parameters, pick a
baseline, and see how the system responds.

Run locally:  streamlit run demo/app.py
"""
from __future__ import annotations

import sys
import math
import pathlib

import numpy as np
import streamlit as st

# ── path resolution (works locally, on Streamlit Cloud, or from repo root) ───
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
    TSRD_PERIODIC_CLASSES, TSRD_PERIODIC_CLASSES as _TPC,
)
from src.evaluation.runner import WIQLPolicy, policy_round_robin
from src.baselines.random_policy import RandomPolicy
from src.baselines.round_robin import RoundRobinPolicy

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Scan — PS 26055 Prototype",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.metric-card {
    background: #161b22; border-radius: 10px; padding: 14px 18px;
    text-align: center; border: 1px solid #30363d; margin-bottom: 6px;
}
.metric-label { font-size: 0.7rem; color: #8b949e; text-transform: uppercase;
                letter-spacing: 1px; margin-bottom: 4px; }
.metric-value { font-size: 1.8rem; font-weight: 700; }
.green  { color: #3fb950; }
.blue   { color: #58a6ff; }
.orange { color: #f0883e; }
.red    { color: #f85149; }
.grey   { color: #8b949e; }
.delta-pos { color: #3fb950; font-size: 0.95rem; font-weight: 600; }
.delta-neg { color: #f85149; font-size: 0.95rem; font-weight: 600; }
hr.rule { border: none; border-top: 1px solid #30363d; margin: 1rem 0; }
.warn-box { background:#2d1a00; border:1px solid #f0883e; border-radius:8px;
            padding:8px 14px; font-size:0.82rem; color:#f0883e; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — three collapsible control layers
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 📡 Smart Scan")
    st.caption("PS 26055 · Coding Saints (ID 120303)")
    st.markdown("---")

    # ── Layer 1: Scenario editor ──────────────────────────────────────────────
    with st.expander("🌍 Layer 1 — Scenario Editor", expanded=True):
        emitter_type = st.radio(
            "Emitter type",
            ["random", "periodic", "frequency-agile", "mixed"],
            index=0,
            horizontal=True,
            help="random: Bernoulli occupancy. periodic: TSRD-grounded radar PRI. "
                 "freq-agile: hopping emitter. mixed: all three types.",
        )
        n_emitters = st.slider("Number of emitters", 1, 5, 3,
                               help="Total emitters placed in the environment.")
        duty_cycle = st.slider("Emitter duty cycle", 0.1, 0.9, 0.5, 0.05,
                               help="P(active per slot) for random/mixed emitters.")
        period_slots = st.slider(
            "Period (slots, periodic/mixed)", 5, 50, 10,
            help="Nominal PRI in time slots. dt=100μs for periodic, 1ms for others."
        )
        hop_size = st.slider("Hop set size (freq-agile/mixed)", 2, 8, 4,
                             help="Number of bands the agile emitter cycles through.")
        n_bands = st.slider("Frequency bands (N)", 4, 32, 8, 2)
        k_scan  = st.slider("Simultaneous scans (K)", 1, min(5, n_bands - 1), 3)
        t_steps = st.slider("Time steps", 200, 2000, 500, 100)
        seed    = st.slider("Random seed", 0, 99, 42)

    # ── Layer 2: Scheduler parameters ────────────────────────────────────────
    with st.expander("⚙️ Layer 2 — Scheduler Configuration", expanded=False):
        c_ucb        = st.slider("UCB exploration constant c", 0.1, 5.0, 1.0, 0.1,
                                 help="Higher c = more exploration.")
        epsilon      = st.slider("UCB floor ε", 0.00, 0.10, 0.01, 0.005,
                                 help="Minimum UCB bonus — prevents band lockout.")
        gamma        = st.slider("Periodic-bias weight γ", 0.0, 10.0, 5.0, 0.5,
                                 help="γ=0 disables the periodic module. Paper default: 5.")
        warmup_steps = st.slider(
            "RR warmup steps", 0, 200, 50, 10,
            help="Round-robin pre-phase before WIQL takes over. "
                 "Set to 0 to reproduce the convergence failure (9/30 seeds)."
        )
        prior    = st.slider("Belief prior b₀", 0.1, 0.9, 0.5, 0.05,
                             help="Initial P(band occupied) before any observation.")
        p_detect = st.slider("Detection probability Pd", 0.5, 1.0, 0.9, 0.05,
                             help="P(detect | scanned ∧ occupied).")
        p_fa     = st.slider("False-alarm rate Pfa", 0.00, 0.10, 0.01, 0.005,
                             help="P(detect | scanned ∧ idle).")

        st.caption(
            "**Try these to reproduce paper findings:**\n"
            "- γ=0 → periodic module off → see intercept rate drop\n"
            "- warmup=0 → convergence failure on Periodic scenario\n"
            "- c=0.1 → under-exploration → misses dynamic emitters\n"
            "- c=5.0 → over-exploration → slow convergence"
        )

    # ── Layer 3: Baseline selector ────────────────────────────────────────────
    with st.expander("📊 Layer 3 — Baseline Selector", expanded=False):
        baseline_choice = st.selectbox(
            "Compare WIQL-UCB against",
            ["Round-Robin", "Random", "Clarkson (2003)", "Teissier (2026)"],
            index=0,
            help="Clarkson and Teissier are analytical bounds — "
                 "not implemented as schedulers, falls back to round-robin.",
        )
        if baseline_choice in ("Clarkson (2003)", "Teissier (2026)"):
            st.markdown(
                '<div class="warn-box">⚠️ '
                f'{baseline_choice} is an analytical bound, not a runnable scheduler. '
                'Showing Round-Robin as the closest implemented baseline.</div>',
                unsafe_allow_html=True,
            )
            _effective_baseline = "Round-Robin"
            _baseline_note = f"{baseline_choice} → falls back to Round-Robin"
        else:
            _effective_baseline = baseline_choice
            _baseline_note = None

    st.markdown("---")
    run_clicked = st.button("▶ Run Episode", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        "**Code & paper:** "
        "[github.com/Vedant-lab-15/smartscan-ps26055]"
        "(https://github.com/Vedant-lab-15/smartscan-ps26055)"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_custom_emitters(
    emitter_type: str, n_emitters: int, n_bands: int,
    duty_cycle: float, period_slots: int, hop_size: int,
    dt_us: float, rng: np.random.Generator,
) -> list[Emitter]:
    """Build emitter list from prototype controls."""
    emitters: list[Emitter] = []

    if emitter_type == "random":
        for i in range(n_emitters):
            emitters.append(Emitter(
                emitter_type="background",
                band=int(rng.integers(0, n_bands)),
                p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)),
                label=f"bg_{i}",
            ))

    elif emitter_type == "periodic":
        cls = _TPC[0]  # P1-TSRD
        T_slots = period_slots  # user-defined, not TSRD-locked
        sigma_slots = max(0.5, T_slots * 0.08)
        for i in range(n_emitters):
            emitters.append(Emitter(
                emitter_type="periodic",
                band=int(rng.integers(0, n_bands)),
                T_slots=float(T_slots),
                sigma_T_slots=float(sigma_slots),
                snr_db=float(rng.uniform(12.0, 22.0)),
                label=f"periodic_{i}",
            ))

    elif emitter_type == "frequency-agile":
        for i in range(n_emitters):
            start = int(rng.integers(0, max(1, n_bands - hop_size)))
            hops  = list(range(start, min(start + hop_size, n_bands)))
            emitters.append(Emitter(
                emitter_type="freq_agile",
                band=hops[0],
                hop_bands=hops,
                p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)),
                label=f"agile_{i}",
            ))

    else:  # mixed
        per_type = max(1, n_emitters // 3)
        remainder = n_emitters - per_type * 3
        # background
        for i in range(per_type):
            emitters.append(Emitter(
                emitter_type="background",
                band=int(rng.integers(0, n_bands)),
                p_active=duty_cycle,
                snr_db=float(rng.uniform(5.0, 18.0)),
                label=f"bg_{i}",
            ))
        # periodic
        for i in range(per_type):
            T = period_slots
            emitters.append(Emitter(
                emitter_type="periodic",
                band=int(rng.integers(0, n_bands)),
                T_slots=float(T),
                sigma_T_slots=max(0.5, T * 0.08),
                snr_db=float(rng.uniform(12.0, 22.0)),
                label=f"periodic_{i}",
            ))
        # freq-agile
        for i in range(per_type + remainder):
            start = int(rng.integers(0, max(1, n_bands - hop_size)))
            hops  = list(range(start, min(start + hop_size, n_bands)))
            emitters.append(Emitter(
                emitter_type="freq_agile",
                band=hops[0], hop_bands=hops,
                p_active=duty_cycle,
                snr_db=float(rng.uniform(8.0, 20.0)),
                label=f"agile_{i}",
            ))

    return emitters


def run_episode_core(
    emitters, n_bands, k_scan, t_steps, seed, dt_us,
    c_ucb, epsilon, gamma, warmup_steps, prior, p_detect, p_fa,
    baseline: str,
) -> dict:
    """Run one full episode. Returns per-step logs for WIQL and baseline."""

    # ── WIQL-UCB ──────────────────────────────────────────────────────────────
    env_w = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed,
                       p_false_alarm=p_fa)
    env_w.set_emitters(emitters)

    policy_w = WIQLPolicy(
        n_bands=n_bands, k=k_scan, use_periodic_bias=(gamma > 0),
        c_ucb=c_ucb, epsilon_explore=epsilon,
        periodic_gamma=gamma, rr_warmup_steps=warmup_steps,
        prior=prior, p_detect=p_detect, p_fa=p_fa,
    )
    policy_w.reset()

    wiql_rewards: list[float] = []
    wiql_intercepts: list[int] = []
    wiql_occupied: list[int] = []
    wiql_actions: list[set] = []
    wiql_beliefs: list[np.ndarray] = []
    wiql_indices: list[np.ndarray] = []
    occ_grid: list[list[bool]] = []

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
            if ta and sc and de:   si += 1; sr += 1.0
            elif sc and de and not ta: sr -= 0.1

        wiql_rewards.append(sr / max(k_scan, 1))
        wiql_intercepts.append(si)
        wiql_occupied.append(so)
        wiql_actions.append(action)
        wiql_beliefs.append(policy_w.bt.get_all().copy())
        raw_idx = policy_w.sched.compute_indices(policy_w.bt.get_all())
        wiql_indices.append(np.where(np.isinf(raw_idx), 5.0, raw_idx).copy())
        occ_grid.append(occ_row)

    # ── Baseline ──────────────────────────────────────────────────────────────
    env_b = RenewalEnv(n_bands=n_bands, dt_us=dt_us, episode_len=t_steps,
                       k_scan=k_scan, seed=seed, p_false_alarm=p_fa)
    env_b.set_emitters(emitters)
    env_b.reset()

    if baseline == "Random":
        base_pol = RandomPolicy(n_bands=n_bands, k_scan=k_scan, seed=seed + 1)
    else:
        base_pol = RoundRobinPolicy(n_bands=n_bands, k_scan=k_scan)

    base_rewards: list[float] = []
    base_intercepts: list[int] = []
    base_occupied: list[int] = []

    for t in range(t_steps):
        action = base_pol.select_arms()
        obs    = env_b.step(action)

        si = so = 0; sr = 0.0
        for b in range(n_bands):
            ta = obs[b]["true_active"]; sc = obs[b]["scanned"]; de = obs[b]["detected"]
            if ta: so += 1
            if ta and sc and de:   si += 1; sr += 1.0
            elif sc and de and not ta: sr -= 0.1

        base_rewards.append(sr / max(k_scan, 1))
        base_intercepts.append(si)
        base_occupied.append(so)

    total_occ_w = max(sum(wiql_occupied), 1)
    total_occ_b = max(sum(base_occupied),  1)

    return {
        "wiql_rate":       sum(wiql_intercepts) / total_occ_w,
        "base_rate":       sum(base_intercepts) / total_occ_b,
        "wiql_rewards":    wiql_rewards,
        "base_rewards":    base_rewards,
        "wiql_intercepts": wiql_intercepts,
        "base_intercepts": base_intercepts,
        "wiql_occupied":   wiql_occupied,
        "wiql_actions":    wiql_actions,
        "wiql_beliefs":    wiql_beliefs,
        "wiql_indices":    wiql_indices,
        "occ_grid":        occ_grid,
    }


def cum_rate(intercepts, occupied):
    rates = []; ci = 0; co = 0
    for i, o in zip(intercepts, occupied):
        ci += i; co += o
        rates.append(ci / max(co, 1))
    return rates


def gamma_sweep(emitters, n_bands, k_scan, t_steps, seed, dt_us,
                c_ucb, epsilon, warmup_steps, prior, p_detect, p_fa):
    """Quick 5-point γ sweep for the sensitivity panel."""
    gammas = [0.0, 1.0, 3.0, 5.0, 10.0]
    rates = []
    for g in gammas:
        r = run_episode_core(
            emitters, n_bands, k_scan, t_steps, seed, dt_us,
            c_ucb, epsilon, g, warmup_steps, prior, p_detect, p_fa,
            baseline="Round-Robin",
        )
        rates.append(r["wiql_rate"])
    return gammas, rates


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# 📡 Smart Scan Strategy — Interactive Prototype")
st.caption(
    "Whittle-inspired priority scheduler for EW spectrum scanning · "
    "Coding Saints (ID 120303) · SIH 2026 DRDO · "
    "Adjust the controls and press ▶ Run to see how the system responds."
)
st.markdown('<hr class="rule">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# IDLE STATE
# ═══════════════════════════════════════════════════════════════════════════════
if not run_clicked:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### What you can do")
        st.markdown("""
1. **Pick a scenario** — random occupancy, periodic radar emitters, frequency-agile hopping, or a mix
2. **Tune the scheduler** — adjust UCB exploration, periodic-bias weight γ, warmup steps
3. **Pick a baseline** — round-robin, random, or Clarkson / Teissier (analytical bounds)
4. **Press ▶ Run** — see the intercept rates, occupancy heatmap, belief state, and priority index

**Try these experiments:**
- Set **γ = 0** on a Periodic scenario → periodic module turns off → watch the rate drop
- Set **warmup = 0** on a Periodic scenario → convergence failure (9/30 seeds in the paper)
- Set **N = 32, K = 3** → harder coverage problem
- Set **Pd = 0.6** → noisy receiver → see belief uncertainty increase
- Mix emitter types → see the scheduler handle heterogeneous traffic
        """)
    with col_b:
        st.markdown("### What the visualizations show")
        st.markdown("""
| Panel | What it shows |
|---|---|
| 📊 Intercept counter | WIQL-UCB vs baseline, head-to-head |
| 📈 Cumulative rate | Who pulls ahead, and when |
| 🗺️ Occupancy heatmap | Ground-truth activity + scan decisions |
| 🧠 Belief & priority | Scheduler's internal state at episode end |
| 🎛️ γ sensitivity | How intercept rate responds to the periodic-bias weight |
        """)
        st.info("Parameters match the paper's defaults. Change them to explore the design space.")

# ═══════════════════════════════════════════════════════════════════════════════
# RUN STATE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    # Determine dt_us
    DT_PERIODIC = 100.0
    DT_DEFAULT  = 1_000.0
    dt_us = DT_PERIODIC if emitter_type == "periodic" else DT_DEFAULT

    with st.spinner("Running episode..."):
        rng = np.random.default_rng(seed)
        emitters = build_custom_emitters(
            emitter_type, n_emitters, n_bands, duty_cycle,
            period_slots, hop_size, dt_us, rng,
        )
        results = run_episode_core(
            emitters, n_bands, k_scan, t_steps, seed, dt_us,
            c_ucb, epsilon, gamma, warmup_steps, prior, p_detect, p_fa,
            baseline=_effective_baseline,
        )

    T   = t_steps
    n   = n_bands
    wr  = results["wiql_rate"]
    br  = results["base_rate"]
    delta = wr - br
    base_label = baseline_choice  # display name (may differ from effective)

    # ── Baseline fallback note ────────────────────────────────────────────────
    if _baseline_note:
        st.markdown(
            f'<div class="warn-box">ℹ️ {_baseline_note}</div>',
            unsafe_allow_html=True,
        )

    # ── Headline metrics ──────────────────────────────────────────────────────
    st.markdown("### 📊 Episode Results")
    st.caption(
        f"Single-seed prototype run · {t_steps} steps · {emitter_type} emitters · "
        f"N={n_bands} bands · K={k_scan} · γ={gamma} · warmup={warmup_steps} steps · "
        "Adjust controls and re-run to explore — "
        "full N=30 results: [headline_numbers.json]"
        "(https://github.com/Vedant-lab-15/smartscan-ps26055/blob/main/results/headline_numbers.json)"
    )

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
          <div class="metric-label">{base_label} Rate</div>
          <div class="metric-value orange">{br:.3f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        cls = "delta-pos" if delta >= 0 else "delta-neg"
        arrow = "▲" if delta >= 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">WIQL Advantage</div>
          <div class="metric-value {cls}">{arrow} {abs(delta):.3f}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        scan_frac = k_scan / n_bands * 100
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Scan Fraction K/N</div>
          <div class="metric-value blue">{scan_frac:.0f}%</div>
          <div class="metric-label">{k_scan}/{n_bands} bands/step</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Cumulative intercept rate ─────────────────────────────────────────────
    import pandas as pd
    st.markdown(f"### 📈 Cumulative Intercept Rate — WIQL-UCB vs {base_label}")

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
                  color=["#3fb950", "#8b949e"])
    st.caption("Higher = better. Watch for the gap to open as WIQL-UCB learns which bands are active.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Band occupancy heatmaps ───────────────────────────────────────────────
    st.markdown("### 🗺️ Band Occupancy — Ground Truth + Scan Decisions")

    sample_steps = min(120, T)
    step_h = max(1, T // sample_steps)
    sampled = list(range(0, T, step_h))[:sample_steps]

    occ_arr = np.array([[1.0 if results["occ_grid"][t][b] else 0.0
                         for t in sampled] for b in range(n)])

    wiql_scan_arr = occ_arr.copy()
    base_scan_arr = occ_arr.copy()
    for ti, t in enumerate(sampled):
        w_act = results["wiql_actions"][t]
        # Reconstruct baseline action for display
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

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap4 = ListedColormap(["#0d1117", "#21262d", "#f0883e", "#3fb950"])

    def make_heatmap(arr, title, fig_n):
        fh = max(1.8, n * 0.28)
        fig, ax = plt.subplots(figsize=(7, fh), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.imshow(np.clip(arr, 0, 2.0), aspect="auto", cmap=cmap4,
                  vmin=0, vmax=2, interpolation="nearest")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"B{b}" for b in range(n)], color="#8b949e", fontsize=6)
        ax.set_xticks([])
        ax.set_title(title, color="#e6edf3", fontsize=8, pad=3)
        for sp in ax.spines.values(): sp.set_visible(False)
        fig.tight_layout(pad=0.3)
        return fig

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        fig_w = make_heatmap(wiql_scan_arr, "WIQL-UCB  🟢hit  🟠active  ▒scanned-idle", 1)
        st.pyplot(fig_w, use_container_width=True); plt.close(fig_w)
    with col_h2:
        fig_b = make_heatmap(base_scan_arr, f"{base_label}  🟢hit  🟠active  ▒scanned-idle", 2)
        st.pyplot(fig_b, use_container_width=True); plt.close(fig_b)
    st.caption(
        "Row = band · Column = time step (downsampled). "
        "WIQL-UCB concentrates on active bands; "
        f"{base_label} distributes uniformly."
    )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Belief state & priority index ─────────────────────────────────────────
    st.markdown("### 🧠 Final Belief State & Priority Index")
    final_belief = results["wiql_beliefs"][-1]
    final_index  = results["wiql_indices"][-1]
    band_labels  = [f"B{b}" for b in range(n)]

    def bar_fig(values, labels, title, color):
        fw = max(3.5, n * 0.45)
        fig, ax = plt.subplots(figsize=(fw, 2.6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.bar(labels, values, color=color, alpha=0.85, width=0.6)
        ax.set_ylim(0, max(float(np.max(values)) * 1.15, 0.1))
        ax.tick_params(colors="#8b949e", labelsize=6)
        ax.set_title(title, color="#e6edf3", fontsize=8, pad=3)
        for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        fig.tight_layout(pad=0.4)
        return fig

    col_bl, col_pr = st.columns(2)
    with col_bl:
        fig_b2 = bar_fig(final_belief, band_labels, "Belief b_i = P(band occupied)", "#58a6ff")
        st.pyplot(fig_b2, use_container_width=True); plt.close(fig_b2)
        st.caption("Converges toward true occupancy pattern. High belief → prioritised for scanning.")
    with col_pr:
        fig_p2 = bar_fig(final_index, band_labels, "Priority index I_i = W·b + UCB", "#3fb950")
        st.pyplot(fig_p2, use_container_width=True); plt.close(fig_p2)
        st.caption("Highest bar = band scanned next. Combines Whittle index, belief, and UCB bonus.")

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── Rolling reward ────────────────────────────────────────────────────────
    st.markdown(f"### 🏆 Rolling Reward — WIQL-UCB vs {base_label} (20-step window)")
    w_roll = pd.Series(results["wiql_rewards"]).rolling(20, min_periods=1).mean()
    b_roll = pd.Series(results["base_rewards"]).rolling(20, min_periods=1).mean()
    roll_df = pd.DataFrame({
        "Step": list(range(T)),
        "WIQL-UCB": w_roll.tolist(),
        base_label: b_roll.tolist(),
    }).set_index("Step").iloc[::step_every]
    st.line_chart(roll_df, use_container_width=True, height=180,
                  color=["#3fb950", "#8b949e"])
    st.caption(
        "20-step rolling reward. Positive spikes = intercepts. "
        "Full N=30 results: "
        "[headline_numbers.json](https://github.com/Vedant-lab-15/smartscan-ps26055/blob/main/results/headline_numbers.json)"
    )

    st.markdown('<hr class="rule">', unsafe_allow_html=True)

    # ── γ sensitivity panel ────────────────────────────────────────────────────
    st.markdown("### 🎛️ Parameter Sensitivity — Intercept Rate vs Periodic-Bias Weight γ")
    st.caption(
        "Shows how the intercept rate changes across 5 values of γ, "
        "holding all other parameters fixed. "
        "γ=0 = periodic module off; γ=5 is the paper's recommended value."
    )

    with st.spinner("Computing γ sweep (5 points)..."):
        gs, gr = gamma_sweep(
            emitters, n_bands, k_scan,
            min(t_steps, 400),  # cap sweep at 400 steps for speed
            seed, dt_us, c_ucb, epsilon, warmup_steps, prior, p_detect, p_fa,
        )

    sens_df = pd.DataFrame({"γ": gs, "Intercept Rate": gr}).set_index("γ")
    st.line_chart(sens_df, use_container_width=True, height=180,
                  color=["#f0883e"])
    # Annotate current γ
    current_g_rate = None
    for gv, grate in zip(gs, gr):
        if abs(gv - gamma) < 0.01:
            current_g_rate = grate
    if current_g_rate is not None:
        st.caption(f"Current γ = {gamma} → Intercept Rate = {current_g_rate:.3f}")
    else:
        # Interpolate
        idx_nearest = int(np.argmin([abs(gv - gamma) for gv in gs]))
        st.caption(f"Nearest γ = {gs[idx_nearest]} → Rate = {gr[idx_nearest]:.3f}  "
                   f"(current γ={gamma} is between sweep points)")

    # Periodic emitter info
    if emitter_type in ("periodic", "mixed"):
        st.markdown('<hr class="rule">', unsafe_allow_html=True)
        st.markdown("### 🎯 Periodic Emitter Configuration")
        T_us = period_slots * dt_us
        sigma_us = T_us * 0.08
        st.info(
            f"**User-defined periodic emitter:** T = {T_us:.0f} μs ({period_slots} slots at dt={dt_us:.0f} μs), "
            f"σ = {sigma_us:.0f} μs (CoV = 8%)\n\n"
            f"**TSRD-grounded reference values:**\n"
            f"- P1: T = {_TPC[0]['T_us']} μs, σ = {_TPC[0]['sigma_T_us']} μs\n"
            f"- P2: T = {_TPC[1]['T_us']} μs, σ = {_TPC[1]['sigma_T_us']} μs\n\n"
            "The Welford estimator learns the PRI online from observed inter-arrival times. "
            "Convergence: 30/30 seeds (with warmup ≥ 50 steps, worst-case 0.419)."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="rule">', unsafe_allow_html=True)
st.markdown(
    "This is an interactive prototype of the scheduler described in the paper. "
    "Full results, code, and methodology: "
    "[github.com/Vedant-lab-15/smartscan-ps26055]"
    "(https://github.com/Vedant-lab-15/smartscan-ps26055) · "
    "SIH 2026 PS 26055 (DRDO) · MIT License"
)
