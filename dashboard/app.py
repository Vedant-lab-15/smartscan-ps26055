"""
EW Smart Scan Strategy — Live Demo Dashboard
SIH 2026 · PS 26055 (DRDO) · Alan Turing Institute / Dstl dataset

Two tabs:
  Tab 1: Main scheduler — WIQL-UCB vs Round-Robin on 8 frequency bands
  Tab 2: Periodic interception — Module C catching a periodic radar emitter

Run:  streamlit run dashboard/app.py
"""
import sys
import os
import pathlib
import time
import math

import numpy as np
import streamlit as st

# ── ensure project root is importable ────────────────────────────────────────
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ew_smart_scan.models.pdw_generator import PDWGenerator, PeriodicBandConfig
from ew_smart_scan.env.rf_environment import RFEnvironment
from ew_smart_scan.env.belief_tracker import BeliefTracker
from ew_smart_scan.env.receiver_model import ReceiverModel
from ew_smart_scan.env.wiql_ucb import WIQLScheduler, ACTION_SCAN, ACTION_PASSIVE
from ew_smart_scan.env.periodic_intercept import PeriodicInterceptModule

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EW Smart Scan — SIH 2026",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── dark-theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background: #0d1117; color: #e6edf3; }
.metric-box {
    background: #161b22; border-radius: 8px; padding: 12px 18px;
    margin: 4px; text-align: center;
}
.metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
.metric-value.green { color: #3fb950; }
.metric-value.orange { color: #f0883e; }
.metric-value.red   { color: #f85149; }
.caught-flash {
    background: #388bfd33; border: 2px solid #388bfd;
    border-radius: 8px; padding: 10px; text-align: center;
    font-size: 1.4rem; font-weight: 800; color: #79c0ff;
    animation: pulse 0.5s ease-in-out;
}
@keyframes pulse { 0%{opacity:0.3} 50%{opacity:1} 100%{opacity:0.3} }
.band-cell {
    display: inline-block; width: 56px; height: 56px; margin: 3px;
    border-radius: 8px; text-align: center; line-height: 56px;
    font-size: 0.7rem; font-weight: 600; color: #fff;
    border: 2px solid transparent; transition: all 0.2s;
}
.band-scanning { border: 3px solid #f0f0f0 !important; box-shadow: 0 0 14px #ffffffaa; }
.legend-dot {
    display: inline-block; width: 14px; height: 14px;
    border-radius: 50%; margin-right: 6px; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def belief_to_color(b: float) -> str:
    """Map belief P(occupied) ∈ [0,1] to a hex color (blue→red gradient)."""
    b = max(0.0, min(1.0, b))
    # Cool blue (free) → warm red (occupied)
    r = int(20  + b * 200)
    g = int(100 - b * 90)
    bl = int(200 - b * 160)
    return f"rgb({r},{g},{bl})"


def make_band_html(beliefs: np.ndarray, scanning: set[int],
                   occupied: dict[int, bool] | None = None,
                   band_labels: list[str] | None = None) -> str:
    n = len(beliefs)
    cells = []
    for i in range(n):
        bg    = belief_to_color(beliefs[i])
        label = band_labels[i] if band_labels else f"B{i}"
        scanning_cls = " band-scanning" if i in scanning else ""
        indicator = ""
        if occupied and occupied.get(i):
            indicator = "●"
        cells.append(
            f'<div class="band-cell{scanning_cls}" style="background:{bg};">'
            f'{label}<br><span style="font-size:1rem">{indicator}</span></div>'
        )
    return '<div style="display:flex;flex-wrap:wrap;gap:2px;">' + "".join(cells) + "</div>"


def build_synth_env(n_bands: int, k_scan: int, seed: int,
                    p_emit_hot: float = 0.75, p_emit_cold: float = 0.10,
                    n_hot: int = 3):
    """Build a synthetic environment with n_hot hot bands and the rest cold."""
    band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    p_emit = [p_emit_hot if i < n_hot else p_emit_cold for i in range(n_bands)]
    gen = PDWGenerator(n_bands=n_bands, band_cf_mhz=band_cf, dt_us=10.0,
                       p_emit=p_emit, seed=seed)
    env = RFEnvironment(source=gen, n_bands=n_bands,
                        band_edges_mhz=band_edges, dt_us=10.0)
    env.initialize()
    return env, gen


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 state
# ─────────────────────────────────────────────────────────────────────────────

def init_tab1(n_bands, k_scan, seed):
    """Initialise / reset the Tab 1 simulation state."""
    env, _ = build_synth_env(n_bands, k_scan, seed)

    def _make_receiver(env_):
        bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
        rec = ReceiverModel(rf_env=env_, belief_tracker=bt,
                             n_bands=n_bands, k_scan=k_scan)
        rec.reset()
        return rec, bt

    # WIQL-UCB
    env_w, _ = build_synth_env(n_bands, k_scan, seed)
    rec_w, bt_w = _make_receiver(env_w)
    sched_w = WIQLScheduler(n_bands=n_bands, k_scan=k_scan)

    # Round-robin
    env_r, _ = build_synth_env(n_bands, k_scan, seed)
    rec_r, bt_r = _make_receiver(env_r)
    rr_cursor = [0]

    return dict(
        step=0,
        env_w=env_w, rec_w=rec_w, bt_w=bt_w, sched_w=sched_w,
        env_r=env_r, rec_r=rec_r, bt_r=bt_r, rr_cursor=rr_cursor,
        rewards_w=[], rewards_r=[],
        action_w=set(), action_r=set(),
        true_occ={},
    )


def step_tab1(state, n_bands, k_scan):
    """Advance one time step for both WIQL and RR."""
    t = state["step"]

    # WIQL-UCB
    b_prev_w = state["rec_w"].get_belief().copy()
    action_w = state["sched_w"].select_arms(b_prev_w)
    obs_w, rew_w, _ = state["rec_w"].step(action_w)
    b_next_w = state["rec_w"].get_belief()
    for b in action_w:
        obs = 1 if len(obs_w[b]) > 0 else 0
        state["sched_w"].update(b, ACTION_SCAN, float(obs),
                                float(b_prev_w[b]), float(b_next_w[b]))
    for b in range(n_bands):
        if b not in action_w:
            state["sched_w"].update(b, ACTION_PASSIVE, 0.0,
                                    float(b_prev_w[b]), float(b_next_w[b]))
    state["rewards_w"].append(rew_w)
    state["action_w"] = action_w

    # Round-robin
    c = state["rr_cursor"][0]
    action_r = {(c + j) % n_bands for j in range(k_scan)}
    state["rr_cursor"][0] = (c + 1) % n_bands
    _, rew_r, _ = state["rec_r"].step(action_r)
    state["rewards_r"].append(rew_r)
    state["action_r"] = action_r

    # Ground truth (for occupied indicator)
    state["true_occ"] = state["env_w"].get_true_state(t)
    state["step"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 state
# ─────────────────────────────────────────────────────────────────────────────

def init_tab2(T_us: float, sigma_us: float, seed: int):
    """Initialise Module C demo."""
    n_bands, k_scan = 4, 2
    periodic_band = 0
    dt_us = 50.0  # 10 steps per nominal period

    band_cf = [900.0 + i * 100.0 for i in range(n_bands)]
    band_edges = [(900.0 + i * 100.0, 1000.0 + i * 100.0) for i in range(n_bands)]
    gen = PDWGenerator(
        n_bands=n_bands, band_cf_mhz=band_cf, dt_us=dt_us,
        p_emit=[0.1] * n_bands, seed=seed,
        periodic_config={
            periodic_band: PeriodicBandConfig(period_us=T_us, sigma_us=sigma_us,
                                              t_start_us=T_us)
        },
    )
    env = RFEnvironment(source=gen, n_bands=n_bands,
                        band_edges_mhz=band_edges, dt_us=dt_us)
    env.initialize()

    bt = BeliefTracker(n_bands=n_bands, p_stay_occ=0.9, p_stay_idle=0.85)
    rec = ReceiverModel(rf_env=env, belief_tracker=bt,
                         n_bands=n_bands, k_scan=k_scan)
    rec.reset()
    module = PeriodicInterceptModule(n_bands=n_bands)

    return dict(
        step=0, env=env, rec=rec, bt=bt, module=module,
        n_bands=n_bands, k_scan=k_scan, periodic_band=periodic_band,
        dt_us=dt_us, T_us=T_us, sigma_us=sigma_us,
        intercept_count=0, emission_count=0,
        cyclic_errors=[],
        caught_this_step=False,
        # timeline data for plot (last N steps)
        timeline_t=[], timeline_true=[], timeline_scanned=[],
        timeline_predicted=[], timeline_dwell_lo=[], timeline_dwell_hi=[],
        timeline_caught=[],
    )


def step_tab2(state):
    """Advance one step of the periodic interception demo."""
    t = state["step"]
    t_now_us = t * state["dt_us"]
    pb = state["periodic_band"]
    k = state["k_scan"]
    n = state["n_bands"]
    T = state["T_us"]

    # Action: dwell-window enforcement when estimates exist
    periodic_idx = state["module"].compute_all_indices(t_now_us)
    in_dwell = periodic_idx[pb] > 0.4

    if in_dwell:
        action = {pb}
        others = [b for b in range(n) if b != pb]
        for j in range(k - 1):
            action.add(others[j % len(others)])
    else:
        cursor = (t * k) % n
        action = {(cursor + j) % n for j in range(k)}

    obs_dict, _, _ = state["rec"].step(action)
    true_state = state["env"].get_true_state(t)

    # Ingest pulses
    for b in action:
        for pulse in obs_dict[b]:
            state["module"].ingest_pulse(b, pulse.toa_us)

    # Scoring
    caught = False
    predicted = state["module"].predict_next_arrival(pb)
    mu, sigma = state["module"].estimate_period(pb)

    true_pulse_toa = None
    if true_state.get(pb, False):
        state["emission_count"] += 1
        if pb in action and obs_dict[pb]:
            state["intercept_count"] += 1
            true_pulse_toa = obs_dict[pb][0].toa_us
            caught = True
            if predicted is not None:
                raw_e = abs(true_pulse_toa - predicted)
                cyc_e = min(raw_e, abs(T - raw_e))
                state["cyclic_errors"].append(cyc_e)

    state["caught_this_step"] = caught

    # Update timeline (keep last 80 steps)
    state["timeline_t"].append(t_now_us)
    state["timeline_true"].append(true_pulse_toa)
    state["timeline_scanned"].append(pb in action)
    state["timeline_predicted"].append(predicted)

    if mu is not None and sigma is not None:
        t_last = state["module"]._estimators[pb].t_last
        t_exp = (t_last + mu) if t_last is not None else None
        half = 3 * sigma
        state["timeline_dwell_lo"].append((t_exp - half) if t_exp else None)
        state["timeline_dwell_hi"].append((t_exp + half) if t_exp else None)
    else:
        state["timeline_dwell_lo"].append(None)
        state["timeline_dwell_hi"].append(None)

    state["timeline_caught"].append(caught)

    # Trim to last 80 steps
    for key in ["timeline_t", "timeline_true", "timeline_scanned",
                "timeline_predicted", "timeline_dwell_lo",
                "timeline_dwell_hi", "timeline_caught"]:
        if len(state[key]) > 80:
            state[key] = state[key][-80:]

    state["step"] += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["📡  Main Scheduler — Learning vs Blind Sweep",
                       "🎯  Periodic Intercept — Predicting the Radar"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📡 Frequency Band Scheduling")
    st.markdown(
        "**WIQL-UCB** learns *which* bands are most active and focuses scanning there. "
        "**Round-Robin** sweeps bands blindly. Watch the band colours and scan choices diverge."
    )

    # Sidebar-style controls in a row
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        n_bands = st.selectbox("Number of bands", [8, 12, 18], index=0, key="t1_bands")
        k_scan  = st.selectbox("Simultaneous scans (K)", [2, 3, 4], index=1, key="t1_k")
    with c2:
        seed_t1 = st.slider("Random seed", 0, 99, 42, key="t1_seed")
        speed   = st.slider("Speed (steps/sec)", 1, 20, 4, key="t1_speed")
    with c3:
        running = st.toggle("▶ Play", key="t1_run")
    with c4:
        if st.button("↺ Reset", key="t1_reset"):
            for k in list(st.session_state.keys()):
                if k.startswith("t1_state"):
                    del st.session_state[k]
            st.rerun()

    # Init state
    state_key = f"t1_state_{n_bands}_{k_scan}_{seed_t1}"
    if state_key not in st.session_state:
        st.session_state[state_key] = init_tab1(n_bands, k_scan, seed_t1)
    s = st.session_state[state_key]

    # Layout: two columns (WIQL left, RR right)
    col_w, col_r = st.columns(2)

    # Legend
    st.markdown("""
    <div style="margin:8px 0 12px 0">
    <span class="legend-dot" style="background:#1450c8"></span>P(active)=Low — free band &nbsp;&nbsp;
    <span class="legend-dot" style="background:#c85014"></span>P(active)=High — likely active &nbsp;&nbsp;
    <span class="legend-dot" style="background:rgba(0,0,0,0);border:3px solid #fff;display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:6px;vertical-align:middle"></span>Scanning now &nbsp;&nbsp;
    ● Active this step (ground truth)
    </div>
    """, unsafe_allow_html=True)

    beliefs_w = s["rec_w"].get_belief()
    beliefs_r = s["rec_r"].get_belief()

    with col_w:
        st.markdown("### 🔵 WIQL-UCB  *(Learns from experience)*")
        st.markdown(make_band_html(beliefs_w, s["action_w"], s["true_occ"]),
                    unsafe_allow_html=True)
        avg_w = np.mean(s["rewards_w"]) if s["rewards_w"] else 0.0
        cum_w = np.sum(s["rewards_w"])
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Avg Reward / Step</div>
        <div class="metric-value green">{avg_w:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### ⚪ Round-Robin  *(Sweeps blindly)*")
        st.markdown(make_band_html(beliefs_r, s["action_r"], s["true_occ"]),
                    unsafe_allow_html=True)
        avg_r = np.mean(s["rewards_r"]) if s["rewards_r"] else 0.0
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Avg Reward / Step</div>
        <div class="metric-value orange">{avg_r:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Step counter and advantage
    st.markdown(f"**Step {s['step']}** &nbsp;|&nbsp; "
                f"WIQL advantage: **{avg_w - avg_r:+.4f}** reward/step", unsafe_allow_html=True)

    # Reward chart
    if len(s["rewards_w"]) > 1:
        import pandas as pd
        chart_data = pd.DataFrame({
            "WIQL-UCB": pd.Series(s["rewards_w"]).rolling(20, min_periods=1).mean(),
            "Round-Robin": pd.Series(s["rewards_r"]).rolling(20, min_periods=1).mean(),
        })
        st.line_chart(chart_data, use_container_width=True, height=200,
                      color=["#3fb950", "#8b949e"])
        st.caption("Reward per step (20-step rolling average) — higher is better")

    # Auto-step
    if running:
        step_tab1(s, n_bands, k_scan)
        time.sleep(1.0 / max(speed, 1))
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🎯 Catching a Periodic Radar Signal")
    st.markdown(
        "The system learns the radar's **pulse repetition interval** (PRI) "
        "and opens a precise scan window at the predicted arrival time — "
        "instead of checking randomly. Watch it go from guessing to predicting."
    )

    cc1, cc2, cc3, cc4 = st.columns([2, 2, 2, 2])
    with cc1:
        T_us_    = st.selectbox("Radar period T (μs)", [300, 500, 750, 1000], index=1, key="t2_T") * 1.0
        sigma_us_ = st.selectbox("Timing jitter σ (μs)", [5, 10, 20, 50], index=2, key="t2_sig") * 1.0
    with cc2:
        seed_t2 = st.slider("Seed", 0, 99, 42, key="t2_seed")
        speed2  = st.slider("Speed (steps/sec)", 1, 20, 5, key="t2_speed")
    with cc3:
        running2 = st.toggle("▶ Play", key="t2_run")
    with cc4:
        if st.button("↺ Reset", key="t2_reset"):
            for k in list(st.session_state.keys()):
                if k.startswith("t2_state"):
                    del st.session_state[k]
            st.rerun()

    state2_key = f"t2_state_{T_us_}_{sigma_us_}_{seed_t2}"
    if state2_key not in st.session_state:
        st.session_state[state2_key] = init_tab2(T_us_, sigma_us_, seed_t2)
    s2 = st.session_state[state2_key]

    # Caught flash
    if s2["caught_this_step"]:
        st.markdown('<div class="caught-flash">🎯 CAUGHT!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="height:52px"></div>', unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    mu_val, sig_val = s2["module"].estimate_period(s2["periodic_band"])
    mu_str  = f"{mu_val:.1f} μs"  if mu_val  is not None else "Learning..."
    sig_str = f"{sig_val:.1f} μs" if sig_val is not None else "Learning..."
    rate = s2["intercept_count"] / max(s2["emission_count"], 1)
    err  = float(np.mean(s2["cyclic_errors"])) if s2["cyclic_errors"] else float("nan")
    err_str = f"{err:.1f} μs" if not math.isnan(err) else "—"

    with m1:
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Estimated Period</div>
        <div class="metric-value" style="font-size:1.4rem">{mu_str}</div>
        <div class="metric-label">true: {T_us_:.0f} μs</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Estimated Jitter σ</div>
        <div class="metric-value" style="font-size:1.4rem">{sig_str}</div>
        <div class="metric-label">true: {sigma_us_:.0f} μs</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Intercept Rate</div>
        <div class="metric-value green">{rate:.0%}</div>
        <div class="metric-label">{s2['intercept_count']}/{s2['emission_count']}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-box">
        <div class="metric-label">Timing Error</div>
        <div class="metric-value orange">{err_str}</div>
        <div class="metric-label">converges → ~{sigma_us_:.0f} μs</div>
        </div>""", unsafe_allow_html=True)

    # Timeline chart
    if len(s2["timeline_t"]) > 2:
        import pandas as pd
        import plotly.graph_objects as go

        ts   = s2["timeline_t"]
        n_ts = len(ts)

        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3", size=12),
            height=280, margin=dict(l=40, r=20, t=30, b=30),
            showlegend=True, legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=11),
            ),
            xaxis=dict(title="Time (μs)", gridcolor="#21262d",
                       color="#8b949e", tickformat=".0f"),
            yaxis=dict(visible=False),
        )

        # Dwell window shading
        dw_lo = s2["timeline_dwell_lo"]
        dw_hi = s2["timeline_dwell_hi"]
        valid_dw = [(i, ts[i], dw_lo[i], dw_hi[i])
                    for i in range(n_ts)
                    if dw_lo[i] is not None and dw_hi[i] is not None]
        if valid_dw:
            for idx, t_, lo_, hi_ in valid_dw[-1:]:  # just last window
                fig.add_vrect(
                    x0=lo_, x1=hi_,
                    fillcolor="#388bfd22", layer="below",
                    line_width=0,
                    annotation_text="Scan<br>window",
                    annotation_font=dict(color="#79c0ff", size=10),
                    annotation_position="top left",
                )

        # Predicted next arrival line
        preds = [(ts[i], s2["timeline_predicted"][i])
                 for i in range(n_ts) if s2["timeline_predicted"][i] is not None]
        if preds:
            last_pred_t = preds[-1][1]
            fig.add_vline(
                x=last_pred_t, line_dash="dash",
                line_color="#58a6ff", line_width=2,
                annotation_text="Predicted<br>next signal",
                annotation_font=dict(color="#58a6ff", size=10),
                annotation_position="top right",
            )

        # True emission ticks
        true_ticks = [(ts[i], s2["timeline_true"][i])
                      for i in range(n_ts)
                      if s2["timeline_true"][i] is not None]
        if true_ticks:
            fig.add_trace(go.Scatter(
                x=[t[1] for t in true_ticks],
                y=[0.5] * len(true_ticks),
                mode="markers",
                marker=dict(symbol="line-ns", size=20, color="#f0883e",
                            line=dict(color="#f0883e", width=3)),
                name="Radar pulse (ground truth)",
                showlegend=True,
            ))

        # Caught markers
        caught_ticks = [(ts[i], s2["timeline_true"][i])
                        for i in range(n_ts)
                        if s2["timeline_caught"][i]
                        and s2["timeline_true"][i] is not None]
        if caught_ticks:
            fig.add_trace(go.Scatter(
                x=[t[1] for t in caught_ticks],
                y=[0.5] * len(caught_ticks),
                mode="markers",
                marker=dict(symbol="star", size=16, color="#3fb950",
                            line=dict(color="#3fb950", width=2)),
                name="✓ CAUGHT",
                showlegend=True,
            ))

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Orange ticks = radar pulses (ground truth) · "
            "Blue dashed line = system's predicted next arrival · "
            "Blue shading = scan window (±3σ) · "
            "Green stars = successful intercepts"
        )
    else:
        st.info("Press ▶ Play to start the simulation")

    # Step counter
    st.markdown(f"**Step {s2['step']}** — "
                f"{'🎯 In dwell window — scanning!' if s2['caught_this_step'] else 'Observing...'}")

    # Auto-step
    if running2:
        step_tab2(s2)
        time.sleep(1.0 / max(speed2, 1))
        st.rerun()
