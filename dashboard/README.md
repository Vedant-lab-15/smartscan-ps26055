# EW Smart Scan Strategy — Live Demo Dashboard

**SIH 2026 · PS 26055 (DRDO)**  
Visualises the Whittle-index scheduler (WIQL-UCB) and periodic-emitter interception module (Module C) live, step by step.

---

## Local run

```bash
# 1. Install dependencies (if not already done)
pip install -e ".[dev]"
pip install streamlit plotly

# 2. Run the dashboard
streamlit run dashboard/app.py
```

Opens at http://localhost:8501

**No HuggingFace token or real data needed** — the demo runs fully on synthetic data by default.

---

## What you're looking at

### Tab 1 — Main Scheduler
| Element | Meaning |
|---|---|
| Coloured cells (blue→red) | P(band is active) — darker red = system thinks this band has traffic |
| White glowing border | This band is being scanned *right now* |
| Orange dot ● | Band is truly active this step (ground truth) |
| WIQL-UCB | Learns which bands are worth watching and concentrates scans there |
| Round-Robin | Sweeps bands mechanically, ignoring what it's learned |

The reward chart shows 20-step rolling average. The gap between the lines is the system's performance advantage.

### Tab 2 — Periodic Interception
| Element | Meaning |
|---|---|
| Orange ticks | True radar pulses (ground truth) |
| Blue dashed line | System's predicted next pulse arrival |
| Blue shaded region | The ±3σ "scan window" — we scan when we expect a pulse |
| Green stars ✓ | Successful intercepts — pulse caught within the window |
| 🎯 CAUGHT! | Flash when an intercept succeeds |

The system starts knowing nothing, estimates the radar's period from observations, then opens a precise timing window. Watch the intercept rate climb as estimates converge.

---

## Deployment (public, synthetic-only)

### Streamlit Community Cloud
1. Push to a public GitHub repo
2. Go to https://share.streamlit.io → "New app"
3. Set main file: `dashboard/app.py`
4. No secrets needed (synthetic mode, no HF token required)

### HuggingFace Spaces
1. Create a new Space → SDK: Streamlit
2. Upload the repo contents
3. Add `streamlit` and `plotly` to `requirements.txt`
4. Set `app_file: dashboard/app.py` in `README.md` YAML front-matter

---

## Dependencies
- `streamlit >= 1.30`
- `plotly >= 5.0`
- `numpy`
- All `ew_smart_scan` package dependencies (see `pyproject.toml`)

No real TSRD data or HuggingFace credentials required for the default demo.
