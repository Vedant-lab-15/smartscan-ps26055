---
title: Smart Scan Demo
emoji: 📡
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.30.0"
app_file: demo/app.py
pinned: false
---

# Smart Scan Strategy — PS 26055

Whittle-inspired priority scheduler for Electronic Support (ES) receivers operating with no prior emitter intelligence.

**SIH 2026 · PS 26055 (DRDO) · Coding Saints, Team ID 120303**

## What this demo shows

- **Band occupancy heatmap** — ground truth RF activity across N frequency bands
- **Scan decisions overlay** — WIQL-UCB vs Round-Robin side by side
- **Belief state** — per-band P(occupied) learned from hit/miss feedback
- **Priority index** — combined Whittle + UCB score driving scan decisions
- **Intercept counter** — head-to-head comparison with live delta

## How to use

1. Select a scenario (Background / Periodic / Frequency-Agile)
2. Adjust bands, scan capacity, and time steps
3. Press **▶ Run Episode**

Results render instantly — no live loop.

## Code and paper

Full source: [github.com/Vedant-lab-15/smartscan-ps26055](https://github.com/Vedant-lab-15/smartscan-ps26055)
