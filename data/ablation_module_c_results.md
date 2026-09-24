========================================================================
  Module C Ablation — Real-Data Failure Mode Analysis
  SIH 2026 — PS 26055 (DRDO)
========================================================================

  Three conditions, one variable changed per condition:
    A) Baseline: standard K=3-of-18 competitive access
    B) Guaranteed dedicated: periodic band always scanned
    C) Active-window only (emitter 31): eval restricted to emitter's
       actual active period under guaranteed access

  Warmup = second half of EACH CONDITION'S evaluation window
  (not the full file — the previous run used the wrong denominator)

────────────────────────────────────────────────────────────────────────
  EMITTER 10  (config_0.h5)  T≈1512μs  σ≈133μs  CoV=0.088  DT=750μs
  Active window: 0.198s (0.68% of 29s file)  |  131 periods  |  132 pulses
────────────────────────────────────────────────────────────────────────

  NOTE: Conditions A/B below are evaluated over the ACTIVE WINDOW only
  (17.2M–17.4M μs). At DT=750μs this gives 264 steps.
  Warmup = first 132 steps; mature = last 132 steps.

  Running Condition A (competitive access, active window)...
  Running Condition B (guaranteed access, active window)...

  Cond    Intercept rate   CycErr late (μs)  n_mature  Emissions
  ───── ──────────────── ────────────────── ───────── ──────────
  A               0.9618              397.8       125        262
  B               1.0000              384.6       130        262

  A→B lift: 0.9618 → 1.0000 (+0.0382)
  dwell window = 6σ = 799μs = 53% of T  (tight: leaves ~24% of period outside dwell)

────────────────────────────────────────────────────────────────────────
  EMITTER 31  (config_2.h5)  T≈503μs  σ≈36μs  CoV=0.071  DT=250μs
  Active window: 0.099s (0.36% of 27.9s file)  |  198 periods  |  199 pulses
────────────────────────────────────────────────────────────────────────

  Running Condition A (competitive access, full file — original)...
  Running Condition B (guaranteed access, full file)...
  Running Condition C (guaranteed access, active window only)...

  Cond       Window           Intercept rate   CycErr late (μs)  n_mature  Emissions
  ────────── ────────────── ──────────────── ────────────────── ───────── ──────────
  A          Full file                0.0481                N/A         0       1394
  B          Full file                1.0000            14953.1       698       1394
  C          Active only              1.0000              117.2       123        256

────────────────────────────────────────────────────────────────────────
  DIAGNOSIS
────────────────────────────────────────────────────────────────────────

  EMITTER 10:
  → JITTER-DOMINATED: A=0.962 → B=1.000 (+0.038)
    Even guaranteed access does not materially improve intercept rate.
    σ=133μs means 6σ dwell window = 798μs = 53% of T — nearly half the period
    is always in-window, so the dwell-window selection provides little benefit.

  EMITTER 31:
  → COVERAGE-RATIO-DOMINATED: B(full)=1.000
    Guaranteed access already gives good intercept rate on full file.

────────────────────────────────────────────────────────────────────────
  COMPARISON vs SYNTHETIC (reference: 0.9838 rate, 16.6μs error)
────────────────────────────────────────────────────────────────────────

  Case                                          Rate   CycErr(μs)
  ────────────────────────────────────────── ─────── ────────────
  Synthetic (σ=20μs, designed)                0.9838         16.6
  Em10 Cond-B (full-access, σ=133μs)          1.0000        384.6
  Em31 Cond-C (active-win, σ=36μs)            1.0000        117.2