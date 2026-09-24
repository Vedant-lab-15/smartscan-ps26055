# Limitations

**SIH 2026 PS 26055 — Honest disclosure for reviewers**
**Team Coding Saints, ID 120303**

This document lists the known limitations of the current implementation. These are intentional disclosures, not omissions.

---

## 1. Indexability Unverified

The priority index I_i(t) = W_i·b_i(t) + γ·periodic_bonus_i(t) + UCB_i(t) is labelled "Whittle-inspired heuristic" throughout the paper. Formal Whittle indexability requires proving that the per-arm optimization problem is indexable (Whittle 1988), which has not been verified for this belief-augmented formulation. The empirical results confirm the index works well in practice; the theoretical guarantee is left as future work.

---

## 2. Sensitivity — Measured at Receiver Threshold −1 dB

The minimum receiver SNR threshold at which Pd ≥ 0.90 is maintained was measured in a dedicated SNR sweep (background scenario, K=3, 8 bands, N=30 seeds). Model: Pd(snr) = sigmoid((emitter_snr − threshold) / 3 dB), emitter SNR ~ Uniform(8, 20) dB.

**Result: Sensitivity = −1 dB receiver threshold** (last point where 95% CI lower bound ≥ 0.90). The current operating point (p_detect = 0.9 fixed) corresponds to a receiver threshold of ≈ 8.4 dB, where mean Pd ≈ 0.89.

Full data in `results/task2_snr_sweep.csv` and `results/task2_sensitivity.json`.

---

## 3. Frequency-Agile Handling is Reactive, Not Predictive

The scheduler's frequency-agile advantage (+0.14 in the baseline scenario) comes from UCB's natural de-prioritization of recently-missed bands — a reactive mechanism. When a hopping emitter moves to a new band, the UCB bonus on the vacated band decays, and the newly active band's UCB bonus grows. This gives a passive advantage but does not predict the next hop.

A correlated-transition model (P ∈ ℝ^{K×K}) would explicitly track band-to-band hop patterns and is on the roadmap. The current advantage degrades moderately at high hop rates (+0.107 at 4× baseline hop rate, down from +0.140) — the boundary is characterized in `docs/evaluation.md`.

---

## 4. Multi-Band Periodic Convergence Required an Explicit Discovery Phase

In the 8-band, K=3 multi-band harness, the periodic module converged in only 21/30 seeds under the default UCB cold-start (baseline), and 9/30 seeds in the original configuration. The root cause is the UCB cold-start's stable argsort tie-breaking: bands with higher indices (5, 6, 7) were systematically under-explored in the first 50 steps, leaving the periodic estimator with zero observations and zero Gaussian bonus — a feedback stall.

**Fix applied:** A 50-step round-robin pre-phase was added to `WIQLPolicy.select()` in `src/evaluation/runner.py`. This guarantees ≥1 scan of every band before WIQL takes over, seeding the Welford estimator.

**Result after fix: 30/30 seeds converge, worst-case intercept rate 0.419.**

The standalone K=2/4-band configuration remains at 30/30 seeds (0.975 ± 0.006) — unchanged.

**Disclosure for judges:** The 0.975 standalone intercept rate is for the K=2/4-band configuration. The multi-band result with the RR pre-phase is 30/30 seeds, worst-case 0.419, mean 0.631.

---

## 5. TSRD Periodic Emitter Statistics are Thin

The periodic emitter parameters (T₁ = 1511.5 μs, σ₁ = 133.2 μs; T₂ = 502.5 μs, σ₂ = 35.9 μs) are grounded in 2 characterised instances from 3 TSRD files. This is a thin statistical basis — the parameters may not be representative of the full TSRD periodic emitter population. 95% of TSRD emitters (125 total from 3 files) have CoV ≥ 1.0 and are not periodic in the classical fixed-PRI sense.

---

## 6. Hardware Timing Assumptions

The computational feasibility analysis (24 μs/step at K=8) is measured in a Python interpreter on a commodity laptop CPU. The DRDO problem involves hardware receivers with microsecond-scale LO retuning. A production implementation would require:
- C or C++ implementation of the priority index computation
- Direct interface to the receiver's band-switching hardware
- Accounting for LO retuning dead-time (not modeled here)

The current analysis demonstrates that the algorithmic complexity is compatible with EW latency budgets; hardware integration is out of scope for this submission.

---

## Not Implemented (Roadmap)

| Feature | Status | Notes |
|---|---|---|
| Correlated-transition model (P matrix) | Roadmap | Would improve frequency-agile tracking |
| Threat-weighted scheduling (DQWIC) | Roadmap | Multi-objective reward, Module B |
| Adversarial-jitter robustness (EXP3) | Roadmap | Module D — reactive emitter evasion |
| Hardware LO retuning dead-time | Out of scope | Implementation-specific |
