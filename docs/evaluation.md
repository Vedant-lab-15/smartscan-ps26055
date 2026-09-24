# Evaluation Methodology

**SIH 2026 PS 26055 — Statistical Harness**
**Team Coding Saints, ID 120303**

---

## Harness Design

| Parameter | Value | Rationale |
|---|---|---|
| Seeds | N = 30 | Sufficient power for Wilcoxon signed-rank (paired) |
| Episode length | 5000 slots | Allows WIQL-UCB to mature past cold-start phase |
| Time step (background/freq-agile) | 1000 μs (1 ms) | Practical EW dwell time |
| Time step (periodic) | 100 μs | Matched to TSRD emitter periods (T ≈ 500–1500 μs) |
| Bands | N = 8 | Matches validated codebase coverage ratio |
| Simultaneous scans | K = 3 | K/N = 37.5% scan fraction |
| CI method | Bootstrap, B = 2000 | Accounts for non-normal distributions |
| Confidence level | 95% | Standard |
| Significance test | Wilcoxon signed-rank (paired by seed) | Non-parametric, matched-pair |
| Worst-case reporting | min across seeds | Robustness against hostile-reviewer scrutiny |

---

## Scenarios

### 1. Background (Random Occupancy)
- 6 background emitters per episode, Bernoulli(p) per slot
- p drawn from Beta(1, 3) per emitter (mean ≈ 0.25, heavy tail toward 0)
- No structured emitters
- Baseline test of WIQL-UCB belief convergence

### 2. Periodic
- 2 periodic emitters (TSRD-grounded parameters) + 4 background
- Class P1: T = 1511.5 μs, σ_T = 133.2 μs
- Class P2: T = 502.5 μs, σ_T = 35.9 μs
- Occupancy from renewal process with TruncNormal inter-arrivals
- Tests the γ-bias and periodic module integration
- **Note:** 2 characterised instances is a thin basis — disclosed in `docs/limitations.md`

### 3. Frequency-Agile
- 3 agile emitters + 3 background
- Hop mechanism: deterministic cyclic (+1 band index)
- Hop set: 4 contiguous bands
- Mean dwell: Poisson(λ=10) slots per band
- Active rate while on band: Bernoulli(0.6)
- No P-matrix (correlated hop transition modeling is roadmap)
- Tests UCB's reactive de-prioritization advantage

---

## Seven Figures of Merit

| FoM | Definition | Computation |
|---|---|---|
| **Pd** | P(detection \| scanned ∧ occupied) | tp / (tp + fn), over scanned bands only |
| **Pfa** | P(detection \| scanned ∧ idle) | fp / (fp + tn), modeled at p_fa = 0.01 |
| **Sensitivity** | Min SNR threshold for Pd ≥ 0.90 | **−1 dB** (95% CI lower bound ≥ 0.90; background scenario, N=30) |
| **Avg Intercept Rate** | Detections / total steps | tp / n_steps (equivalent to Pd in this formulation) |
| **Avg Reward/Cost** | Net reward per step | raw_reward − 0.05·\|scanned\| per step |
| **% Correct Predictions** | Belief accuracy | Fraction of (band, step) pairs where b_i > 0.5 matches ground truth |
| **Avg Intercept Time Error** | Mean \|predicted − actual\| ToA | Cyclic (mod-T) correction applied; periodic scenario only |

---

## Measured Values (N=30, WIQL-UCB no-bias)

| FoM | Background | Periodic | Freq-Agile |
|---|---|---|---|
| Pd | 0.503 ± 0.048 | 0.511 ± 0.050 | 0.456 ± 0.032 |
| Pfa | ≈ 0.010 | ≈ 0.010 | ≈ 0.010 |
| Sensitivity | −1 dB threshold | −1 dB threshold | −1 dB threshold |
| Avg Intercept Rate | 0.503 ± 0.048 | 0.511 ± 0.050 | 0.456 ± 0.032 |
| Avg Reward (net) | 0.233 ± 0.028 | 0.197 ± 0.026 | 0.333 ± 0.024 |
| % Correct Predictions | 0.652 ± 0.033 | 0.654 ± 0.026 | 0.685 ± 0.021 |
| Avg Time Error | N/A | 16.6 μs (standalone) | N/A |

---

## Wilcoxon Results (WIQL-UCB no-bias vs Round-Robin)

| Scenario | Δ intercept rate | p-value |
|---|---|---|
| Background | +0.1984 | < 0.001 |
| Periodic | +0.1923 | < 0.001 |
| Frequency-agile | +0.1402 | < 0.001 |

All three scenarios pass p < 0.001 (paired Wilcoxon, N=30 seeds).

---

## γ-Sweep (Periodic Multi-Band Integration)

| γ | Mean intercept rate | Δ vs γ=0 | Wilcoxon p |
|---|---|---|---|
| 0 (baseline) | 0.223 | — | — |
| 0.5 | 0.317 | +0.094 | 0.020 |
| 1.0 | 0.315 | +0.092 | 0.012 |
| 2.0 | 0.320 | +0.098 | 0.005 |
| 5.0 | 0.365 | +0.142 | 0.001 |
| 10.0 | 0.362 | +0.140 | 0.002 |

All γ > 0 produce p < 0.05. **Recommended: γ = 5.**

---

## Periodic Module Standalone Results

| Metric | Value |
|---|---|
| Mean intercept rate | 0.975 |
| 95% CI half-width | ± 0.006 |
| Worst-case (min seed) | 0.939 |
| Avg intercept time error | 16.6 μs |

Standalone configuration: K=2, 4 bands, 30 seeds, 5000 steps, T=1511.5 μs, σ=133.2 μs (P1-TSRD).

---

## Frequency-Agile Sensitivity Sweep

| Hop rate multiplier | WIQL-UCB mean | Round-Robin mean | Δ | Wilcoxon p |
|---|---|---|---|---|
| 1× (baseline) | 0.456 | 0.316 | +0.140 | < 0.001 |
| 2× | 0.434 | 0.318 | +0.116 | < 0.001 |
| 4× | 0.423 | 0.316 | +0.107 | < 0.001 |

Minimum gain across all configs: **+0.107** (p < 0.001). UCB reactive de-prioritization degrades moderately at high hop rates but remains significant throughout.

---

## Reproducing Results

```bash
# Full N=30-seed harness (writes to results/headline_numbers.json)
bash scripts/run_evaluation.sh

# Regenerate all figures from existing CSVs
python scripts/generate_figures.py
```
