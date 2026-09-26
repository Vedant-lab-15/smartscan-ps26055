# Smart Scan Strategy for Electronic Warfare

**SIH 2026 Problem Statement 26055 — DRDO**
**Team Coding Saints (ID 120303)**

A Whittle-inspired priority scheduler for Electronic Support (ES) receivers operating with no prior emitter intelligence. We formulate spectrum-scan interception as a restless multi-armed bandit with a POMDP belief state, and combine a priority index with an additive periodic-recurrence bias.

---

## Headline Results

| Result | Value | Significance |
|---|---|---|
| Core scheduler vs. round-robin (3 scenarios) | +0.14 to +0.20 intercept rate | N=30, p<0.001 |
| Periodic module, standalone | 0.975 ± 0.006 intercept rate | worst-case 0.939 |
| Multi-band convergence (8-band, K=3) | 30/30 seeds, worst-case 0.419 | 50-step RR pre-phase fix |
| Frequency-agile handling | +0.107 min across sensitivity sweep | p<0.001, all configs |

---

## What It Does

Each frequency band is an arm of a restless multi-armed bandit. Occupancy evolves whether or not the band is observed. The scheduler tracks a per-band belief state from hit/miss feedback, computes a priority index per band, and scans the top-M bands each step.

The priority index combines three terms:
- **Belief value** — W_i · b_i(t)
- **Additive periodic bias** — γ · periodic_bonus_i(t)
- **UCB exploration** — max(c · sqrt(ln t / N_i(t)), ε)

Where ε = 0.01 is a minimum exploration floor that prevents permanent lockout of bands in non-stationary environments.

---

## Quick Start

```bash
git clone https://github.com/Vedant-lab-15/smartscan-ps26055.git
cd smartscan-ps26055
pip install -r requirements.txt
pytest tests/
```

---

## Reproduce the Paper Results

```bash
bash scripts/run_evaluation.sh
```

This runs the full N=30-seed harness across the three scenarios (background, periodic, frequency-agile) and writes results to `results/headline_numbers.json`.

---

## Run the Demo

```bash
streamlit run demo/app.py
```

(A deployed interactive prototype is available at [smartscan-ps26055.streamlit.app](https://smartscan-ps26055-eedhs42fmrbm5or5uragaw.streamlit.app/))

---

## Architecture

See `docs/architecture.md` for the full system diagram and design notes.

```
I_i(t) = W_i · b_i(t)  +  γ · periodic_bonus_i(t)  +  max(c·√(ln t / N_i(t)), ε)
         ─────────────    ──────────────────────────    ──────────────────────────
          Belief value       Periodic recurrence              UCB exploration
                                   bias                    (with floor ε = 0.01)
```

---

## Project Structure

```
smartscan-ps26055/
├── src/
│   ├── scheduler/          # WIQLScheduler, BeliefTracker, PeriodicInterceptModule
│   ├── environment/        # RFEnvironment, ReceiverModel, PDWGenerator, TSRDLoader
│   ├── evaluation/         # EvaluationHarness (7 FoMs), statistical harness, outputs
│   └── baselines/          # RoundRobinPolicy, RandomPolicy
├── demo/                   # Interactive prototype — scenario editor, scheduler config, baseline selector
├── tests/                  # 134 tests — unit, property-based, integration
├── scripts/                # run_evaluation.sh, generate_figures.py
├── docs/                   # Architecture, evaluation methodology, limitations, deployment
├── results/                # Headline numbers, CSVs, plots
│   └── figures/            # Generated plots (reproducible via generate_figures.py)
└── data/                   # Data directory — see data/README.md for TSRD download
```

---

## What's Built vs. Roadmap

**Built and validated:**
- Core WIQL-UCB scheduler (tabular, ~320 bytes/arm, ~24 μs/step at K=8)
- Belief-state update with realistic Pd=0.9, Pfa=0.01 (HMM POMDP)
- Periodic recurrence module — standalone (0.975 ± 0.006) and multi-band convergence 30/30 seeds with 50-step RR pre-phase (worst-case 0.419); γ=5 gives +0.142 over γ=0, p=0.0011
- Statistical evaluation harness (N=30 seeds, 95% bootstrap CI, Wilcoxon signed-rank)
- Frequency-agile handling via UCB de-prioritization (+0.107 min across hop-rate sweep)
- UCB non-stationarity fix (exploration floor ε=0.01)
- Interactive prototype deployed at [smartscan-ps26055.streamlit.app](https://smartscan-ps26055-eedhs42fmrbm5or5uragaw.streamlit.app/) — scenario editor, scheduler config panel, baseline selector

**Roadmap:**
- Frequency-agile correlated-transition modeling (P ∈ ℝ^{K×K})
- Threat-weighted scheduling (DQWIC)
- Adversarial-jitter robustness (EXP3/minimax)
- Hardware LO retuning dead-time penalties

---

## Limitations

- Indexability of the priority index is unverified; labeled "Whittle-inspired heuristic."
- Sensitivity: −1 dB receiver threshold (minimum SNR at which Pd ≥ 0.90 with 95% CI, N=30 seeds).
- Frequency-agile handling is reactive, not predictive of cross-band hops.
- Multi-band periodic convergence required a 50-step round-robin discovery pre-phase to reach 30/30 seeds (8-band, K=3 harness); 30/30 also in standalone K=2 configuration.

See `docs/limitations.md` for the full list.

---

## Citation

If you use this code, please cite the work in `CITATION.cff`.

```bibtex
@misc{codingsaints2026smartscan,
  title  = {Smart Scan Strategy for Electronic Warfare — SIH 2026, PS 26055},
  author = {Coding Saints, Team ID 120303},
  year   = {2026},
  url    = {https://github.com/Vedant-lab-15/smartscan-ps26055}
}
```

---

## License

MIT License — see `LICENSE`.
