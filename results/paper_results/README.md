# Paper Results

These CSVs and the summary JSON contain **every number in the paper**, produced by the standard harness (post-fix, warmup=50).

All results: N=30 seeds, T=5000 steps, n_bands=8, K=3, warmup=50, γ=5, Pd=0.9, Pfa=0.01, 95% bootstrap CI (B=2000).

Reproduce all results in one command:
```bash
bash scripts/reproduce_paper_results.sh
```

---

## Files

| File | Content | Paper table |
|---|---|---|
| `core_scheduler.csv` | Per-seed intercept rates for WIQL, RR, Random, Clarkson (3 scenarios × 30 seeds) | Table I |
| `gamma_sweep_perseeds.csv` | Per-seed intercept rates for γ ∈ {0, 0.5, 1, 2, 5, 10} × 30 seeds | Section II.E |
| `seven_foms.csv` | Per-seed Pd, Pfa, reward, belief accuracy (3 scenarios × 30 seeds) | Table III |
| `snr_sweep_perseeds.csv` | Per-seed Pd at each SNR threshold × 30 seeds | Section III.D |
| `snr_sweep_summary.csv` | Aggregated Pd mean/CI per threshold | Section III.D |
| `summary.json` | All aggregated results in one file | All tables |

---

## Configuration

```
n_bands       = 8
K_scan        = 3
T             = 5000 slots
dt_periodic   = 100 μs
dt_other      = 1000 μs
warmup        = 50 steps (RR pre-phase)
gamma         = 5 (periodic bias weight)
p_detect      = 0.9
p_fa          = 0.01
prior         = 0.5
N             = 30 seeds
CI            = 95% bootstrap, B=2000
test          = Wilcoxon signed-rank, one-sided (greater), exact method
```
