# Controlled Demonstrations

These results come from **bespoke scripts** with non-standard configurations.
They are NOT part of the main evaluation harness. See `scripts/controlled_demonstrations/` for the scripts that produced them.

---

## standalone_0975_demonstration

**Files:** `module_c_standalone_rigorous.csv`, `module_c_standalone_allseeds.csv`

**What it measures:** Periodic module (Module C) intercept rate in an idealized standalone configuration — a single synthetic emitter, dedicated receiver, no band competition.

**Configuration:**
- N = 30 seeds
- K = 2 simultaneous scans, N_bands = 4
- Episode length: T = 2000 steps
- dt = 50 μs → T_slots = 10 for T=500 μs emitter
- Emitter: single periodic, T=500 μs, σ=20 μs
- warmup = 0 (natural UCB cold-start)
- γ = 5 (periodic module on)
- Metric: intercepts / total_occupied_all_bands

**Result:** mean = 0.975 ± 0.006, worst-case = 0.939

**Why it's bespoke:** This configuration uses a hand-crafted single-emitter environment and a 2000-step episode, not the standard `make_periodic_scenario` factory or the 5000-step harness. It demonstrates the periodic module's maximum performance under ideal conditions (dedicated receiver). The paper should cite this as a "controlled demonstration" distinct from the multi-band operational results.

---

## ask3_gamma_sweep.csv

**What it measures:** Pre-fix γ-sweep (warmup=0) — the numbers in the original paper γ-sweep table.

**Configuration:** warmup=0, N=30, T=5000, 8 bands, K=3, periodic scenario.

**Superseded by:** `results/paper_results/gamma_sweep_perseeds.csv` (post-fix, warmup=50).

---

## headline_numbers_prefix.json

**What it is:** The original (pre-fix) headline_numbers.json. Preserved for audit trail.
Superseded by `results/paper_results/summary.json`.
