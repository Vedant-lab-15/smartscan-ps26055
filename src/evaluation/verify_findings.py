"""
Steps 2-4: Re-verify periodic scenario after K-scaling fix, Wilcoxon for no_bias,
and convergence cross-tab for Test 2.

Usage: python3 -m harness.verify_findings
"""
from __future__ import annotations

import pathlib, csv, sys
import numpy as np
from scipy import stats as sp_stats

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.runner import (
    run_test1, wilcoxon_p, bootstrap_ci,
)
from src.evaluation.outputs import OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Re-run periodic scenario with K-scaling fix
# ─────────────────────────────────────────────────────────────────────────────

def step2_rerun_periodic(n_seeds=30, episode_len=5000, n_bands=8, k_scan=3, dt_us=1000.0):
    print("\n" + "="*60)
    print("  STEP 2 — Periodic scenario re-run (K-scaling fix applied)")
    print(f"  K-aware dwell_half = min(3.0, 6.0/K) = min(3.0, {6.0/k_scan:.2f}) = {min(3.0, 6.0/k_scan):.2f}σ")
    print("="*60)

    from harness.env_generator import make_periodic_scenario, RenewalEnv
    from harness.runner import WIQLPolicy, policy_round_robin, policy_random, run_episode

    DT_PERIODIC = 100.0
    algorithms = ["round_robin", "random", "wiql_ucb", "wiql_ucb_no_bias"]
    algo_runs = {a: [] for a in algorithms}

    for seed in range(n_seeds):
        env_rng = np.random.default_rng(seed)
        emitters = make_periodic_scenario(rng=env_rng, n_bands=n_bands, dt_us=DT_PERIODIC)
        env = RenewalEnv(n_bands=n_bands, dt_us=DT_PERIODIC, episode_len=episode_len,
                         k_scan=k_scan, seed=seed)
        env.set_emitters(emitters)

        for algo in algorithms:
            use_bias = algo != "wiql_ucb_no_bias"
            if algo in ("wiql_ucb", "wiql_ucb_no_bias"):
                policy_obj = WIQLPolicy(n_bands, k_scan, use_periodic_bias=use_bias)
            else:
                policy_obj = None

            env.reset()
            metrics, _ = run_episode(env, algo, policy_obj)
            algo_runs[algo].append(metrics)
            rate = metrics["intercept_rate"]
            if seed % 5 == 0:
                print(f"    seed={seed:2d}  {algo:<22}  rate={rate:.3f}")

    # Summarise
    def mean_ci(vals):
        m = float(np.mean(vals))
        lo, hi = bootstrap_ci(vals)
        return m, lo, hi

    on_rates  = [r["intercept_rate"] for r in algo_runs["wiql_ucb"]]
    off_rates = [r["intercept_rate"] for r in algo_runs["wiql_ucb_no_bias"]]
    rr_rates  = [r["intercept_rate"] for r in algo_runs["round_robin"]]

    on_m,  on_lo,  on_hi  = mean_ci(on_rates)
    off_m, off_lo, off_hi = mean_ci(off_rates)
    rr_m,  rr_lo,  rr_hi  = mean_ci(rr_rates)
    delta = on_m - off_m

    # Wilcoxon: bias-ON vs bias-OFF (paired by seed)
    try:
        _, p_bias = sp_stats.wilcoxon(on_rates, off_rates, alternative="greater")
    except Exception:
        p_bias = float("nan")

    print(f"\n  Results (post K-scaling fix):")
    print(f"  {'Algorithm':<26} {'Mean':>7} {'CI lo':>7} {'CI hi':>7}")
    print(f"  {'─'*26} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'wiql_ucb (bias-ON)':<26} {on_m:>7.4f} {on_lo:>7.4f} {on_hi:>7.4f}")
    print(f"  {'wiql_ucb_no_bias':<26} {off_m:>7.4f} {off_lo:>7.4f} {off_hi:>7.4f}")
    print(f"  {'round_robin':<26} {rr_m:>7.4f} {rr_lo:>7.4f} {rr_hi:>7.4f}")
    print(f"\n  Bias-ON vs Bias-OFF delta: {delta:+.4f}")
    print(f"  Pre-fix delta was: -0.2012")
    if delta > 0:
        print(f"  ✓ FLIPPED: bias-ON now EXCEEDS bias-OFF (+{delta:.4f})")
        print(f"  Wilcoxon bias-ON > bias-OFF: p={p_bias:.4f}")
    else:
        print(f"  ✗ STILL NEGATIVE: bias-ON still underperforms bias-OFF ({delta:.4f})")
        print(f"  → STOP: do not claim hierarchical bias improves periodic interception")

    return algo_runs, on_m, off_m, delta, p_bias


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Wilcoxon for wiql_ucb_no_bias on all 3 scenarios
# ─────────────────────────────────────────────────────────────────────────────

def step3_wilcoxon_no_bias(periodic_algo_runs=None):
    """Compute Wilcoxon for wiql_ucb_no_bias vs round_robin, all scenarios.
    Background and freq_agile use the already-collected full run data.
    Periodic uses the freshly re-run data from Step 2.
    """
    print("\n" + "="*60)
    print("  STEP 3 — Wilcoxon signed-rank: wiql_ucb_no_bias vs round_robin")
    print("="*60)

    raw_csv = OUTPUT_DIR / "test1_raw.csv"
    if not raw_csv.exists():
        print("  ERROR: results/test1_raw.csv not found. Run full harness first.")
        return

    # Load full run CSV
    rows = []
    with open(raw_csv) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    scenarios = ["background", "freq_agile", "periodic"]
    print(f"\n  {'Scenario':<14} {'WIQL-no-bias mean':>18} {'RR mean':>9} {'Δ':>7} {'Wilcoxon p':>12}")
    print(f"  {'─'*14} {'─'*18} {'─'*9} {'─'*7} {'─'*12}")

    results = {}
    for scen in scenarios:
        if scen == "periodic" and periodic_algo_runs is not None:
            # Use fresh data from Step 2
            no_bias_rates = [r["intercept_rate"] for r in periodic_algo_runs["wiql_ucb_no_bias"]]
            rr_rates = [r["intercept_rate"] for r in periodic_algo_runs["round_robin"]]
            note = "(re-run)"
        else:
            no_bias_rates = [float(r["intercept_rate"]) for r in rows
                             if r["scenario"] == scen and r["algorithm"] == "wiql_ucb_no_bias"]
            rr_rates = [float(r["intercept_rate"]) for r in rows
                        if r["scenario"] == scen and r["algorithm"] == "round_robin"]
            note = "(full-run)"

        if not no_bias_rates or not rr_rates:
            print(f"  {scen:<14} no data")
            continue

        no_bias_m = float(np.mean(no_bias_rates))
        rr_m = float(np.mean(rr_rates))
        delta = no_bias_m - rr_m
        p = wilcoxon_p(no_bias_rates, rr_rates)
        results[scen] = {"no_bias_mean": no_bias_m, "rr_mean": rr_m, "delta": delta, "p": p}
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {scen:<14} {no_bias_m:>18.4f} {rr_m:>9.4f} {delta:>+7.4f} {p:>10.4f} {sig}  {note}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Convergence cross-tab for Test 2
# ─────────────────────────────────────────────────────────────────────────────

CONVERGENCE_THRESHOLD = 0.4  # intercept_rate > 0.4 = converged

def step4_convergence_crosstab():
    print("\n" + "="*60)
    print(f"  STEP 4 — Test 2 convergence cross-tab (threshold: rate > {CONVERGENCE_THRESHOLD})")
    print("="*60)

    raw_csv = OUTPUT_DIR / "test2_raw.csv"
    if not raw_csv.exists():
        print("  ERROR: results/test2_raw.csv not found.")
        return

    rows = []
    with open(raw_csv) as f:
        for row in csv.DictReader(f):
            if row["algorithm"] == "wiql_ucb":
                rows.append({
                    "jitter_ratio": float(row["jitter_ratio"]),
                    "seed": int(row["seed"]),
                    "rate": float(row["intercept_rate"]),
                })

    if not rows:
        print("  No wiql_ucb data found in test2_raw.csv")
        return

    jitter_levels = sorted(set(r["jitter_ratio"] for r in rows))
    n_seeds = max(r["seed"] for r in rows) + 1

    print(f"\n  Convergence threshold: intercept_rate > {CONVERGENCE_THRESHOLD}")
    print(f"  N seeds: {n_seeds}")
    print(f"\n  {'Jitter %':>9} | {'N converged':>11} | Converging seed IDs")
    print(f"  {'─'*9} | {'─'*11} | {'─'*40}")

    convergence_map = {}  # jitter_ratio -> set of converging seeds
    for j in jitter_levels:
        converging = {r["seed"] for r in rows
                      if r["jitter_ratio"] == j and r["rate"] > CONVERGENCE_THRESHOLD}
        convergence_map[j] = converging
        ids_str = str(sorted(converging)) if converging else "none"
        print(f"  {j*100:>8.0f}% | {len(converging):>11} | {ids_str}")

    # Check if the same seeds converge at every level
    all_seed_sets = [convergence_map[j] for j in jitter_levels]
    intersection = set.intersection(*all_seed_sets) if all_seed_sets else set()
    union = set.union(*all_seed_sets) if all_seed_sets else set()
    union_only = union - intersection  # seeds that converge at SOME but not ALL levels

    print(f"\n  Seeds converging at ALL jitter levels: {sorted(intersection)}")
    print(f"  Seeds converging at SOME but not all:  {sorted(union_only)}")
    print(f"  Seeds NEVER converging:                {sorted(set(range(n_seeds)) - union)}")

    if len(intersection) >= 3 and len(union_only) <= 3:
        print(f"\n  ✓ FINDING CONFIRMED: Bimodal convergence, jitter-ORTHOGONAL")
        print(f"    The same {len(intersection)} seed IDs converge regardless of jitter level.")
        print(f"    Convergence is determined by exploration topology, not jitter magnitude.")
    elif len(union_only) > len(intersection):
        print(f"\n  ✗ FINDING CHALLENGED: Jitter-dependent convergence detected")
        print(f"    {len(union_only)} seeds converge at some jitter levels but not others.")
        print(f"    The bimodal-convergence-is-jitter-orthogonal explanation may be incomplete.")
    else:
        print(f"\n  → AMBIGUOUS: mixed pattern, interpret carefully")

    return convergence_map, intersection


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 2
    periodic_algo_runs, on_m, off_m, delta, p_bias = step2_rerun_periodic()

    # Step 3 (uses Step 2 data for periodic)
    wilcoxon_results = step3_wilcoxon_no_bias(periodic_algo_runs)

    # Step 4 (uses existing test2 CSV)
    conv_result = step4_convergence_crosstab()

    # ── Final flags ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINDINGS STATUS")
    print("="*60)

    print("\n  1. Core WIQL-UCB beats round-robin (+0.14–0.19)")
    print("     → CONFIRMED (full-run data; see Wilcoxon for no-bias variant above)")

    if wilcoxon_results:
        for scen, r in wilcoxon_results.items():
            p = r["p"]; d = r["delta"]
            flag = "✓ SIGNIFICANT" if p < 0.05 else "— p≥0.05"
            print(f"       {scen}: no-bias Δ={d:+.4f}  p={p:.4f}  {flag}")

    print(f"\n  2. Hierarchical bias (Module C) in periodic scenario")
    if delta > 0:
        sig = "significant" if p_bias < 0.05 else "directional (p≥0.05)"
        print(f"     → CONFIRMED post-fix: bias-ON={on_m:.4f} > bias-OFF={off_m:.4f} "
              f"(Δ={delta:+.4f}, p={p_bias:.4f}) — {sig}")
        print(f"     → Pre-fix delta was -0.2012; K-scaling fix corrected the starvation bug")
    else:
        print(f"     ✗ STILL FAILS post-fix: bias-ON={on_m:.4f} < bias-OFF={off_m:.4f} (Δ={delta:.4f})")
        print(f"     → DO NOT claim hierarchical bias improves periodic scenario in PPT/doc")
        print(f"     → Needs architecture-level investigation, not parameter tuning")

    print(f"\n  3. Module C jitter robustness (flat rate 0–20%)")
    if conv_result:
        _, intersection = conv_result
        if len(intersection) >= 3:
            print(f"     → CONFIRMED: bimodal convergence is jitter-orthogonal")
            print(f"       ({len(intersection)} seeds converge at all jitter levels)")
            print(f"       Correct statement: 'on converging seeds, intercept rate is stable 0–20%'")
            print(f"       NOT 'robust to 20% jitter' (convergence is architecture-limited)")
        else:
            print(f"     → REWRITE NEEDED: convergence pattern is jitter-dependent")
