"""
Harness Runner — Test 1 (Core Scheduler) + Test 2 (Jitter Degradation)
=======================================================================
Implements three schedulers and runs them in the renewal-process environment.

Schedulers:
  round_robin  — cycles bands in fixed order, ignores observations
  random       — selects K bands uniformly at random each slot
  wiql_ucb     — Whittle Index Q-Learning with UCB (our system)

Each scheduler pair (scenario, algorithm) is run across N_SEEDS seeds.
Statistics: mean ± 95% CI via bootstrap; worst-case (min) per pair.
Significance: Wilcoxon signed-rank test (paired by seed) vs baselines.

Logging: one run per seed captures belief/priority/scan-decision time series
for the demo visualisation (first seed of the periodic scenario only).
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
from scipy import stats as sp_stats

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.env_generator import (
    RenewalEnv, make_background_scenario, make_periodic_scenario,
    make_freq_agile_scenario,
)
from src.scheduler.belief import BeliefTracker
from src.scheduler.wiql_ucb import WIQLScheduler, ACTION_SCAN, ACTION_PASSIVE
from src.scheduler.periodic import PeriodicInterceptModule


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dataclass
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    intercepts: list[int],
    total_slots_occupied: list[int],
    rewards: list[float],
    intercept_times_us: list[float],
    dt_us: float,
) -> dict:
    """Aggregate per-slot lists into episode-level scalars."""
    total_occ = sum(total_slots_occupied)
    total_int = sum(intercepts)
    intercept_rate = total_int / max(total_occ, 1)
    avg_intercept_time = float(np.mean(intercept_times_us)) if intercept_times_us else float("nan")
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return {
        "intercept_rate": intercept_rate,
        "avg_intercept_time_ms": avg_intercept_time / 1000.0,  # μs → ms
        "avg_reward": avg_reward,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler policies
# ─────────────────────────────────────────────────────────────────────────────

def policy_round_robin(t: int, n_bands: int, k: int, **_) -> set[int]:
    start = (t * k) % n_bands
    return {(start + j) % n_bands for j in range(k)}


def policy_random(t: int, n_bands: int, k: int, rng: np.random.Generator, **_) -> set[int]:
    return set(rng.choice(n_bands, size=k, replace=False).tolist())


class WIQLPolicy:
    """Stateful WIQL-UCB + optional periodic bias.

    Additive-index formulation (fixed — replaces hard threshold bypass):
      I_i(t) = W_i(t) + gamma * periodic_bonus_i(t)
    where:
      W_i(t)              = WIQL belief-derived Whittle index (includes UCB bonus)
      periodic_bonus_i(t) = Gaussian proximity bonus near predicted arrival
                          = exp(-0.5 * ((t_now - t_hat_i) / sigma_i)^2)
                            scaled by PERIODIC_GAMMA so it can shift rankings
                            without permanently dominating

    Every band is scored every step — no band is ever bypassed or frozen
    out of Q-updates. Bands not selected receive the passive Q-update as
    normal (this is unchanged from base WIQL).

    K-aware dwell_half_sigma (carried over from K-scaling fix, now unused
    as the hard dwell gate is removed, but dwell_half_sigma still controls
    the Gaussian bonus width):
      dwell_half_sigma = min(3.0, 6.0/K)
    """
    PERIODIC_GAMMA = 1.0  # tunable weight for the periodic bonus

    def __init__(self, n_bands: int, k: int, use_periodic_bias: bool = True,
                 c_ucb: float = 1.0, epsilon_explore: float = 0.01,
                 periodic_gamma: float = 1.0, rr_warmup_steps: int = 50,
                 prior: float = 0.5, p_stay_occ: float = 0.9,
                 p_stay_idle: float = 0.85, p_detect: float = 0.9,
                 p_fa: float = 0.01):
        self.n_bands = n_bands
        self.k = k
        self.use_periodic_bias = use_periodic_bias
        self.periodic_gamma = periodic_gamma
        self.rr_warmup_steps = rr_warmup_steps
        self._c_ucb = c_ucb
        self._epsilon = epsilon_explore
        self._prior = prior
        self._p_stay_occ = p_stay_occ
        self._p_stay_idle = p_stay_idle
        self._p_detect = p_detect
        self._p_fa = p_fa
        self.bt = BeliefTracker(n_bands=n_bands, p_stay_occ=p_stay_occ,
                                p_stay_idle=p_stay_idle, prior=prior,
                                p_detect=p_detect, p_fa=p_fa)
        self.sched = WIQLScheduler(n_bands=n_bands, k_scan=k,
                                   c_ucb=c_ucb, epsilon_explore=epsilon_explore)
        # K-aware sigma for Gaussian bonus width
        k_aware_dwell_half = min(3.0, 6.0 / k)
        self.pmod = (PeriodicInterceptModule(n_bands=n_bands,
                                              dwell_half_sigma=k_aware_dwell_half)
                     if use_periodic_bias else None)
        self._prev_belief: np.ndarray | None = None

    def reset(self):
        self.bt = BeliefTracker(self.n_bands, p_stay_occ=self._p_stay_occ,
                                p_stay_idle=self._p_stay_idle, prior=self._prior,
                                p_detect=self._p_detect, p_fa=self._p_fa)
        self.sched = WIQLScheduler(self.n_bands, k_scan=self.k,
                                   c_ucb=self._c_ucb,
                                   epsilon_explore=self._epsilon)
        k_aware_dwell_half = min(3.0, 6.0 / self.k)
        self.pmod = (PeriodicInterceptModule(self.n_bands,
                                              dwell_half_sigma=k_aware_dwell_half)
                     if self.use_periodic_bias else None)
        self._prev_belief = None

    def _periodic_bonus(self, band_id: int, t_now_us: float) -> float:
        """Gaussian proximity bonus toward predicted next arrival.

        Returns a value in [0, 1] — 1.0 exactly at t_hat, decaying smoothly
        away. This is ADDITIVE to W_i(t), not a gate.
        """
        if self.pmod is None:
            return 0.0
        mu, sigma = self.pmod.estimate_period(band_id)
        if mu is None or mu <= 0 or sigma is None:
            return 0.0
        t_last = self.pmod._estimators[band_id].t_last
        if t_last is None:
            return 0.0
        t_expected = t_last + mu
        sigma_us = max(sigma, 1e-6)
        dist = abs(t_now_us - t_expected)
        # Normalise: sigma_us is the natural scale; use it directly
        return float(np.exp(-0.5 * (dist / sigma_us) ** 2))

    def select(self, t: int, dt_us: float) -> set[int]:
        belief = self.bt.get_all()
        self._prev_belief = belief.copy()

        # ── Round-robin pre-phase (configurable warmup steps) ─────────────────
        if t < self.rr_warmup_steps:
            cursor = (t * self.k) % self.n_bands
            return {(cursor + j) % self.n_bands for j in range(self.k)}

        # ── WIQL + periodic additive index (after warmup) ─────────────────────
        wiql_indices = self.sched.compute_indices(belief)

        if self.pmod is not None:
            t_now_us = t * dt_us
            bonuses = np.array([
                self.periodic_gamma * self._periodic_bonus(b, t_now_us)
                for b in range(self.n_bands)
            ], dtype=np.float64)
            combined = wiql_indices + bonuses
        else:
            combined = wiql_indices

        # Top-K selection with +inf handling and stable tie-breaking (ascending band_id)
        # Treat +inf (unvisited WIQL arms) as very large but finite for sorting
        finite_combined = np.where(np.isinf(combined), 1e15, combined)
        order = np.argsort(-finite_combined, kind="stable")
        return set(int(order[j]) for j in range(self.k))

    def update(self, t: int, action: set[int], obs: dict, dt_us: float):
        """Update belief tracker, WIQL Q-tables, and periodic module."""
        belief_prev = self._prev_belief if self._prev_belief is not None else self.bt.get_all()

        for band in range(self.n_bands):
            if band in action:
                detected = obs[band]["detected"]
                self.bt.update(band, 1 if detected else 0)
                if self.pmod is not None and detected:
                    t_now_us = t * dt_us
                    self.pmod.ingest_pulse(band, t_now_us)
            else:
                self.bt.update(band, None)

        belief_next = self.bt.get_all()
        for band in action:
            obs_bin = 1 if obs[band]["detected"] else 0
            self.sched.update(
                band, ACTION_SCAN, float(obs_bin),
                float(belief_prev[band]), float(belief_next[band])
            )
        for band in range(self.n_bands):
            if band not in action:
                self.sched.update(
                    band, ACTION_PASSIVE, 0.0,
                    float(belief_prev[band]), float(belief_next[band])
                )


# ─────────────────────────────────────────────────────────────────────────────
# Single episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    env: RenewalEnv,
    policy_name: str,
    policy_obj,          # callable(t, n_bands, k, rng, **kw) or WIQLPolicy
    capture_trace: bool = False,
) -> tuple[dict, dict | None]:
    """Run one episode. Returns (metrics_dict, trace_dict_or_None)."""

    n = env.n_bands
    k = env.k_scan
    dt = env.dt_us
    T = env.episode_len
    rng_pol = np.random.default_rng(env.seed + 99_999)  # separate RNG for random policy

    if hasattr(policy_obj, "reset"):
        policy_obj.reset()

    # Accumulators
    intercepts = []
    occupied_slots = []
    rewards = []
    intercept_times_us = []

    # Trace (belief, priority, scan, hit/miss for band 0)
    trace: dict | None = {"belief": [], "priority": [], "scanned": [], "hit": [], "miss": [],
                          "true_active": [], "predicted_toa": []} if capture_trace else None

    env.reset()

    for t in range(T):
        # Select action
        if policy_name == "round_robin":
            action = policy_round_robin(t, n, k)
        elif policy_name == "random":
            action = policy_random(t, n, k, rng=rng_pol)
        elif policy_name in ("wiql_ucb", "wiql_ucb_no_bias"):
            action = policy_obj.select(t, dt)
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        obs = env.step(action)

        # Per-band scoring
        slot_intercepts = 0
        slot_occupied   = 0
        slot_reward     = 0.0

        for band in range(n):
            true_active = obs[band]["true_active"]
            scanned     = obs[band]["scanned"]
            detected    = obs[band]["detected"]

            if true_active:
                slot_occupied += 1
            if true_active and scanned and detected:
                slot_intercepts += 1
                intercept_times_us.append(t * dt)
                slot_reward += 1.0
            elif scanned and detected and not true_active:
                slot_reward -= 0.1  # false-alarm penalty

        intercepts.append(slot_intercepts)
        occupied_slots.append(slot_occupied)
        rewards.append(slot_reward / max(k, 1))

        # Update WIQL policy state
        if policy_name in ("wiql_ucb", "wiql_ucb_no_bias"):
            policy_obj.update(t, action, obs, dt)

        # Capture trace for band 0
        if capture_trace and trace is not None:
            b0_obs = obs[0]
            true_a = b0_obs["true_active"]
            scan_b0 = b0_obs["scanned"]
            det_b0  = b0_obs["detected"]

            if policy_name in ("wiql_ucb", "wiql_ucb_no_bias"):
                belief_b0 = float(policy_obj.bt.get_all()[0])
                idx_arr = policy_obj.sched.compute_indices(policy_obj.bt.get_all())
                priority_b0 = float(idx_arr[0]) if not np.isinf(idx_arr[0]) else 5.0
                pred_toa = policy_obj.pmod.predict_next_arrival(0) if policy_obj.pmod else None
            else:
                belief_b0 = 0.5
                priority_b0 = 0.0
                pred_toa = None

            trace["belief"].append(belief_b0)
            trace["priority"].append(priority_b0)
            trace["scanned"].append(scan_b0)
            trace["hit"].append(scan_b0 and det_b0 and true_a)
            trace["miss"].append(scan_b0 and not det_b0 and true_a)
            trace["true_active"].append(true_a)
            trace["predicted_toa"].append(pred_toa)

    metrics = compute_metrics(intercepts, occupied_slots, rewards, intercept_times_us, dt)
    return metrics, trace


# ─────────────────────────────────────────────────────────────────────────────
# Statistical summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Bootstrap 95% CI. Returns (lower, upper)."""
    arr = np.array(values)
    means = [np.mean(arr[np.random.default_rng(i).integers(0, len(arr), len(arr))])
             for i in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


def wilcoxon_p(wiql_vals: list[float], baseline_vals: list[float]) -> float:
    """Wilcoxon signed-rank test (paired). Returns p-value."""
    try:
        _, p = sp_stats.wilcoxon(wiql_vals, baseline_vals, alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def summarise(results: dict[str, list[dict]]) -> dict[str, dict]:
    """Compute mean, CI, worst-case for each (algo, metric) combination."""
    summary = {}
    for algo, runs in results.items():
        summary[algo] = {}
        for metric in runs[0].keys():
            vals = [r[metric] for r in runs]
            vals_finite = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if not vals_finite:
                summary[algo][metric] = {"mean": float("nan"), "ci_lo": float("nan"),
                                         "ci_hi": float("nan"), "worst": float("nan")}
                continue
            mean = float(np.mean(vals_finite))
            ci_lo, ci_hi = bootstrap_ci(vals_finite)
            worst = float(np.min(vals_finite))
            summary[algo][metric] = {"mean": mean, "ci_lo": ci_lo,
                                     "ci_hi": ci_hi, "worst": worst}
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Core Scheduler vs Baselines
# ─────────────────────────────────────────────────────────────────────────────

def run_test1(
    n_seeds: int = 5,        # smoke: 5; full: 30
    episode_len: int = 500,  # smoke: 500; full: 5000
    n_bands: int = 16,
    k_scan: int = 3,
    dt_us: float = 1_000.0,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Returns:
      raw_rows: list of dicts (long format — one row per seed×scenario×algo)
      summary:  nested dict  {scenario: {algo: {metric: {mean,ci_lo,ci_hi,worst}}}}

    dt_us override per scenario:
      background / freq_agile: use supplied dt_us (1ms default is fine)
      periodic: use dt_us=100μs so T_TSRD_P1≈15 slots, T_TSRD_P2≈5 slots
    """
    DT_PERIODIC = 100.0  # μs — matched to TSRD periodic emitter periods

    scenarios = {
        "background":  (make_background_scenario, dt_us),
        "periodic":    (make_periodic_scenario,   DT_PERIODIC),
        "freq_agile":  (make_freq_agile_scenario, dt_us),
    }
    algorithms = ["round_robin", "random", "wiql_ucb", "wiql_ucb_no_bias"]

    raw_rows: list[dict] = []
    summary: dict = {}
    demo_trace: dict | None = None  # captured from first seed of periodic scenario

    for scen_name, (scenario_factory, scen_dt_us) in scenarios.items():
        if verbose:
            print(f"\n  Scenario: {scen_name} | seeds={n_seeds} T={episode_len} dt={scen_dt_us:.0f}μs")
        algo_runs: dict[str, list[dict]] = {a: [] for a in algorithms}

        for seed in range(n_seeds):
            env_rng = np.random.default_rng(seed)
            emitters_kwargs: dict = {"rng": env_rng, "n_bands": n_bands}
            if scen_name == "periodic":
                emitters_kwargs["dt_us"] = scen_dt_us

            emitters = scenario_factory(**emitters_kwargs)

            env = RenewalEnv(
                n_bands=n_bands, dt_us=scen_dt_us, episode_len=episode_len,
                k_scan=k_scan, seed=seed,
            )
            env.set_emitters(emitters)

            for algo in algorithms:
                use_bias = algo != "wiql_ucb_no_bias"
                if algo in ("wiql_ucb", "wiql_ucb_no_bias"):
                    policy_obj = WIQLPolicy(n_bands, k_scan, use_periodic_bias=use_bias)
                else:
                    policy_obj = None

                capture = (
                    demo_trace is None
                    and scen_name == "periodic"
                    and algo == "wiql_ucb"
                    and seed == 0
                )

                env.reset()  # same emitters, same seed = same ground truth per seed
                metrics, trace = run_episode(env, algo, policy_obj, capture_trace=capture)

                if trace is not None:
                    demo_trace = trace

                algo_runs[algo].append(metrics)
                raw_rows.append({
                    "scenario": scen_name, "algorithm": algo, "seed": seed,
                    **metrics,
                })
                if verbose:
                    print(f"    seed={seed:2d}  {algo:<22}  "
                          f"rate={metrics['intercept_rate']:.3f}  "
                          f"reward={metrics['avg_reward']:.3f}")

        summary[scen_name] = summarise(algo_runs)

        # Wilcoxon tests
        wiql_rates = [r["intercept_rate"] for r in algo_runs["wiql_ucb"]]
        for base in ("round_robin", "random"):
            base_rates = [r["intercept_rate"] for r in algo_runs[base]]
            p = wilcoxon_p(wiql_rates, base_rates)
            summary[scen_name].setdefault("_wilcoxon", {})[f"wiql_vs_{base}"] = p
            if verbose:
                print(f"    Wilcoxon WIQL vs {base}: p={p:.4f}")

    return raw_rows, summary, demo_trace


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Periodic Module vs Jitter
# ─────────────────────────────────────────────────────────────────────────────

JITTER_RATIOS = [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]


def run_test2(
    n_seeds: int = 5,
    episode_len: int = 500,
    n_bands: int = 16,
    k_scan: int = 3,
    dt_us: float = 1_000.0,   # kept for API compat; overridden internally
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """Test 2: Periodic Module vs Jitter.

    Always uses DT_PERIODIC=100μs so T_TSRD_P1≈15 slots, T_TSRD_P2≈5 slots.
    The dt_us argument is accepted but ignored; DT_PERIODIC is used instead.

    Returns:
      raw_rows: long-format list of dicts [jitter_ratio, algorithm, seed, ...]
      summary:  {jitter_ratio: {algo: {metric: {mean,ci_lo,ci_hi,worst}}}}
    """
    DT_PERIODIC = 100.0  # fixed — matched to TSRD periodic emitter periods
    raw_rows: list[dict] = []
    summary: dict = {}

    for j_ratio in JITTER_RATIOS:
        if verbose:
            print(f"\n  Jitter ratio: {j_ratio:.2f} | seeds={n_seeds} dt={DT_PERIODIC:.0f}μs")

        algo_runs: dict[str, list[dict]] = {
            "wiql_ucb": [], "round_robin": [], "random": []
        }

        for seed in range(n_seeds):
            env_rng = np.random.default_rng(seed)
            emitters = make_periodic_scenario(
                rng=env_rng, n_bands=n_bands, dt_us=DT_PERIODIC,
                n_background=2, n_periodic=1, jitter_ratio=j_ratio,
            )
            env = RenewalEnv(
                n_bands=n_bands, dt_us=DT_PERIODIC, episode_len=episode_len,
                k_scan=k_scan, seed=seed,
            )
            env.set_emitters(emitters)

            for algo in ("wiql_ucb", "round_robin", "random"):
                if algo == "wiql_ucb":
                    policy_obj = WIQLPolicy(n_bands, k_scan, use_periodic_bias=True)
                else:
                    policy_obj = None

                env.reset()
                metrics, _ = run_episode(env, algo, policy_obj)
                algo_runs[algo].append(metrics)
                raw_rows.append({
                    "jitter_ratio": j_ratio, "algorithm": algo, "seed": seed,
                    **metrics,
                })

            if verbose:
                r_wiql = algo_runs["wiql_ucb"][-1]["intercept_rate"]
                r_rr   = algo_runs["round_robin"][-1]["intercept_rate"]
                print(f"    seed={seed:2d}  wiql={r_wiql:.3f}  rr={r_rr:.3f}")

        summary[j_ratio] = summarise(algo_runs)

    return raw_rows, summary
