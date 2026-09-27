#!/usr/bin/env bash
# ============================================================
# Reproduce all paper results (post-fix system, warmup=50)
# SIH 2026 PS 26055 — Smart Scan Strategy for EW
# Coding Saints, Team ID 120303
#
# Usage:  bash scripts/reproduce_paper_results.sh
#
# Writes fresh CSVs to:  results/paper_results/
# Writes summary JSON:   results/paper_results/summary.json
#
# Runtime: ~20–30 minutes on a commodity laptop
#          (N=30 seeds × 5000 steps × 3 scenarios + γ-sweep + SNR sweep)
# ============================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "  Smart Scan Strategy — PS 26055 — Paper Reproduction"
echo "  Coding Saints, Team ID 120303"
echo "  Config: N=30, T=5000, n_bands=8, K=3, warmup=50, γ=5"
echo "============================================================"
echo ""

# Verify dependencies
python3 -c "import numpy, scipy, matplotlib, pandas" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

python3 -c "import sys; sys.path.insert(0,'.');
from src.evaluation.runner import WIQLPolicy, run_episode
from src.evaluation.env_generator import RenewalEnv
" || {
    echo "ERROR: Cannot import src/. Run from repo root."
    exit 1
}

mkdir -p results/paper_results

# ============================================================
python3 - << 'PYEOF'
import sys, csv, json, pathlib, numpy as np
from scipy import stats as sp_stats
sys.path.insert(0, '.')
from src.evaluation.env_generator import (
    RenewalEnv, make_background_scenario, make_periodic_scenario,
    make_freq_agile_scenario,
)
from src.evaluation.runner import WIQLPolicy, run_episode
from src.baselines.round_robin import RoundRobinPolicy
from src.baselines.random_policy import RandomPolicy
from src.baselines.clarkson import ClarksonPolicy

RESULTS  = pathlib.Path('results/paper_results')
N        = 30
T        = 5000
NB       = 8
K        = 3
DT_P     = 100.0
DT_D     = 1000.0
WARMUP   = 50
GAMMA    = 5.0
P_DET    = 0.9
P_FA     = 0.01
PRIOR    = 0.5
CI_BOOTS = 2000

def bci(vals, n=CI_BOOTS):
    arr = np.array(vals)
    ms  = [np.mean(arr[np.random.default_rng(i).integers(0,len(arr),len(arr))]) for i in range(n)]
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))

def wxon(a, b):
    try:
        r = sp_stats.wilcoxon(a, b, alternative='greater', method='exact')
        return float(r.statistic), float(r.pvalue)
    except:
        return float('nan'), float('nan')

print("=" * 60)
print("STEP 1/4 — Core Scheduler vs Baselines (3 scenarios)")
print("=" * 60)

scenario_cfgs = [
    ('background', make_background_scenario, DT_D),
    ('periodic',   make_periodic_scenario,   DT_P),
    ('freq_agile', make_freq_agile_scenario,  DT_D),
]
core_rows = []
core_summary = {}

for scen, factory, dt in scenario_cfgs:
    print(f"  {scen}  (dt={dt:.0f}μs)...", flush=True)
    w_r=[]; rr_r=[]; ra_r=[]; cl_r=[]
    for seed in range(N):
        rng = np.random.default_rng(seed)
        kw  = {'rng': rng, 'n_bands': NB}
        if scen == 'periodic': kw['dt_us'] = dt
        emitters = factory(**kw)

        env_w = RenewalEnv(n_bands=NB, dt_us=dt, episode_len=T, k_scan=K, seed=seed)
        env_w.set_emitters(emitters)
        pol_w = WIQLPolicy(n_bands=NB, k=K, use_periodic_bias=True,
                           periodic_gamma=GAMMA, rr_warmup_steps=WARMUP,
                           prior=PRIOR, p_detect=P_DET, p_fa=P_FA)
        m_w, _ = run_episode(env_w, 'wiql_ucb', pol_w)
        w_r.append(m_w['intercept_rate'])

        for BaseCls, lst, kw2 in [
            (RoundRobinPolicy, rr_r, {}),
            (RandomPolicy,     ra_r, {'seed': seed+999}),
        ]:
            env_b = RenewalEnv(n_bands=NB, dt_us=dt, episode_len=T, k_scan=K, seed=seed)
            env_b.set_emitters(emitters)
            pol_b = BaseCls(n_bands=NB, k_scan=K, **kw2)
            m_b, _ = run_episode(env_b, 'round_robin', pol_b)
            lst.append(m_b['intercept_rate'])

        env_cl = RenewalEnv(n_bands=NB, dt_us=dt, episode_len=T, k_scan=K, seed=seed)
        env_cl.set_emitters(emitters)
        pol_cl = ClarksonPolicy.from_renewal_env(env_cl, k_scan=K)
        env_cl.reset(); pol_cl.reset()
        ci2=co2=0
        for t in range(T):
            act=pol_cl.select_arms(); obs=env_cl.step(act)
            for b in range(NB):
                if obs[b]['true_active']: co2+=1
                if obs[b]['true_active'] and obs[b]['scanned'] and obs[b]['detected']: ci2+=1
        cl_r.append(ci2/max(co2,1))

        core_rows.append({'scenario':scen,'seed':seed,
                          'wiql_rate':w_r[-1],'rr_rate':rr_r[-1],
                          'random_rate':ra_r[-1],'clarkson_rate':cl_r[-1]})

    lo_w,hi_w=bci(w_r); W_rr,p_rr=wxon(w_r,rr_r)
    core_summary[scen] = {
        'wiql':    {'mean':round(float(np.mean(w_r)),4),'ci_lo':round(lo_w,4),'ci_hi':round(hi_w,4),'worst':round(float(min(w_r)),4),'N':N},
        'rr':      {'mean':round(float(np.mean(rr_r)),4),'ci_lo':round(bci(rr_r)[0],4),'ci_hi':round(bci(rr_r)[1],4),'delta_vs_wiql':round(float(np.mean(w_r))-float(np.mean(rr_r)),4),'wilcoxon_p':round(p_rr,10),'N':N},
        'random':  {'mean':round(float(np.mean(ra_r)),4),'N':N},
        'clarkson':{'mean':round(float(np.mean(cl_r)),4),'N':N},
    }
    print(f"    WIQL={np.mean(w_r):.4f}[{lo_w:.4f},{hi_w:.4f}]  "
          f"RR={np.mean(rr_r):.4f}  Δ={float(np.mean(w_r))-float(np.mean(rr_r)):+.4f}  p={p_rr:.2e}")

p = RESULTS/'core_scheduler.csv'
with open(p,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(core_rows[0].keys())); w.writeheader(); w.writerows(core_rows)
print(f"  → {p}  ({len(core_rows)} rows)")

print()
print("=" * 60)
print("STEP 2/4 — γ-Sweep (periodic scenario, γ ∈ {0,0.5,1,2,5,10})")
print("=" * 60)

GAMMAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
gamma_rows = []; gamma_summary = {}
baseline_rates = None

for gamma in GAMMAS:
    rates = []
    for seed in range(N):
        rng = np.random.default_rng(seed)
        emitters = make_periodic_scenario(rng=rng, n_bands=NB, dt_us=DT_P)
        env = RenewalEnv(n_bands=NB, dt_us=DT_P, episode_len=T, k_scan=K, seed=seed)
        env.set_emitters(emitters)
        pol = WIQLPolicy(n_bands=NB, k=K, use_periodic_bias=(gamma>0),
                         periodic_gamma=gamma, rr_warmup_steps=WARMUP,
                         prior=PRIOR, p_detect=P_DET, p_fa=P_FA)
        m, _ = run_episode(env, 'wiql_ucb', pol)
        rates.append(m['intercept_rate'])
        gamma_rows.append({'gamma':gamma,'seed':seed,'intercept_rate':m['intercept_rate']})
    if gamma == 0.0:
        baseline_rates = rates[:]
    lo, hi = bci(rates)
    d = float(np.mean(rates)) - float(np.mean(baseline_rates)) if baseline_rates else 0.0
    W, p_val = (float('nan'), float('nan')) if gamma==0 else wxon(rates, baseline_rates)
    p_bonf = min(1.0, p_val * 5) if not (p_val != p_val) else float('nan')  # M=5
    gamma_summary[str(gamma)] = {
        'mean':round(float(np.mean(rates)),4),'ci_lo':round(lo,4),'ci_hi':round(hi,4),
        'worst':round(float(min(rates)),4),'delta':round(d,4),
        'wilcoxon_W': W if W==W else None,
        'p_uncorrected': round(p_val,10) if p_val==p_val else None,
        'p_bonferroni_m5': round(p_bonf,10) if p_bonf==p_bonf else None,
        'N': N,
    }
    p_str = f"{p_val:.2e}" if p_val==p_val else "—"
    pb_str = f"{p_bonf:.2e}" if p_bonf==p_bonf else "—"
    print(f"  γ={gamma:>5.1f}  mean={np.mean(rates):.4f}[{lo:.4f},{hi:.4f}]  Δ={d:+.4f}  p={p_str}  Bonf={pb_str}")

pg = RESULTS/'gamma_sweep_perseeds.csv'
with open(pg,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['gamma','seed','intercept_rate']); w.writeheader(); w.writerows(gamma_rows)
print(f"  → {pg}  ({len(gamma_rows)} rows)")

print()
print("=" * 60)
print("STEP 3/4 — Seven FoMs (3 scenarios)")
print("=" * 60)

fom_rows=[]; fom_summary={}
for scen, factory, dt in scenario_cfgs:
    print(f"  {scen}...", flush=True)
    pd_v=[]; pfa_v=[]; rew_v=[]; bc_v=[]
    for seed in range(N):
        rng = np.random.default_rng(seed)
        kw = {'rng':rng,'n_bands':NB}
        if scen=='periodic': kw['dt_us']=dt
        emitters = factory(**kw)
        env = RenewalEnv(n_bands=NB, dt_us=dt, episode_len=T, k_scan=K, seed=seed)
        env.set_emitters(emitters)
        pol = WIQLPolicy(n_bands=NB, k=K, use_periodic_bias=True,
                         periodic_gamma=GAMMA, rr_warmup_steps=WARMUP,
                         prior=PRIOR, p_detect=P_DET, p_fa=P_FA)
        pol.reset(); env.reset()
        tp=fp=fn=tn=0; rewards=[]; bc_all=[]
        for t in range(T):
            action=pol.select(t,dt); obs=env.step(action); pol.update(t,action,obs,dt)
            gt=env.get_true_occupancy(t); beliefs=pol.bt.get_all()
            bc_all.append(sum(1 for b in range(NB) if (beliefs[b]>0.5)==gt[b])/NB)
            sr=0.0
            for b in range(NB):
                ta=obs[b]['true_active']; sc=obs[b]['scanned']; de=obs[b]['detected']
                if sc:
                    if ta and de:           tp+=1; sr+=1.0
                    elif not ta and de:     fp+=1; sr-=0.1
                    elif ta and not de:     fn+=1
                    else:                   tn+=1
            rewards.append(sr/max(K,1))
        pd_v.append(tp/max(tp+fn,1)); pfa_v.append(fp/max(fp+tn,1))
        rew_v.append(float(np.mean(rewards))); bc_v.append(float(np.mean(bc_all)))
        fom_rows.append({'scenario':scen,'seed':seed,'pd':pd_v[-1],'pfa':pfa_v[-1],
                         'avg_reward':rew_v[-1],'pct_correct':bc_v[-1]})
    lo_pd,hi_pd=bci(pd_v); lo_r,hi_r=bci(rew_v); lo_b,hi_b=bci(bc_v)
    fom_summary[scen] = {
        'pd':{'mean':round(float(np.mean(pd_v)),4),'ci_lo':round(lo_pd,4),'ci_hi':round(hi_pd,4)},
        'pfa':{'mean':round(float(np.mean(pfa_v)),6)},
        'avg_reward':{'mean':round(float(np.mean(rew_v)),4),'ci_lo':round(lo_r,4),'ci_hi':round(hi_r,4)},
        'pct_correct':{'mean':round(float(np.mean(bc_v)),4),'ci_lo':round(lo_b,4),'ci_hi':round(hi_b,4)},
    }
    print(f"    Pd={np.mean(pd_v):.4f}[{lo_pd:.4f},{hi_pd:.4f}]  Pfa={np.mean(pfa_v):.5f}  "
          f"reward={np.mean(rew_v):.4f}  belief_acc={np.mean(bc_v):.4f}")

pf = RESULTS/'seven_foms.csv'
with open(pf,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(fom_rows[0].keys())); w.writeheader(); w.writerows(fom_rows)
print(f"  → {pf}  ({len(fom_rows)} rows)")

print()
print("=" * 60)
print("STEP 4/4 — SNR Sensitivity Sweep (background, receiver threshold -5→+21 dB)")
print("=" * 60)

import math
def pd_sig(snr, thr, slope=3.0):
    return 1.0 / (1.0 + math.exp(-(snr - thr) / slope))

snr_rows=[]; snr_summary=[]
for thr in range(-5, 22, 2):
    pds=[]
    for seed in range(N):
        rng = np.random.default_rng(seed)
        emitters = make_background_scenario(rng=rng, n_bands=NB)
        env = RenewalEnv(n_bands=NB, dt_us=DT_D, episode_len=T, k_scan=K, seed=seed)
        env.set_emitters(emitters)
        pol = WIQLPolicy(n_bands=NB, k=K, use_periodic_bias=False,
                         rr_warmup_steps=WARMUP, prior=PRIOR, p_detect=P_DET, p_fa=P_FA)
        pol.reset(); env.reset()
        tp_s=occ_s=0
        for t in range(T):
            action=pol.select(t,DT_D); gt=env._get_ground_truth(t)
            obs=env.step(action); pol.update(t,action,obs,DT_D)
            for b in action:
                ta,snr=gt[b]
                if ta:
                    occ_s+=1
                    if env.rng.random() < pd_sig(snr, thr): tp_s+=1
        pds.append(tp_s/max(occ_s,1))
        snr_rows.append({'snr_threshold_db':thr,'seed':seed,'pd':pds[-1]})
    lo, hi = bci(pds)
    snr_summary.append({'snr_threshold_db':thr,'pd_mean':round(float(np.mean(pds)),4),
                         'pd_ci_lo':round(lo,4),'pd_ci_hi':round(hi,4),
                         'pd_worst':round(float(min(pds)),4),'N':N})
    sig = '✓' if lo >= 0.90 else ''
    print(f"  thr={thr:>4} dB  Pd={np.mean(pds):.4f}  CI=[{lo:.4f},{hi:.4f}]  {sig}")

sens_thr = next((r['snr_threshold_db'] for r in snr_summary if r['pd_ci_lo'] >= 0.90), None)

ps1 = RESULTS/'snr_sweep_perseeds.csv'
with open(ps1,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['snr_threshold_db','seed','pd']); w.writeheader(); w.writerows(snr_rows)
ps2 = RESULTS/'snr_sweep_summary.csv'
with open(ps2,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(snr_summary[0].keys())); w.writeheader(); w.writerows(snr_summary)
print(f"  Sensitivity threshold (CI lo ≥ 0.90): {sens_thr} dB")
print(f"  → {ps1}  ({len(snr_rows)} rows)")
print(f"  → {ps2}")

# Write summary JSON
summary = {
    'metadata': {
        'description': 'Post-fix paper results (warmup=50, gamma=5)',
        'N': N, 'T': T, 'n_bands': NB, 'K': K,
        'warmup': WARMUP, 'gamma': GAMMA,
        'p_detect': P_DET, 'p_fa': P_FA, 'prior': PRIOR,
        'ci_method': f'bootstrap B={CI_BOOTS} 95%',
        'test': 'Wilcoxon signed-rank one-sided exact N=30 pairs',
        'metric': 'intercepts / total_occupied_all_bands',
        'note_standalone': (
            'The 0.975 standalone result is a CONTROLLED DEMONSTRATION '
            'from scripts/controlled_demonstrations/. It uses a single synthetic '
            'emitter (T=500us, sigma=20us, dt=50us, 2000 steps) — not the standard harness. '
            'See results/controlled_demonstrations/README.md.'
        ),
    },
    'core_scheduler_vs_baselines': core_summary,
    'gamma_sweep': gamma_summary,
    'seven_foms': fom_summary,
    'snr_sensitivity': {
        'sensitivity_threshold_db': sens_thr,
        'model': 'Pd(snr) = sigmoid((emitter_snr - threshold_db) / 3.0)',
        'sweep': snr_summary,
    },
}
pj = RESULTS/'summary.json'
pj.write_text(json.dumps(summary, indent=2))
print()
print(f"  → {pj}")

print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
print()
print("Core Scheduler vs Round-Robin:")
for scen, v in core_summary.items():
    print(f"  {scen:<15} WIQL={v['wiql']['mean']:.4f}[{v['wiql']['ci_lo']:.4f},{v['wiql']['ci_hi']:.4f}]  "
          f"RR={v['rr']['mean']:.4f}  Δ={v['rr']['delta_vs_wiql']:+.4f}  p={v['rr']['wilcoxon_p']:.2e}")
print()
print("γ-Sweep (periodic, Bonferroni M=5):")
for g, v in gamma_summary.items():
    p_str = f"{v['p_bonferroni_m5']:.2e}" if v.get('p_bonferroni_m5') else "—"
    print(f"  γ={float(g):>5.1f}  mean={v['mean']:.4f}  Δ={v['delta']:+.4f}  Bonf-p={p_str}")
print()
print("Seven FoMs (WIQL-UCB, post-fix):")
for scen, v in fom_summary.items():
    print(f"  {scen:<15} Pd={v['pd']['mean']:.4f}[{v['pd']['ci_lo']:.4f},{v['pd']['ci_hi']:.4f}]  "
          f"Pfa={v['pfa']['mean']:.5f}  reward={v['avg_reward']['mean']:.4f}")
print()
print(f"SNR Sensitivity: {sens_thr} dB (receiver threshold, Pd ≥ 0.90 at 95% CI)")
print()
print("NOTE: Standalone 0.975 is a controlled demonstration — see")
print("      results/controlled_demonstrations/README.md")
print()
print("All CSVs written to results/paper_results/")
print("Summary JSON: results/paper_results/summary.json")
PYEOF

echo ""
echo "============================================================"
echo "  Reproduction complete."
echo "  Verify: check results/paper_results/summary.json"
echo "============================================================"
