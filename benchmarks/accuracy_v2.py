#!/usr/bin/env python
"""
WP5: accuracy benchmark v2, exactly as fixed in benchmarks/PLAN_accuracy_v2.md (written before any result).
  python accuracy_v2.py OUT_DIR --default_coupling X [--procs 4] [--reps 20]     run (resumable: one JSON per dataset)
  python accuracy_v2.py OUT_DIR --summary                                          tables + verdict by the plan's rule
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

H2 = [0.2, 0.5]; PI = [0.001, 0.01, 0.1]; RHO = [0.4, 0.8]; MULT = [('div3', 1 / 3), ('1', 1.0), ('x3', 3.0)]
SCEN = ['h%s_pi%s_rho%s_m%s' % (h, p, r, m) for h, p, r, (m, _) in itertools.product(H2, PI, RHO, MULT)]
PHIS = [('1e-6', 1e-6), ('1e-4', 1e-4), ('1e-2', 1e-2)]
COMPARE = ['snr', 'effect', 'learned', 'default_2ch']


def parse_scen(s):
    h, p, r, m = s.split('_')
    return dict(h2=float(h[1:]), pi=float(p[2:]), rho=float(r[3:]), mult=dict(MULT)[m[1:]])


def run_one(args):
    out_dir, scen, rep, default = args
    fn = os.path.join(out_dir, 'runs', '%s_r%02d.json' % (scen, rep))
    if os.path.exists(fn):
        return fn
    import synthbench as sb
    seed = 100000 + 100 * SCEN.index(scen) + rep
    t0 = time.time()
    data = sb.make_dataset(seed, M=2000, block=100, **parse_scen(scen))
    est, secs = {}, {}
    for m in ('prscsx', 'snr', 'effect', 'learned'):
        t1 = time.time()
        est[m] = {lab: sb.fit(data, m, phi, seed=seed) for lab, phi in PHIS}
        secs[m] = time.time() - t1
    t1 = time.time()
    second = {lab: sb.fit(data, default, phi, seed=seed + 1000) for lab, phi in PHIS}   # chain 2 of --n_chains=2
    est['default_2ch'] = {lab: (est[default][lab] + second[lab]) / 2 for lab, _ in PHIS}
    secs['default_2ch'] = secs[default] + time.time() - t1
    res = {'scenario': scen, 'rep': rep, 'seed': seed, 'n_causal': data['n_causal'], 'default_coupling': default,
           'seconds': secs, 'methods': {m: sb.evaluate(data, e, n_val=3000, val_seed=seed) for m, e in est.items()}}
    res['total_seconds'] = time.time() - t0
    json.dump(res, open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    return fn


def summary(out_dir):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json')))]
    md = ['# Accuracy benchmark v2: results (rule from PLAN_accuracy_v2.md)', '',
          '%d datasets of %d planned (%d scenarios x 20 replicates).' % (len(runs), 20 * len(SCEN), len(SCEN)), '']
    verdicts, pooled, per_scen = {}, [], []
    for m in COMPARE:
        for t in ('EUR', 'EAS', 'AFR'):
            d = np.array([r['methods'][m][t]['r2_tuned_validation'] - r['methods']['prscsx'][t]['r2_tuned_validation'] for r in runs])
            mean = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(len(d)))
            sm = {}
            for s in SCEN:
                x = [r['methods'][m][t]['r2_tuned_validation'] - r['methods']['prscsx'][t]['r2_tuned_validation'] for r in runs if r['scenario'] == s]
                if x:
                    sm[s] = float(np.mean(x))
            lo, hi = mean - 1.96 * se, mean + 1.96 * se
            if mean > 0.01 and lo > 0 and min(sm.values()) >= -0.01:
                v = 'better'
            elif mean < -0.01 and hi < 0 and max(sm.values()) <= 0.01:
                v = 'worse'
            else:
                v = 'no demonstrated difference'
            verdicts['%s/%s' % (m, t)] = v
            pooled.append(dict(method=m, target=t, n=len(d), mean_diff=mean, ci_low=lo, ci_high=hi, worst_scenario_mean=min(sm.values()),
                               worst_scenario=min(sm, key=sm.get), best_scenario_mean=max(sm.values()), best_scenario=max(sm, key=sm.get),
                               n_better=int((d > 0).sum()), verdict=v))
            for s, v_ in sm.items():
                per_scen.append(dict(method=m, target=t, scenario=s, mean_diff=v_))
    md += ['## Pooled paired differences in tuned r2_g (method - prscsx)', '',
           '| method | target | n | mean diff | 95% CI | worst scenario (mean) | best scenario (mean) | better in | verdict |', '|---|---|---|---|---|---|---|---|---|']
    for p in pooled:
        md.append('| %s | %s | %d | %+.4f | [%+.4f, %+.4f] | %s (%+.4f) | %s (%+.4f) | %d/%d | **%s** |' % (
            p['method'], p['target'], p['n'], p['mean_diff'], p['ci_low'], p['ci_high'], p['worst_scenario'], p['worst_scenario_mean'],
            p['best_scenario'], p['best_scenario_mean'], p['n_better'], p['n'], p['verdict']))
    # per-scenario means of every method
    md += ['', '## Mean tuned r2_g per scenario and target', '', '| scenario | target | prscsx | snr | effect | learned | default_2ch |', '|---|---|---|---|---|---|---|']
    tab = []
    for s in SCEN:
        rr = [r for r in runs if r['scenario'] == s]
        if not rr:
            continue
        for t in ('EUR', 'EAS', 'AFR'):
            v = {m: float(np.mean([r['methods'][m][t]['r2_tuned_validation'] for r in rr])) for m in ['prscsx'] + COMPARE}
            tab.append(dict(scenario=s, target=t, n=len(rr), **v))
            md.append('| %s | %s | %.4f | %.4f | %.4f | %.4f | %.4f |' % (s, t, v['prscsx'], v['snr'], v['effect'], v['learned'], v['default_2ch']))
    general = all(verdicts['%s/%s' % (runs[0]['default_coupling'] if runs else 'snr', t)] == 'better' for t in ('EUR', 'EAS', 'AFR')) if runs else False
    md += ['', 'General claim "improved (default coupling %s, 1 chain) predicts better than PRS-CSx": **%s**' % (
        runs[0]['default_coupling'] if runs else '?', 'yes' if general else 'no')]
    secs = {m: float(np.mean([r['seconds'][m] for r in runs])) for m in ['prscsx'] + COMPARE} if runs else {}
    md += ['', 'Mean seconds per dataset (3 phi values, 1 core): ' + ', '.join('%s %.1f' % kv for kv in secs.items())]
    json.dump(dict(pooled=pooled, per_scenario=per_scen, table=tab, verdicts=verdicts, general_claim=general, n_runs=len(runs),
                   seconds=secs), open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    open(os.path.join(out_dir, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md[:20 + 12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--default_coupling'); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--reps', type=int, default=20); ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'runs'), exist_ok=True)
    if not a.summary:
        jobs = [(a.out, s, r, a.default_coupling) for r in range(1, a.reps + 1) for s in SCEN]
        with get_context('spawn').Pool(a.procs) as pool:
            for fn in pool.imap_unordered(run_one, jobs):
                print(time.strftime('%H:%M:%S'), os.path.basename(fn), flush=True)
    summary(a.out)


if __name__ == '__main__':
    main()
