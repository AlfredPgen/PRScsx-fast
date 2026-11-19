#!/usr/bin/env python
"""
WP4 step 3: stress test of the cross-ancestry coupling (extends the review's stress.py, re-implemented here because
the review files were not available). Five scenarios x 20 replicates; methods improved/ with --coupling snr (the
PRS-CSx model), effect and learned (everything else equal: RB means, single precision, default chain length);
synthetic LD, 2,000 SNPs (10 blocks of 200), GWAS 300k / 100k / 30k; oracle tuning over phi in {1e-4, 1e-2, 1}.
Scenarios (h2 = 0.5, pi = 1%):
  equal      : equal effect variance, rho = 0.8 (the assumption of --coupling=effect)
  afr_x3     : AFR effect variance 3 times larger, rho = 0.8
  afr_x10    : AFR effect variance 10 times larger, rho = 0.8
  afr_div3   : AFR effect variance 3 times smaller, rho = 0.8
  rho0.4     : equal effect variance, rho = 0.4
Usage: python coupling_stress.py OUT_DIR [--reps 20] [--procs 4] [--summary]
One JSON per (scenario, replicate) in OUT_DIR/runs (resumable); --summary writes OUT_DIR/summary.json and .md.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCEN = {'equal': dict(mult=1.0, rho=0.8), 'afr_x3': dict(mult=3.0, rho=0.8), 'afr_x10': dict(mult=10.0, rho=0.8),
        'afr_div3': dict(mult=1 / 3, rho=0.8), 'rho0.4': dict(mult=1.0, rho=0.4)}
METHODS = ['snr', 'effect', 'learned']
PHIS = [('1e-4', 1e-4), ('1e-2', 1e-2), ('1', 1.0)]


def run_one(args):
    out_dir, scen, rep = args
    fn = os.path.join(out_dir, 'runs', '%s_r%02d.json' % (scen, rep))
    if os.path.exists(fn):
        return fn
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import synthbench as sb
    t0 = time.time()
    seed = 1000 * list(SCEN).index(scen) + rep
    data = sb.make_dataset(seed, **SCEN[scen])
    res = {'scenario': scen, 'rep': rep, 'seed': seed, 'n_causal': data['n_causal'], 'methods': {}}
    for m in METHODS:
        est = {lab: sb.fit(data, m, phi, seed=seed) for lab, phi in PHIS}
        res['methods'][m] = sb.evaluate(data, est, val_seed=seed)
    res['seconds'] = time.time() - t0
    tmp = fn + '.tmp'
    json.dump(res, open(tmp, 'w'), indent=0)
    os.replace(tmp, fn)
    return fn


def summary(out_dir):
    import glob
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'runs', '*.json')))]
    rows, md = [], ['| scenario | target | n | snr | effect | learned | effect - snr (SE) | learned - snr (SE) | learned - best(snr,effect) |',
                    '|---|---|---|---|---|---|---|---|---|']
    for scen in SCEN:
        rr = [r for r in runs if r['scenario'] == scen]
        if not rr:
            continue
        for t in ('EUR', 'EAS', 'AFR'):
            v = {m: np.array([r['methods'][m][t]['r2_tuned_oracle'] for r in rr]) for m in METHODS}
            mean = {m: float(v[m].mean()) for m in METHODS}
            d_es = v['effect'] - v['snr']; d_ls = v['learned'] - v['snr']
            se = lambda d: float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')
            best = max(mean['snr'], mean['effect'])
            rec = dict(scenario=scen, target=t, n=len(rr), **mean, effect_minus_snr=float(d_es.mean()), se_effect_minus_snr=se(d_es),
                       learned_minus_snr=float(d_ls.mean()), se_learned_minus_snr=se(d_ls), learned_minus_best=mean['learned'] - best,
                       effect_worse_in=int((d_es < 0).sum()), learned_worse_than_snr_in=int((d_ls < 0).sum()))
            rows.append(rec)
            md.append('| %s | %s | %d | %.4f | %.4f | %.4f | %+.4f (%.4f) | %+.4f (%.4f) | %+.4f |' % (
                scen, t, len(rr), mean['snr'], mean['effect'], mean['learned'], rec['effect_minus_snr'], rec['se_effect_minus_snr'],
                rec['learned_minus_snr'], rec['se_learned_minus_snr'], rec['learned_minus_best']))
    lb = [r['learned_minus_best'] for r in rows]; ls = [r['learned_minus_snr'] for r in rows]
    verdict = ('learned within 0.005 of the better of snr and effect everywhere' if min(lb) >= -0.005 else
               'learned never more than 0.005 below snr' if min(ls) >= -0.005 else 'learned more than 0.005 below snr somewhere')
    json.dump(dict(rows=rows, min_learned_minus_best=min(lb), min_learned_minus_snr=min(ls), verdict=verdict,
                   n_runs=len(runs)), open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    md += ['', 'Mean r2_g (= R2/h2, oracle tuning over phi in {1e-4, 1e-2, 1}) over replicates; SE of the paired difference.',
           'min(learned - best of snr/effect) = %+.4f; min(learned - snr) = %+.4f -> %s' % (min(lb), min(ls), verdict)]
    open(os.path.join(out_dir, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--reps', type=int, default=20); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--summary', action='store_true'); ap.add_argument('--scenarios', default=','.join(SCEN))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'runs'), exist_ok=True)
    if not a.summary:
        jobs = [(a.out, s, r) for r in range(1, a.reps + 1) for s in a.scenarios.split(',')]
        with get_context('spawn').Pool(a.procs) as pool:
            for fn in pool.imap_unordered(run_one, jobs):
                print(time.strftime('%H:%M:%S'), os.path.basename(fn), flush=True)
    summary(a.out)


if __name__ == '__main__':
    main()
