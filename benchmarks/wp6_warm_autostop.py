#!/usr/bin/env python
"""
WP6.1 / 6.2: evaluation of --warm_start and --auto_stop of PRScsx_improved.py on the WP5 simulated datasets
(scenarios with rho = 0.8 and multiplier 1, the same seeds as benchmarks/accuracy_v2.py, so the datasets are identical).

  python wp6_warm_autostop.py OUT_DIR warm     --coupling C [--reps 5]  [--procs 4]
  python wp6_warm_autostop.py OUT_DIR warm_asc --coupling C [--reps 5]  [--procs 4]   (grid in increasing order)
  python wp6_warm_autostop.py OUT_DIR autostop --coupling C [--reps 20] [--procs 4]
  python wp6_warm_autostop.py OUT_DIR --summary

warm:     phi grid 1, 1e-2, 1e-4, 1e-6 (in this order), PRS-CSx default length (3,000 iterations, 1,500 burn-in).
          cold  = every phi from scratch (seed s);  cold2 = the same with seed s+1000 (Monte Carlo noise reference);
          warm  = phi 2..4 start from the previous phi's final state with burn-in 750 (--warm_burnin default n_burnin/2)
          and the same 1,500 kept iterations.
warm_asc: the same with the grid in increasing order (1e-6, 1e-4, 1e-2, 1); added after the decreasing order had
          been run, to check whether the order matters (it cannot change the speed result).
autostop: 2 chains (seeds s, s+1000) run side by side to 3,000 iterations; the --auto_stop rule (AutoStopRule) is
          evaluated on the way, and the estimate at the stop point is stored ('auto2'); 'fixed2' = both chains at
          3,000 (= --n_chains=2), 'fixed1' = chain 1 at 3,000 (= the default single chain). Because chains own their
          random streams, 'auto2' is exactly what --auto_stop=TRUE --n_chains=2 returns.
Accuracy: synthbench.evaluate (oracle R2/h2 per phi, and phi + weights tuned in a validation sample of 3,000).
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accuracy_v2 as acc  # noqa: E402  (scenario names and seeds)

SCEN6 = [s for s in acc.SCEN if '_rho0.8_' in s and s.endswith('_m1')]
GRID = [('1', 1.0), ('1e-2', 1e-2), ('1e-4', 1e-4), ('1e-6', 1e-6)]
N_ITER, N_BURNIN, WARM_BURNIN = 3000, 1500, 750


def _setup(data):
    import synthbench as sb
    from csximp.mcmc_improved import Solver
    snp, bt, fr, ix = sb._common(data)
    nb = data['M'] // data['block']
    solver = Solver([data['R_ref'][p] for p in sb.POPS], [[data['block']] * nb for _ in sb.POPS], 1, precision='single')
    return sb, snp, bt, fr, ix, solver


def _chain(data, setup, coupling, phi, seed, n_iter, n_burnin, init=None):
    from csximp.mcmc_improved import ChainState
    sb, snp, bt, fr, ix, solver = setup
    return ChainState(1, 0.5, phi, bt, fr, ix, data['n_gwas'], solver, n_iter, n_burnin, 5, list(sb.POPS), data['M'],
                      seed, coupling, True, 1.0, 'FALSE', 'FALSE', init=init)


def _est(chains):
    return np.mean([np.column_stack([ch.running_mean(k) for k in range(3)]) for ch in chains], axis=0)


def run_warm(args):
    out_dir, scen, rep, coupling = args[:4]
    asc = len(args) > 4 and args[4]
    grid = GRID[::-1] if asc else GRID
    fn = os.path.join(out_dir, 'warm_asc' if asc else 'warm', '%s_r%02d.json' % (scen, rep))
    if os.path.exists(fn):
        return fn
    import synthbench as sb
    seed = 100000 + 100 * acc.SCEN.index(scen) + rep
    data = sb.make_dataset(seed, M=2000, block=100, **acc.parse_scen(scen))
    setup = _setup(data)
    est, secs, iters = {}, {}, {}
    for mode, sd in (('cold', seed), ('cold2', seed + 1000), ('warm', seed)):
        est[mode], state, secs[mode], iters[mode] = {}, None, 0.0, 0
        for i, (lab, phi) in enumerate(grid):
            nb = WARM_BURNIN if (mode == 'warm' and i > 0) else N_BURNIN
            ni = N_ITER - N_BURNIN + nb
            t0 = time.perf_counter()
            ch = _chain(data, setup, coupling, phi, sd, ni, nb, init=state if mode == 'warm' else None)
            for _ in range(ni):
                ch.step()
            secs[mode] += time.perf_counter() - t0
            iters[mode] += ni
            est[mode][lab] = _est([ch])
            state = ch.export_state()
    setup[-1].close()
    res = {'scenario': scen, 'rep': rep, 'seed': seed, 'coupling': coupling, 'order': [g[0] for g in grid],
           'seconds': secs, 'iterations': iters,
           'methods': {m: sb.evaluate(data, e, n_val=3000, val_seed=seed) for m, e in est.items()},
           'corr_to_cold': {m: {lab: [float(np.corrcoef(est[m][lab][:, k], est['cold'][lab][:, k])[0, 1]) for k in range(3)]
                                for lab, _ in GRID} for m in ('cold2', 'warm')}}
    json.dump(res, open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    return fn


def run_autostop(args):
    out_dir, scen, rep, coupling = args[:4]
    fn = os.path.join(out_dir, 'autostop', '%s_r%02d.json' % (scen, rep))
    if os.path.exists(fn):
        return fn
    import synthbench as sb
    from csximp.mcmc_improved import AutoStopRule
    seed = 100000 + 100 * acc.SCEN.index(scen) + rep
    data = sb.make_dataset(seed, M=2000, block=100, **acc.parse_scen(scen))
    setup = _setup(data)
    est = {'auto2': {}, 'fixed2': {}, 'fixed1': {}}
    info = {}
    for lab, phi in acc.PHIS:
        chains = [_chain(data, setup, coupling, phi, seed + 1000 * c, N_ITER, N_BURNIN) for c in range(2)]
        rule = AutoStopRule(N_BURNIN)
        stop, t_stop = None, None
        t0 = time.perf_counter()
        for itr in range(1, N_ITER + 1):
            for ch in chains:
                ch.step()
            rule.check(itr, chains)
            if rule.converged and stop is None:
                stop, t_stop = itr, time.perf_counter() - t0
                est['auto2'][lab] = _est(chains)
        t_all = time.perf_counter() - t0
        if stop is None:
            est['auto2'][lab] = _est(chains)
        est['fixed2'][lab] = _est(chains)
        est['fixed1'][lab] = _est(chains[:1])
        info[lab] = {'stop_iteration': stop, 'converged': stop is not None, 'seconds_to_stop': t_stop if stop else t_all,
                     'seconds_full': t_all, 'rhat99_history': rule.history}
    setup[-1].close()
    res = {'scenario': scen, 'rep': rep, 'seed': seed, 'coupling': coupling, 'phi': info,
           'methods': {m: sb.evaluate(data, e, n_val=3000, val_seed=seed) for m, e in est.items()}}
    json.dump(res, open(fn + '.tmp', 'w'))
    os.replace(fn + '.tmp', fn)
    return fn


def _ci(x):
    x = np.asarray(x, float)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return m, m - 1.96 * se, m + 1.96 * se


def summary(out_dir):
    md = ['# WP6: warm start and automatic stopping (improved/)', '']
    res = {}
    for sub, title, first in (('warm', 'phi grid 1, 1e-2, 1e-4, 1e-6 (decreasing)', '1'),
                              ('warm_asc', 'phi grid 1e-6, 1e-4, 1e-2, 1 (increasing)', '1e-6')):
        W = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, sub, '*.json')))]
        if not W:
            continue
        md += ['## 6.1 --warm_start, %s; %d datasets x 3 target ancestries' % (title, len(W)), '']
        speed_it = np.mean([w['iterations']['cold'] / w['iterations']['warm'] for w in W])
        speed_t = np.mean([w['seconds']['cold'] / w['seconds']['warm'] for w in W])
        md += ['Speed-up (cold / warm): iterations %.3fx, measured MCMC time %.3fx (mean over datasets).' % (speed_it, speed_t), '']
        md += ['| phi | warm - cold: mean R2/h2 diff [95% CI] | cold2 - cold (MC noise): mean diff [95% CI] | '
               'mean abs diff warm vs cold | mean abs diff cold2 vs cold |', '|---|---|---|---|---|']
        rows = {}
        for lab, _ in GRID + [('tuned_validation', None)]:
            key = 'r2_tuned_validation' if lab == 'tuned_validation' else 'r2_oracle_phi%s' % lab
            dw, dn = [], []
            for w in W:
                for t in ('EUR', 'EAS', 'AFR'):
                    c = w['methods']['cold'][t][key]
                    dw.append(w['methods']['warm'][t][key] - c); dn.append(w['methods']['cold2'][t][key] - c)
            a, b = _ci(dw), _ci(dn)
            rows[lab] = {'warm_minus_cold': a, 'cold2_minus_cold': b, 'abs_warm': float(np.mean(np.abs(dw))),
                         'abs_noise': float(np.mean(np.abs(dn)))}
            md.append('| %s | %+.5f [%+.5f, %+.5f] | %+.5f [%+.5f, %+.5f] | %.5f | %.5f |'
                      % (lab, *a, *b, np.mean(np.abs(dw)), np.mean(np.abs(dn))))
        later = [lab for lab, _ in GRID if lab != first]
        cc = {m: np.mean([np.mean(w['corr_to_cold'][m][lab]) for w in W for lab in later]) for m in ('warm', 'cold2')}
        md += ['', 'Correlation of posterior means with the cold run (warm-started phi values, mean over datasets and '
               'ancestries): warm %.4f, cold2 (other seed) %.4f.' % (cc['warm'], cc['cold2']), '']
        res[sub] = {'n': len(W), 'speedup_iterations': speed_it, 'speedup_time': speed_t, 'rows': rows, 'corr': cc}
    A = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out_dir, 'autostop', '*.json')))]
    if A:
        md += ['## 6.2 --auto_stop (2 chains, max 3,000 iterations, burn-in 1,500; %d datasets x 3 phi)' % len(A), '']
        stops = [a['phi'][lab]['stop_iteration'] for a in A for lab, _ in acc.PHIS]
        conv = [s is not None for s in stops]
        it = np.array([s if s is not None else N_ITER for s in stops], float)
        md += ['Converged before the maximum: %d of %d runs; iterations used: median %d, mean %.0f, range %d-%d.'
               % (sum(conv), len(conv), np.median(it), it.mean(), it.min(), it.max()),
               'Cost relative to --n_chains=2 fixed length: %.3f (iterations); relative to the default single chain: %.3f.'
               % (it.mean() / N_ITER, 2 * it.mean() / N_ITER), '']
        md += ['| comparison | metric | mean diff [95% CI] |', '|---|---|---|']
        rows = {}
        for a_, b_ in (('auto2', 'fixed2'), ('auto2', 'fixed1'), ('fixed2', 'fixed1')):
            for key in ('r2_tuned_oracle', 'r2_tuned_validation'):
                d = [x['methods'][a_][t][key] - x['methods'][b_][t][key] for x in A for t in ('EUR', 'EAS', 'AFR')]
                ci = _ci(d)
                rows['%s-%s:%s' % (a_, b_, key)] = ci
                md.append('| %s - %s | %s | %+.5f [%+.5f, %+.5f] |' % (a_, b_, key, *ci))
        md += ['']
        by = {}
        for s in SCEN6:
            v = [a['phi'][lab]['stop_iteration'] or N_ITER for a in A if a['scenario'] == s for lab, _ in acc.PHIS]
            if v:
                by[s] = float(np.mean(v))
        md += ['Mean stop iteration by scenario: ' + ', '.join('%s %.0f' % kv for kv in by.items()), '']
        res['autostop'] = {'n': len(A), 'converged': int(sum(conv)), 'runs': len(conv), 'iter_mean': float(it.mean()),
                           'iter_median': float(np.median(it)), 'rows': rows, 'by_scenario': by}
    open(os.path.join(out_dir, 'wp6_summary.md'), 'w').write('\n'.join(md) + '\n')
    json.dump(res, open(os.path.join(out_dir, 'wp6_summary.json'), 'w'), indent=1)
    print('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('mode', nargs='?', choices=['warm', 'warm_asc', 'autostop'])
    ap.add_argument('--coupling', default=None)
    ap.add_argument('--reps', type=int, default=None)
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--summary', action='store_true')
    a = ap.parse_args()
    if a.summary:
        return summary(a.out_dir)
    if a.coupling is None:
        sys.exit('--coupling is required (the default chosen in WP4)')
    reps = a.reps or (20 if a.mode == 'autostop' else 5)
    os.makedirs(os.path.join(a.out_dir, a.mode), exist_ok=True)
    jobs = [(a.out_dir, s, r, a.coupling, a.mode == 'warm_asc') for r in range(reps) for s in SCEN6]
    fn = run_autostop if a.mode == 'autostop' else run_warm
    t0 = time.time()
    with get_context('fork').Pool(a.procs) as pool:
        for i, f in enumerate(pool.imap_unordered(fn, jobs)):
            print('%d/%d %s (%.0f s)' % (i + 1, len(jobs), os.path.basename(f), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
