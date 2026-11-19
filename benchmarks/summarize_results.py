#!/usr/bin/env python
"""
Summarise benchmark results into CSV/Markdown tables and PNG plots.

  python summarize_results.py runtime  --bench /work/bench --out /work/bench/results
  python summarize_results.py accuracy --bench /work/bench --out /work/bench/results [--sim /work/sim] [--ref_dir /work/ref/1kg]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'sim'))
from compare_outputs import compare  # noqa: E402


def _w(path, rows, cols):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def _md(rows, cols, fmt=None):
    fmt = fmt or {}
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join((fmt.get(c, '%s') % r[c]) if r.get(c) is not None else '' for c in cols) + ' |')
    return '\n'.join(out)


# ------------------------------------------------------------------------------------------------------------ runtime
def runtime(bench, out):
    os.makedirs(out, exist_ok=True)
    md = []
    # ---- example data
    rows = []
    for v in ['original_t1', 'fixed_t1', 'fast_t1', 'fast_t4', 'improved_t1', 'improved_t4']:
        ts = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(bench, 'runtime/ex/%s_r*/job_timing.json' % v)))]
        if not ts:
            continue
        w = np.array([t['wall_s'] for t in ts]); c = np.array([t['user_s'] + t['sys_s'] for t in ts])
        rows.append(dict(variant=v, n=len(ts), wall_mean=w.mean(), wall_sd=w.std(ddof=1) if len(w) > 1 else 0, cpu_mean=c.mean(),
                         rss_max=max(t['max_rss_mb'] for t in ts)))
    base = [r for r in rows if r['variant'] == 'original_t1']
    for r in rows:
        r['speedup'] = base[0]['wall_mean'] / r['wall_mean'] if base else None
    cols = ['variant', 'n', 'wall_mean', 'wall_sd', 'cpu_mean', 'rss_max', 'speedup']
    _w(os.path.join(out, 'runtime_example.csv'), rows, cols)
    md += ['## Example data (chr22, 1,000 SNPs, EUR+EAS, 2,000 iterations, phi=1e-2)', '',
           _md(rows, cols, {'wall_mean': '%.1f', 'wall_sd': '%.1f', 'cpu_mean': '%.1f', 'rss_max': '%.0f', 'speedup': '%.2fx'}), '']
    # exactness on the example (same seed): fast vs original, improved vs original
    ex_rows = []
    for rep in (1, 2, 3):
        od = os.path.join(bench, 'runtime/ex/original_t1_r%d' % rep)
        for v in ['fixed_t1', 'fast_t1', 'fast_t4', 'improved_t1', 'improved_t4']:
            nd = os.path.join(bench, 'runtime/ex/%s_r%d' % (v, rep))
            for f in sorted(glob.glob(os.path.join(od, '*_pst_eff_*.txt'))):
                g = os.path.join(nd, os.path.basename(f))
                if os.path.exists(g):
                    c = compare(f, g)
                    ex_rows.append(dict(seed=rep, variant=v, file=os.path.basename(f).split('_pst')[0], identical=c['identical_text'],
                                        max_abs=c['max_abs'], mean_abs=c['mean_abs'], pearson_r=c['pearson_r']))
    _w(os.path.join(out, 'exactness_example.csv'), ex_rows, list(ex_rows[0]) if ex_rows else ['seed'])
    agg = {}
    for r in ex_rows:
        a = agg.setdefault(r['variant'], dict(variant=r['variant'], files=0, identical=0, max_abs=0.0, mean_abs=[], r=[]))
        a['files'] += 1; a['identical'] += int(r['identical']); a['max_abs'] = max(a['max_abs'], r['max_abs'])
        a['mean_abs'].append(r['mean_abs']); a['r'].append(r['pearson_r'])
    arows = [dict(variant=a['variant'], files=a['files'], identical=a['identical'], max_abs=a['max_abs'],
                  mean_abs=float(np.mean(a['mean_abs'])), min_r=float(np.min(a['r']))) for a in agg.values()]
    md += ['### Output agreement with the ORIGINAL (same seed), example data', '',
           _md(arows, ['variant', 'files', 'identical', 'max_abs', 'mean_abs', 'min_r'], {'max_abs': '%.3g', 'mean_abs': '%.3g', 'min_r': '%.6f'}), '']
    # ---- probe chr22
    prow = []
    ot = json.load(open('/work/baseline/fullchr22_seed1_t1/timing.json'))
    prow.append(dict(variant='original_t1', wall=ot['wall_s'], cpu=ot['user_s'] + ot['sys_s'], rss=ot['max_rss_mb']))
    for v in ['fast_t1', 'fast_t4', 'improved_t1', 'improved_t4']:
        f = os.path.join(bench, 'runtime/probe/%s/job_timing.json' % v)
        if os.path.exists(f):
            t = json.load(open(f))
            prow.append(dict(variant=v, wall=t['wall_s'], cpu=t['user_s'] + t['sys_s'], rss=t['max_rss_mb']))
    for r in prow:
        r['speedup'] = prow[0]['wall'] / r['wall']
        d = os.path.join(bench, 'runtime/probe', r['variant'])
        ident = []
        for f in sorted(glob.glob('/work/baseline/fullchr22_seed1_t1/probe_*_pst_eff_*.txt')):
            g = os.path.join(d, os.path.basename(f))
            if os.path.exists(g):
                c = compare(f, g); ident.append((c['identical_text'], c['max_abs'], c['pearson_r']))
        r['identical'] = '%d/%d' % (sum(x[0] for x in ident), len(ident)) if ident else ''
        r['min_r'] = min(x[2] for x in ident) if ident else None
    cols = ['variant', 'wall', 'cpu', 'rss', 'speedup', 'identical', 'min_r']
    _w(os.path.join(out, 'runtime_probe_chr22.csv'), prow, cols)
    md += ['## All HapMap3 SNPs of chr22 (17,007 SNPs, EUR+EAS, 2,000 iterations, phi=1e-2, seed 1)', '',
           _md(prow, cols, {'wall': '%.1f', 'cpu': '%.1f', 'rss': '%.0f', 'speedup': '%.2fx', 'min_r': '%.6f'}), '']
    # ---- scaling
    srows = []
    for d in sorted(glob.glob(os.path.join(bench, 'runtime/scale/*'))):
        tag = os.path.basename(d)
        nsnp = sum(1 for _ in open('/work/sim/scale_%s/target.bim' % tag))
        for m in ['original_t4', 'fast_t4', 'improved_t4']:
            t20 = os.path.join(d, '%s_it20/job_timing.json' % m); t80 = os.path.join(d, '%s_it80/job_timing.json' % m)
            if not (os.path.exists(t20) and os.path.exists(t80)):
                continue
            a = json.load(open(t20)); b = json.load(open(t80))
            per_it = (b['wall_s'] - a['wall_s']) / 60.0
            setup = a['wall_s'] - 20 * per_it
            srows.append(dict(region=tag.replace('_', ',').replace('chr', 'chr '), n_snps=nsnp, method=m, setup_s=setup,
                              per_iter_s=per_it, wall_it80=b['wall_s'], rss_mb=max(a['max_rss_mb'], b['max_rss_mb']),
                              est_full_run_h=(setup + 3000 * per_it) / 3600.0))
    cols = ['region', 'n_snps', 'method', 'setup_s', 'per_iter_s', 'wall_it80', 'rss_mb', 'est_full_run_h']
    _w(os.path.join(out, 'runtime_scaling.csv'), srows, cols)
    md += ['## Scaling (simulated EUR+EAS+AFR, 4 threads/cores each; setup and per-iteration time from runs of 20 and 80 iterations;',
           '   est_full_run_h = setup + 3,000 x per-iteration, i.e. EXTRAPOLATED to the default chain length)', '',
           _md(srows, cols, {'setup_s': '%.1f', 'per_iter_s': '%.3f', 'wall_it80': '%.1f', 'rss_mb': '%.0f', 'est_full_run_h': '%.2f'}), '']
    if srows:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for m, mk in [('original_t4', 'o-'), ('fast_t4', 's-'), ('improved_t4', '^-')]:
            rr = [r for r in srows if r['method'] == m]
            x = [r['n_snps'] for r in rr]
            ax[0].plot(x, [r['est_full_run_h'] for r in rr], mk, label=m)
            ax[1].plot(x, [r['per_iter_s'] for r in rr], mk, label=m)
            ax[2].plot(x, [r['rss_mb'] / 1024 for r in rr], mk, label=m)
        from matplotlib.ticker import NullFormatter
        xs = sorted({r['n_snps'] for r in srows})
        for a, t in zip(ax, ['est. full run (h, 3000 it; extrapolated)', 'seconds / iteration', 'peak RSS (GB)']):
            a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('SNPs (union of 3 ancestries)'); a.set_title(t); a.legend(); a.grid(alpha=.3)
            a.set_xticks(xs); a.set_xticklabels(['%dk' % round(x / 1000) for x in xs]); a.xaxis.set_minor_formatter(NullFormatter())
        fig.tight_layout(); fig.savefig(os.path.join(out, 'runtime_scaling.png'), dpi=120)
    mrows = []
    for d in sorted(glob.glob(os.path.join(bench, 'runtime/mem_after_fix/*'))):
        tag = os.path.basename(d)
        rec = dict(region=tag.replace('_', ',').replace('chr', 'chr '))
        o = [r for r in srows if r['region'] == rec['region'] and r['method'] == 'original_t4']
        rec['original_t4'] = o[0]['rss_mb'] if o else None
        for m in ('fast', 'improved'):
            f = os.path.join(d, m, 'job_timing.json')
            rec[m + '_t4'] = json.load(open(f))['max_rss_mb'] if os.path.exists(f) else None
        mrows.append(rec)
    if mrows:
        _w(os.path.join(out, 'memory_final.csv'), mrows, ['region', 'original_t4', 'fast_t4', 'improved_t4'])
        md += ['## Peak RSS (MB) of the FINAL code (parser keeps only requested ancestries/numeric columns), 4 threads, 20 iterations', '',
               _md(mrows, ['region', 'original_t4', 'fast_t4', 'improved_t4'], {'original_t4': '%.0f', 'fast_t4': '%.0f', 'improved_t4': '%.0f'}), '']
    open(os.path.join(out, 'runtime.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


# ------------------------------------------------------------------------------------------------------------ accuracy
def accuracy(bench, out, sim, ref_dir):
    from evaluate import Truth, evaluate_method
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, 'accuracy_rows.json')
    allrows = json.load(open(cache)) if os.path.exists(cache) else []
    done = {(r['dataset'], r['method'], r['seed']) for r in allrows}
    for dd in sorted(glob.glob(os.path.join(bench, 'accuracy/*'))):
        dname = os.path.basename(dd)
        truth = None
        for md_ in sorted(glob.glob(os.path.join(dd, '*_s*'))):
            mname, seed = os.path.basename(md_).rsplit('_s', 1)
            if (dname, mname, int(seed)) in done:
                continue
            phis = [p for p in os.listdir(md_) if os.path.exists(os.path.join(md_, p, 'job_timing.json'))]
            if len(phis) < 5:
                continue                    # incomplete grid
            truth = truth or Truth(os.path.join(sim, 'acc', dname), ref_dir)
            rows = evaluate_method(truth, md_, 'sim', n_val=3000, val_seed=int(seed))
            for r in rows:
                r.update(dataset=dname, scenario=dname.rsplit('_r', 1)[0], method=mname, seed=int(seed))
            allrows += rows
            done.add((dname, mname, int(seed)))
            json.dump(allrows, open(cache, 'w'))
            print('evaluated', dname, mname, seed, flush=True)
    return allrows


def accuracy_tables(out):
    rows = json.load(open(os.path.join(out, 'accuracy_rows.json')))
    import collections
    md = []
    key_metrics = [('r2_tuned_validation', 'tuned (phi grid + validation, n=3000)'),
                   ('r2_stack_validation', 'stacked over phi x ancestry (NNLS, validation n=3000)'),
                   ('r2_tuned_oracle', 'tuned (oracle phi & weights)'), ('r2_auto_meta', 'auto + META (no tuning)')]
    main = [r for r in rows if r['seed'] == int(r['dataset'][-1])]       # MCMC seed = replicate index
    methods = ['prscsx', 'improved', 'rbonly', 'cplonly', 'ldshrink']
    scen = sorted({r['scenario'] for r in main})
    table = []
    for met, label in key_metrics:
        for s in scen:
            for t in ['EUR', 'EAS', 'AFR']:
                rec = dict(metric=label, scenario=s, target=t)
                for m in methods:
                    v = [r['value'] for r in main if r['metric'] == met and r['scenario'] == s and r['target'] == t and r['method'] == m]
                    rec[m] = float(np.mean(v)) if v else None
                    rec[m + '_n'] = len(v)
                table.append(rec)
    _w(os.path.join(out, 'accuracy_by_scenario.csv'), table, ['metric', 'scenario', 'target'] + methods + [m + '_n' for m in methods])
    md += ['## Accuracy r2_g = R2/h2 (mean over replicates; each replicate = new data + MCMC seed)', '']
    for met, label in key_metrics:
        sub = [r for r in table if r['metric'] == label]
        cols = ['scenario', 'target'] + methods
        md += ['### ' + label, '', _md(sub, cols, {m: '%.4f' for m in methods}), '']
    # paired differences vs prscsx
    diff_rows = []
    for met, label in key_metrics:
        for m in methods[1:]:
            for t in ['EUR', 'EAS', 'AFR']:
                d = []
                for ds in sorted({r['dataset'] for r in main}):
                    a = [r['value'] for r in main if r['metric'] == met and r['dataset'] == ds and r['target'] == t and r['method'] == 'prscsx']
                    b = [r['value'] for r in main if r['metric'] == met and r['dataset'] == ds and r['target'] == t and r['method'] == m]
                    if a and b:
                        d.append(b[0] - a[0])
                if d:
                    d = np.array(d)
                    diff_rows.append(dict(metric=label, method=m, target=t, n_datasets=len(d), mean_diff=d.mean(),
                                          se=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'),
                                          n_better=int((d > 0).sum()), min_diff=d.min(), max_diff=d.max()))
    _w(os.path.join(out, 'accuracy_paired_diff.csv'), diff_rows, list(diff_rows[0]) if diff_rows else ['metric'])
    md += ['## Paired differences vs PRS-CSx (method - prscsx, same datasets)', '',
           _md(diff_rows, ['metric', 'method', 'target', 'n_datasets', 'mean_diff', 'se', 'n_better', 'min_diff', 'max_diff'],
               {'mean_diff': '%+.4f', 'se': '%.4f', 'min_diff': '%+.4f', 'max_diff': '%+.4f'}), '']
    # MCMC seed variability (dataset h0.5_pi0.01_r1)
    seed_rows = []
    for m in ['prscsx', 'improved']:
        for met, label in key_metrics:
            for t in ['EUR', 'EAS', 'AFR']:
                v = [r['value'] for r in rows if r['dataset'] == 'h0.5_pi0.01_r1' and r['method'] == m and r['metric'] == met and r['target'] == t]
                if len(v) > 1:
                    seed_rows.append(dict(method=m, metric=label, target=t, n_seeds=len(v), mean=np.mean(v), sd=np.std(v, ddof=1), min=min(v), max=max(v)))
    _w(os.path.join(out, 'accuracy_mcmc_seed_variability.csv'), seed_rows, list(seed_rows[0]) if seed_rows else ['method'])
    md += ['## MCMC-seed variability of accuracy (dataset h0.5_pi0.01_r1, same data, different MCMC seeds)', '',
           _md(seed_rows, ['method', 'metric', 'target', 'n_seeds', 'mean', 'sd', 'min', 'max'],
               {'mean': '%.4f', 'sd': '%.4f', 'min': '%.4f', 'max': '%.4f'}), '']
    # multi-chain averaging (post hoc, dataset h0.5_pi0.01_r1): single chains (seeds 1, 11, 12) vs averages
    ch_rows = []
    for m in ['prscsx', 'improved']:
        for met, label in key_metrics:
            for t in ['EUR', 'EAS', 'AFR']:
                single = [r['value'] for r in rows if r['dataset'] == 'h0.5_pi0.01_r1' and r['method'] == m and r['metric'] == met and r['target'] == t]
                a2 = [r['value'] for r in rows if r['dataset'] == 'h0.5_pi0.01_r1' and r['method'] == m + '-avg2ch' and r['metric'] == met and r['target'] == t]
                a3 = [r['value'] for r in rows if r['dataset'] == 'h0.5_pi0.01_r1' and r['method'] == m + '-avg3ch' and r['metric'] == met and r['target'] == t]
                if len(single) == 3 and a2 and a3:
                    ch_rows.append(dict(method=m, metric=label, target=t, single_chain_mean=np.mean(single), single_chain_sd=np.std(single, ddof=1),
                                        avg_2_chains=a2[0], avg_3_chains=a3[0]))
    if ch_rows:
        _w(os.path.join(out, 'accuracy_chain_averaging.csv'), ch_rows, list(ch_rows[0]))
        md += ['## Averaging independent chains (post hoc from MCMC seeds 1, 11, 12; dataset h0.5_pi0.01_r1)', '',
               _md(ch_rows, ['method', 'metric', 'target', 'single_chain_mean', 'single_chain_sd', 'avg_2_chains', 'avg_3_chains'],
                   {'single_chain_mean': '%.4f', 'single_chain_sd': '%.4f', 'avg_2_chains': '%.4f', 'avg_3_chains': '%.4f'}), '']
    # estimation error at the oracle-best phi of each method
    err_rows = []
    for s in scen:
        for t in ['EUR', 'EAS', 'AFR']:
            rec = dict(scenario=s, ancestry=t)
            for m in methods:
                v = []
                for ds in sorted({r['dataset'] for r in main if r['scenario'] == s}):
                    e = [r['value'] for r in main if r['metric'] == 'est_error' and r['dataset'] == ds and r['target'] == t
                         and r['method'] == m and r['phi'] != 'phiauto']
                    if e:
                        v.append(min(e))
                rec[m] = float(np.mean(v)) if v else None
            err_rows.append(rec)
    _w(os.path.join(out, 'accuracy_estimation_error.csv'), err_rows, ['scenario', 'ancestry'] + methods)
    md += ['## Relative estimation error ||w - beta||^2/||beta||^2 of the ancestry-specific posterior (best phi; lower is better)', '',
           _md(err_rows, ['scenario', 'ancestry'] + methods, {m: '%.4f' for m in methods}), '']
    # main table: tuned (validation) r2_g, phenotypic R2 = r2_g * h2, correlation r = sqrt(R2), estimation error
    mt = []
    for s in scen:
        h2 = float(s.split('_')[0][1:])
        for t in ['EUR', 'EAS', 'AFR']:
            for m in ['prscsx', 'improved']:
                v = [r['value'] for r in main if r['metric'] == 'r2_tuned_validation' and r['scenario'] == s and r['target'] == t and r['method'] == m]
                e = [rec[m] for rec in err_rows if rec['scenario'] == s and rec['ancestry'] == t]
                if v:
                    r2g = float(np.mean(v))
                    mt.append(dict(scenario=s, target=t, method=m, n=len(v), r2_g=r2g, R2=r2g * h2, r=np.sqrt(r2g * h2),
                                   est_error=e[0] if e else None))
    _w(os.path.join(out, 'accuracy_main_table.csv'), mt, ['scenario', 'target', 'method', 'n', 'r2_g', 'R2', 'r', 'est_error'])
    md += ['## Main table: tuned PRS (phi grid + validation n=3000, README procedure). r2_g = R2/h2; R2 = phenotypic R2 of a '
           'genome-wide PRS with this per-SNP accuracy; r = corr(PRS, phenotype); est_error at best phi', '',
           _md(mt, ['scenario', 'target', 'method', 'n', 'r2_g', 'R2', 'r', 'est_error'],
               {'r2_g': '%.4f', 'R2': '%.4f', 'r': '%.4f', 'est_error': '%.4f'}), '']
    open(os.path.join(out, 'accuracy.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))
    # plot
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(key_metrics), 3, figsize=(15, 4 * len(key_metrics)), squeeze=False)
    for i, (met, label) in enumerate(key_metrics):
        for j, t in enumerate(['EUR', 'EAS', 'AFR']):
            ax = axes[i][j]
            x = np.arange(len(scen)); wdt = 0.8 / len(methods)
            for k, m in enumerate(methods):
                vals = [[r['value'] for r in main if r['metric'] == met and r['scenario'] == s and r['target'] == t and r['method'] == m] for s in scen]
                mean = [np.mean(v) if v else np.nan for v in vals]
                ax.bar(x + (k - 2) * wdt, mean, wdt, label=m)
                for xi, v in zip(x, vals):
                    ax.plot([xi + (k - 2) * wdt] * len(v), v, 'k.', ms=3)
            ax.set_xticks(x); ax.set_xticklabels(scen, rotation=20, fontsize=8); ax.set_title('%s - target %s' % (label, t), fontsize=9)
            ax.set_ylabel('r2_g = R2/h2'); ax.grid(axis='y', alpha=.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'accuracy.png'), dpi=110)
    # paired differences improved - prscsx (one dot per dataset, colour = scenario), mean +- SE
    fig, axes = plt.subplots(1, len(key_metrics), figsize=(4.2 * len(key_metrics), 4), squeeze=False)
    cols = dict(zip(scen, ['C0', 'C1', 'C2', 'C3', 'C4']))
    for i, (met, label) in enumerate(key_metrics):
        ax = axes[0][i]
        for j, t in enumerate(['EUR', 'EAS', 'AFR']):
            ds_list = sorted({r['dataset'] for r in main})
            d = []
            for ds in ds_list:
                a = [r['value'] for r in main if r['dataset'] == ds and r['target'] == t and r['metric'] == met and r['method'] == 'prscsx']
                b = [r['value'] for r in main if r['dataset'] == ds and r['target'] == t and r['metric'] == met and r['method'] == 'improved']
                if a and b:
                    d.append(b[0] - a[0])
                    ax.plot(j + np.random.default_rng(len(d)).uniform(-.15, .15), b[0] - a[0], 'o', ms=5, color=cols[ds.rsplit('_r', 1)[0]],
                            label=ds.rsplit('_r', 1)[0] if (i == 0 and j == 0 and ds.endswith('r1')) else None)
            if d:
                d = np.array(d); se = d.std(ddof=1) / np.sqrt(len(d))
                ax.errorbar(j + 0.3, d.mean(), yerr=1.96 * se, fmt='ks', capsize=4)
        ax.axhline(0, color='grey', lw=1); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['EUR', 'EAS', 'AFR'])
        ax.set_title(label, fontsize=9); ax.set_ylabel('improved - PRS-CSx (r2_g)'); ax.grid(axis='y', alpha=.3)
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Paired accuracy differences per dataset (dots) and mean +- 95% CI (black)', fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'accuracy_paired_diff.png'), dpi=110)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['runtime', 'accuracy', 'accuracy-tables'])
    ap.add_argument('--bench', default='/work/bench'); ap.add_argument('--out', default='/work/bench/results')
    ap.add_argument('--sim', default='/work/sim'); ap.add_argument('--ref_dir', default='/work/ref/1kg')
    a = ap.parse_args()
    if a.what == 'runtime':
        runtime(a.bench, a.out)
    elif a.what == 'accuracy':
        accuracy(a.bench, a.out, a.sim, a.ref_dir)
        accuracy_tables(a.out)
    else:
        accuracy_tables(a.out)
