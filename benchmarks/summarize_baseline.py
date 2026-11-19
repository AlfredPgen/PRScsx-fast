#!/usr/bin/env python
"""Collect baseline timing JSONs + output comparisons into CSV/Markdown tables."""
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_outputs import compare  # noqa: E402

B = sys.argv[1] if len(sys.argv) > 1 else '/work/baseline'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(B, 'summary')
os.makedirs(OUT, exist_ok=True)

rows = []
for tj in sorted(glob.glob(os.path.join(B, '*', 'timing.json'))):
    run = os.path.basename(os.path.dirname(tj))
    t = json.load(open(tj))
    thr = 'default(4)' if not any('NUM_THREADS=1' in e for e in t.get('env_overrides', [])) else '1'
    if run == 'readme_example':
        thr = 'default(4)'
    nfiles = len(glob.glob(os.path.join(os.path.dirname(tj), '*_pst_eff_*.txt')))
    rows.append(dict(run=run, blas_threads=thr, exit_code=t['exit_code'], wall_s=round(t['wall_s'], 2),
                     cpu_user_s=round(t['user_s'], 2), max_rss_mb=round(t['max_rss_mb'], 1), output_files=nfiles))
with open(os.path.join(OUT, 'baseline_runtime.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

md = ['| run | BLAS threads | exit | wall (s) | CPU user (s) | peak RSS (MB) | output files |', '|---|---|---|---|---|---|---|']
for r in rows:
    md.append('| %s | %s | %d | %.2f | %.2f | %.1f | %d |' % (r['run'], r['blas_threads'], r['exit_code'], r['wall_s'],
                                                           r['cpu_user_s'], r['max_rss_mb'], r['output_files']))
ex = [r['wall_s'] for r in rows if r['run'].startswith('phi1e-2_seed') and r['run'].endswith('_t1') and 'rerun' not in r['run']]
md.append('')
md.append('Example data, phi=1e-2, seeds 1-5, 1 BLAS thread: wall mean %.2f s, SD %.2f s, min %.2f, max %.2f (n=%d)'
          % (np.mean(ex), np.std(ex, ddof=1), min(ex), max(ex), len(ex)))

# reproducibility + MC variability
f = lambda run, pop: os.path.join(B, run, 'test_%s_pst_eff_a1_b0.5_phi1e-02_chr22.txt' % pop)
md += ['', '| comparison | pop | byte-identical | max abs diff |', '|---|---|---|---|']
for a, b in [('phi1e-2_seed1_t1', 'phi1e-2_seed1_t1_rerun'), ('phi1e-2_seed1_t1', 'phi1e-2_seed1_tdefault'),
             ('phi1e-2_seed1_t1', 'meta_phi1e-2_seed1_t1')]:
    for pop in ['EUR', 'EAS']:
        r = compare(f(a, pop), f(b, pop))
        md.append('| %s vs %s | %s | %s | %.3g |' % (a, b, pop, r['identical_text'], r['max_abs']))
md += ['', 'Seed-to-seed Monte-Carlo variability of posterior means (10 pairs of seeds 1-5):', '',
       '| pop | n SNPs | pearson r median [min,max] | max abs diff median | mean abs diff median | median |beta| |',
       '|---|---|---|---|---|---|']
for pop in ['EUR', 'EAS']:
    rs = [compare(f('phi1e-2_seed%d_t1' % i, pop), f('phi1e-2_seed%d_t1' % j, pop)) for i, j in itertools.combinations(range(1, 6), 2)]
    Bm = np.array([np.loadtxt(f('phi1e-2_seed%d_t1' % i, pop), usecols=5) for i in range(1, 6)])
    r_ = [x['pearson_r'] for x in rs]
    md.append('| %s | %d | %.4f [%.4f, %.4f] | %.3g | %.3g | %.3g |' % (
        pop, rs[0]['n'], np.median(r_), min(r_), max(r_), np.median([x['max_abs'] for x in rs]),
        np.median([x['mean_abs'] for x in rs]), np.median(np.abs(Bm.mean(0)))))
open(os.path.join(OUT, 'baseline_summary.md'), 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
