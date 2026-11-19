#!/usr/bin/env python
"""
Post-hoc multi-chain averages from independent MCMC-seed runs of the same method on the same dataset
(equivalent to --n_chains=K for the ancestry-specific posterior means; META files are averaged, which approximates
--n_chains's pooled-variance META weighting).

  python make_chain_average.py --dataset_dir /work/bench/accuracy/h0.5_pi0.01_r1 --method improved --seeds 1,11 --name improved_avg2
creates DATASET_DIR/<name>_s<first seed>/phi*/  with averaged sim_*_pst_eff_* files (+ a job_timing.json marker).
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True); ap.add_argument('--method', required=True)
    ap.add_argument('--seeds', required=True); ap.add_argument('--name', required=True)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    src = [os.path.join(a.dataset_dir, '%s_s%d' % (a.method, s)) for s in seeds]
    out = os.path.join(a.dataset_dir, '%s_s%d' % (a.name, seeds[0]))
    for phi_dir in sorted(glob.glob(os.path.join(src[0], 'phi*'))):
        phi = os.path.basename(phi_dir)
        if not all(os.path.exists(os.path.join(s, phi, 'job_timing.json')) for s in src):
            continue
        od = os.path.join(out, phi); os.makedirs(od, exist_ok=True)
        for fn in glob.glob(os.path.join(phi_dir, 'sim_*_pst_eff_*.txt')):
            rows = [l.split() for l in open(fn)]
            vals = np.mean([[float(l.split()[5]) for l in open(os.path.join(s, phi, os.path.basename(fn)))] for s in src], axis=0)
            with open(os.path.join(od, os.path.basename(fn)), 'w') as f:
                for r, v in zip(rows, vals):
                    f.write('%s\t%s\t%s\t%s\t%s\t%.6e\n' % (r[0], r[1], r[2], r[3], r[4], v))
        json.dump({'averaged_from': [os.path.join(s, phi) for s in src]}, open(os.path.join(od, 'job_timing.json'), 'w'))
    print('wrote', out)


if __name__ == '__main__':
    main()
