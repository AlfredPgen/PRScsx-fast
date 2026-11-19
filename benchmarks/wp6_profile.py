#!/usr/bin/env python
"""
WP6.3: where does the time go in an improved/ chromosome run with the low-rank LD update?
Runs PRScsx_improved.py in-process with timing wrappers around (a) the LD-block kernels (Solver.update: all
Cholesky / Woodbury block updates incl. the normal draws) and (b) the GIG sampler (sample_psi); everything else in
the Gibbs iteration (ridge/xx vectors, sigma/delta/tau/phi draws, accumulators, Python overhead) is 'other'.
Decision rule (instructions): add numba prange to the remaining Python code only if 'other' > 15% of MCMC time.

  python wp6_profile.py OUT.json --ref_dir DIR --bim_prefix P --sst EUR.txt,EAS.txt --n_gwas 200000,100000 \
         --pop EUR,EAS --chrom 22 [--n_iter 600 --threads 1,4 --linalg lowrank]
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'improved'))
sys.path.insert(0, os.path.join(HERE, '..', 'fast'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--ref_dir', required=True)
    ap.add_argument('--bim_prefix', required=True)
    ap.add_argument('--sst', required=True)
    ap.add_argument('--n_gwas', required=True)
    ap.add_argument('--pop', required=True)
    ap.add_argument('--chrom', default='22')
    ap.add_argument('--n_iter', type=int, default=600)
    ap.add_argument('--threads', default='1,4')
    ap.add_argument('--linalg', default='lowrank')
    a = ap.parse_args()

    import PRScsx_improved as P
    import csximp.mcmc_improved as M
    acc = {'kernel': 0.0, 'gig': 0.0, 'step': 0.0}
    orig_update, orig_gig = M.Solver.update, M.sample_psi

    def update(self, *x, **k):
        t = time.perf_counter(); r = orig_update(self, *x, **k); acc['kernel'] += time.perf_counter() - t; return r

    def gig(*x, **k):
        t = time.perf_counter(); r = orig_gig(*x, **k); acc['gig'] += time.perf_counter() - t; return r

    orig_step = M.ChainState.step

    def step(self):
        t = time.perf_counter(); orig_step(self); acc['step'] += time.perf_counter() - t

    M.Solver.update, M.sample_psi, M.ChainState.step = update, gig, step
    out = []
    for th in [int(x) for x in a.threads.split(',')]:
        acc['kernel'] = acc['gig'] = acc['step'] = 0.0
        with tempfile.TemporaryDirectory() as d:
            t0 = time.perf_counter()
            P.main(['--ref_dir=' + a.ref_dir, '--bim_prefix=' + a.bim_prefix, '--sst_file=' + a.sst, '--n_gwas=' + a.n_gwas,
                    '--pop=' + a.pop, '--chrom=' + a.chrom, '--phi=1e-2', '--seed=1', '--n_iter=%d' % a.n_iter,
                    '--n_burnin=%d' % (a.n_iter // 2), '--threads=%d' % th, '--linalg=' + a.linalg, '--quiet',
                    '--out_dir=' + d, '--out_name=p'])
            wall = time.perf_counter() - t0
        m = acc['step']                       # time inside Gibbs iterations
        other = m - acc['kernel'] - acc['gig']
        r = {'threads': th, 'n_iter': a.n_iter, 'wall_total_s': wall, 'mcmc_s': m, 'kernel_s': acc['kernel'],
             'gig_s': acc['gig'], 'other_s': other, 'other_share': other / m, 'kernel_share': acc['kernel'] / m,
             'gig_share': acc['gig'] / m}
        print(json.dumps(r), flush=True)
        out.append(r)
    json.dump({'chrom': a.chrom, 'pop': a.pop, 'linalg': a.linalg, 'runs': out,
               'rule': 'prange only if other_share > 0.15'}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
