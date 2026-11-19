#!/usr/bin/env python
"""
WP3 step 4: an eigen panel built by tools/build_ld_panel.py from simulated genotypes
  (a) reproduces the sample LD matrix to 1e-5 (--rank_var=1), and
  (b) gives the same posterior as the matching b x b panel (1000 Genomes format), by the T2c criterion:
      mean corr(eigen_i, dense_j) >= mean corr(dense_i, dense_j) - 0.005 over seeds 1..S.
Both fits use fast/ with --linalg=auto (the b x b panel is eigen-decomposed at load time; blocks with b > 1.7 r use the
low-rank update); a full factorisation of the 4,000-SNP block at every iteration would take hours (T2c already compares
full and low-rank).

Usage: python wp3_eigen_panel_check.py OUT_JSON WORK_DIR [--sizes 300,1500,4000 --n 600 --seeds 5 --n_iter 1000 --threads 4]
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import build_ld_panel as blp  # noqa: E402

POPS = ('AFR', 'AMR', 'EAS', 'EUR', 'SAS')


def simulate_genotypes(rng, n, sizes, rho=0.95):
    blocks = []
    for b in sizes:
        f = rng.uniform(0.05, 0.5, b)
        z = np.empty((2 * n, b)); z[:, 0] = rng.standard_normal(2 * n)
        for j in range(1, b):
            z[:, j] = rho * z[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(2 * n)
        from scipy.stats import norm
        h = (z < norm.ppf(f)).astype(float)
        blocks.append(h[:n] + h[n:])
    return np.hstack(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('work')
    ap.add_argument('--sizes', default='300,1500,4000'); ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--n_iter', type=int, default=1000)
    ap.add_argument('--threads', type=int, default=4); ap.add_argument('--n_gwas', type=int, default=50000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    sizes = [int(x) for x in a.sizes.split(',')]
    os.makedirs(a.work, exist_ok=True)
    G = simulate_genotypes(rng, a.n, sizes)
    keep = (G.std(0) > 0) & (np.minimum(G.mean(0) / 2, 1 - G.mean(0) / 2) >= 0.01)
    m = G.shape[1]
    snps = ['rs%d' % (j + 1) for j in range(m)]
    bps = np.arange(1, m + 1) * 100
    gap = np.cumsum([0] + sizes)
    for k in range(1, len(sizes)):
        bps[gap[k]:] += 10 ** 6                                   # block boundaries at clear gaps
    prefix = os.path.join(a.work, 'geno')
    blp.write_bed(prefix, G, snps, chrom=22, bps=bps)
    with open(os.path.join(a.work, 'blocks.txt'), 'w') as f:
        for k in range(len(sizes)):
            f.write('22\t%d\t%d\n' % (bps[gap[k]], bps[gap[k + 1] - 1] + 1))
    eig_dir = os.path.join(a.work, 'panel_eigen')
    t0 = time.time()
    blp.main(['build', '--bfile', prefix, '--pop', 'EUR', '--chrom', '22', '--out', eig_dir, '--blocks',
              os.path.join(a.work, 'blocks.txt'), '--rank_var', '1', '--maf', '0.01'])
    blp.main(['merge', '--out', eig_dir])
    build_s = time.time() - t0

    # (a) reproduce the sample LD; also write the matching b x b panel (1000 Genomes format)
    dense_dir = os.path.join(a.work, 'panel_dense'); os.makedirs(os.path.join(dense_dir, 'ldblk_1kg_eur'), exist_ok=True)
    max_err = 0.0
    info = [l.split() for l in open(os.path.join(eig_dir, 'snpinfo_mult_eigen'))]
    open(os.path.join(dense_dir, 'snpinfo_mult_1kg_hm3'), 'w').write(''.join('\t'.join(r) + '\n' for r in info))
    with h5py.File(os.path.join(eig_dir, 'ldeig_eur', 'ldeig_chr22.hdf5'), 'r') as he, \
            h5py.File(os.path.join(dense_dir, 'ldblk_1kg_eur', 'ldblk_1kg_chr22.hdf5'), 'w') as hd:
        for k in range(1, len(sizes) + 1):
            g = he['blk_%d' % k]
            sl = [s.decode() for s in g['snplist'][:]]
            idx = [snps.index(s) for s in sl]
            X = G[:, idx]; X = (X - X.mean(0)) / X.std(0)
            R = X.T @ X / a.n                                      # sample LD (ddof 0, as the builder)
            V = np.asarray(g['evec'][:], dtype=np.float64); lam = np.asarray(g['eval'][:])
            max_err = max(max_err, float(np.abs((V * lam) @ V.T - R).max()))
            gd = hd.create_group('blk_%d' % k)
            gd.create_dataset('ldblk', data=R); gd.create_dataset('snplist', data=np.array([s.encode() for s in sl]))

    # summary statistics simulated from the sample LD
    kept = [r[1] for r in info[1:]]
    pos = {s: i for i, s in enumerate(snps)}
    Xall = G[:, [pos[s] for s in kept]]; Xall = (Xall - Xall.mean(0)) / Xall.std(0)
    beta = np.zeros(len(kept)); causal = rng.random(len(kept)) < 0.01
    beta[causal] = rng.standard_normal(causal.sum()); beta *= np.sqrt(0.01 / max(beta @ (Xall.T @ (Xall @ beta)) / a.n, 1e-12))
    ld_beta = Xall.T @ (Xall @ beta) / a.n
    noise = Xall.T @ rng.standard_normal(a.n) / a.n * np.sqrt(a.n / a.n_gwas)     # ~ N(0, R/N_gwas)
    bhat = ld_beta + noise
    with open(os.path.join(a.work, 'EUR_sumstats.txt'), 'w') as f:
        f.write('SNP\tA1\tA2\tBETA\tSE\n')
        for s, r, bh in zip(kept, info[1:], bhat):
            f.write('%s\t%s\t%s\t%.8e\t%.8e\n' % (s, r[3], r[4], bh, 1 / np.sqrt(a.n_gwas)))
    with open(os.path.join(a.work, 'target.bim'), 'w') as f:
        for r in info[1:]:
            f.write('22\t%s\t0\t%s\t%s\t%s\n' % (r[1], r[2], r[3], r[4]))

    est, times = {}, {}
    for kind, panel in (('dense', dense_dir), ('eigen', eig_dir)):
        for seed in range(1, a.seeds + 1):
            od = os.path.join(a.work, 'fit_%s_%d' % (kind, seed)); os.makedirs(od, exist_ok=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, os.path.join(ROOT, 'fast', 'PRScsx_fast.py'), '--ref_dir=' + panel,
                                '--bim_prefix=' + os.path.join(a.work, 'target'), '--sst_file=' + os.path.join(a.work, 'EUR_sumstats.txt'),
                                '--n_gwas=%d' % a.n_gwas, '--pop=EUR', '--chrom=22', '--phi=1e-2', '--seed=%d' % seed,
                                '--linalg=auto', '--n_iter=%d' % a.n_iter, '--n_burnin=%d' % (a.n_iter // 2),
                                '--threads=%d' % a.threads, '--quiet', '--out_name=t', '--out_dir=' + od], capture_output=True, text=True)
            assert r.returncode == 0, r.stdout[-3000:]
            times.setdefault(kind, []).append(time.time() - t0)
            est[(kind, seed)] = np.loadtxt(os.path.join(od, 't_EUR_pst_eff_a1_b0.5_phi1e-02_chr22.txt'), usecols=5)
    S = range(1, a.seeds + 1)
    dd = [np.corrcoef(est[('dense', i)], est[('dense', j)])[0, 1] for i, j in itertools.combinations(S, 2)]
    de = [np.corrcoef(est[('dense', i)], est[('eigen', j)])[0, 1] for i in S for j in S]
    ee = [np.corrcoef(est[('eigen', i)], est[('eigen', j)])[0, 1] for i, j in itertools.combinations(S, 2)]
    res = dict(sizes=sizes, n_people=a.n, snps_after_qc=len(kept), build_seconds=build_s, max_abs_ld_error=max_err,
               ld_reproduced_to_1e5=bool(max_err < 1e-5), mean_corr_dense_dense=float(np.mean(dd)),
               mean_corr_dense_eigen=float(np.mean(de)), mean_corr_eigen_eigen=float(np.mean(ee)),
               posterior_criterion_passed=bool(np.mean(de) >= np.mean(dd) - 0.005),
               seconds_per_fit=dict((k, float(np.mean(v))) for k, v in times.items()), n_iter=a.n_iter, seeds=a.seeds)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
