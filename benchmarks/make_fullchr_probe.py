#!/usr/bin/env python
"""
Quick RUNTIME PROBE data: GWAS summary statistics for ALL HapMap3 SNPs on one chromosome, simulated
from the real 1000G LD blocks of each population, in the original PRS-CSx input formats.

This is NOT the accuracy benchmark (that uses genotype-level simulation with LD-reference mismatch);
it only produces realistically sized, realistically correlated inputs for timing/profiling.

Model (standardized scale, per population k, per LD block):
    beta_k  : causal fraction pi, shared causal SNPs, cross-pop effect correlation rho
    bhat_k  = R_k beta_k + e_k,  e_k ~ N(0, R_k / N_k)       (R_k aligned to snpinfo A1 via FLP)
Written as  SNP A1 A2 BETA SE  with BETA = bhat_k * sqrt(N_k) * SE, SE = 1/sqrt(N_k)
so that PRS-CSx's  beta/se/sqrt(N)  recovers bhat_k exactly.
"""
import argparse
import os
import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref_dir', required=True)
    ap.add_argument('--chrom', type=int, default=22)
    ap.add_argument('--pops', default='EUR,EAS')
    ap.add_argument('--n_gwas', default='200000,100000')
    ap.add_argument('--h2', type=float, default=0.02, help='heritability explained by this chromosome')
    ap.add_argument('--pi', type=float, default=0.01, help='causal fraction')
    ap.add_argument('--rho', type=float, default=0.8, help='cross-population genetic correlation of causal effects')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    pops = a.pops.split(','); ns = [int(x) for x in a.n_gwas.split(',')]
    os.makedirs(a.out_dir, exist_ok=True)

    ref = []
    with open(os.path.join(a.ref_dir, 'snpinfo_mult_1kg_hm3')) as f:
        hdr = next(f).split()
        for line in f:
            ll = line.split()
            if int(ll[0]) == a.chrom:
                ref.append(ll)
    col = {c: i for i, c in enumerate(hdr)}
    snps = [r[1] for r in ref]
    M = len(snps)

    # shared causal set over the union of SNPs; correlated effects across populations
    causal = rng.random(M) < a.pi
    K = len(pops)
    C = np.full((K, K), a.rho); np.fill_diagonal(C, 1.0)
    eff = np.zeros((M, K))
    eff[causal] = rng.multivariate_normal(np.zeros(K), C, size=causal.sum())

    # target bim = union of SNPs present (frq>0) in any population
    present = np.zeros(M, bool)
    for p in pops:
        present |= np.array([float(r[col['FRQ_' + p]]) > 0 for r in ref])
    with open(os.path.join(a.out_dir, 'target.bim'), 'w') as f:
        for r, ok in zip(ref, present):
            if ok:
                f.write('%s\t%s\t0\t%s\t%s\t%s\n' % (r[0], r[1], r[2], r[3], r[4]))

    for k, (p, n) in enumerate(zip(pops, ns)):
        idx = [i for i, r in enumerate(ref) if float(r[col['FRQ_' + p]]) > 0]
        flp = np.array([int(ref[i][col['FLP_' + p]]) for i in idx], float)
        b = eff[idx, k].copy()
        nz = b != 0
        b[nz] *= np.sqrt(a.h2 / np.sum(b[nz] ** 2))  # approx per-chrom h2 (ignores LD inflation)
        bhat = np.zeros(len(idx))
        h = h5py.File(os.path.join(a.ref_dir, 'ldblk_1kg_%s' % p.lower(), 'ldblk_1kg_chr%d.hdf5' % a.chrom), 'r')
        mm = 0
        for blk in range(1, len(h) + 1):
            R = np.array(h['blk_%d' % blk]['ldblk'])
            sz = R.shape[0]
            fl = flp[mm:mm + sz]
            R = R * np.outer(fl, fl)
            w, V = np.linalg.eigh(R)
            w = np.clip(w, 0, None)
            noise = V @ (np.sqrt(w / n) * rng.standard_normal(sz))
            bhat[mm:mm + sz] = R @ b[mm:mm + sz] + noise
            mm += sz
        assert mm == len(idx)
        se = 1.0 / np.sqrt(n)
        with open(os.path.join(a.out_dir, '%s_sumstats.txt' % p), 'w') as f:
            f.write('SNP\tA1\tA2\tBETA\tSE\n')
            for j, i in enumerate(idx):
                f.write('%s\t%s\t%s\t%.8e\t%.8e\n' % (ref[i][1], ref[i][3], ref[i][4], bhat[j] * np.sqrt(n) * se, se))
        print('%s: %d SNPs written (N=%d), %d causal' % (p, len(idx), n, int(nz.sum())))


if __name__ == '__main__':
    main()
