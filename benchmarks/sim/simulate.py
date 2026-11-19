#!/usr/bin/env python
"""
Multi-ancestry simulation with realistic LD, written in the ORIGINAL PRS-CSx input formats.

1) make-ref : simulated LD reference panel (finite reference sample => realistic LD mismatch)
   python simulate.py make-ref --ref_dir /work/ref/1kg --chroms 22 --pops EUR,EAS,AFR --n_ref 500,500,650 \
          --seed 11 --out /work/sim/ref_chr22
   -> OUT/snpinfo_mult_1kg_hm3 (chosen chromosomes; FLP=1 because the simulated LD is in snpinfo-A1 coding)
      OUT/ldblk_1kg_<pop>/ldblk_1kg_chr<c>.hdf5 (same blocks/SNP lists as 1KG; LD = sample correlation of
      n_ref individuals drawn from N(0, R_pop))

2) make-data : true effects + GWAS summary statistics + target .bim + truth file
   python simulate.py make-data --ref_dir /work/ref/1kg --chroms 22 --pops EUR,EAS,AFR \
          --n_gwas 300000,100000,30000 --h2 0.5 --pi 0.01 --rho 0.8 --seed 1 --out /work/sim/data/X
   Model (standardised genotypes, per ancestry k):
     causal_j ~ Bernoulli(pi), shared across ancestries;
     (beta_j1..beta_jK) | causal ~ N(0, C), C_kk = 1, C_kl = rho;  beta_k rescaled so beta_k' R_k beta_k = h2_region_k
     h2_region_k = h2 * M_region_k / M_genome_k  (per-SNP heritability of a genome-wide trait with heritability h2)
     bhat_k = R_k beta_k + e_k,  e_k ~ N(0, R_k / N_k)    (R_k = true population LD, see simcore)
     a random fraction --miss of SNPs is dropped from each sumstats file independently.
   Sumstats are written per allele: BETA = bhat/sd, SE = 1/(sqrt(N) sd), sd = sqrt(2f(1-f)) => z = bhat*sqrt(N).
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore  # noqa: E402


def make_ref(a):
    pops = a.pops.split(',')
    n_ref = [int(x) for x in a.n_ref.split(',')]
    chroms = [int(c) for c in a.chroms.split(',')]
    os.makedirs(a.out, exist_ok=True)
    header, rows, col = simcore.read_snpinfo(a.ref_dir, chroms)
    with open(os.path.join(a.out, 'snpinfo_mult_1kg_hm3'), 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            r = list(r)
            for p in simcore.POPS:
                r[col['FLP_' + p]] = '1'
            f.write('\t'.join(r) + '\n')
    rng = np.random.default_rng(a.seed)
    for pop, nr in zip(pops, n_ref):
        d = os.path.join(a.out, 'ldblk_1kg_%s' % pop.lower())
        os.makedirs(d, exist_ok=True)
        for c in chroms:
            src = os.path.join(a.ref_dir, 'ldblk_1kg_%s' % pop.lower(), 'ldblk_1kg_chr%d.hdf5' % c)
            with h5py.File(src, 'r') as hs:
                n_blk = len(hs)
                all_snps = [[s for s in hs['blk_%d' % k]['snplist'][:]] for k in range(1, n_blk + 1)]
            blocks = {tuple(s): (R, L) for s, R, L in simcore.true_ld_blocks(a.ref_dir, pop, c)}
            with h5py.File(os.path.join(d, 'ldblk_1kg_chr%d.hdf5' % c), 'w') as ho:
                for k in range(n_blk):
                    snps = [s.decode() for s in all_snps[k]]
                    g = ho.create_group('blk_%d' % (k + 1))
                    if snps:
                        R, L = blocks[tuple(snps)]
                        for attempt in range(20):
                            X = rng.standard_normal((nr, L.shape[1])) @ L.T       # n_ref x b, Cov = R_pop
                            Rr = np.atleast_2d(np.corrcoef(X, rowvar=False))
                            if not a.check_svd:
                                break
                            try:      # the original PRS-CSx needs LAPACK gesdd to converge on each block (see report)
                                from scipy import linalg as _la
                                _la.svd(Rr)
                                break
                            except np.linalg.LinAlgError:
                                print('make-ref: %s chr%d blk_%d (b=%d): gesdd did not converge, re-drawing' % (pop, c, k + 1, len(snps)), flush=True)
                    else:
                        Rr = np.zeros((0, 0))
                    g.create_dataset('ldblk', data=Rr)
                    g.create_dataset('snplist', data=np.array([s.encode() for s in snps], dtype='S'))
            print('make-ref: %s chr%d: %d blocks, n_ref=%d' % (pop, c, n_blk, nr), flush=True)
    json.dump(vars(a), open(os.path.join(a.out, 'sim_ref_params.json'), 'w'), indent=1)


def _restrict(snps, R, present_k, pos):
    keep = np.array([present_k[pos[x]] for x in snps])
    if not keep.any():
        return None
    if keep.all():
        w, V = np.linalg.eigh(R)
        return snps, R, V * np.sqrt(np.clip(w, 0, None))
    Rs = R[np.ix_(keep, keep)]
    w, V = np.linalg.eigh(Rs)
    return [s for s, k in zip(snps, keep) if k], Rs, V * np.sqrt(np.clip(w, 0, None))


def make_data(a):
    pops = a.pops.split(',')
    ns = [int(x) for x in a.n_gwas.split(',')]
    chroms = [int(c) for c in a.chroms.split(',')]
    K = len(pops)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    _, rows, col = simcore.read_snpinfo(a.ref_dir, chroms)
    m_gen = simcore.genome_m(a.ref_dir) if a.m_genome is None else {p: a.m_genome for p in simcore.POPS}
    snps = [r[1] for r in rows]
    pos = {s: i for i, s in enumerate(snps)}
    frq = np.array([[float(r[col['FRQ_' + p]]) for p in pops] for r in rows])
    present = frq > 0
    if a.max_bp is not None:                       # restrict the simulated region (effects, sumstats, target SNPs)
        present &= (np.array([int(r[2]) for r in rows]) <= a.max_bp)[:, None]
    keep = present.any(1)
    M = len(snps)

    causal = (rng.random(M) < a.pi) & keep
    C = np.full((K, K), a.rho); np.fill_diagonal(C, 1.0)
    beta = np.zeros((M, K))
    beta[causal] = rng.multivariate_normal(np.zeros(K), C, size=int(causal.sum()))
    beta[~present] = 0.0

    bhat = np.full((M, K), np.nan)
    h2_region = {}
    for k, pop in enumerate(pops):
        blocks = [(s, R, L) for c in chroms for s, R, L in simcore.true_ld_blocks(a.ref_dir, pop, c)]
        gvar = 0.0
        blocks = [_restrict(s, R, present[:, k], pos) for s, R, L in blocks]
        blocks = [x for x in blocks if x is not None]
        for s, R, L in blocks:
            ii = np.array([pos[x] for x in s]); b = beta[ii, k]
            gvar += b @ R @ b
        m_reg = int(present[:, k].sum())
        h2_region[pop] = a.h2 * m_reg / m_gen[pop]
        scale = np.sqrt(h2_region[pop] / gvar) if gvar > 0 else 0.0
        beta[:, k] *= scale
        for s, R, L in blocks:
            ii = np.array([pos[x] for x in s])
            noise = L @ rng.standard_normal(L.shape[1]) / np.sqrt(ns[k])
            bhat[ii, k] = R @ beta[ii, k] + noise
        # write sumstats (per-allele scale), with random missingness
        drop = rng.random(M) < a.miss
        with open(os.path.join(a.out, '%s_sumstats.txt' % pop), 'w') as f:
            f.write('SNP\tA1\tA2\tBETA\tSE\n')
            for j in np.flatnonzero(present[:, k] & ~drop):
                sd = np.sqrt(2 * frq[j, k] * (1 - frq[j, k]))
                f.write('%s\t%s\t%s\t%.8e\t%.8e\n' % (snps[j], rows[j][3], rows[j][4], bhat[j, k] / sd, 1.0 / (np.sqrt(ns[k]) * sd)))
        print('make-data: %s M=%d causal=%d h2_region=%.5f N=%d' % (pop, m_reg, int((causal & present[:, k]).sum()), h2_region[pop], ns[k]), flush=True)
    with open(os.path.join(a.out, 'target.bim'), 'w') as f:
        for j in np.flatnonzero(keep):
            f.write('%s\t%s\t0\t%s\t%s\t%s\n' % (rows[j][0], snps[j], rows[j][2], rows[j][3], rows[j][4]))
    np.savez_compressed(os.path.join(a.out, 'truth.npz'), snps=np.array(snps), beta=beta, frq=frq, present=present,
                        causal=causal, pops=np.array(pops), chroms=np.array(chroms))
    meta = dict(vars(a)); meta['h2_region'] = h2_region; meta['n_causal'] = int(causal.sum()); meta['M'] = int(keep.sum())
    json.dump(meta, open(os.path.join(a.out, 'sim_params.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('make-ref')
    r.add_argument('--ref_dir', required=True); r.add_argument('--chroms', required=True)
    r.add_argument('--pops', default='EUR,EAS,AFR'); r.add_argument('--n_ref', default='500,500,650')
    r.add_argument('--seed', type=int, default=11); r.add_argument('--out', required=True)
    r.add_argument('--check_svd', action='store_true', help='re-draw blocks on which LAPACK gesdd fails to converge')
    d = sub.add_parser('make-data')
    d.add_argument('--ref_dir', required=True); d.add_argument('--chroms', required=True)
    d.add_argument('--pops', default='EUR,EAS,AFR'); d.add_argument('--n_gwas', default='300000,100000,30000')
    d.add_argument('--h2', type=float, default=0.5); d.add_argument('--pi', type=float, default=0.01)
    d.add_argument('--rho', type=float, default=0.8); d.add_argument('--miss', type=float, default=0.02)
    d.add_argument('--max_bp', type=int, default=None, help='only SNPs with BP <= max_bp (sub-region of the chromosome)')
    d.add_argument('--m_genome', type=int, default=None, help='override genome-wide SNP count (default: HM3 count per pop)')
    d.add_argument('--seed', type=int, default=1); d.add_argument('--out', required=True)
    a = ap.parse_args()
    make_ref(a) if a.cmd == 'make-ref' else make_data(a)


if __name__ == '__main__':
    main()
