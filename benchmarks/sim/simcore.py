"""
Shared core of the multi-ancestry simulation + evaluation.

True population LD per ancestry and LD block = real 1000 Genomes block (PRS-CSx hdf5), sign-aligned to the
snpinfo A1 allele, projected to PSD and lightly ridged so that it is full rank:
    R_pop = (1-eps) * PSD(R_1kg) + eps * I,  rescaled to unit diagonal   (eps = 0.03 by default)
(1KG blocks are estimated from ~500 people and are rank-deficient; real population LD is full rank.)
"""
import os

import h5py
import numpy as np

POPS = ('AFR', 'AMR', 'EAS', 'EUR', 'SAS')
EPS_RIDGE = 0.03


def read_snpinfo(ref_dir, chroms):
    chroms = set(int(c) for c in chroms)
    rows = []
    with open(os.path.join(ref_dir, 'snpinfo_mult_1kg_hm3')) as f:
        header = next(f).split()
        for line in f:
            ll = line.split()
            if int(ll[0]) in chroms:
                rows.append(ll)
    col = {c: i for i, c in enumerate(header)}
    return header, rows, col


def genome_m(ref_dir):
    """Number of HapMap3 SNPs with frq>0 genome-wide, per ancestry (for per-SNP heritability scaling)."""
    cnt = {p: 0 for p in POPS}
    with open(os.path.join(ref_dir, 'snpinfo_mult_1kg_hm3')) as f:
        header = next(f).split()
        idx = {p: header.index('FRQ_' + p) for p in POPS}
        for line in f:
            ll = line.split()
            for p in POPS:
                if float(ll[idx[p]]) > 0:
                    cnt[p] += 1
    return cnt


def true_ld_blocks(ref_dir, pop, chrom, eps=EPS_RIDGE):
    """
    Yields (snp_list, R_pop, L) per non-empty block, in hdf5 order; L @ L.T == R_pop (sampling factor).
    R_pop is in snpinfo-A1 coding.
    """
    _, rows, col = read_snpinfo(ref_dir, [chrom])
    flp = {r[1]: int(r[col['FLP_' + pop]]) for r in rows if float(r[col['FRQ_' + pop]]) > 0}
    fn = os.path.join(ref_dir, 'ldblk_1kg_%s' % pop.lower(), 'ldblk_1kg_chr%d.hdf5' % chrom)
    with h5py.File(fn, 'r') as h:
        for k in range(1, len(h) + 1):
            snps = [s.decode() for s in h['blk_%d' % k]['snplist'][:]]
            if not snps:
                continue
            R = np.array(h['blk_%d' % k]['ldblk'], dtype=np.float64)
            f = np.array([flp[s] for s in snps], dtype=np.float64)
            R = R * np.outer(f, f)
            R = (R + R.T) / 2
            w, V = np.linalg.eigh(R)
            w = np.clip(w, 0, None)
            R = (V * w) @ V.T
            R = (1 - eps) * R + eps * np.eye(len(snps))
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)
            w, V = np.linalg.eigh(R)
            w = np.clip(w, 0, None)
            L = V * np.sqrt(w)
            yield snps, R, L
