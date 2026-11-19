#!/usr/bin/env python
"""
Synthetic PRS-CSx reference panel for the repository's 1,000 test SNPs (chr22), in the original 1000 Genomes format,
so that exactness tests run anywhere without the 9 GB panel.

Re-implemented from the description in the review of 18 Nov 2026 (the review's own generator was not available):
  * the 1,000 SNPs of test_data/test.bim plus 20 panel-only SNPs that no sumstats/target file contains;
  * 7 LD blocks per ancestry: sizes 150, 200, 0 (empty block), 20 (panel-only SNPs, empty after matching),
    350 (larger than the reference sample, n_ref = 200, so rank-deficient), 100, 200;
  * LD = sample correlation of n_ref simulated genotypes (latent AR(1) Gaussian thresholded to 0/1/2);
  * 10% of SNPs per ancestry stored in flipped allele coding (FLP = -1);
  * 5% of SNPs with A1/A2 swapped in the SNP-info file relative to the .bim (exercises the 1-FRQ path);
  * 3% of SNPs absent (frequency 0) in EAS and in AFR.

Usage: python make_synth_ref.py OUT_DIR [--seed 1]
"""
import argparse
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BIM = os.path.join(HERE, '..', 'test_data', 'test.bim')
POPS = ('AFR', 'AMR', 'EAS', 'EUR', 'SAS')
BLOCKS = [150, 200, 0, 'extra', 350, 100, 200]       # 'extra' = the 20 panel-only SNPs
N_REF = 200


def genotypes(rng, b, n, frq, rho):
    """n x b genotype matrix with AR(1) latent correlation rho and allele frequencies frq."""
    from scipy.stats import norm
    z = np.empty((2 * n, b))
    z[:, 0] = rng.standard_normal(2 * n)
    for j in range(1, b):
        z[:, j] = rho * z[:, j - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(2 * n)
    hap = (z < norm.ppf(frq)[None, :]).astype(float)          # P(allele) = frq
    g = hap[:n] + hap[n:]
    for j in np.flatnonzero(g.std(0) == 0):                    # avoid monomorphic columns in the sample
        g[rng.integers(n), j] = 1.0 - g[rng.integers(n), j] % 2
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--bim', default=DEFAULT_BIM)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    bim = [l.split() for l in open(a.bim)]
    snps = [r[1] for r in bim]
    m = len(snps)
    assert m == sum(x for x in BLOCKS if x != 'extra'), 'block sizes must add up to the number of target SNPs'

    # SNP-info rows: real SNPs in bim order; 20 panel-only SNPs inserted as block 4
    rows, blocks = [], []
    pos = 0
    for size in BLOCKS:
        if size == 'extra':
            last_bp = int(rows[-1]['BP']) if rows else 1
            blk = []
            for k in range(20):
                rows.append(dict(SNP='rsSYNTH%d' % k, BP=last_bp + 1 + k, A1='A', A2='G', extra=True))
                blk.append(len(rows) - 1)
            blocks.append(blk)
            continue
        blk = []
        for j in range(pos, pos + size):
            r = bim[j]
            a1, a2 = r[4], r[5]
            if rng.random() < 0.05:                            # swapped alleles in SNP info vs .bim
                a1, a2 = a2, a1
            rows.append(dict(SNP=r[1], BP=int(r[3]), A1=a1, A2=a2, extra=False))
            blk.append(len(rows) - 1)
        blocks.append(blk)
        pos += size
    # SNP-info rows are written in construction order, which is also the HDF5 block order (as in the real panel)
    nrow = len(rows)
    frq = {p: rng.uniform(0.05, 0.95, nrow) for p in POPS}
    flp = {p: np.where(rng.random(nrow) < 0.10, -1, 1) for p in POPS}
    for p in ('EAS', 'AFR'):
        frq[p][rng.random(nrow) < 0.03] = 0.0
    for p in ('AMR', 'SAS'):
        frq[p][:] = 0.0

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'snpinfo_mult_1kg_hm3'), 'w') as f:
        f.write('CHR\tSNP\tBP\tA1\tA2\t' + '\t'.join('FRQ_' + p for p in POPS) + '\t' + '\t'.join('FLP_' + p for p in POPS) + '\n')
        for i, r in enumerate(rows):
            f.write('22\t%s\t%d\t%s\t%s\t' % (r['SNP'], r['BP'], r['A1'], r['A2']) +
                    '\t'.join('%.6f' % frq[p][i] for p in POPS) + '\t' + '\t'.join('%d' % (flp[p][i] if frq[p][i] > 0 else 0) for p in POPS) + '\n')

    rho_ld = {'EUR': 0.95, 'EAS': 0.93, 'AFR': 0.85}
    for p in ('EUR', 'EAS', 'AFR'):
        d = os.path.join(a.out, 'ldblk_1kg_%s' % p.lower())
        os.makedirs(d, exist_ok=True)
        with h5py.File(os.path.join(d, 'ldblk_1kg_chr22.hdf5'), 'w') as h:
            for k, blk in enumerate(blocks):
                present = [i for i in blk if frq[p][i] > 0]
                g = h.create_group('blk_%d' % (k + 1))
                if not present:
                    g.create_dataset('ldblk', data=np.zeros((0, 0)))
                    g.create_dataset('snplist', data=np.array([], dtype='S10'))
                    continue
                X = genotypes(rng, len(present), N_REF, frq[p][present], rho_ld[p])
                R = np.atleast_2d(np.corrcoef(X, rowvar=False))
                f = flp[p][present].astype(float)
                R = R * np.outer(f, f)                          # stored in the ancestry's own allele coding
                g.create_dataset('ldblk', data=R)
                g.create_dataset('snplist', data=np.array([rows[i]['SNP'].encode() for i in present]))
    print('synthetic panel written to %s (%d SNP-info rows, %d blocks per ancestry)' % (a.out, nrow, len(blocks)))


if __name__ == '__main__':
    main()
