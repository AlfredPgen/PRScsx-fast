#!/usr/bin/env python
"""
WP3 step 6 support: measured cost of tools/build_ld_panel.py per LD block, for docs/05_building_a_dense_panel.md.
For (n people, b SNPs) in a grid: simulate dense-SNP-like genotypes (latent AR(1), correlation 0.98 between neighbouring
SNPs), write PLINK files, build one block with --rank_var 0.999 and 1, record wall time, kept rank and file size.
Single thread unless OPENBLAS_NUM_THREADS is set. Usage: python wp3_builder_timing.py OUT_JSON WORK_DIR
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
import build_ld_panel as blp  # noqa: E402


def geno(rng, n, b, rho=0.98):
    from scipy.stats import norm
    f = rng.uniform(0.02, 0.5, b)
    z = np.empty((2 * n, b)); z[:, 0] = rng.standard_normal(2 * n)
    e = rng.standard_normal((2 * n, b)) * np.sqrt(1 - rho ** 2)
    for j in range(1, b):
        z[:, j] = rho * z[:, j - 1] + e[:, j]
    h = (z < norm.ppf(f)).astype(float)
    return h[:n] + h[n:]


def main():
    out, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    for n, b in ((600, 2000), (2000, 2000), (2000, 6000), (5000, 2000), (5000, 6000)):
        G = geno(rng, n, b)
        pre = os.path.join(work, 'g_%d_%d' % (n, b))
        blp.write_bed(pre, G, ['s%d' % j for j in range(b)], chrom=1)
        for rv in (0.999, 1.0):
            od = os.path.join(work, 'p_%d_%d_%s' % (n, b, rv))
            t0 = time.time()
            blp.main(['build', '--bfile', pre, '--pop', 'EUR', '--chrom', '1', '--out', od, '--block_snps', str(b),
                      '--rank_var', str(rv), '--maf', '0.01'])
            dt = time.time() - t0
            import h5py
            fn = os.path.join(od, 'ldeig_eur', 'ldeig_chr1.hdf5')
            with h5py.File(fn, 'r') as h:
                r = h['blk_1']['eval'].shape[0]; bb = h['blk_1']['evec'].shape[0]
            rows.append(dict(n=n, b=b, b_after_qc=bb, rank_var=rv, rank=r, seconds=dt, mb=os.path.getsize(fn) / 1e6))
            print(rows[-1], flush=True)
    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
