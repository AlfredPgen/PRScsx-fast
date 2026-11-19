#!/usr/bin/env python
"""
WP3 step 6 support (extrapolation inputs for docs/05): single-thread cost of the three parts of building one LD block
with tools/build_ld_panel.py at biobank sample sizes:
  (1) reading + standardising genotypes from a PLINK .bed (seconds per SNP, n = 20,000 and 76,000 people),
  (2) the cross-product X'X/n (n >= b; seconds for b = 700, 2000, 3000),
  (3) the symmetric eigendecomposition (b = 700, 2000, 3000, 6000).
Usage: OPENBLAS_NUM_THREADS=1 python wp3_builder_scaling.py OUT_JSON WORK_DIR
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
import build_ld_panel as blp  # noqa: E402


def main():
    out, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(1)
    res = {'read_seconds_per_snp': {}, 'crossprod_seconds': {}, 'eigh_seconds': {}}
    for n in (20000, 76000):
        b = 300
        G = rng.binomial(2, 0.3, size=(n, b)).astype(float)
        pre = os.path.join(work, 'r_%d' % n)
        blp.write_bed(pre, G, ['s%d' % j for j in range(b)], chrom=1)
        bed = blp.Bed(pre)
        t0 = time.perf_counter()
        X = bed.read(list(range(b)))
        dt = time.perf_counter() - t0
        res['read_seconds_per_snp'][str(n)] = dt / b
        print('read n=%d: %.2e s/SNP' % (n, dt / b), flush=True)
        del G, X
    for n, b in ((20000, 700), (20000, 2000), (20000, 3000), (76000, 700), (76000, 2000)):
        X = rng.standard_normal((n, b))
        t0 = time.perf_counter()
        C = X.T @ X / n
        dt = time.perf_counter() - t0
        res['crossprod_seconds']['%d_%d' % (n, b)] = dt
        print('X\'X n=%d b=%d: %.2f s' % (n, b, dt), flush=True)
        del X, C
    for b in (700, 2000, 3000, 6000):
        A = rng.standard_normal((b, b)); A = A @ A.T / b
        t0 = time.perf_counter()
        np.linalg.eigh(A)
        dt = time.perf_counter() - t0
        res['eigh_seconds'][str(b)] = dt
        print('eigh b=%d: %.2f s' % (b, dt), flush=True)
    res['threads'] = os.environ.get('OPENBLAS_NUM_THREADS', 'unset')
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
