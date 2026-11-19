#!/usr/bin/env python
"""
WP2 test T2d: time of ONE LD-block update, full factorisation vs low-rank, single thread.

V is built from the standardised genotypes of 500 simulated people (rank r = 499 ~ 500) for block sizes
b in {500, 1000, 2000, 5000, 10000}; the full update is fast/'s exact kernel (dpotrf + 2 dtrtrs on D + R), the low-rank
update is fastcsx.lowrank (double and single precision). Also measures the break-even ratio b/r for --linalg=auto.

Usage: OPENBLAS_NUM_THREADS=1 python lowrank_speed.py OUT_JSON
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fast'))
from fastcsx.lapack import block_update, new_workspace  # noqa: E402
from fastcsx.lowrank import eigen_block, lowrank_update  # noqa: E402


def make_block(rng, n, b):
    """Genotype-like data with local LD: AR(1) latent Gaussian, standardised; returns V, lam, D (if small enough)."""
    z = np.empty((n, b))
    z[:, 0] = rng.standard_normal(n)
    for j in range(1, b):
        z[:, j] = 0.9 * z[:, j - 1] + np.sqrt(1 - 0.81) * rng.standard_normal(n)
    X = (z - z.mean(0)) / z.std(0)
    U, s, Vt = np.linalg.svd(X / np.sqrt(n), full_matrices=False)
    lam = s ** 2
    keep = lam > 1e-13 * lam[0]
    return np.ascontiguousarray(Vt[keep].T), lam[keep]


def time_full(V, lam, ridge, bhat, z, s, reps):
    b = V.shape[0]
    D = (V * lam) @ V.T
    ws = new_workspace(b)
    Dw = ws['dinvt'][:b * b].reshape(b, b); F = ws['F'][:b * b].reshape(b, b); out = np.empty(b)
    block_update(D, 1.0 / ridge, bhat, z, s, Dw, F, out, ws['chars'], ws['ints'])      # warm-up / compile
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        block_update(D, 1.0 / ridge, bhat, z, s, Dw, F, out, ws['chars'], ws['ints'])
        t.append(time.perf_counter() - t0)
    del D, ws
    return min(t)


def time_lowrank(V, lam, ridge, bhat, e1, e2, s, dt, reps):
    b, r = V.shape
    Vd = V.astype(dt); sq = np.sqrt(lam).astype(dt); U = np.empty((b, r), dtype=dt)
    beta = np.empty(b); mean = np.empty(b)
    lowrank_update(Vd, sq, ridge, bhat, e1, e2, s, beta, mean, U)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lowrank_update(Vd, sq, ridge, bhat, e1, e2, s, beta, mean, U)
        t.append(time.perf_counter() - t0)
    return min(t)


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(1)
    rows = []
    for b in (500, 1000, 2000, 5000, 10000):
        V, lam = make_block(rng, 500, b)
        r = V.shape[1]
        ridge = 1.0 / np.minimum(np.exp(rng.normal(-3, 2, b)), 1.0)
        bhat = rng.normal(0, 0.01, b); e1 = rng.standard_normal(r); e2 = rng.standard_normal(b); s = 0.003
        reps = 5 if b <= 2000 else 2
        tf = time_full(V, lam, ridge, bhat, e2, s, reps)
        tl = time_lowrank(V, lam, ridge, bhat, e1, e2, s, np.float64, reps)
        ts = time_lowrank(V, lam, ridge, bhat, e1, e2, s, np.float32, reps)
        rows.append(dict(b=b, r=r, full_s=tf, lowrank_double_s=tl, lowrank_single_s=ts, speedup_double=tf / tl, speedup_single=tf / ts))
        print('b=%5d r=%3d  full %.4f s  low-rank double %.4f s (%.1fx)  single %.4f s (%.1fx)' % (b, r, tf, tl, tf / tl, ts, ts and tf / ts), flush=True)
    # break-even b/r for auto: fixed r, increase b until low-rank (double) is faster
    be = []
    for r_target in (100, 300, 500):
        prev = None
        for ratio in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0):
            b = int(round(ratio * r_target))
            V, lam = make_block(rng, r_target + 1, b)
            r = V.shape[1]
            ridge = 1.0 / np.minimum(np.exp(rng.normal(-3, 2, b)), 1.0)
            bhat = rng.normal(0, 0.01, b); e1 = rng.standard_normal(r); e2 = rng.standard_normal(b)
            tf = time_full(V, lam, ridge, bhat, e2, 0.003, 7)
            tl = time_lowrank(V, lam, ridge, bhat, e1, e2, 0.003, np.float64, 7)
            be.append(dict(r=r, b=b, ratio=b / r, full_s=tf, lowrank_double_s=tl, lowrank_faster=bool(tl < tf)))
            print('  break-even scan r=%d b=%d (b/r=%.2f): full %.5f  low-rank %.5f  %s' % (r, b, b / r, tf, tl, 'LOW-RANK FASTER' if tl < tf else ''), flush=True)
    json.dump(dict(speed=rows, break_even=be), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
