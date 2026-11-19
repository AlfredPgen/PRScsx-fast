"""
Start-up self-check: is this platform bit-compatible with the original PRS-CSx?

Exactness of fast/ depends on the compiled code (numba/LLVM), the C maths library and the LAPACK build producing
the same floating-point results as the original's Python/SciPy code. It has been verified on Linux (glibc, SciPy's
OpenBLAS) and independently on Windows 11 (review of 18 Nov 2026). On any other platform it is checked here:
  1. 2,000 GIG variates from the compiled sampler vs the original Python gigrnd (verbatim copy), same seed, compared
     bit for bit, together with the position of the random stream afterwards;
  2. one compiled LD-block update vs the original SciPy calls (cholesky, solve_triangular, np.dot) on a 60 x 60 block.
The global numpy random state is saved and restored, so the check never changes the results of a run.
"""
import time

import numpy as np
from numpy import random

from . import gigrnd_reference
from .gig import sample_psi
from .lapack import block_update, new_workspace


def _gig_ok(n, seed):
    rng = np.random.default_rng(seed)
    P = 1.0 - 0.5 * rng.integers(0, 4, n)
    A = 2.0 * np.exp(rng.normal(0.0, 3.0, n))
    B = np.exp(rng.normal(-2.0, 5.0, n))
    random.seed(seed)
    ref = np.empty(n)
    for j in range(n):
        while True:
            try:
                ref[j] = gigrnd_reference.gigrnd(P[j], A[j], B[j])
            except (ArithmeticError, ValueError):
                continue
            break
    tail_ref = random.random_sample(3)
    random.seed(seed)
    out = sample_psi(P, A, B)
    tail = random.random_sample(3)
    return bool(np.array_equal(ref, out) and np.array_equal(tail_ref, tail))


def _block_ok(seed, b=60):
    from scipy import linalg
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((40, b)) + 0.5 * rng.standard_normal((40, 1))
    ld = np.corrcoef(X, rowvar=False)                    # rank-deficient (40 < 60), like small reference panels
    _, s, v = linalg.svd(ld)
    ld = (ld + np.dot(v.T, np.dot(np.diag(s), v))) / 2   # the original's PSD projection
    psi = np.minimum(np.exp(rng.normal(-3.0, 2.0, (b, 1))), 1.0)
    bhat = rng.normal(0.0, 0.01, (b, 1)); z = rng.standard_normal((b, 1)); sig = np.array([[0.97]]); n = 100000
    dinvt = ld + np.diag(1.0 / psi.T[0])
    U = linalg.cholesky(dinvt)
    ref = linalg.solve_triangular(U, linalg.solve_triangular(U, bhat, trans='T') + np.sqrt(sig / n) * z, trans='N')
    qref = np.dot(np.dot(ref.T, dinvt), ref)
    ws = new_workspace(b)
    D = ws['dinvt'][:b * b].reshape(b, b); F = ws['F'][:b * b].reshape(b, b)
    out = np.empty((b, 1))
    info = block_update(ld, 1.0 / psi[:, 0], bhat[:, 0], z[:, 0], float(np.sqrt(sig / n)[0, 0]), D, F, out[:, 0],
                        ws['chars'], ws['ints'])
    return bool(info == 0 and np.array_equal(out, ref) and np.array_equal(np.dot(np.dot(out.T, D), out), qref))


def self_check(n_gig=2000, seed=20261118):
    """Returns (gig_ok, block_ok, seconds). Never changes the global random state.
    Test hook: environment variable PRSCSX_FAST_SELFCHECK_SIMULATE_MISMATCH=1 reports a GIG mismatch."""
    import os
    state = random.get_state()
    t0 = time.perf_counter()
    try:
        gig_ok = _gig_ok(n_gig, seed) and os.environ.get('PRSCSX_FAST_SELFCHECK_SIMULATE_MISMATCH') != '1'
        blk_ok = _block_ok(seed)
    finally:
        random.set_state(state)
    return gig_ok, blk_ok, time.perf_counter() - t0
