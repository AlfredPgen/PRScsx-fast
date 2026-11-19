"""
GIL-free per-LD-block beta update that is bit-identical to PRS-CSx's

    dinvt = ld_blk + np.diag(1.0/psi_blk)
    dinvt_chol = scipy.linalg.cholesky(dinvt)                        # f2py: dpotrf('U') on a Fortran copy
    beta_tmp = solve_triangular(dinvt_chol, bhat_blk, trans='T') + sqrt(sigma/n)*z   # dtrtrs('U','T','N')
    beta_blk = solve_triangular(dinvt_chol, beta_tmp, trans='N')                       # dtrtrs('U','N','N')

SciPy's f2py LAPACK wrappers hold the GIL (measured: 0.9x "speed-up" with 4 threads), so we call the very
same LAPACK routines from the same shared library (scipy's bundled OpenBLAS, via scipy.linalg.cython_lapack)
inside a nogil numba function. Blocks can then be processed concurrently by Python threads.
"""
import ctypes

import numba as nb
import numpy as np
from numba.extending import get_cython_function_address

_vp = ctypes.c_void_p
_dpotrf = ctypes.CFUNCTYPE(None, _vp, _vp, _vp, _vp, _vp)(
    get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrf'))
_dtrtrs = ctypes.CFUNCTYPE(None, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp)(
    get_cython_function_address('scipy.linalg.cython_lapack', 'dtrtrs'))


@nb.njit(nogil=True, error_model='numpy')
def block_update(ld, invpsi, bhat, z, s, dinvt, F, beta, chars, ints):
    """
    ld     (b,b) C-contiguous LD block            invpsi (b,) = 1/psi on this block
    bhat   (b,)  marginal standardized effects    z (b,) standard normals, s = sqrt(sigma/n)
    dinvt  (b,b) C-contiguous output: ld + diag(invpsi) (kept for the quadratic form)
    F      (b,b) workspace: Fortran-ordered copy of dinvt, overwritten by its Cholesky factor
    beta   (b,)  output
    chars  uint8 workspace [ 'U', 'T', 'N' ];  ints int32 workspace (5)
    returns LAPACK info (0 = success)
    """
    b = ld.shape[0]
    for i in range(b):
        for j in range(b):
            dinvt[i, j] = ld[i, j]+0.0          # + diag(...) has exact zeros off the diagonal
        dinvt[i, i] = ld[i, i]+invpsi[i]
    for i in range(b):                          # F (C memory) == dinvt in Fortran order
        for j in range(b):
            F[j, i] = dinvt[i, j]
    ints[0] = b      # n = lda = ldb
    ints[1] = 1      # nrhs
    ints[2] = 0      # info
    _dpotrf(chars[0:].ctypes, ints[0:].ctypes, F.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2]
    for i in range(b):
        beta[i] = bhat[i]
    _dtrtrs(chars[0:].ctypes, chars[1:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[1:].ctypes,
            F.ctypes, ints[0:].ctypes, beta.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2]
    for i in range(b):
        beta[i] = beta[i]+s*z[i]
    _dtrtrs(chars[0:].ctypes, chars[2:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[1:].ctypes,
            F.ctypes, ints[0:].ctypes, beta.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    return ints[2]


def new_workspace(max_b):
    """Per-thread workspace."""
    return {'dinvt': np.empty(max_b*max_b), 'F': np.empty(max_b*max_b),
            'chars': np.frombuffer(b'UTN', dtype=np.uint8).copy(), 'ints': np.zeros(5, dtype=np.int32)}
