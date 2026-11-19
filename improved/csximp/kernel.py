"""
GIL-free per-LD-block update for PRS-CSx-improved.

For one block with factor U'U = D + kappa*Psi^-1 (kappa = Nbar/N_k under effect-scale coupling, 1 under PRS-CSx):
    x1   = U^-T bhat
    m    = U^-1 x1                   = (D + kappa Psi^-1)^-1 bhat      (conditional posterior mean, Rao-Blackwell)
    beta = U^-1 (x1 + s z)           ~ N(m, (sigma2/N_k)(D + kappa Psi^-1)^-1)
    quad = beta' (D + kappa Psi^-1) beta
Both triangular back-solves are done in one LAPACK call (nrhs = 2).
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
def block_update_rb(ld, ridge, bhat, z, s, F, rhs, beta, mean, chars, ints):
    """
    ld (b,b) C-contiguous; ridge (b,) = kappa/psi; bhat, z (b,); s = sqrt(sigma2/N_k)
    F (b,b) workspace, rhs (2,b) workspace; beta, mean (b,) outputs. Returns (info, quad).
    """
    b = ld.shape[0]
    for i in range(b):                    # F (C memory) == (ld + diag(ridge)) in Fortran order
        for j in range(b):
            F[j, i] = ld[i, j]
        F[i, i] = ld[i, i]+ridge[i]
    ints[0] = b; ints[1] = 1; ints[2] = 0; ints[3] = 2
    _dpotrf(chars[0:].ctypes, ints[0:].ctypes, F.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    for i in range(b):
        rhs[0, i] = bhat[i]
    _dtrtrs(chars[0:].ctypes, chars[1:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[1:].ctypes,
            F.ctypes, ints[0:].ctypes, rhs.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    for i in range(b):
        rhs[1, i] = rhs[0, i]+s*z[i]
    _dtrtrs(chars[0:].ctypes, chars[2:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[3:].ctypes,
            F.ctypes, ints[0:].ctypes, rhs.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    quad = 0.0
    for i in range(b):
        mean[i] = rhs[0, i]
        beta[i] = rhs[1, i]
    for i in range(b):                    # beta' (ld + diag(ridge)) beta
        acc = 0.0
        for j in range(b):
            acc += ld[i, j]*beta[j]
        quad += beta[i]*(acc+ridge[i]*beta[i])
    return 0, quad


def new_workspace(max_b):
    return {'F': np.empty(max_b*max_b), 'rhs': np.empty(2*max_b),
            'chars': np.frombuffer(b'UTN', dtype=np.uint8).copy(), 'ints': np.zeros(5, dtype=np.int32)}


# ------------------------------------------------------------------ single precision (2x faster Cholesky, half the memory)
_spotrf = ctypes.CFUNCTYPE(None, _vp, _vp, _vp, _vp, _vp)(
    get_cython_function_address('scipy.linalg.cython_lapack', 'spotrf'))
_strtrs = ctypes.CFUNCTYPE(None, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp, _vp)(
    get_cython_function_address('scipy.linalg.cython_lapack', 'strtrs'))


@nb.njit(nogil=True, error_model='numpy')
def block_update_rb32(ld, ridge, bhat, z, s, F, rhs, beta, mean, chars, ints):
    """Same as block_update_rb with ld, F, rhs in float32 (factorisation and solves in single precision);
    inputs/outputs bhat, z, beta, mean, quad stay float64. Relative error of the solves ~1e-7 (measured),
    negligible against Monte Carlo error; D + kappa*Psi^-1 has eigenvalues >= kappa >= 1 when psi <= 1."""
    b = ld.shape[0]
    for i in range(b):
        for j in range(b):
            F[j, i] = ld[i, j]
        F[i, i] = np.float32(ld[i, i]+ridge[i])
    ints[0] = b; ints[1] = 1; ints[2] = 0; ints[3] = 2
    _spotrf(chars[0:].ctypes, ints[0:].ctypes, F.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    for i in range(b):
        rhs[0, i] = bhat[i]
    _strtrs(chars[0:].ctypes, chars[1:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[1:].ctypes,
            F.ctypes, ints[0:].ctypes, rhs.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    for i in range(b):
        rhs[1, i] = rhs[0, i]+s*z[i]
    _strtrs(chars[0:].ctypes, chars[2:].ctypes, chars[2:].ctypes, ints[0:].ctypes, ints[3:].ctypes,
            F.ctypes, ints[0:].ctypes, rhs.ctypes, ints[0:].ctypes, ints[2:].ctypes)
    if ints[2] != 0:
        return ints[2], 0.0
    quad = 0.0
    for i in range(b):
        mean[i] = rhs[0, i]
        beta[i] = rhs[1, i]
    for i in range(b):
        acc = 0.0
        for j in range(b):
            acc += ld[i, j]*beta[j]
        quad += beta[i]*(acc+ridge[i]*beta[i])
    return 0, quad


def new_workspace32(max_b):
    return {'F': np.empty(max_b*max_b, dtype=np.float32), 'rhs': np.empty(2*max_b, dtype=np.float32),
            'chars': np.frombuffer(b'UTN', dtype=np.uint8).copy(), 'ints': np.zeros(5, dtype=np.int32)}
