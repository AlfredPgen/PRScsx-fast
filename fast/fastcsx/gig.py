"""
Numba port of PRS-CSx's GIG sampler (gigrnd.py, Devroye 2014) that reproduces the original
*bit for bit*, including its consumption of the numpy legacy random stream.

How exactness is achieved
-------------------------
* Same arithmetic, same operation order, no fast-math (LLVM keeps IEEE semantics, no FMA contraction).
* exp/log/cosh/sinh/sqrt resolve to the same C maths library the CPython `math` module uses (glibc on Linux).
  Bit identity was verified on Linux/glibc (tests in this repository) and independently on Windows 11 with a
  different C library and OpenBLAS build (review of 18 Nov 2026). Because it depends on numba, LLVM and the C
  library, it is re-checked at every start-up (fastcsx/selfcheck.py; --strict turns a mismatch into an error).
  pow(x, 2) is called with the exponent supplied at run time: LLVM would otherwise rewrite
  pow(x, 2.0) as x*x, which differs from glibc pow in ~0.09% of inputs (measured).
* Python exceptions raised inside gigrnd (OverflowError, ZeroDivisionError, ValueError) are emulated
  by explicit checks; as in the fixed PRScsx.py they count as a failed attempt and sampling continues
  with fresh uniforms (max MAX_FAIL consecutive failures per SNP).
* Uniforms: the original calls numpy.random.random() three times per rejection attempt. We draw a
  buffer with numpy.random.random_sample(L) (identical stream), consume it in compiled code, then
  restore the generator state and advance it by exactly the number of uniforms consumed.
"""
import math

import numba as nb
import numpy as np
from numpy import random

MAX_FAIL = 1000
_TWO = np.array([2.0])   # passed at run time, see _pow2
OK, NEED_MORE, SETUP_ERROR, TOO_MANY_FAILURES, NAN_ERROR = 0, 1, 2, 3, 4
_JIT = dict(nogil=True, cache=True, error_model='numpy')


@nb.njit(**_JIT)
def _bad_unary(x, r):
    # CPython math_1(): ValueError if NaN produced from non-NaN input, OverflowError if inf from finite input
    if math.isnan(r) and not math.isnan(x):
        return True
    if math.isinf(r) and not math.isinf(x):
        return True
    return False


@nb.njit(**_JIT)
def _pow2(x, two):
    # CPython math.pow(x, 2) for finite x = libm pow(x, 2.0); OverflowError if result is inf.
    # `two` (== 2.0) is passed at run time so LLVM cannot rewrite pow(x, 2.0) as x*x.
    r = math.pow(x, two)
    bad = math.isinf(r) and not math.isinf(x)
    return r, bad


@nb.njit(**_JIT)
def _psi(x, alpha, lam):
    c = math.cosh(x)
    if _bad_unary(x, c):
        return 0.0, True
    e = math.exp(x)
    if _bad_unary(x, e):
        return 0.0, True
    return -alpha*(c-1.0)-lam*(e-x-1.0), False


@nb.njit(**_JIT)
def _dpsi(x, alpha, lam):
    sh = math.sinh(x)
    if _bad_unary(x, sh):
        return 0.0, True
    e = math.exp(x)
    if _bad_unary(x, e):
        return 0.0, True
    return -alpha*sh-lam*(e-1.0), False


@nb.njit(**_JIT)
def _log(x):
    # CPython math.log: ValueError for x <= 0
    if x <= 0.0:
        return 0.0, True
    return math.log(x), False


@nb.njit(**_JIT)
def _sqrt(x):
    if x < 0.0:
        return 0.0, True
    return math.sqrt(x), False


@nb.njit(**_JIT)
def _gig_one(p, a, b, U, pos, max_fail, two):
    """One GIG(p, a, b) variate. Returns (value, new_pos, status)."""
    # ---- setup (deterministic; any Python exception here would repeat forever -> error) ----
    lam = p
    omega, bad = _sqrt(a*b)
    if bad:
        return 0.0, pos, SETUP_ERROR
    swap = False
    if lam < 0:
        lam = -lam
        swap = True
    po, b1 = _pow2(omega, two)
    pl, b2 = _pow2(lam, two)
    if b1 or b2:
        return 0.0, pos, SETUP_ERROR
    tmp, bad = _sqrt(po+pl)
    if bad:
        return 0.0, pos, SETUP_ERROR
    alpha = tmp-lam

    # find t
    x, bad = _psi(1.0, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    x = -x
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            den = alpha+lam
            if den == 0.0:
                return 0.0, pos, SETUP_ERROR
            t, bad = _sqrt(2.0/den)
            if bad:
                return 0.0, pos, SETUP_ERROR
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            den = alpha+2.0*lam
            if den == 0.0:
                return 0.0, pos, SETUP_ERROR
            t, bad = _log(4.0/den)
            if bad:
                return 0.0, pos, SETUP_ERROR
    else:
        return 0.0, pos, NAN_ERROR

    # find s
    x, bad = _psi(-1.0, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    x = -x
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            den = alpha*math.cosh(1)+lam
            if den == 0.0:
                return 0.0, pos, SETUP_ERROR
            s, bad = _sqrt(4.0/den)
            if bad:
                return 0.0, pos, SETUP_ERROR
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        else:
            pa, bad = _pow2(alpha, two)
            if bad or pa == 0.0:
                return 0.0, pos, SETUP_ERROR
            inner, bad = _sqrt(1.0/pa+2.0/alpha)
            if bad:
                return 0.0, pos, SETUP_ERROR
            lg, bad = _log(1.0+1.0/alpha+inner)
            if bad:
                return 0.0, pos, SETUP_ERROR
            if lam == 0:
                s = lg
            else:
                s1 = 1.0/lam
                s = lg if lg < s1 else s1      # Python min(s1, lg)
    else:
        return 0.0, pos, NAN_ERROR

    # auxiliary parameters
    v, bad = _psi(t, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    eta = -v
    v, bad = _dpsi(t, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    zeta = -v
    v, bad = _psi(-s, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    theta = -v
    xi, bad = _dpsi(-s, alpha, lam)
    if bad:
        return 0.0, pos, SETUP_ERROR
    if xi == 0.0 or zeta == 0.0:
        return 0.0, pos, SETUP_ERROR
    pp = 1.0/xi
    r = 1.0/zeta
    td = t-r*eta
    sd = s-pp*theta
    q = td+sd

    # ---- rejection sampling; a Python exception = failed call, retried with fresh uniforms ----
    n_fail = 0
    nU = U.shape[0]
    while True:
        if pos+3 > nU:
            return 0.0, pos, NEED_MORE
        u = U[pos]
        vv = U[pos+1]
        w = U[pos+2]
        pos += 3
        failed = False
        if u < q/(pp+q+r):
            rnd = -sd+q*vv
        elif u < (q+r)/(pp+q+r):
            lv, failed = _log(vv)
            rnd = td-r*lv
        else:
            lv, failed = _log(vv)
            rnd = -sd+pp*lv
        if not failed:
            arg = -eta-zeta*(rnd-t)
            f1 = math.exp(arg)
            failed = _bad_unary(arg, f1)
        if not failed:
            arg = -theta+xi*(rnd+s)
            f2 = math.exp(arg)
            failed = _bad_unary(arg, f2)
        if not failed:
            if (rnd >= -sd) and (rnd <= td):
                gv = 1.0
            elif rnd > td:
                gv = f1
            elif rnd < -sd:
                gv = f2
            else:
                return 0.0, pos, NAN_ERROR
            ps, failed = _psi(rnd, alpha, lam)
        if not failed:
            eps = math.exp(ps)
            failed = _bad_unary(ps, eps)
        if not failed:
            if not (w*gv <= eps):
                continue                          # ordinary rejection
            # accepted: transform back to the three-parameter GIG (may also raise in Python)
            er = math.exp(rnd)
            failed = _bad_unary(rnd, er)
            if not failed and omega == 0.0:
                failed = True
            if not failed:
                pl2, b1 = _pow2(lam, two)
                po2, b2 = _pow2(omega, two)
                failed = b1 or b2 or (po2 == 0.0)
            if not failed:
                sq, failed = _sqrt(1.0+pl2/po2)
            if not failed:
                val = er*(lam/omega+sq)
                if swap:
                    if val == 0.0:
                        failed = True
                    else:
                        val = 1.0/val
            if not failed:
                if b == 0.0:
                    failed = True
                else:
                    sq, failed = _sqrt(a/b)
                    if not failed and sq == 0.0:
                        failed = True
            if not failed:
                return val/sq, pos, OK
        n_fail += 1
        if n_fail >= max_fail:
            return 0.0, pos, TOO_MANY_FAILURES


@nb.njit(**_JIT)
def _gig_batch(P, A, B, U, pos, j0, out, max_fail, two):
    for j in range(j0, P.shape[0]):
        val, newpos, status = _gig_one(P[j], A[j], B[j], U, pos, max_fail, two)
        if status != OK:
            return j, pos, status          # pos = start of SNP j (restart point)
        out[j] = val
        pos = newpos
    return P.shape[0], pos, OK


def sample_psi(P, A, B, max_fail=MAX_FAIL, rs=None):
    """psi_j ~ GIG(P_j, A_j, B_j), j = 0..p-1, consuming numpy's *global* legacy stream exactly as
    `for jj: gigrnd.gigrnd(P[jj], A[jj], B[jj])` in the (fixed) original would.
    rs: optional numpy.random.RandomState to use instead of the global stream (same stream for the same seed)."""
    if rs is None:
        rs = random
    P = np.ascontiguousarray(P, dtype=np.float64)
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    n = P.shape[0]
    state = rs.get_state()
    chunk = 4*n+300                       # expected use is ~3.5 uniforms per SNP
    U = rs.random_sample(chunk)
    out = np.empty(n)
    j, pos = 0, 0
    while True:
        j, pos, status = _gig_batch(P, A, B, U, pos, j, out, max_fail, _TWO[0])
        if status == OK:
            break
        if status == NEED_MORE:
            U = np.concatenate([U, rs.random_sample(chunk)])
            continue
        if status == TOO_MANY_FAILURES:
            raise RuntimeError('GIG sampling failed %d times in a row for SNP index %d (p=%r, a=%r, b=%r)'
                               % (max_fail, j, P[j], A[j], B[j]))
        raise RuntimeError('GIG sampling: invalid parameters for SNP index %d (p=%r, a=%r, b=%r)' % (j, P[j], A[j], B[j]))
    rs.set_state(state)
    rs.random_sample(pos)                 # advance the stream by exactly the uniforms consumed
    return out
