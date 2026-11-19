"""
Low-rank LD-block update, shared by fast/ (--linalg=lowrank|auto) and improved/.

For one block of b SNPs with D = V Lambda V' (V: b x r, Lambda: r positive eigenvalues), ridge R = diag(ridge) and
s = sqrt(sigma2/N), the beta update of PRS-CS(x) needs, with A = D + R:
    mean  m    = A^-1 bhat
    draw  beta = A^-1 (bhat + s*eta),  eta = V Lambda^1/2 e1 + R^1/2 e2,  e1 ~ N(0, I_r), e2 ~ N(0, I_b)
          => Cov(eta) = A, so beta ~ N(m, s^2 A^-1) exactly as the full-factorisation update
    quad       = beta' A beta = || Lambda^1/2 V' beta ||^2 + sum_j ridge_j beta_j^2
Woodbury identity, written with U = R^-1/2 V Lambda^1/2 (b x r) and K = I_r + U'U (r x r):
    A = R^1/2 (I + U U') R^1/2,    A^-1 x = R^-1/2 ( y - U K^-1 U' y ),  y = R^-1/2 x.
K equals Lambda^1/2 (Lambda^-1 + V' R^-1 V) Lambda^1/2, i.e. the r x r matrix of the instructions rescaled on both
sides; its eigenvalues are >= 1, so its Cholesky factor is well conditioned even in single precision (Lambda^-1
itself would reach 1e13).  Cost: b r^2 (U'U) + r^3/3 (Cholesky) + O(b r), against b^3/3 for the full factorisation.

Eigen-decomposition at load time (eigen_block): eigenvalues <= EIG_TOL * lambda_max are dropped. On the 1000 Genomes
chr22 blocks (EUR/EAS/AFR, 49,112 eigenvalues) after the original PSD projection, the exact zeros it creates all lie
below 1e-15 * lambda_max and there is no eigenvalue between 1e-15 and 1e-12 (measured), so EIG_TOL = 1e-13 removes only
those zeros ("no truncation"). A cut at 1e-8 would drop 766 genuine eigenvalues.
"""
import numba as nb
import numpy as np

EIG_TOL = 1e-13
AUTO_RATIO = 1.7           # --linalg=auto uses low-rank when b > AUTO_RATIO * r. Measured single-thread break-even
                           # b/r: 1.4-1.6 (r=100), 1.6-1.8 (r=300, 500); theory b^3/3 = b r^2 + r^3/3 gives 1.9.
                           # benchmarks/results_v2/wp2_T2d_lowrank_speed.json
_JIT = dict(nogil=True, cache=True, error_model='numpy')


@nb.njit(**_JIT)
def _chol_solve(L, T):
    """Solve (L L') X = T in place of T; L lower triangular r x r, T r x k."""
    r, k = T.shape
    for c in range(k):
        for i in range(r):                    # forward: L z = t
            acc = T[i, c]
            for j in range(i):
                acc -= L[i, j] * T[j, c]
            T[i, c] = acc / L[i, i]
        for i in range(r - 1, -1, -1):        # backward: L' x = z
            acc = T[i, c]
            for j in range(i + 1, r):
                acc -= L[j, i] * T[j, c]
            T[i, c] = acc / L[i, i]


@nb.njit(**_JIT)
def lowrank_update(V, sqlam, ridge, bhat, e1, e2, s, beta, mean, U):
    """
    V (b,r) and sqlam (r,) = sqrt(eigenvalues): float64 or float32 (sets the working precision of U, K, solves).
    ridge, bhat (b,), e1 (r,), e2 (b,): float64. U: (b,r) workspace, same dtype as V.
    Writes beta and mean (float64, (b,)); returns quad = beta' A beta (float64).
    """
    b, r = V.shape
    rs = np.empty(b)
    for i in range(b):
        rs[i] = 1.0 / np.sqrt(ridge[i])
    for i in range(b):
        for j in range(r):
            U[i, j] = V[i, j] * sqlam[j] * rs[i]
    K = np.dot(U.T, U)
    for j in range(r):
        K[j, j] += 1.0
    L = np.linalg.cholesky(K)
    Y = np.empty((b, 2), dtype=U.dtype)
    Ue1 = np.dot(U, e1.astype(U.dtype))
    for i in range(b):
        y1 = rs[i] * bhat[i]
        Y[i, 0] = y1
        Y[i, 1] = y1 + s * (Ue1[i] + e2[i])
    T = np.dot(U.T, Y)                          # r x 2
    _chol_solve(L, T)
    UT = np.dot(U, T)                           # b x 2
    for i in range(b):
        mean[i] = rs[i] * (Y[i, 0] - UT[i, 0])
        beta[i] = rs[i] * (Y[i, 1] - UT[i, 1])
    c = np.dot(V.T, beta.astype(V.dtype))       # r
    quad = 0.0
    for j in range(r):
        quad += float(sqlam[j]) ** 2 * float(c[j]) ** 2
    for i in range(b):
        quad += ridge[i] * beta[i] * beta[i]
    return quad


def eigen_block(D, tol=EIG_TOL):
    """Eigen-decomposition of a (PSD-projected) LD block; keeps eigenvalues > tol * lambda_max, descending order."""
    w, Q = np.linalg.eigh(D)
    w = w[::-1]; Q = Q[:, ::-1]
    keep = w > tol * w[0] if w.size and w[0] > 0 else np.zeros(w.size, bool)
    return np.ascontiguousarray(Q[:, keep]), w[keep].copy()


def truncate(V, lam, rank_var):
    """Keep the largest eigenvalues that explain the share rank_var of the trace (rank_var >= 1: keep all)."""
    if rank_var >= 1 or lam.size == 0:
        return V, lam
    cs = np.cumsum(lam) / lam.sum()
    k = int(np.searchsorted(cs, rank_var) + 1)
    k = min(k, lam.size)
    return np.ascontiguousarray(V[:, :k]), lam[:k].copy()


def use_lowrank(b, r, mode, ratio=AUTO_RATIO):
    if mode == 'lowrank':
        return True
    if mode == 'auto':
        return b > ratio * r
    return False


def prepare_blocks(ld_blk, blk_size, bhat, mode, rank_var=1.0, ratio=AUTO_RATIO, eig=None):
    """
    Decide, per non-empty LD block of one ancestry, whether the update is full or low-rank.
      ld_blk, blk_size : as returned by parse_ldblk;  bhat : (p,1) marginal effects of this ancestry (block order)
      mode             : 'full' | 'lowrank' | 'auto';  rank_var : share of the trace kept (>= 1: no truncation)
      eig              : optional list of precomputed (V, lam) per block (e.g. from a cache or an eigen panel)
    Returns (blocks, bhat_out, eig_list): blocks = list of (D or None, V or None, lam or None) for non-empty blocks;
    bhat_out = bhat, projected on span(V) block by block when rank_var < 1 (data consistent with the truncated LD).
    With mode == 'full' and rank_var >= 1 nothing is computed and bhat is returned unchanged (exact original path).
    """
    if mode == 'full' and rank_var >= 1:
        return [(ld, None, None) for ld, b in zip(ld_blk, blk_size) if b > 0], bhat, None
    bhat_out = np.array(bhat, dtype=np.float64, copy=True)
    out, eig_list = [], []
    mm = 0; kk = 0
    for k, (ld, b) in enumerate(zip(ld_blk, blk_size)):
        if b == 0:
            eig_list.append(None)
            continue
        if eig is not None and eig[k] is not None:
            V, lam = eig[k]
        else:
            V, lam = eigen_block(ld)
        eig_list.append((V, lam))
        V, lam = truncate(V, lam, rank_var)
        if rank_var < 1:
            bb = bhat_out[mm:mm + b, 0]
            bhat_out[mm:mm + b, 0] = V @ (V.T @ bb)
        if use_lowrank(b, V.shape[1], mode, ratio):
            out.append((None, V, lam))
        else:
            D = ld if (rank_var >= 1 and ld is not None) else np.ascontiguousarray((V * lam) @ V.T)
            out.append((D, None, None))
        mm += b; kk += 1
    return out, bhat_out, eig_list
