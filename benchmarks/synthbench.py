"""
In-process synthetic benchmark engine (WP4 coupling stress test, WP5 accuracy benchmark v2).

Data (one replicate):
  * 3 ancestries EUR, EAS, AFR with GWAS sizes N = 300,000 / 100,000 / 30,000 (default).
  * M SNPs in blocks of `block` SNPs. In each block and ancestry, genotypes follow a latent Gaussian AR(1) with
    ancestry-specific correlation (EUR 0.95, EAS 0.93, AFR 0.85: shorter LD in AFR) and allele frequencies
    U(0.05, 0.5) drawn per ancestry. "True" LD = correlation in 10,000 simulated people; the reference LD given to the
    methods = correlation in n_ref = 500 / 500 / 650 other people (finite reference panel => LD mismatch), then the
    PRS-CSx PSD projection.
  * Causal SNPs shared (probability pi); effects (standardised) ~ N(0, C), C_kl = rho, C_kk = 1, with an extra variance
    multiplier for the smallest GWAS (AFR). Effects are scaled so that EUR has b'R b = h2 * M / M_genome
    (per-SNP heritability of a genome-wide trait, M_genome = 1.1e6); the same scale is used in every ancestry, so
    AFR's region heritability is `mult` times larger (up to LD).
  * bhat_k = R_true,k beta_k + N(0, R_true,k / N_k).
Methods: 'prscsx' = the exact PRS-CSx sampler of fast/ (= the original); improved/ with coupling snr / effect /
learned (RB means, single precision), optionally 2 chains. All use PRS-CSx's default chain length (3,000 / 1,500).
Accuracy (exact, from true effects and true LD): r2_g = R2/h2 of the combined PRS in each target ancestry,
  * oracle: best phi with population-optimal combination weights;
  * validation: README procedure, phi and OLS weights chosen in a validation sample of n_val people (genome-equivalent
    selection noise, as in benchmarks/sim/evaluate.py).
"""
import os
import sys
import tempfile

import numpy as np
from scipy import linalg
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
sys.path.insert(0, os.path.join(ROOT, 'improved'))
sys.path.insert(0, os.path.join(HERE, 'sim'))

POPS = ('EUR', 'EAS', 'AFR')
RHO_LD = {'EUR': 0.95, 'EAS': 0.93, 'AFR': 0.85}
N_REF = {'EUR': 500, 'EAS': 500, 'AFR': 650}
M_GENOME = 1.1e6


def _genotypes(rng, n, frq, rho):
    b = frq.size
    z = np.empty((2 * n, b))
    z[:, 0] = rng.standard_normal(2 * n)
    e = rng.standard_normal((2 * n, b)) * np.sqrt(1 - rho ** 2)
    for j in range(1, b):
        z[:, j] = rho * z[:, j - 1] + e[:, j]
    h = (z < norm.ppf(frq)).astype(np.float64)
    return h[:n] + h[n:]


def _corr(G):
    X = G - G.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X /= sd
    R = X.T @ X / G.shape[0]
    np.fill_diagonal(R, 1.0)
    return R


def _psd(R):
    try:
        _, s, v = linalg.svd(R)
    except np.linalg.LinAlgError:
        _, s, v = linalg.svd(R, lapack_driver='gesvd')
    return (R + np.dot(v.T, np.dot(np.diag(s), v))) / 2


def make_dataset(seed, M=2000, block=200, h2=0.5, pi=0.01, rho=0.8, mult=1.0, n_gwas=(300000, 100000, 30000), n_true=10000):
    rng = np.random.default_rng(seed)
    nb = M // block
    data = {'M': M, 'block': block, 'h2': h2, 'pi': pi, 'rho': rho, 'mult': mult, 'n_gwas': list(n_gwas)}
    R_true = {p: [] for p in POPS}; R_ref = {p: [] for p in POPS}
    for p in POPS:
        for k in range(nb):
            frq = rng.uniform(0.05, 0.5, block)
            R_true[p].append(_corr(_genotypes(rng, n_true, frq, RHO_LD[p])))
            R_ref[p].append(_psd(_corr(_genotypes(rng, N_REF[p], frq, RHO_LD[p]))))
    causal = rng.random(M) < pi
    if not causal.any():
        causal[rng.integers(M)] = True
    C = np.full((3, 3), rho); np.fill_diagonal(C, 1.0)
    C[2, :] *= np.sqrt(mult); C[:, 2] *= np.sqrt(mult)          # AFR (smallest GWAS) variance multiplier
    beta = np.zeros((M, 3))
    beta[causal] = rng.multivariate_normal(np.zeros(3), C, size=int(causal.sum()))
    gv = sum(beta[k * block:(k + 1) * block, 0] @ R_true['EUR'][k] @ beta[k * block:(k + 1) * block, 0] for k in range(nb))
    h2_region = h2 * M / M_GENOME
    beta *= np.sqrt(h2_region / gv)
    bhat = np.zeros((M, 3))
    for i, p in enumerate(POPS):
        for k in range(nb):
            sl = slice(k * block, (k + 1) * block)
            w, V = np.linalg.eigh(R_true[p][k]); w = np.clip(w, 0, None)
            bhat[sl, i] = R_true[p][k] @ beta[sl, i] + V @ (np.sqrt(w / n_gwas[i]) * rng.standard_normal(block))
    data.update(R_true=R_true, R_ref=R_ref, beta=beta, bhat=bhat, h2_region=h2_region, n_causal=int(causal.sum()))
    return data


# --------------------------------------------------------------------------------------------- fitting
def _common(data):
    M = data['M']
    snp = {'CHR': [22] * M, 'SNP': ['rs%d' % i for i in range(M)], 'BP': list(range(M)), 'A1': ['A'] * M, 'A2': ['G'] * M}
    bt = {k: data['bhat'][:, k].reshape(M, 1).copy() for k in range(3)}
    fr = {k: np.full((M, 1), 0.5) for k in range(3)}           # het = sqrt(0.5) for all: standardised = per-allele * het
    ix = {k: list(range(M)) for k in range(3)}
    return snp, bt, fr, ix


def fit(data, method, phi, seed, n_chains=1, n_iter=3000):
    """Posterior mean effects on the standardised scale, (M, 3). phi=None means auto."""
    from fastcsx.mcmc import mcmc as mcmc_exact
    from csximp.mcmc_improved import Solver, _run_chain
    snp, bt, fr, ix = _common(data)
    n = data['n_gwas']
    nb = data['M'] // data['block']
    ld = [data['R_ref'][p] for p in POPS]
    bs = [[data['block']] * nb for _ in POPS]
    het = np.sqrt(0.5)
    if method == 'prscsx':
        with tempfile.TemporaryDirectory() as d:
            files = mcmc_exact(1, 0.5, phi, snp, bt, fr, ix, n, ld, bs, n_iter, n_iter // 2, 5, list(POPS), 22, d, 't', 'FALSE',
                               'FALSE', seed, n_threads=1, log=lambda *x: None)
            return np.column_stack([np.loadtxt(f, usecols=5) for f in files[:3]]) * het
    coupling = method.split('_')[0]                    # snr | effect | learned, optional suffix _2ch
    solver = Solver(ld, bs, 1, precision='single')
    est = []
    for c in range(n_chains):
        r = _run_chain(1, 0.5, phi, snp, bt, fr, ix, n, solver, n_iter, n_iter // 2, 5, list(POPS), seed + 1000 * c, coupling,
                       True, 1.0, lambda *x: None, 'FALSE', 'FALSE')
        est.append(np.column_stack([r['beta_est'][k][:, 0] for k in range(3)]) * het)
    solver.close()
    return np.mean(est, axis=0)


# --------------------------------------------------------------------------------------------- evaluation
def moments(data, W, t):
    """W: (M, m) standardised weights; target ancestry index t. Returns S_WW, S_Wb, var_g."""
    block = data['block']; R = data['R_true'][POPS[t]]; b = data['beta'][:, t]
    m = W.shape[1]
    SWW = np.zeros((m, m)); SWb = np.zeros(m); vg = 0.0
    for k in range(data['M'] // block):
        sl = slice(k * block, (k + 1) * block)
        RW = R[k] @ W[sl]
        SWW += W[sl].T @ RW; SWb += RW.T @ b[sl]; vg += b[sl] @ R[k] @ b[sl]
    return SWW, SWb, vg


def evaluate(data, est_by_phi, n_val=3000, val_seed=0):
    """est_by_phi: {phi_label: (M,3) estimates}. Returns {target: {metric: value, 'phi_...': label}}."""
    from evaluate import r2_oracle, validation_select
    labels = [l for l in est_by_phi if l != 'auto']
    out = {}
    for t, pop in enumerate(POPS):
        cols, W = [], []
        for l in labels:
            W.append(est_by_phi[l]); cols.append(list(range(3 * len(cols), 3 * len(cols) + 3)))
        Wm = np.hstack(W)
        SWW, SWb, vg = moments(data, Wm, t)
        orc = [r2_oracle(SWW, SWb, vg, c)[0] for c in cols]
        bi = int(np.argmax(orc))
        rng = np.random.default_rng(val_seed + 1000 * t)
        r2v, gi = validation_select(SWW, SWb, vg, cols, data['h2'], n_val, rng)
        out[pop] = {'r2_tuned_oracle': float(orc[bi]), 'phi_oracle': labels[bi], 'r2_tuned_validation': float(r2v),
                    'phi_validation': labels[gi], **{'r2_oracle_phi%s' % l: float(v) for l, v in zip(labels, orc)}}
    return out
