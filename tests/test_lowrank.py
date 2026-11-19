"""
WP2 tests of the low-rank LD-block update (fast/fastcsx/lowrank.py), used by fast (--linalg) and improved.
  T2a  kernel vs dense linear algebra: conditional mean m within 1e-9 (double) / 1e-4 (single) relative error,
       on 20 random rank-deficient blocks and 20 real (PRSCSX_REF_DIR) or synthetic LD blocks
  T2b  distribution: 20,000 draws for a fixed psi on a 40-SNP block; mean within 4 SE of m; covariance within 5%
       (Frobenius) of s^2 A^-1
  T2c  same posterior as the original: fast --linalg=full vs --linalg=lowrank, seeds 1-5 each, 2,000 iterations, on the
       chr22 example (real panel if available, else the synthetic panel); mean corr(full, lowrank) must not be lower
       than mean corr(full, full) by more than 0.005  (numbers written to benchmarks/results_v2/wp2_T2c_*.json)
"""
import itertools
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy import linalg

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
from fastcsx.lowrank import eigen_block, lowrank_update  # noqa: E402

REF_DIR = os.environ.get('PRSCSX_REF_DIR', '')
HAVE_REF = all(os.path.exists(os.path.join(REF_DIR, x)) for x in ('snpinfo_mult_1kg_hm3', 'ldblk_1kg_eur', 'ldblk_1kg_eas'))
TD = os.path.join(ROOT, 'test_data')


def _psd(R):
    _, s, v = linalg.svd(R)
    return (R + np.dot(v.T, np.dot(np.diag(s), v))) / 2


def _random_blocks(rng, n_blocks=20):
    out = []
    for _ in range(n_blocks):
        b = int(rng.integers(30, 400)); n = int(rng.integers(10, b))          # n < b: rank-deficient
        X = rng.standard_normal((n, b)) @ np.linalg.cholesky(0.8 ** np.abs(np.subtract.outer(np.arange(b), np.arange(b)))).T
        out.append(_psd(np.corrcoef(X, rowvar=False)))
    return out


def _ld_blocks(rng, n_blocks=20, tmpdir=None):
    """20 real 1000 Genomes chr22 blocks (EUR, EAS, AFR) if available, otherwise synthetic-panel blocks."""
    out = []
    if HAVE_REF:
        for pop in ('eur', 'eas', 'afr'):
            fn = os.path.join(REF_DIR, 'ldblk_1kg_%s' % pop, 'ldblk_1kg_chr22.hdf5')
            if not os.path.exists(fn):
                continue
            with h5py.File(fn, 'r') as h:
                for k in range(1, len(h) + 1):
                    R = np.array(h['blk_%d' % k]['ldblk'])
                    if R.size and len(out) < n_blocks:
                        out.append(_psd(R))
    if len(out) < n_blocks:
        for seed in range(1, 10):
            d = os.path.join(tmpdir, 'synth%d' % seed)
            subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), d, '--seed', str(seed)], check=True, capture_output=True)
            for pop in ('eur', 'eas', 'afr'):
                with h5py.File(os.path.join(d, 'ldblk_1kg_%s' % pop, 'ldblk_1kg_chr22.hdf5'), 'r') as h:
                    for k in range(1, len(h) + 1):
                        R = np.array(h['blk_%d' % k]['ldblk'])
                        if R.size > 1 and len(out) < n_blocks:
                            out.append(_psd(R))
            if len(out) >= n_blocks:
                break
    return out


def _check(D, rng, dtype, tol):
    b = D.shape[0]
    V, lam = eigen_block(D)
    ridge = 1.0 / np.minimum(np.exp(rng.normal(-3, 2, b)), 1.0)
    bhat = rng.normal(0, 0.01, b)
    m_ref = np.linalg.solve(D + np.diag(ridge), bhat)                  # full factorisation of the original D
    beta = np.empty(b); mean = np.empty(b); U = np.empty(V.shape, dtype=dtype)
    lowrank_update(V.astype(dtype), np.sqrt(lam).astype(dtype), ridge, bhat, rng.standard_normal(V.shape[1]),
                   rng.standard_normal(b), 0.003, beta, mean, U)
    err = np.linalg.norm(mean - m_ref) / np.linalg.norm(m_ref)
    assert err < tol, (b, V.shape[1], err)
    return err


@pytest.mark.parametrize('dtype,tol', [(np.float64, 1e-9), (np.float32, 1e-4)])
def test_T2a_kernel_vs_dense(tmp_path, dtype, tol):
    rng = np.random.default_rng(2)
    errs = [_check(D, rng, dtype, tol) for D in _random_blocks(rng)]
    errs += [_check(D, rng, dtype, tol) for D in _ld_blocks(rng, tmpdir=str(tmp_path))]
    assert len(errs) == 40


def test_T2b_draw_distribution():
    rng = np.random.default_rng(3)
    b, n = 40, 25
    X = rng.standard_normal((n, b)) @ np.linalg.cholesky(0.9 ** np.abs(np.subtract.outer(np.arange(b), np.arange(b)))).T
    D = _psd(np.corrcoef(X, rowvar=False))
    V, lam = eigen_block(D)
    r = V.shape[1]
    assert r < b
    ridge = 1.0 / np.minimum(np.exp(rng.normal(-2, 1.5, b)), 1.0)
    bhat = rng.normal(0, 0.02, b); s = 0.01
    A = D + np.diag(ridge)
    m = np.linalg.solve(A, bhat); C = s ** 2 * np.linalg.inv(A)
    nd = 20000
    draws = np.empty((nd, b)); beta = np.empty(b); mean = np.empty(b); U = np.empty((b, r))
    sq = np.sqrt(lam)
    for i in range(nd):
        lowrank_update(V, sq, ridge, bhat, rng.standard_normal(r), rng.standard_normal(b), s, beta, mean, U)
        draws[i] = beta
    se = draws.std(0, ddof=1) / np.sqrt(nd)
    assert np.all(np.abs(draws.mean(0) - m) < 4 * se)
    Cs = np.cov(draws, rowvar=False)
    assert np.linalg.norm(Cs - C) / np.linalg.norm(C) < 0.05


@pytest.fixture(scope='module')
def t2c_panel(tmp_path_factory):
    if HAVE_REF:
        return REF_DIR, 'real 1000 Genomes'
    d = tmp_path_factory.mktemp('synth')
    subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(d)], check=True, capture_output=True)
    return str(d), 'synthetic'


def test_T2c_same_posterior_as_original(t2c_panel, tmp_path):
    panel, kind = t2c_panel
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    est = {}
    for mode in ('full', 'lowrank'):
        for seed in range(1, 6):
            d = tmp_path / ('%s_%d' % (mode, seed)); d.mkdir()
            r = subprocess.run([sys.executable, os.path.join(ROOT, 'fast', 'PRScsx_fast.py'), '--ref_dir=' + panel,
                                '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD),
                                '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=%d' % seed,
                                '--linalg=' + mode, '--threads=2', '--quiet', '--out_name=t', '--out_dir=%s' % d],
                               capture_output=True, text=True, timeout=1800, env=env)
            assert r.returncode == 0, r.stdout[-2000:]
            for pop in ('EUR', 'EAS'):
                est[(mode, seed, pop)] = np.loadtxt(str(d / ('t_%s_pst_eff_a1_b0.5_phi1e-02_chr22.txt' % pop)), usecols=5)
    res = {'panel': kind}
    for pop in ('EUR', 'EAS'):
        ff = [np.corrcoef(est[('full', i, pop)], est[('full', j, pop)])[0, 1] for i, j in itertools.combinations(range(1, 6), 2)]
        fl = [np.corrcoef(est[('full', i, pop)], est[('lowrank', j, pop)])[0, 1] for i in range(1, 6) for j in range(1, 6)]
        ll = [np.corrcoef(est[('lowrank', i, pop)], est[('lowrank', j, pop)])[0, 1] for i, j in itertools.combinations(range(1, 6), 2)]
        res[pop] = dict(mean_corr_full_full=float(np.mean(ff)), mean_corr_full_lowrank=float(np.mean(fl)),
                        mean_corr_lowrank_lowrank=float(np.mean(ll)), difference=float(np.mean(ff) - np.mean(fl)), passed=bool(np.mean(fl) >= np.mean(ff) - 0.005))
    out = os.path.join(ROOT, 'benchmarks', 'results_v2', 'wp2_T2c_same_posterior_%s.json' % ('real' if HAVE_REF else 'synthetic'))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
    for pop in ('EUR', 'EAS'):
        assert res[pop]['passed'], res
