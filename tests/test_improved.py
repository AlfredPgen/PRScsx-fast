"""
Tests for improved/ (PRS-CSx-improved). Correctness (not bit identity, which is not intended):
 * block kernel = direct linear algebra (double and single precision)
 * effect-scale coupling == PRS-CSx coupling for a single ancestry (same seed -> identical output)
 * same posterior as the exact original sampler: long-run posterior means agree within Monte Carlo error
 * Ledoit-Wolf intensity formula; CLI output format/SNP set identical to the original's
Run:  PRSCSX_REF_DIR=/path/to/ref python -m pytest -v tests/test_improved.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
sys.path.insert(0, os.path.join(ROOT, 'improved'))
REF_DIR = os.environ.get('PRSCSX_REF_DIR', '')
TD = os.path.join(ROOT, 'test_data')
REF_OUT = os.path.join(HERE, 'reference_outputs')

from csximp.kernel import block_update_rb, block_update_rb32, new_workspace, new_workspace32  # noqa: E402
from csximp.mcmc_improved import Solver, lw_shrinkage, mcmc_improved  # noqa: E402
from fastcsx.mcmc import mcmc as mcmc_exact  # noqa: E402

have_ref = all(os.path.exists(os.path.join(REF_DIR, x)) for x in ('snpinfo_mult_1kg_hm3', 'ldblk_1kg_eur', 'ldblk_1kg_eas'))
needs_ref = pytest.mark.skipif(not have_ref, reason='set PRSCSX_REF_DIR')


def _psd_corr(rng, n, b):
    return np.atleast_2d(np.corrcoef(rng.standard_normal((n, b)), rowvar=False))


@pytest.mark.parametrize('single', [False, True])
def test_kernel_matches_direct_linear_algebra(single):
    rng = np.random.default_rng(0)
    for b in (1, 5, 120, 300):
        ld = _psd_corr(rng, 400, b)
        ridge = 1.0 / np.minimum(np.exp(rng.normal(-3, 2, b)), 1.0)
        bhat = rng.normal(0, 0.01, b); z = rng.standard_normal(b); s = 0.003
        A = ld + np.diag(ridge)
        m_ref = np.linalg.solve(A, bhat)
        U = np.linalg.cholesky(A).T                       # A = U'U
        beta_ref = np.linalg.solve(U, np.linalg.solve(U.T, bhat) + s * z)
        ws = (new_workspace32 if single else new_workspace)(b)
        fn = block_update_rb32 if single else block_update_rb
        ldx = ld.astype(np.float32) if single else ld
        beta = np.empty(b); mean = np.empty(b)
        info, q = fn(ldx, ridge, bhat, z, s, ws['F'][:b * b].reshape(b, b), ws['rhs'][:2 * b].reshape(2, b),
                     beta, mean, ws['chars'], ws['ints'])
        tol = 2e-5 if single else 1e-10
        assert info == 0
        assert np.allclose(mean, m_ref, rtol=tol, atol=tol * np.abs(m_ref).max())
        assert np.allclose(beta, beta_ref, rtol=tol, atol=tol * np.abs(beta_ref).max())
        assert np.isclose(q, beta_ref @ A @ beta_ref, rtol=tol)


def test_lw_shrinkage_formula():
    R = np.array([[1, .5, 0], [.5, 1, .2], [0, .2, 1.]])
    n = 101
    off = R[~np.eye(3, dtype=bool)]
    expect = np.sum((1 - off ** 2) ** 2) / (n - 1) / np.sum(off ** 2)
    assert np.isclose(lw_shrinkage(R, n), min(1, expect))
    assert lw_shrinkage(np.eye(4), 100) == 0.0


def tiny(n_pop=2, p=8, seed=0):
    rng = np.random.default_rng(seed)
    snp = {'CHR': [22] * p, 'SNP': ['rs%d' % i for i in range(p)], 'BP': list(range(p)), 'A1': ['A'] * p, 'A2': ['G'] * p}
    bt, fr, ix, ld, bs = {}, {}, {}, {}, {}
    truth = np.zeros(p); truth[[0, 5]] = [0.03, -0.02]
    for k in range(n_pop):
        R = _psd_corr(rng, 300, p)
        bt[k] = (R @ truth + rng.multivariate_normal(np.zeros(p), R / 20000)).reshape(p, 1)
        fr[k] = np.full((p, 1), 0.25); ix[k] = list(range(p)); ld[k] = [R[:4, :4], R[4:, 4:]]; bs[k] = [4, p - 4]
    return snp, bt, fr, ix, ld, bs


def _read(fn):
    return np.loadtxt(fn, usecols=5)


def test_effect_coupling_equals_snr_for_one_ancestry(tmp_path):
    snp, bt, fr, ix, ld, bs = tiny(1)
    outs = []
    for c in ('effect', 'snr'):
        d = tmp_path / c; d.mkdir()
        solver = Solver([ld[0]], [bs[0]], 1, precision='double')
        f = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, [20000], solver, 400, 200, 5, ['EUR'], 22, str(d), 't', 'FALSE',
                          'FALSE', 3, coupling=c, rb=True, log=lambda *x: None)
        outs.append(_read(f[0]))
    assert np.array_equal(outs[0], outs[1])


def test_same_posterior_as_exact_original_sampler(tmp_path):
    """coupling=snr: same model as PRS-CSx. Compare long-run posterior means (improved-RB vs exact original sampler,
    several independent chains each) within Monte Carlo error."""
    snp, bt, fr, ix, ld, bs = tiny(2)
    n = [20000, 8000]
    orig, imp = [], []
    for s in range(4):
        d = tmp_path / ('o%d' % s); d.mkdir()
        f = mcmc_exact(1, 0.5, 1e-2, snp, bt, fr, ix, n, ld, bs, 6000, 1000, 1, ['EUR', 'EAS'], 22, str(d), 't',
                       'FALSE', 'FALSE', 100 + s, log=lambda *x: None)
        orig.append(np.concatenate([_read(x) for x in f]))
        d = tmp_path / ('i%d' % s); d.mkdir()
        solver = Solver([ld[0], ld[1]], [bs[0], bs[1]], 1, precision='double')
        f = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, 6000, 1000, 1, ['EUR', 'EAS'], 22, str(d), 't',
                          'FALSE', 'FALSE', 200 + s, coupling='snr', rb=True, log=lambda *x: None)
        imp.append(np.concatenate([_read(x) for x in f]))
    orig = np.array(orig); imp = np.array(imp)
    se = np.sqrt(orig.var(0, ddof=1) / 4 + imp.var(0, ddof=1) / 4)
    z = (orig.mean(0) - imp.mean(0)) / np.maximum(se, 1e-12)
    assert np.all(np.abs(z) < 4.5), z
    assert np.corrcoef(orig.mean(0), imp.mean(0))[0, 1] > 0.99


@needs_ref
def test_cli_output_format_and_snp_set(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--ref_dir=' + REF_DIR,
           '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD),
           '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=1', '--meta=TRUE',
           '--n_iter=200', '--n_burnin=100', '--out_dir=%s' % tmp_path, '--out_name=test', '--threads=2', '--quiet']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for f in os.listdir(os.path.join(REF_OUT, 'metapst_phi1e-2_seed1_t1')):
        ref_rows = [l.split()[:5] for l in open(os.path.join(REF_OUT, 'metapst_phi1e-2_seed1_t1', f))]
        new_rows = [l.split() for l in open(os.path.join(tmp_path, f))]
        assert [x[:5] for x in new_rows] == ref_rows
        assert all(len(x) == 6 and np.isfinite(float(x[5])) for x in new_rows)


def test_multiple_chains_average_and_rhat(tmp_path):
    """--n_chains=2 equals the average of the two single chains (seeds s and s+1000) and logs R-hat."""
    snp, bt, fr, ix, ld, bs = tiny(2)
    n = [20000, 8000]
    msgs = []
    outs = {}
    for name, nc, seed in (('two', 2, 5), ('c1', 1, 5), ('c2', 1, 1005)):
        d = tmp_path / name; d.mkdir()
        solver = Solver([ld[0], ld[1]], [bs[0], bs[1]], 1, precision='double')
        f = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, 300, 100, 5, ['EUR', 'EAS'], 22, str(d), 't',
                          'FALSE', 'FALSE', seed, n_chains=nc, log=msgs.append)
        outs[name] = np.concatenate([_read(x) for x in f])
    assert np.allclose(outs['two'], (outs['c1'] + outs['c2']) / 2, rtol=1e-5, atol=1e-12)
    assert any('R-hat' in m for m in msgs)


def test_psi_max_inf_forces_double_precision(tmp_path):
    if not have_ref:
        pytest.skip('needs PRSCSX_REF_DIR')
    cmd = [sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--ref_dir=' + REF_DIR,
           '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt' % TD, '--n_gwas=200000', '--pop=EUR',
           '--chrom=22', '--phi=1e-2', '--seed=1', '--n_iter=50', '--n_burnin=25', '--psi_max=inf',
           '--out_dir=%s' % tmp_path, '--out_name=t', '--threads=1', '--quiet']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'switching to --precision=double' in r.stdout


def test_auto_stop_equals_prefix_of_fixed_length_run(tmp_path):
    """--auto_stop: chains own their random streams, so an interleaved run stopped at iteration T writes exactly the
    output of a fixed-length run of T iterations with the same seeds; the stop is logged."""
    snp, bt, fr, ix, ld, bs = tiny(2)
    n = [20000, 8000]
    msgs = []
    d1 = tmp_path / 'auto'; d1.mkdir()
    solver = Solver([ld[0], ld[1]], [bs[0], bs[1]], 1, precision='double')
    f1 = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, 5000, 200, 5, ['EUR', 'EAS'], 22, str(d1), 't',
                       'FALSE', 'FALSE', 7, n_chains=2, auto_stop=True, log=msgs.append)
    stop = [m for m in msgs if 'auto_stop: converged' in m]
    assert stop, msgs[-5:]
    T = int(stop[0].split('iteration ')[1].split()[0])
    assert 200 + 600 <= T < 5000 and (T - 200) % 200 == 0
    d2 = tmp_path / 'fixed'; d2.mkdir()
    f2 = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, T, 200, 5, ['EUR', 'EAS'], 22, str(d2), 't',
                       'FALSE', 'FALSE', 7, n_chains=2, log=lambda *x: None)
    for a, b in zip(f1, f2):
        assert open(a).read() == open(b).read()


def test_warm_start_continues_from_previous_state(tmp_path):
    """state_io: the second phi starts from the first phi's final state (warm start); without it, outputs differ."""
    snp, bt, fr, ix, ld, bs = tiny(2)
    n = [20000, 8000]
    solver = Solver([ld[0], ld[1]], [bs[0], bs[1]], 1, precision='double')
    io = {}
    d = tmp_path / 'w'; d.mkdir()
    mcmc_improved(1, 0.5, 1.0, snp, bt, fr, ix, n, solver, 300, 100, 5, ['EUR', 'EAS'], 22, str(d), 't', 'FALSE', 'FALSE',
                  3, log=lambda *x: None, state_io=io)
    assert len(io['states']) == 1 and io['states'][0]['psi'].shape == (8,)
    first_state = io['states'][0]['psi'].copy()
    fw = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, 250, 50, 5, ['EUR', 'EAS'], 22, str(d), 'w', 'FALSE',
                       'FALSE', 3, log=lambda *x: None, state_io=io)
    fc = mcmc_improved(1, 0.5, 1e-2, snp, bt, fr, ix, n, solver, 250, 50, 5, ['EUR', 'EAS'], 22, str(d), 'c', 'FALSE',
                       'FALSE', 3, log=lambda *x: None)
    assert not np.array_equal(io['states'][0]['psi'], first_state)
    assert not np.array_equal(_read(fw[0]), _read(fc[0]))
    assert np.all(np.isfinite(_read(fw[0])))


def test_cli_auto_stop_needs_two_chains(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--ref_dir=%s' % tmp_path,
           '--bim_prefix=x', '--sst_file=a.txt', '--n_gwas=1000', '--pop=EUR', '--out_dir=%s' % tmp_path,
           '--out_name=t', '--auto_stop=TRUE']
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and '--auto_stop=TRUE needs --n_chains >= 2' in r.stdout
