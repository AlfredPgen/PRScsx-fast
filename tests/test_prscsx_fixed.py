"""
Regression tests for the NumPy-2-compatible PRS-CSx (the root PRScsx.py).

Fast unit tests run on a tiny synthetic problem (no reference panel needed).
End-to-end tests compare against outputs of the ORIGINAL, unmodified PRS-CSx (produced with NumPy 1.26,
stored in tests/reference_outputs/) and need the 1000G EUR/EAS LD panels; they are skipped unless
PRSCSX_REF_DIR points to a directory with snpinfo_mult_1kg_hm3, ldblk_1kg_eur/, ldblk_1kg_eas/.

Run:  PRSCSX_REF_DIR=/path/to/ref python -m pytest -v tests/test_prscsx_fixed.py
"""
import filecmp
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED = os.environ.get('PRSCSX_FIXED_DIR', os.path.join(HERE, '..'))
REF_DIR = os.environ.get('PRSCSX_REF_DIR', '')
REF_OUT = os.path.join(HERE, 'reference_outputs')
sys.path.insert(0, FIXED)

import gigrnd  # noqa: E402
import mcmc_gtb  # noqa: E402
import parse_genet  # noqa: E402

have_ref = os.path.isfile(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3')) and \
    os.path.isdir(os.path.join(REF_DIR, 'ldblk_1kg_eur')) and os.path.isdir(os.path.join(REF_DIR, 'ldblk_1kg_eas'))
needs_ref = pytest.mark.skipif(not have_ref, reason='set PRSCSX_REF_DIR to the 1000G LD reference directory')


# ---------------------------------------------------------------- tiny synthetic problem
def tiny_problem(n_pop=2, p=6, seed=0):
    rng = np.random.default_rng(seed)
    snp_dict = {'CHR': [22] * p, 'SNP': ['rs%d' % i for i in range(p)], 'BP': list(range(100, 100 + p)),
                'A1': ['A'] * p, 'A2': ['G'] * p}
    beta, frq, idx, ld, bs = {}, {}, {}, {}, {}
    for k in range(n_pop):
        X = rng.standard_normal((200, p)); X[:, 1] += X[:, 0]
        R = np.corrcoef(X, rowvar=False)
        beta[k] = (R @ np.r_[0.05, 0, 0, -0.03, 0, 0][:p] + rng.standard_normal(p) * 0.01).reshape(p, 1)
        frq[k] = np.full((p, 1), 0.3)
        idx[k] = list(range(p))
        ld[k] = [R[:3, :3], R[3:, 3:]]
        bs[k] = [3, p - 3]
    return snp_dict, beta, frq, idx, ld, bs


def run_tiny(tmp_path, n_iter=40, n_burnin=10, thin=5, meta='FALSE', write_pst='FALSE', phi=1e-2, seed=1):
    snp_dict, beta, frq, idx, ld, bs = tiny_problem()
    mcmc_gtb.mcmc(1, 0.5, phi, snp_dict, beta, frq, idx, [10000, 5000], ld, bs, n_iter, n_burnin, thin,
                  ['EUR', 'EAS'], 22, str(tmp_path), 'tiny', meta, write_pst, seed)
    return sorted(os.listdir(tmp_path))


# ---------------------------------------------------------------- unit tests
def test_gigrnd_accepts_numpy_scalars_and_is_positive():
    np.random.seed(3)
    x = np.array([[0.5], [2.0]])
    for _ in range(200):
        v = gigrnd.gigrnd(1 - 0.5 * x[0, 0], 2.0 * x[1, 0], 0.7)
        assert np.isfinite(v) and v > 0


def test_numpy2_root_cause_is_shape1_float_conversion():
    """Documents the root cause: under NumPy>=2, float() of a shape-(1,) array raises TypeError,
    so the ORIGINAL call gigrnd(a-0.5*n_grp[jj], ...) failed on every call."""
    arr = np.ones((3, 1))
    if int(np.__version__.split('.')[0]) >= 2:
        with pytest.raises(TypeError):
            gigrnd.gigrnd(arr[0], arr[1], arr[2])
    assert gigrnd.gigrnd(arr[0, 0], arr[1, 0], arr[2, 0]) > 0   # the fixed call pattern always works


def test_mcmc_runs_and_writes_outputs(tmp_path):
    files = run_tiny(tmp_path, meta='TRUE')
    assert files == ['tiny_EAS_pst_eff_a1_b0.5_phi1e-02_chr22.txt', 'tiny_EUR_pst_eff_a1_b0.5_phi1e-02_chr22.txt',
                     'tiny_META_pst_eff_a1_b0.5_phi1e-02_chr22.txt']
    for f in files:
        vals = np.loadtxt(os.path.join(tmp_path, f), usecols=5)
        assert vals.shape == (6,) and np.all(np.isfinite(vals))


def test_mcmc_phi_auto_prints_scalar_phi(tmp_path, capsys):
    run_tiny(tmp_path, phi=None, meta='TRUE')
    out = capsys.readouterr().out
    assert 'Estimated global shrinkage parameter' in out


def test_non_numerical_error_propagates_instead_of_hanging(tmp_path, monkeypatch):
    """The original bare `except: continue` retried any exception forever (this is what hung on NumPy 2)."""
    def boom(*a):
        raise TypeError('programming error')
    monkeypatch.setattr(mcmc_gtb.gigrnd, 'gigrnd', boom)
    with pytest.raises(TypeError):
        run_tiny(tmp_path)


def test_transient_numerical_error_is_retried(tmp_path, monkeypatch):
    real = gigrnd.gigrnd
    state = {'n': 0}

    def flaky(*a):
        state['n'] += 1
        if state['n'] % 7 == 0:
            raise ZeroDivisionError('transient')
        return real(*a)
    monkeypatch.setattr(mcmc_gtb.gigrnd, 'gigrnd', flaky)
    run_tiny(tmp_path)
    assert state['n'] > 0


def test_deterministic_numerical_error_gives_up_with_message(tmp_path, monkeypatch):
    def always(*a):
        raise OverflowError('deterministic')
    monkeypatch.setattr(mcmc_gtb.gigrnd, 'gigrnd', always)
    with pytest.raises(RuntimeError, match='GIG sampling failed'):
        run_tiny(tmp_path)


@pytest.mark.parametrize('n_iter,n_burnin,thin', [(60, 12, 5), (61, 13, 4), (50, 0, 5), (40, 10, 5)])
def test_number_of_written_draws_matches_retained_draws(tmp_path, n_iter, n_burnin, thin):
    """Original: n_pst=int((n_iter-n_burnin)/thin) undercounts when n_burnin % thin != 0 -> IndexError."""
    run_tiny(tmp_path, n_iter=n_iter, n_burnin=n_burnin, thin=thin, meta='TRUE', write_pst='TRUE')
    expected = len([i for i in range(1, n_iter + 1) if i > n_burnin and i % thin == 0])
    first = open(os.path.join(tmp_path, 'tiny_META_pst_eff_a1_b0.5_phi1e-02_chr22.txt')).readline().split()
    assert len(first) - 5 == expected


def test_parse_bim_tolerates_nonnumeric_chromosomes(tmp_path):
    bim = tmp_path / 'x.bim'
    bim.write_text('22\trs1\t0\t1\tA\tG\nX\trs2\t0\t2\tA\tG\nchr22\trs3\t0\t3\tC\tT\nMT\trs4\t0\t4\tA\tC\n21\trs5\t0\t5\tA\tG\n')
    d = parse_genet.parse_bim(str(tmp_path / 'x'), 22)
    assert d['SNP'] == ['rs1', 'rs3'] and d['A1'] == ['A', 'C'] and d['A2'] == ['G', 'T']


def _cli(args, cwd):
    return subprocess.run([sys.executable, os.path.join(FIXED, 'PRScsx.py')] + args, cwd=cwd,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize('bad,msg', [(['--n_iter=100', '--n_burnin=200'], 'must be larger than'),
                                     (['--thin=0'], 'positive integer'),
                                     (['--n_iter=9', '--n_burnin=6', '--thin=5'], 'No posterior draws')])
def test_cli_rejects_invalid_mcmc_lengths(tmp_path, bad, msg):
    r = _cli(['--ref_dir=%s' % tmp_path, '--bim_prefix=x', '--sst_file=s', '--n_gwas=1', '--pop=EUR',
              '--out_dir=%s' % tmp_path, '--out_name=t'] + bad, tmp_path)
    assert r.returncode == 2 and msg in r.stdout


def test_cli_missing_reference_gives_clear_error(tmp_path):
    td = os.path.join(FIXED, 'test_data')
    r = _cli(['--ref_dir=%s' % tmp_path, '--bim_prefix=%s/test' % td, '--sst_file=%s/EUR_sumstats.txt' % td,
              '--n_gwas=200000', '--pop=EUR', '--chrom=22', '--out_dir=%s' % tmp_path, '--out_name=t'], tmp_path)
    assert r.returncode == 2 and 'SNP information file not found' in r.stdout


# ---------------------------------------------------------------- end-to-end vs ORIGINAL outputs
E2E = {'phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1'],
       'metapst_phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1', '--meta=TRUE', '--write_pst=TRUE'],
       'phiauto_meta_seed1_t1': ['--seed=1', '--meta=TRUE']}


@needs_ref
@pytest.mark.parametrize('run', sorted(E2E))
def test_example_byte_identical_to_original(tmp_path, run):
    td = os.path.join(FIXED, 'test_data')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable, os.path.join(FIXED, 'PRScsx.py'), '--ref_dir=' + REF_DIR,
                        '--bim_prefix=%s/test' % td, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (td, td),
                        '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--out_dir=%s' % tmp_path,
                        '--out_name=test'] + E2E[run], capture_output=True, text=True, timeout=1800, env=env)
    assert r.returncode == 0, r.stdout[-2000:]
    ref_files = sorted(os.listdir(os.path.join(REF_OUT, run)))
    assert sorted(f for f in os.listdir(tmp_path) if '_pst_eff_' in f) == ref_files
    for f in ref_files:
        assert filecmp.cmp(os.path.join(REF_OUT, run, f), os.path.join(tmp_path, f), shallow=False), f
