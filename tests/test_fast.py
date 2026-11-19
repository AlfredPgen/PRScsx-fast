"""
Tests for fast/ (PRS-CSx-fast): exactness against the original PRS-CSx.

Unit tests need no data. Tests marked `needs_ref` need the 1000G reference (EUR, EAS) via PRSCSX_REF_DIR and
compare with outputs of the ORIGINAL program stored in tests/reference_outputs/.
Run:  PRSCSX_REF_DIR=/path/to/ref python -m pytest -v tests/test_fast.py
"""
import filecmp
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
FAST = os.path.join(ROOT, 'fast')
FIXED = os.path.join(ROOT)
REF_DIR = os.environ.get('PRSCSX_REF_DIR', '')
REF_OUT = os.path.join(HERE, 'reference_outputs')
TD = os.path.join(FIXED, 'test_data')
sys.path.insert(0, FAST)
sys.path.insert(0, FIXED)

from fastcsx import parse as fparse  # noqa: E402
from fastcsx.gig import sample_psi  # noqa: E402
from fastcsx.lapack import block_update, new_workspace  # noqa: E402
from fastcsx.mcmc import seqsum  # noqa: E402
import gigrnd  # noqa: E402  (fixed original)

have_ref = all(os.path.exists(os.path.join(REF_DIR, x)) for x in ('snpinfo_mult_1kg_hm3', 'ldblk_1kg_eur', 'ldblk_1kg_eas'))
needs_ref = pytest.mark.skipif(not have_ref, reason='set PRSCSX_REF_DIR to the 1000G reference directory')
ENV1 = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


# ------------------------------------------------------------------------------------------------ unit: exact kernels
def test_gig_bit_identical_and_same_stream_position():
    rng = np.random.default_rng(7)
    n = 20000
    P = 1.0 - 0.5 * rng.integers(0, 6, n)
    A = 2.0 * np.exp(rng.normal(0, 4, n))
    B = np.exp(rng.normal(-2, 6, n))
    np.random.seed(99)
    ref = np.empty(n)
    for j in range(n):
        while True:
            try:
                ref[j] = gigrnd.gigrnd(P[j], A[j], B[j])
            except (ArithmeticError, ValueError):
                continue
            break
    tail_ref = np.random.random_sample(3)
    np.random.seed(99)
    out = sample_psi(P, A, B)
    tail = np.random.random_sample(3)
    assert np.array_equal(ref, out)
    assert np.array_equal(tail_ref, tail)


def test_gig_buffer_extension_path():
    """Force the 'need more uniforms' path (tiny first buffer is impossible to force from outside, so use many SNPs
    with low acceptance) and check exactness still holds."""
    rng = np.random.default_rng(3)
    n = 3000
    P = np.full(n, -2.0); A = np.exp(rng.normal(-8, 2, n)); B = np.exp(rng.normal(8, 2, n))
    np.random.seed(5)
    ref = np.array([gigrnd.gigrnd(P[j], A[j], B[j]) for j in range(n)])
    np.random.seed(5)
    assert np.array_equal(ref, sample_psi(P, A, B))


def test_block_kernel_bit_identical_to_scipy():
    from scipy import linalg
    rng = np.random.default_rng(1)
    ws = new_workspace(400)
    for b in (1, 2, 17, 150, 400):
        X = rng.standard_normal((300, b))
        ld = np.atleast_2d(np.corrcoef(X, rowvar=False))
        _, s, v = linalg.svd(ld); ld = (ld + np.dot(v.T, np.dot(np.diag(s), v))) / 2
        psi = np.minimum(np.exp(rng.normal(-3, 2, (b, 1))), 1.0)
        bhat = rng.normal(0, 0.01, (b, 1)); z = rng.standard_normal((b, 1)); sig = np.array([[0.98]]); n = 100000
        dinvt = ld + np.diag(1.0 / psi.T[0]); U = linalg.cholesky(dinvt)
        ref = linalg.solve_triangular(U, linalg.solve_triangular(U, bhat, trans='T') + np.sqrt(sig / n) * z, trans='N')
        qref = np.dot(np.dot(ref.T, dinvt), ref)
        D = ws['dinvt'][:b * b].reshape(b, b); F = ws['F'][:b * b].reshape(b, b); out = np.empty((b, 1))
        info = block_update(ld, 1.0 / psi[:, 0], bhat[:, 0], z[:, 0], float(np.sqrt(sig / n)[0, 0]), D, F, out[:, 0],
                            ws['chars'], ws['ints'])
        assert info == 0
        assert np.array_equal(out, ref)
        assert np.array_equal(np.dot(np.dot(out.T, D), out), qref)


def test_seqsum_equals_builtin_sum():
    rng = np.random.default_rng(2)
    for n in (1, 2, 10, 1001, 50000):
        x = (rng.standard_normal((n, 1)) * np.exp(rng.normal(0, 10, (n, 1))))
        assert seqsum(x) == float(sum(x)[0])


# ------------------------------------------------------------------------------------------------ parsing equivalence
def _write_sst(path, rows, header):
    with open(path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(r) + '\n')


@needs_ref
@pytest.mark.parametrize('fmt', ['BETA_P', 'BETA_SE', 'OR_P', 'OR_SE'])
def test_parse_sumstats_identical_to_original(tmp_path, fmt):
    import parse_genet
    rows = [l.split() for l in open(os.path.join(TD, 'EUR_sumstats.txt')).read().splitlines()[1:]]
    rng = np.random.default_rng(0)
    out = []
    for r in rows:
        beta = float(r[3]); p = float(r[4])
        eff = '%.6f' % np.exp(beta) if fmt.startswith('OR') else r[3]
        last = r[4] if fmt.endswith('_P') else '%.6f' % max(1e-4, abs(beta) / rng.uniform(1, 4))
        out.append([r[0], r[1], r[2], eff, last])
    # add strand flips / allele swaps / duplicates / non-ACGT lines
    out[5][1], out[5][2] = out[5][2], out[5][1]
    m = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    out[7][1], out[7][2] = m[out[7][1]], m[out[7][2]]
    out.append(list(out[10])); out[-1][3] = out[-1][3][:-1] + '7'
    out.append(['rsBAD', 'AT', 'G', '0.1', '0.5'])
    hdr = ['SNP', 'A1', 'A2', fmt.split('_')[0], fmt.split('_')[1]]
    sst = str(tmp_path / 'sst.txt'); _write_sst(sst, out, hdr)
    ref_dict = parse_genet.parse_ref(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), 22, '1kg')
    vld = parse_genet.parse_bim(os.path.join(TD, 'test'), 22)
    a = parse_genet.parse_sumstats(ref_dict, vld, sst, 'EUR', 200000)
    fref = fparse.read_ref_all(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), [22])[22]
    b = fparse.parse_sumstats(fref, fparse.parse_bim(os.path.join(TD, 'test'), 22),
                              fparse.SumStats(sst, set(fref['SNP'])), 'EUR', 200000, log=lambda *x: None)
    assert a['SNP'] == b['SNP'] and a['FLP'] == b['FLP'] and a['FRQ'] == b['FRQ']
    assert np.array_equal(np.array(a['BETA'], dtype=float), np.array(b['BETA'], dtype=float))


@needs_ref
def test_se_zero_is_an_error_like_the_original(tmp_path):
    """Original: ZeroDivisionError. Fast: ValueError with the offending SNP (never silently inf)."""
    import parse_genet
    rows = [l.split() for l in open(os.path.join(TD, 'EUR_sumstats.txt')).read().splitlines()[1:]]
    out = [[r[0], r[1], r[2], r[3], '0.01'] for r in rows]
    out[3][4] = '0'
    sst = str(tmp_path / 'sst.txt'); _write_sst(sst, out, ['SNP', 'A1', 'A2', 'BETA', 'SE'])
    ref_dict = parse_genet.parse_ref(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), 22, '1kg')
    vld = parse_genet.parse_bim(os.path.join(TD, 'test'), 22)
    with pytest.raises(ZeroDivisionError):
        parse_genet.parse_sumstats(ref_dict, vld, sst, 'EUR', 200000)
    fref = fparse.read_ref_all(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), [22])[22]
    with pytest.raises(ValueError, match=out[3][0]):
        fparse.parse_sumstats(fref, vld, fparse.SumStats(sst, set(fref['SNP'])), 'EUR', 200000, log=lambda *x: None)


@needs_ref
def test_parse_ld_and_align_identical_to_original():
    import parse_genet
    ref_dict = parse_genet.parse_ref(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), 22, '1kg')
    vld = parse_genet.parse_bim(os.path.join(TD, 'test'), 22)
    fref = fparse.read_ref_all(os.path.join(REF_DIR, 'snpinfo_mult_1kg_hm3'), [22])[22]
    sa, sb = {}, {}
    for pp, pop in enumerate(['EUR', 'EAS']):
        f = os.path.join(TD, '%s_sumstats.txt' % pop)
        sa[pp] = parse_genet.parse_sumstats(ref_dict, vld, f, pop, 100000)
        sb[pp] = fparse.parse_sumstats(fref, vld, fparse.SumStats(f, set(fref['SNP'])), pop, 100000, log=lambda *x: None)
        la, za = parse_genet.parse_ldblk(REF_DIR, sa[pp], pop, 22, '1kg')
        lb, zb = fparse.parse_ldblk(REF_DIR, sb[pp], pop, 22, '1kg', log=lambda *x: None)
        assert za == zb
        for x, y in zip(la, lb):
            assert np.array_equal(x, y)
    A = parse_genet.align_ldblk(ref_dict, vld, sa, 2, 22)
    B = fparse.align_ldblk(fref, vld, sb, 2, 22, log=lambda *x: None)
    assert A[0] == B[0] and A[3] == B[3]
    for pp in range(2):
        assert np.array_equal(A[1][pp], B[1][pp]) and np.array_equal(A[2][pp], B[2][pp])


# ------------------------------------------------------------------------------------------------ end to end
def _run_fast(tmp_path, extra, threads):
    cmd = [sys.executable, os.path.join(FAST, 'PRScsx_fast.py'), '--ref_dir=' + REF_DIR, '--bim_prefix=%s/test' % TD,
           '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD), '--n_gwas=200000,100000', '--pop=EUR,EAS',
           '--chrom=22', '--out_dir=%s' % tmp_path, '--out_name=test', '--threads=%d' % threads, '--quiet'] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=ENV1)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


E2E = {'phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1'],
       'metapst_phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1', '--meta=TRUE', '--write_pst=TRUE'],
       'phiauto_meta_seed1_t1': ['--seed=1', '--meta=TRUE']}


@needs_ref
@pytest.mark.parametrize('threads', [1, 4])
@pytest.mark.parametrize('run', sorted(E2E))
def test_fast_byte_identical_to_original(tmp_path, run, threads):
    _run_fast(tmp_path, E2E[run], threads)
    for f in sorted(os.listdir(os.path.join(REF_OUT, run))):
        assert filecmp.cmp(os.path.join(REF_OUT, run, f), os.path.join(tmp_path, f), shallow=False), f


@needs_ref
def test_phi_grid_in_one_run_equals_separate_runs(tmp_path):
    grid = tmp_path / 'grid'; grid.mkdir()
    _run_fast(grid, ['--phi=1e-2,auto', '--seed=1', '--meta=TRUE'], 2)
    for f in os.listdir(os.path.join(REF_OUT, 'phiauto_meta_seed1_t1')):
        assert filecmp.cmp(os.path.join(REF_OUT, 'phiauto_meta_seed1_t1', f), str(grid / f), shallow=False), f
    for f in os.listdir(os.path.join(REF_OUT, 'phi1e-2_seed1_t1')):
        assert filecmp.cmp(os.path.join(REF_OUT, 'phi1e-2_seed1_t1', f), str(grid / f), shallow=False), f


@needs_ref
def test_multi_chromosome_n_jobs(tmp_path):
    """Chromosomes without data are skipped cleanly; --n_jobs=2 gives the same chr22 output."""
    _run_fast(tmp_path, ['--phi=1e-2', '--seed=1', '--chrom=21,22', '--n_jobs=2'], 1)
    for f in os.listdir(os.path.join(REF_OUT, 'phi1e-2_seed1_t1')):
        assert filecmp.cmp(os.path.join(REF_OUT, 'phi1e-2_seed1_t1', f), str(tmp_path / f), shallow=False), f


def test_psd_projection_falls_back_when_gesdd_does_not_converge():
    """A simulated LD block (943 SNPs, rank 499) on which LAPACK gesdd fails ("SVD did not converge"; the original
    PRS-CSx stops). fast falls back to gesvd and returns a valid symmetric PSD matrix."""
    from scipy import linalg
    R = np.load(os.path.join(HERE, 'data', 'gesdd_nonconvergent_ld_block.npz'))['R']
    try:
        linalg.svd(R)
        gesdd_failed = False
    except np.linalg.LinAlgError:
        gesdd_failed = True
    msgs = []
    P = fparse._psd(R, log=msgs.append)
    assert np.isfinite(P).all()
    assert np.allclose(P, P.T, atol=1e-12)
    assert np.linalg.eigvalsh((P + P.T) / 2).min() > -1e-8
    assert np.allclose(P, R, atol=1e-6)                 # R is (numerically) PSD already: projection ~ identity
    assert bool(msgs) == gesdd_failed
