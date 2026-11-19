"""
Byte-identity of fast/ against the root PRScsx.py on a synthetic reference panel (no 9 GB download needed).

The panel (tests/make_synth_ref.py) covers: an empty block, a block of panel-only SNPs, a block larger than the
reference sample (rank-deficient LD), 10% allele flips, A1/A2 swaps against the .bim, and SNPs missing in EAS/AFR.
Configurations (instructions of 18 Nov 2026, WP1):
  (a) EUR+EAS, --phi=1e-2 --meta=TRUE, 400/200 iterations, fast with --threads 1 and 4
  (b) EUR+EAS, auto phi, --meta=TRUE --write_pst=TRUE, 300/150
  (c) EUR+EAS+AFR (AFR uses the EAS test sumstats), --phi=1e-4, 300/150
  (d) EUR only, --phi=1 --n_iter=303 --n_burnin=101 --thin=7   (burn-in not a multiple of the thinning)
"""
import filecmp
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
FIXED = os.path.join(ROOT, 'PRScsx.py')
FAST = os.path.join(ROOT, 'fast', 'PRScsx_fast.py')
TD = os.path.join(ROOT, 'test_data')
ENV = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

TWO = ['--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD), '--n_gwas=200000,100000', '--pop=EUR,EAS']
THREE = ['--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD, TD),
         '--n_gwas=200000,100000,50000', '--pop=EUR,EAS,AFR']
ONE = ['--sst_file=%s/EUR_sumstats.txt' % TD, '--n_gwas=200000', '--pop=EUR']
CONFIGS = {
    'a_two_meta': TWO + ['--phi=1e-2', '--meta=TRUE', '--n_iter=400', '--n_burnin=200'],
    'b_auto_writepst': TWO + ['--meta=TRUE', '--write_pst=TRUE', '--n_iter=300', '--n_burnin=150'],
    'c_three': THREE + ['--phi=1e-4', '--n_iter=300', '--n_burnin=150'],
    'd_one_thin7': ONE + ['--phi=1', '--n_iter=303', '--n_burnin=101', '--thin=7'],
}


@pytest.fixture(scope='module')
def panel(tmp_path_factory):
    d = tmp_path_factory.mktemp('synth_ref')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(d), '--seed', '1'],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return str(d)


def _run(script, panel, out, extra):
    cmd = [sys.executable, script, '--ref_dir=' + panel, '--bim_prefix=%s/test' % TD, '--chrom=22', '--seed=7',
           '--out_name=t', '--out_dir=' + str(out)] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=ENV)
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    return sorted(f for f in os.listdir(out) if '_pst_eff_' in f)


@pytest.mark.parametrize('name', sorted(CONFIGS))
def test_fast_byte_identical_on_synthetic_panel(panel, tmp_path, name):
    ref = tmp_path / 'fixed'; ref.mkdir()
    ref_files = _run(FIXED, panel, ref, CONFIGS[name])
    assert ref_files
    for threads in ((1, 4) if name.startswith('a_') else (2,)):
        out = tmp_path / ('fast_t%d' % threads); out.mkdir()
        files = _run(FAST, panel, out, CONFIGS[name] + ['--threads=%d' % threads])
        assert files == ref_files
        for f in ref_files:
            assert filecmp.cmp(str(ref / f), str(out / f), shallow=False), (name, threads, f)
