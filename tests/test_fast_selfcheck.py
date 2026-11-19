"""WP1: start-up platform self-check of fast/ and phi-level parallelism (no reference panel needed)."""
import filecmp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
from fastcsx.selfcheck import self_check  # noqa: E402

FAST = os.path.join(ROOT, 'fast', 'PRScsx_fast.py')
TD = os.path.join(ROOT, 'test_data')


def test_embedded_gigrnd_is_verbatim_copy():
    ref = open(os.path.join(ROOT, 'gigrnd.py')).read()
    emb = open(os.path.join(ROOT, 'fast', 'fastcsx', 'gigrnd_reference.py')).read()
    assert emb.split('\n', 2)[2] == ref


def test_self_check_passes_fast_and_keeps_random_state():
    self_check()                                         # first call may compile
    np.random.seed(5); before = np.random.get_state()[1].copy(); np.random.seed(5)
    t0 = time.perf_counter()
    gig_ok, blk_ok, secs = self_check()
    assert gig_ok and blk_ok
    assert time.perf_counter() - t0 < 2.0
    assert np.array_equal(np.random.get_state()[1], before)


def _cli(tmp_path, extra, env=None):
    panel = tmp_path / 'panel'
    if not panel.exists():
        subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(panel)], check=True, capture_output=True)
    cmd = [sys.executable, FAST, '--ref_dir=%s' % panel, '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt' % TD,
           '--n_gwas=200000', '--pop=EUR', '--chrom=22', '--phi=1e-2', '--n_iter=20', '--n_burnin=10', '--seed=1',
           '--out_dir=%s' % tmp_path, '--out_name=t', '--quiet'] + extra
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, **(env or {})))


def test_strict_turns_mismatch_into_error(tmp_path):
    env = {'PRSCSX_FAST_SELFCHECK_SIMULATE_MISMATCH': '1'}
    r = _cli(tmp_path, ['--strict'], env)
    assert r.returncode == 3 and 'NOT bit-compatible' in r.stdout
    r = _cli(tmp_path, [], env)                          # without --strict: a warning, and the run completes
    assert r.returncode == 0 and 'WARNING' in r.stdout and 'NOT bit-compatible' in r.stdout
    r = _cli(tmp_path, ['--strict'])                     # real check on this platform passes
    assert r.returncode == 0 and 'bit-compatible with the original' in r.stdout


def test_phi_grid_parallel_equals_separate_runs(tmp_path):
    panel = tmp_path / 'panel'
    subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(panel)], check=True, capture_output=True)
    base = ['--ref_dir=%s' % panel, '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD),
            '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--seed=11', '--n_iter=200', '--n_burnin=100',
            '--threads=1', '--quiet', '--out_name=t']
    runs = {'grid': ['--phi=1e-4,1e-2', '--n_jobs=2'], 's1': ['--phi=1e-4'], 's2': ['--phi=1e-2']}
    for k, extra in runs.items():
        d = tmp_path / k; d.mkdir()
        r = subprocess.run([sys.executable, FAST] + base + extra + ['--out_dir=%s' % d], capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:]
    sep = sorted(os.listdir(tmp_path / 's1')) + sorted(os.listdir(tmp_path / 's2'))
    assert len([f for f in sep if '_pst_eff_' in f]) == 4
    for k in ('s1', 's2'):
        for f in os.listdir(tmp_path / k):
            if '_pst_eff_' in f:
                assert filecmp.cmp(str(tmp_path / k / f), str(tmp_path / 'grid' / f), shallow=False), f
