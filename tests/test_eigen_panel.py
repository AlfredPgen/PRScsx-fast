"""WP3: eigen LD panels (tools/build_ld_panel.py; docs/04_eigen_panel_format.md) and reading them in fast/improved."""
import json
import os
import subprocess
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import build_ld_panel as blp  # noqa: E402

TD = os.path.join(ROOT, 'test_data')


def test_bed_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, (37, 11)).astype(float)
    G[3, 4] = np.nan; G[36, 10] = np.nan
    blp.write_bed(str(tmp_path / 'g'), G, ['s%d' % j for j in range(11)])
    bed = blp.Bed(str(tmp_path / 'g'))
    out = bed.read(list(range(11)))
    assert np.array_equal(np.isnan(out), np.isnan(G)) and np.array_equal(np.nan_to_num(out, nan=-1), np.nan_to_num(G, nan=-1))
    assert np.array_equal(bed.read([4, 2], people=np.array([0, 3, 36])), G[[0, 3, 36]][:, [4, 2]], equal_nan=True)


def test_build_merge_and_flips(tmp_path):
    rng = np.random.default_rng(1)
    n, m = 150, 60
    z = rng.standard_normal((2 * n, m)).cumsum(1) / 3
    G = (z[:n] > 0).astype(float) + (z[n:] > 0).astype(float)
    snps = ['rs%d' % j for j in range(m)]
    a1 = ['A'] * m; a2 = ['G'] * m
    blp.write_bed(str(tmp_path / 'eur'), G, snps, chrom=22, a1=a1, a2=a2)
    # second ancestry: same genotypes, alleles swapped for odd SNPs (counts mirrored), strand complement for SNP 2
    G2 = G.copy(); a1b = list(a1); a2b = list(a2)
    for j in range(1, m, 2):
        G2[:, j] = 2 - G2[:, j]; a1b[j], a2b[j] = a2b[j], a1b[j]
    a1b[2], a2b[2] = 'T', 'C'
    blp.write_bed(str(tmp_path / 'eas'), G2, snps, chrom=22, a1=a1b, a2=a2b)
    for pop in ('EUR', 'EAS'):
        blp.main(['build', '--bfile', str(tmp_path / pop.lower()), '--pop', pop, '--chrom', '22', '--out', str(tmp_path / 'panel'),
                  '--block_snps', '25', '--rank_var', '1', '--maf', '0.0'])
    blp.main(['merge', '--out', str(tmp_path / 'panel')])
    info = [l.split() for l in open(tmp_path / 'panel' / 'snpinfo_mult_eigen')]
    hdr = info[0]; rows = {r[1]: r for r in info[1:]}
    assert len(rows) == m
    fe, fa = hdr.index('FRQ_EUR'), hdr.index('FRQ_EAS'); le, la = hdr.index('FLP_EUR'), hdr.index('FLP_EAS')
    for j in range(m):          # reference alleles come from the first ancestry in AFR, AMR, EAS, EUR, SAS order (EAS)
        r = rows[snps[j]]
        assert abs(float(r[fe]) - float(r[fa])) < 1e-6                       # both = frequency of the SNP-info A1
        assert int(r[le]) * int(r[la]) == (-1 if j % 2 == 1 else 1)           # relative flip = allele swap (SNP 2: strand only)
        assert r[la] == '1'
    # LD reproduced (rank_var 1) in each ancestry's own allele coding
    with h5py.File(tmp_path / 'panel' / 'ldeig_eas' / 'ldeig_chr22.hdf5', 'r') as h:
        g = h['blk_1']; idx = [snps.index(s.decode()) for s in g['snplist'][:]]
        X = G2[:, idx]; X = (X - X.mean(0)) / X.std(0)
        V = np.asarray(g['evec'][:], float); lam = g['eval'][:]
        assert np.abs((V * lam) @ V.T - X.T @ X / n).max() < 1e-5


def test_convert_reproduces_psd_projected_ld(tmp_path):
    from scipy import linalg
    subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(tmp_path / 'syn')], check=True, capture_output=True)
    blp.main(['convert', '--ref_dir', str(tmp_path / 'syn'), '--pops', 'EUR,EAS', '--chroms', '22', '--out', str(tmp_path / 'eig')])
    with h5py.File(tmp_path / 'syn' / 'ldblk_1kg_eur' / 'ldblk_1kg_chr22.hdf5', 'r') as hs, \
            h5py.File(tmp_path / 'eig' / 'ldeig_eur' / 'ldeig_chr22.hdf5', 'r') as he:
        for k in range(1, len(hs) + 1):
            R = np.array(hs['blk_%d' % k]['ldblk'])
            if R.size == 0:
                assert he['blk_%d' % k]['eval'].shape == (0,)
                continue
            _, s, v = linalg.svd(R); D = (R + v.T @ np.diag(s) @ v) / 2
            V = np.asarray(he['blk_%d' % k]['evec'][:], float); lam = he['blk_%d' % k]['eval'][:]
            assert np.abs((V * lam) @ V.T - D).max() < 1e-5


def test_fit_from_eigen_panel_matches_dense_panel(tmp_path):
    """Small version of benchmarks/wp3_eigen_panel_check.py (T2c criterion over 3 seeds)."""
    out = tmp_path / 'res.json'
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'benchmarks', 'wp3_eigen_panel_check.py'), str(out), str(tmp_path / 'w'),
                        '--sizes', '100,300,800', '--n', '200', '--seeds', '3', '--n_iter', '300', '--threads', '2'],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.load(open(out))
    assert res['ld_reproduced_to_1e5'] and res['posterior_criterion_passed'], res


def test_cli_with_eigen_panel(tmp_path):
    """fast switches --linalg=full -> auto with a notice; improved refuses --linalg=full; both run on the eigen panel."""
    subprocess.run([sys.executable, os.path.join(HERE, 'make_synth_ref.py'), str(tmp_path / 'syn')], check=True, capture_output=True)
    blp.main(['convert', '--ref_dir', str(tmp_path / 'syn'), '--pops', 'EUR,EAS', '--chroms', '22', '--out', str(tmp_path / 'eig')])
    base = ['--ref_dir=%s' % (tmp_path / 'eig'), '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD),
            '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=1', '--n_iter=100', '--n_burnin=50',
            '--out_name=t', '--out_dir=%s' % tmp_path]
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'fast', 'PRScsx_fast.py')] + base, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and 'using --linalg=auto' in r.stdout and 'not byte-identical' in r.stdout, r.stdout[-2000:]
    assert os.path.exists(tmp_path / 't_EAS_pst_eff_a1_b0.5_phi1e-02_chr22.txt')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py')] + base + ['--linalg=full'],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 2 and 'not available with an eigen LD panel' in r.stdout
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py')] + base, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:]
