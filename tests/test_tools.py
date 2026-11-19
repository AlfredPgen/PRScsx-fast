"""Tests for tools/combine_prs.py (synthetic PLINK2 .sscore and PLINK1.9 .profile files)."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'combine_prs.py')


def _make(tmp, plink1=False):
    rng = np.random.default_rng(0)
    n = 4000
    ids = [('F%d' % i, 'I%d' % i) for i in range(n)]
    g1 = rng.standard_normal(n); g2 = 0.6 * g1 + 0.8 * rng.standard_normal(n)
    cov = rng.standard_normal(n)
    y = 0.4 * g1 + 0.3 * g2 + 0.5 * cov + rng.standard_normal(n)
    (tmp / 'pheno.txt').write_text('FID IID Y\n' + ''.join('%s %s %.5f\n' % (f, i, v) for (f, i), v in zip(ids, y)))
    (tmp / 'covar.txt').write_text('FID IID C1\n' + ''.join('%s %s %.5f\n' % (f, i, v) for (f, i), v in zip(ids, cov)))
    lv, lt = ['phi pop file'], ['phi pop file']
    for phi, noise in [('1e-4', 0.5), ('1e-2', 2.0), ('auto', 1.0)]:
        for pop, g in [('EUR', g1), ('AFR', g2)]:
            s = g + noise * rng.standard_normal(n)
            for part, sl, L in [('val', range(0, 2000), lv), ('test', range(2000, n), lt)]:
                fn = tmp / ('%s_%s_%s.%s' % (part, pop, phi, 'profile' if plink1 else 'sscore'))
                if plink1:
                    fn.write_text('FID IID PHENO CNT CNT2 SCORE\n' + ''.join('%s %s -9 10 5 %.6f\n' % (ids[k] + (s[k],)) for k in sl))
                else:
                    fn.write_text('#FID\tIID\tALLELE_CT\tNAMED_ALLELE_DOSAGE_SUM\tSCORE1_AVG\n' +
                                  ''.join('%s\t%s\t100\t50\t%.6f\n' % (ids[k] + (s[k],)) for k in sl))
                L.append('%s %s %s' % (phi, pop, fn))
    (tmp / 'val.tsv').write_text('\n'.join(lv) + '\n'); (tmp / 'test.tsv').write_text('\n'.join(lt) + '\n')


def _run(tmp):
    r = subprocess.run([sys.executable, TOOL, '--scores_val', str(tmp / 'val.tsv'), '--scores_test', str(tmp / 'test.tsv'),
                        '--pheno', str(tmp / 'pheno.txt'), '--covar', str(tmp / 'covar.txt'), '--out', str(tmp / 'res')],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.load(open(tmp / 'res.json'))


def test_selects_least_noisy_phi_and_reports_test_r2(tmp_path):
    _make(tmp_path)
    res = _run(tmp_path)
    assert res['selected_phi'] == '1e-4'
    assert 0.15 < res['test']['r2_test_increment'] < 0.30          # true increment ~0.21 for the noisy scores
    w = res['phi']['1e-4']['weights_on_standardised_prs']
    assert len(w) == 2 and all(x > 0 for x in w)


def test_plink19_profile_format(tmp_path):
    _make(tmp_path, plink1=True)
    assert _run(tmp_path)['selected_phi'] == '1e-4'


def test_stack_option(tmp_path):
    _make(tmp_path)
    r = subprocess.run([sys.executable, TOOL, '--scores_val', str(tmp_path / 'val.tsv'), '--scores_test', str(tmp_path / 'test.tsv'),
                        '--pheno', str(tmp_path / 'pheno.txt'), '--covar', str(tmp_path / 'covar.txt'), '--out', str(tmp_path / 'res'),
                        '--stack'], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    res = json.load(open(tmp_path / 'res.json'))
    st = res['stack']
    assert len(st['columns']) == 6 and all(w >= 0 for w in st['weights_on_standardised_prs'])
    # the stack can only fit the validation data at least as well as the best single phi (it nests it, up to sign)
    assert st['r2_val_increment'] >= res['phi'][res['selected_phi']]['r2_val_increment'] - 1e-9
    assert 0.15 < st['test']['r2_test_increment'] < 0.30


COMPARE = os.path.join(HERE, '..', 'tools', 'compare_prs.py')


def test_write_test_scores_and_compare(tmp_path):
    """combine_prs --write_test_scores gives per-person test scores; compare_prs reproduces the test R2 increment
    and gives a paired bootstrap interval for the difference between two methods."""
    _make(tmp_path)
    r = subprocess.run([sys.executable, TOOL, '--scores_val', str(tmp_path / 'val.tsv'), '--scores_test', str(tmp_path / 'test.tsv'),
                        '--pheno', str(tmp_path / 'pheno.txt'), '--covar', str(tmp_path / 'covar.txt'), '--out', str(tmp_path / 'res'),
                        '--stack', '--write_test_scores'], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    res = json.load(open(tmp_path / 'res.json'))
    ts = (tmp_path / 'res.test_scores.tsv').read_text().splitlines()
    assert ts[0].split('\t') == ['FID', 'IID', 'SCORE', 'STACK'] and len(ts) == 2001
    # a worse "method": the same score plus noise
    rng = np.random.default_rng(5)
    rows = [l.split('\t') for l in ts[1:]]
    (tmp_path / 'worse.tsv').write_text('FID\tIID\tSCORE\n' + ''.join('%s\t%s\t%.6f\n' % (f, i, float(s) + 1.5 * rng.standard_normal())
                                                                     for f, i, s, _ in rows))
    r = subprocess.run([sys.executable, COMPARE, '--pheno', str(tmp_path / 'pheno.txt'), '--covar', str(tmp_path / 'covar.txt'),
                        '--scores', 'a=%s' % (tmp_path / 'res.test_scores.tsv'), 'b=%s' % (tmp_path / 'worse.tsv'),
                        '--n_boot', '300', '--out', str(tmp_path / 'cmp')], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    cmp_ = json.load(open(tmp_path / 'cmp.json'))
    assert cmp_['n'] == 2000
    assert abs(cmp_['methods']['a']['r2_increment'] - res['test']['r2_test_increment']) < 1e-9
    lo, hi = cmp_['methods']['b']['diff_ci95']
    assert hi < 0 and cmp_['methods']['b']['diff_vs_ref'] < 0 and cmp_['methods']['b']['p_boot'] < 0.05
