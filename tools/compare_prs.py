#!/usr/bin/env python
"""
compare_prs.py: compare polygenic scores of several methods in the same test sample, with paired bootstrap CIs.

For each method, the incremental R2 of the score over a covariate-only model is computed on the people who have the
phenotype, all covariates and every method's score (so all methods use the same people):
    R2_inc = R2(y ~ 1 + covariates + score) - R2(y ~ 1 + covariates)          (OLS; 0/1 traits: linear probability)
Uncertainty: nonparametric bootstrap over people (--n_boot resamples, the same resamples for every method), percentile
95% intervals; for each method versus the first one (the reference), the paired difference of R2_inc, its 95% interval
and a two-sided bootstrap p-value (2 x the share of resamples whose difference has the other sign; >= 1/n_boot).

Usage:
  python compare_prs.py --pheno pheno.txt [--covar covar.txt] \\
         --scores original=orig.test_scores.tsv improved=impr.test_scores.tsv [--column SCORE] \\
         [--n_boot 2000] [--seed 1] [--out compare_prs]
  score files: header with IID and the score column (default SCORE), FID optional: e.g. <out>.test_scores.tsv of
  combine_prs.py --write_test_scores, a PLINK2 .sscore (--column SCORE1_AVG) or a PLINK 1.9 .profile (--column SCORE).
Outputs: <out>.json and a table on stdout.
"""
import argparse
import json
import sys

import numpy as np


def _read(fn, column):
    with open(fn) as f:
        h = [c.lstrip('#') for c in f.readline().split()]
        if column not in h:
            sys.exit('* column %s not found in %s (header: %s)' % (column, fn, ' '.join(h)))
        j, ii = h.index(column), h.index('IID')
        fi = h.index('FID') if 'FID' in h else ii
        out = {}
        for line in f:
            r = line.split()
            if r and r[j] not in ('NA', 'nan', '.'):
                out[(r[fi], r[ii])] = float(r[j])
    return out


def _read_pheno(fn):
    with open(fn) as f:
        h = f.readline().split()
        out = {}
        for line in f:
            r = line.split()
            if len(r) > 2 and r[2] not in ('NA', 'nan', '-9', '.'):
                out[(r[0], r[1])] = float(r[2])
    return out, h[2]


def _read_covar(fn):
    with open(fn) as f:
        h = f.readline().split()
        out = {}
        for line in f:
            r = line.split()
            try:
                out[(r[0], r[1])] = [float(x) for x in r[2:]]
            except ValueError:
                continue
    return out, h[2:]


def _r2(y, A, w):
    """Weighted R2 of the OLS fit y ~ A (A includes the intercept); w = bootstrap counts."""
    sw = np.sqrt(w)
    c = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    ybar = np.sum(w * y) / np.sum(w)
    return 1.0 - np.sum(w * (y - A @ c) ** 2) / np.sum(w * (y - ybar) ** 2)


def incremental_r2(y, X0, scores, w):
    base = _r2(y, X0, w)
    return np.array([_r2(y, np.column_stack([X0, s]), w) - base for s in scores.T])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pheno', required=True)
    ap.add_argument('--covar')
    ap.add_argument('--scores', nargs='+', required=True, help='LABEL=FILE, the first one is the reference')
    ap.add_argument('--column', default='SCORE')
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='compare_prs')
    a = ap.parse_args(argv)
    labels, files = zip(*[s.split('=', 1) for s in a.scores])
    pheno, pname = _read_pheno(a.pheno)
    covar, cnames = _read_covar(a.covar) if a.covar else ({}, [])
    sc = [_read(f, a.column) for f in files]
    ids = set(pheno).intersection(*[set(s) for s in sc])
    if a.covar:
        ids &= set(covar)
    ids = sorted(ids)
    if len(ids) < 50:
        sys.exit('* only %d people with phenotype, covariates and all scores' % len(ids))
    y = np.array([pheno[i] for i in ids])
    X0 = np.column_stack([np.ones(len(ids))] + ([np.array([covar[i] for i in ids])] if a.covar else []))
    S = np.column_stack([[s[i] for i in ids] for s in sc])
    n = len(ids)
    est = incremental_r2(y, X0, S, np.ones(n))
    rng = np.random.default_rng(a.seed)
    boot = np.empty((a.n_boot, len(labels)))
    for b in range(a.n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        boot[b] = incremental_r2(y, X0, S, w)
    res = {'phenotype': pname, 'covariates': cnames, 'n': n, 'n_boot': a.n_boot, 'methods': {}}
    print('%-16s %10s %24s %12s %24s %10s' % ('method', 'R2_inc', '95% CI', 'diff_vs_ref', '95% CI', 'p'))
    for k, lab in enumerate(labels):
        lo, hi = np.percentile(boot[:, k], [2.5, 97.5])
        r = {'r2_increment': float(est[k]), 'ci95': [float(lo), float(hi)],
             'corr_score_pheno': float(np.corrcoef(S[:, k], y)[0, 1])}
        line = '%-16s %10.5f   [%9.5f, %9.5f]' % (lab, est[k], lo, hi)
        if k > 0:
            d = boot[:, k] - boot[:, 0]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            p = max(2.0 * min(np.mean(d <= 0), np.mean(d >= 0)), 1.0 / a.n_boot)
            r.update(diff_vs_ref=float(est[k] - est[0]), diff_ci95=[float(dlo), float(dhi)], p_boot=float(min(p, 1.0)))
            line += ' %+12.5f   [%+9.5f, %+9.5f] %10.4f' % (est[k] - est[0], dlo, dhi, min(p, 1.0))
        res['methods'][lab] = r
        print(line)
    print('n = %d people (phenotype %s; reference = %s)' % (n, pname, labels[0]))
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    main()
