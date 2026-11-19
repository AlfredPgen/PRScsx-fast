#!/usr/bin/env python
"""
combine_prs.py: the PRS-CSx README's recommended tuning step, as a tool.

For each phi, the ancestry-specific polygenic scores (computed with PLINK --score from the PRS-CSx / fast / improved
output files) are standardised in the validation sample, and a linear combination is fitted:
    y ~ covariates + c_1 PRS_pop1 + ... + c_K PRS_popK      (OLS; y quantitative, or 0/1 treated as linear probability)
The phi with the largest validation R2 increment over the covariate-only model is selected. If test scores are given,
the selected phi, weights and the VALIDATION standardisation are applied unchanged to the test sample and its R2 is reported.

Usage:
  python combine_prs.py --scores_val val_scores.tsv --pheno pheno.txt [--covar covar.txt] [--scores_test test_scores.tsv]
                        [--out result_prefix] [--stack] [--write_test_scores]
  val_scores.tsv / test_scores.tsv : tab/space-separated, header 'phi pop file', one line per (phi, ancestry);
                                     file = PLINK2 .sscore (uses SCORE1_AVG/SCORE1_SUM) or PLINK1.9 .profile (uses SCORE)
  pheno.txt  : header 'FID IID <PHENO>' (first phenotype column used; -9/NA = missing)
  covar.txt  : header 'FID IID <COV1> <COV2> ...' (optional)
  --stack : additionally fit ONE combination of ALL (phi x ancestry) scores by non-negative least squares on the
            validation-standardised scores (stacking / super learner, Breiman 1996), residualised on the covariates;
            it is reported next to the single-best-phi model and applied to the test set as well.
  --write_test_scores : write <out>.test_scores.tsv (FID IID SCORE [STACK]): the combined score of every test person
            (selected phi, frozen validation weights; STACK with --stack). Compare methods with compare_prs.py.
Outputs: <out>.json (all phi: validation R2, weights; selected phi; stacked weights; test R2) and a table on stdout.
"""
import argparse
import json
import sys

import numpy as np


def _read_table(fn):
    with open(fn) as f:
        header = f.readline().split()
        rows = [l.split() for l in f if l.strip()]
    return header, rows


def read_scores(fn):
    h, rows = _read_table(fn)
    h = [c.lstrip('#') for c in h]
    for col in ('SCORE1_AVG', 'SCORE1_SUM', 'SCORE'):
        if col in h:
            j = h.index(col); break
    else:
        cand = [c for c in h if c.upper().startswith('SCORE')]
        if not cand:
            raise ValueError('no score column in %s (header %s)' % (fn, h))
        j = h.index(cand[0])
        print('* using column %s of %s' % (cand[0], fn), file=sys.stderr)
    fi, ii = h.index('FID') if 'FID' in h else None, h.index('IID')
    return {((r[fi] if fi is not None else r[ii]), r[ii]): float(r[j]) for r in rows}


def read_pheno(fn):
    h, rows = _read_table(fn)
    out = {}
    for r in rows:
        v = r[2]
        if v in ('NA', 'nan', '-9', '.'):
            continue
        out[(r[0], r[1])] = float(v)
    return out, h[2]


def read_covar(fn):
    h, rows = _read_table(fn)
    out = {}
    for r in rows:
        try:
            out[(r[0], r[1])] = [float(x) for x in r[2:]]
        except ValueError:
            continue
    return out, h[2:]


def load_score_list(fn):
    h, rows = _read_table(fn)
    idx = {c: h.index(c) for c in ('phi', 'pop', 'file')}
    d = {}
    for r in rows:
        d.setdefault(r[idx['phi']], {})[r[idx['pop']]] = read_scores(r[idx['file']])
    return d


def r2_fit(y, X0, P):
    """R2 of y ~ [1, X0, P] and of y ~ [1, X0]; returns (r2_full, r2_base, coef_P)."""
    n = len(y)
    A0 = np.column_stack([np.ones(n)] + ([X0] if X0 is not None else []))
    A = np.column_stack([A0, P])
    tss = np.sum((y - y.mean()) ** 2)
    c0 = np.linalg.lstsq(A0, y, rcond=None)[0]; c = np.linalg.lstsq(A, y, rcond=None)[0]
    return 1 - np.sum((y - A @ c) ** 2) / tss, 1 - np.sum((y - A0 @ c0) ** 2) / tss, c, A0.shape[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scores_val', required=True); ap.add_argument('--pheno', required=True)
    ap.add_argument('--covar'); ap.add_argument('--scores_test'); ap.add_argument('--out', default='combine_prs')
    ap.add_argument('--stack', action='store_true', help='also fit an NNLS stack over all phi x ancestry scores')
    ap.add_argument('--write_test_scores', action='store_true', help='write <out>.test_scores.tsv (needs --scores_test)')
    a = ap.parse_args()
    if a.write_test_scores and not a.scores_test:
        sys.exit('--write_test_scores needs --scores_test')
    test_scores = {}
    pheno, pname = read_pheno(a.pheno)
    covar, cnames = read_covar(a.covar) if a.covar else ({}, [])
    val = load_score_list(a.scores_val)
    test = load_score_list(a.scores_test) if a.scores_test else None
    res = {'phenotype': pname, 'covariates': cnames, 'phi': {}}
    for phi, pops in sorted(val.items()):
        names = sorted(pops)
        ids = sorted(set(pheno).intersection(*[set(pops[p]) for p in names]).intersection(covar if a.covar else pheno))
        y = np.array([pheno[i] for i in ids])
        X0 = np.array([covar[i] for i in ids]) if a.covar else None
        P = np.column_stack([[pops[p][i] for i in ids] for p in names])
        mu, sd = P.mean(0), P.std(0)
        sd[sd == 0] = 1.0
        Pz = (P - mu) / sd
        r2, r2b, coef, k0 = r2_fit(y, X0, Pz)
        res['phi'][phi] = dict(pops=names, n_val=len(ids), r2_val=r2, r2_val_covariates_only=r2b, r2_val_increment=r2 - r2b,
                               weights_on_standardised_prs=list(map(float, coef[k0:])), val_mean=list(map(float, mu)),
                               val_sd=list(map(float, sd)))
    best = max(res['phi'], key=lambda p: res['phi'][p]['r2_val_increment'])
    res['selected_phi'] = best
    print('%-10s %-22s %8s %10s %12s' % ('phi', 'ancestries', 'n_val', 'R2_val', 'R2_increment'))
    for phi, r in sorted(res['phi'].items()):
        print('%-10s %-22s %8d %10.4f %12.4f %s' % (phi, ','.join(r['pops']), r['n_val'], r['r2_val'], r['r2_val_increment'],
                                                   '<- selected' if phi == best else ''))
    if test is not None:
        r = res['phi'][best]
        pops = test[best]
        ids = sorted(set(pheno).intersection(*[set(pops[p]) for p in r['pops']]).intersection(covar if a.covar else pheno))
        y = np.array([pheno[i] for i in ids])
        X0 = np.array([covar[i] for i in ids]) if a.covar else None
        Pz = (np.column_stack([[pops[p][i] for i in ids] for p in r['pops']]) - np.array(r['val_mean'])) / np.array(r['val_sd'])
        score = Pz @ np.array(r['weights_on_standardised_prs'])        # frozen validation weights
        test_scores['SCORE'] = dict(zip(ids, score))
        r2t, r2bt, _, _ = r2_fit(y, X0, score[:, None])
        res['test'] = dict(n_test=len(ids), r2_test=r2t, r2_test_covariates_only=r2bt, r2_test_increment=r2t - r2bt,
                           corr_prs_pheno=float(np.corrcoef(score, y)[0, 1]))
        print('test: n=%d  R2=%.4f  R2 increment over covariates=%.4f  corr(PRS, y)=%.4f' %
              (len(ids), r2t, r2t - r2bt, res['test']['corr_prs_pheno']))
    if a.stack:
        from scipy.optimize import nnls
        cols = [(phi, p) for phi in sorted(val) for p in sorted(val[phi])]
        ids = sorted(set(pheno).intersection(*[set(val[phi][p]) for phi, p in cols]).intersection(covar if a.covar else pheno))
        y = np.array([pheno[i] for i in ids])
        X0 = np.column_stack([np.ones(len(ids))] + ([np.array([covar[i] for i in ids])] if a.covar else []))
        P = np.column_stack([[val[phi][p][i] for i in ids] for phi, p in cols])
        mu, sd = P.mean(0), P.std(0); sd[sd == 0] = 1.0
        Z = (P - mu) / sd
        ry = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]                 # residualise on covariates
        RZ = Z - X0 @ np.linalg.lstsq(X0, Z, rcond=None)[0]
        sgn = np.sign(RZ.T @ ry); sgn[sgn == 0] = 1.0
        w, _ = nnls(RZ * sgn, ry); w = w * sgn
        r2s, r2b, _, _ = r2_fit(y, X0[:, 1:] if a.covar else None, (Z @ w)[:, None])
        res['stack'] = dict(columns=['%s:%s' % c for c in cols], weights_on_standardised_prs=list(map(float, w)),
                            val_mean=list(map(float, mu)), val_sd=list(map(float, sd)), n_val=len(ids),
                            r2_val=r2s, r2_val_increment=r2s - r2b)
        print('stack (NNLS over %d scores): n_val=%d  R2_val=%.4f  R2_increment=%.4f  nonzero weights=%d'
              % (len(cols), len(ids), r2s, r2s - r2b, int(np.sum(w != 0))))
        if test is not None:
            ids = sorted(set(pheno).intersection(*[set(test[phi][p]) for phi, p in cols]).intersection(covar if a.covar else pheno))
            y = np.array([pheno[i] for i in ids])
            X0 = np.array([covar[i] for i in ids]) if a.covar else None
            Zt = (np.column_stack([[test[phi][p][i] for i in ids] for phi, p in cols]) - mu) / sd
            r2t, r2bt, _, _ = r2_fit(y, X0, (Zt @ w)[:, None])
            test_scores['STACK'] = dict(zip(ids, Zt @ w))
            res['stack']['test'] = dict(n_test=len(ids), r2_test=r2t, r2_test_increment=r2t - r2bt)
            print('stack test: n=%d  R2=%.4f  R2 increment=%.4f' % (len(ids), r2t, r2t - r2bt))
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    if a.write_test_scores:
        cols = sorted(test_scores, key=lambda c: c != 'SCORE')
        ids = sorted(set().union(*[set(test_scores[c]) for c in cols]))
        with open(a.out + '.test_scores.tsv', 'w') as f:
            f.write('FID\tIID\t' + '\t'.join(cols) + '\n')
            for i in ids:
                f.write('%s\t%s\t' % i + '\t'.join('%.8g' % test_scores[c][i] if i in test_scores[c] else 'NA' for c in cols) + '\n')


if __name__ == '__main__':
    main()
