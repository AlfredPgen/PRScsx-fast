#!/usr/bin/env python
"""
Exact (population-level) prediction accuracy of PRS weight files on a simulated dataset.

For target ancestry t with true LD R_t and true standardised effects beta_t, a PRS with per-allele weights u has
standardised weights w_j = u_j * sqrt(2 f_tj (1-f_tj)) and
    r2_g = (w'R_t beta_t)^2 / ((w'R_t w)(beta_t'R_t beta_t))      = squared corr(PRS, genetic value) = R2 / h2,
    R2   = r2_g * h2 (phenotypic R^2 of a genome-wide PRS if all regions behaved like this one).
No test-sample noise. Several PRS can be combined:
  * oracle:      population-optimal linear combination  (upper bound for validation tuning)
  * validation:  weights (and the phi grid point) chosen in a simulated validation sample of n_val people.
                 (PRS_1..m, g) ~ N(0, [W beta]' R_t [W beta]) exactly (Gaussian genotypes); the phenotype is
                 y = c*g + e with c = sqrt(h2/h2_region), Var(e) = 1-h2, i.e. selection noise of a genome-wide PRS.
Estimation error: ||w_k - beta_k||^2 / ||beta_k||^2 on ancestry k's own standardised scale.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simcore  # noqa: E402


class Truth:
    def __init__(self, data_dir, ref_dir):
        z = np.load(os.path.join(data_dir, 'truth.npz'))
        self.snps = list(z['snps']); self.beta = z['beta']; self.frq = z['frq']; self.present = z['present']
        self.pops = list(z['pops']); self.chroms = [int(c) for c in z['chroms']]
        self.pos = {s: i for i, s in enumerate(self.snps)}
        self.params = json.load(open(os.path.join(data_dir, 'sim_params.json')))
        self.ref_dir = ref_dir
        self._ld = {}

    def ld(self, pop):
        if pop not in self._ld:
            k = self.pops.index(pop)
            out = []
            for c in self.chroms:
                for snps, R, L in simcore.true_ld_blocks(self.ref_dir, pop, c):
                    ii = np.array([self.pos[s] for s in snps])
                    keep = self.present[ii, k]                  # restricted region (--max_bp)
                    if keep.any():
                        out.append((ii[keep], R[np.ix_(keep, keep)]))
            self._ld[pop] = out
        return self._ld[pop]

    def read_weights(self, fn):
        """Per-allele weights (M,) aligned to the truth SNP order (0 where absent)."""
        u = np.zeros(len(self.snps))
        with open(fn) as f:
            for line in f:
                ll = line.split()
                u[self.pos[ll[1]]] = float(ll[5])
        return u

    def std_weights(self, u, pop):
        k = self.pops.index(pop)
        f = self.frq[:, k]
        return u * np.sqrt(2 * f * (1 - f)) * self.present[:, k]

    def moments(self, W, pop):
        """W: (M, m) standardised weights in ancestry `pop`. Returns S_WW (m,m), S_Wb (m,), var_g."""
        k = self.pops.index(pop)
        b = self.beta[:, k]
        m = W.shape[1]
        SWW = np.zeros((m, m)); SWb = np.zeros(m); vg = 0.0
        for ii, R in self.ld(pop):
            Wi = W[ii]; bi = b[ii]
            RW = R @ Wi
            SWW += Wi.T @ RW; SWb += RW.T @ bi; vg += bi @ R @ bi
        return SWW, SWb, vg


def r2_single(SWW, SWb, vg):
    d = np.diag(SWW)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(d > 0, SWb ** 2 / (d * vg), 0.0)


def r2_oracle(SWW, SWb, vg, cols):
    S = SWW[np.ix_(cols, cols)]; s = SWb[cols]
    c = np.linalg.lstsq(S, s, rcond=None)[0]
    return float(s @ c / vg), c


def r2_of_combo(SWW, SWb, vg, cols, c):
    S = SWW[np.ix_(cols, cols)]; s = SWb[cols]
    num = (c @ s) ** 2; den = c @ S @ c * vg
    return float(num / den) if den > 0 else 0.0


def validation_select(SWW, SWb, vg, groups, h2, n_val, rng):
    """
    groups: list of column-index lists (e.g. one list per phi value: the ancestry-specific PRSs).
    Simulates one validation sample, fits OLS weights per group, picks the group with the best validation R2.
    Returns (population r2_g of the selected, fitted combination, index of selected group).
    """
    m = SWW.shape[0]
    Sig = np.zeros((m + 1, m + 1)); Sig[:m, :m] = SWW; Sig[:m, m] = SWb; Sig[m, :m] = SWb; Sig[m, m] = vg
    w, V = np.linalg.eigh(Sig); w = np.clip(w, 0, None)
    X = rng.standard_normal((n_val, m + 1)) @ (V * np.sqrt(w)).T
    c_scale = np.sqrt(h2 / vg) if vg > 0 else 0.0
    y = c_scale * X[:, m] + rng.standard_normal(n_val) * np.sqrt(1 - h2)
    best, best_r2v = None, -np.inf
    for gi, cols in enumerate(groups):
        A = np.column_stack([np.ones(n_val), X[:, cols]])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        r2v = 1 - np.sum((y - A @ coef) ** 2) / np.sum((y - y.mean()) ** 2)
        if r2v > best_r2v:
            best_r2v, best = r2v, (gi, coef[1:])
    gi, c = best
    return r2_of_combo(SWW, SWb, vg, groups[gi], c), gi


def validation_stack(SWW, SWb, vg, cols, h2, n_val, rng):
    """
    Stacking (super learner) over ALL given PRS columns (e.g. every phi x ancestry): non-negative least squares of the
    validation phenotype on the validation-standardised PRSs (sign-aligned so a positive weight is the natural
    direction), then the population r2_g of the fitted combination. Uses the same simulated validation design as
    validation_select (draw it with the same rng state to obtain a paired comparison).
    """
    from scipy.optimize import nnls
    m = SWW.shape[0]
    Sig = np.zeros((m + 1, m + 1)); Sig[:m, :m] = SWW; Sig[:m, m] = SWb; Sig[m, :m] = SWb; Sig[m, m] = vg
    w, V = np.linalg.eigh(Sig); w = np.clip(w, 0, None)
    X = rng.standard_normal((n_val, m + 1)) @ (V * np.sqrt(w)).T
    c_scale = np.sqrt(h2 / vg) if vg > 0 else 0.0
    y = c_scale * X[:, m] + rng.standard_normal(n_val) * np.sqrt(1 - h2)
    P = X[:, cols]; P = P - P.mean(0); sd = P.std(0); sd[sd == 0] = 1.0
    Z = P / sd
    sgn = np.sign(Z.T @ (y - y.mean())); sgn[sgn == 0] = 1.0
    coef, _ = nnls(Z * sgn, y - y.mean())
    c = coef * sgn / sd                                   # weights on the raw PRS scale
    if not np.any(c):
        return 0.0
    return r2_of_combo(SWW, SWb, vg, cols, c)


def est_error(truth, u, pop):
    k = truth.pops.index(pop)
    w = truth.std_weights(u, pop); b = truth.beta[:, k]; pr = truth.present[:, k]
    return float(np.sum((w[pr] - b[pr]) ** 2) / np.sum(b[pr] ** 2))


# ----------------------------------------------------------------------------------------------- method-level evaluation
def find_outputs(method_dir, out_name, pops, chrom_tag='chr'):
    """{phi_tag: {pop|'META': [files...]}} from PRS-CSx-style output files in method_dir (all chromosomes)."""
    res = {}
    files = glob.glob(os.path.join(method_dir, '%s_*_pst_eff_*.txt' % out_name)) + \
        glob.glob(os.path.join(method_dir, '*', '%s_*_pst_eff_*.txt' % out_name))      # one sub-folder per phi
    for fn in files:
        base = os.path.basename(fn)[len(out_name) + 1:]
        pop, rest = base.split('_pst_eff_')
        phi_tag = rest.split('_')[2]
        res.setdefault(phi_tag, {}).setdefault(pop, []).append(fn)
    return res


def evaluate_method(truth, method_dir, out_name, n_val=3000, val_seed=0):
    """Rows of metrics for one fitted method (all phi values found) on one simulated dataset."""
    pops = truth.pops
    outs = find_outputs(method_dir, out_name, pops)
    phis = sorted(outs, key=lambda t: (t == 'phiauto', float(t[3:]) if t != 'phiauto' else 0))
    h2 = truth.params['h2']
    # weight vectors (per-allele, summed over chromosome files)
    U = {}
    for ph in phis:
        for src, files in outs[ph].items():
            u = np.zeros(len(truth.snps))
            for fn in files:
                u += truth.read_weights(fn)
            U[(ph, src)] = u
    rows = []
    for t in pops:
        keys = list(U)
        W = np.column_stack([truth.std_weights(U[k], t) for k in keys])
        SWW, SWb, vg = truth.moments(W, t)
        single = r2_single(SWW, SWb, vg)
        col = {k: i for i, k in enumerate(keys)}
        for k in keys:
            rows.append(dict(target=t, phi=k[0], prs=k[1], metric='r2_single', value=float(single[col[k]])))
        groups, gphis = [], []
        for ph in phis:
            cols = [col[(ph, s)] for s in pops if (ph, s) in col]
            r2o, _ = r2_oracle(SWW, SWb, vg, cols)
            rows.append(dict(target=t, phi=ph, prs='combo', metric='r2_oracle_combo', value=r2o))
            if ph != 'phiauto':
                groups.append(cols); gphis.append(ph)
        if groups:
            best = max(range(len(groups)), key=lambda g: r2_oracle(SWW, SWb, vg, groups[g])[0])
            rows.append(dict(target=t, phi=gphis[best], prs='combo', metric='r2_tuned_oracle',
                             value=r2_oracle(SWW, SWb, vg, groups[best])[0]))
            rng = np.random.default_rng(val_seed + 1000 * pops.index(t))
            r2v, gi = validation_select(SWW, SWb, vg, groups, h2, n_val, rng)
            rows.append(dict(target=t, phi=gphis[gi], prs='combo', metric='r2_tuned_validation', value=r2v))
            rng = np.random.default_rng(val_seed + 1000 * pops.index(t))       # same validation sample
            allc = sorted(c for g in groups for c in g)
            rows.append(dict(target=t, phi='all', prs='stack', metric='r2_stack_validation',
                             value=validation_stack(SWW, SWb, vg, allc, h2, n_val, rng)))
        if ('phiauto', 'META') in col:
            rows.append(dict(target=t, phi='phiauto', prs='META', metric='r2_auto_meta', value=float(single[col[('phiauto', 'META')]])))
        if ('phiauto', t) in col:
            rows.append(dict(target=t, phi='phiauto', prs=t, metric='r2_auto_own', value=float(single[col[('phiauto', t)]])))
    for (ph, src), u in U.items():
        if src in pops:
            rows.append(dict(target=src, phi=ph, prs=src, metric='est_error', value=est_error(truth, u, src)))
    return rows


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True); ap.add_argument('--ref_dir', default='/work/ref/1kg')
    ap.add_argument('--method_dir', required=True); ap.add_argument('--out_name', required=True)
    ap.add_argument('--n_val', type=int, default=3000); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    tr = Truth(a.data, a.ref_dir)
    rows = evaluate_method(tr, a.method_dir, a.out_name, a.n_val)
    for r in rows:
        if r['metric'] != 'r2_single' or r['prs'] == r['target']:
            print('%-4s %-9s %-5s %-20s %.4f' % (r['target'], r['phi'], r['prs'], r['metric'], r['value']))
    if a.json:
        json.dump(rows, open(a.json, 'w'), indent=0)
