#!/usr/bin/env python
"""
WP3 step 5: effect of LD rank truncation (--rank_var) on real 1000 Genomes LD (chr22, EUR + EAS).

Data: the 1000 Genomes EUR/EAS chr22 panels converted to eigen form (tools/build_ld_panel.py convert, rank_var 1);
summary statistics for all chr22 HapMap3 SNPs simulated from the same LD (/work/probe_chr22: N = 200k / 100k).
Part A (deterministic, no Monte Carlo noise): conditional posterior mean m = (D_t + R)^-1 P_t bhat with the truncated
        LD D_t (and projected bhat) vs m = (D + R)^-1 bhat untruncated, at fixed shrinkage: R = I/psi with
        psi = 1e-4, 1e-2, 1 for all SNPs, and a sparse setting (psi = 1 for the 1% largest |bhat| of each block,
        1e-4 elsewhere). Correlation over all SNPs, per ancestry.
Part B (fits): improved/ with --coupling=snr (the PRS-CSx model), Rao-Blackwellised means, --linalg=auto,
        phi = 1e-2, 3 seeds per rank_var; correlation of the 3-seed average with the untruncated 3-seed average, and
        the noise floor = correlation between two independent untruncated 3-seed averages; seconds per iteration.
Usage: python wp3_truncation_study.py OUT_JSON WORK_DIR [--threads 4]
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
from fastcsx import parse  # noqa: E402
from fastcsx.lowrank import truncate  # noqa: E402

RANK_VARS = [1.0, 0.999, 0.995, 0.99, 0.95]
REF = '/work/ref/1kg'
PROBE = '/work/probe_chr22'


def part_a(panel):
    ref_all = parse.read_ref_all(os.path.join(panel, 'snpinfo_mult_eigen'), [22], pops=['EUR', 'EAS'])[22]
    vld = parse.parse_bim(PROBE + '/target', 22)
    out = {}
    for pop, n in (('EUR', 200000), ('EAS', 100000)):
        sst = parse.SumStats('%s/%s_sumstats.txt' % (PROBE, pop), set(ref_all['SNP']))
        sd = parse.parse_sumstats(ref_all, vld, sst, pop, n, log=lambda *x: None)
        blk_size, eig = parse.parse_eigblk(panel, sd, pop, 22, log=lambda *x: None)
        bhat = np.array(sd['BETA'])
        settings = {'psi=1e-4': None, 'psi=1e-2': None, 'psi=1': None, 'sparse': None}
        res = {rv: {k: [[], []] for k in settings} for rv in RANK_VARS}
        ranks = {rv: 0 for rv in RANK_VARS}
        mm = 0
        for b, e in zip(blk_size, eig):
            if b == 0:
                continue
            V, lam = e
            bb = bhat[mm:mm + b]
            top = np.abs(bb) >= np.quantile(np.abs(bb), 0.99)
            ridges = {'psi=1e-4': np.full(b, 1e4), 'psi=1e-2': np.full(b, 1e2), 'psi=1': np.ones(b),
                      'sparse': np.where(top, 1.0, 1e4)}
            D = (V * lam) @ V.T
            full = {k: np.linalg.solve(D + np.diag(r), bb) for k, r in ridges.items()}
            for rv in RANK_VARS:
                Vt, lt = truncate(V, lam, rv)
                ranks[rv] += lt.size
                Dt = (Vt * lt) @ Vt.T
                pb = Vt @ (Vt.T @ bb) if rv < 1 else bb
                for k, r in ridges.items():
                    res[rv][k][0].append(full[k]); res[rv][k][1].append(np.linalg.solve(Dt + np.diag(r), pb))
            mm += b
        out[pop] = {str(rv): dict(total_rank=ranks[rv], **{k: float(np.corrcoef(np.concatenate(v[0]), np.concatenate(v[1]))[0, 1])
                                                             for k, v in res[rv].items()}) for rv in RANK_VARS}
        out[pop]['n_snps'] = int(sum(blk_size))
    return out


def part_b(panel, work, threads):
    est, secs = {}, {}
    runs = [(rv, s) for rv in RANK_VARS for s in (1, 2, 3)] + [(1.0, s) for s in (4, 5, 6)]
    for rv, seed in runs:
        od = os.path.join(work, 'fit_rv%s_s%d' % (rv, seed)); os.makedirs(od, exist_ok=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--ref_dir=' + panel,
                            '--bim_prefix=' + PROBE + '/target', '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (PROBE, PROBE),
                            '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=%d' % seed,
                            '--coupling=snr', '--linalg=auto', '--rank_var=%s' % rv, '--threads=%d' % threads, '--quiet',
                            '--out_name=t', '--out_dir=' + od], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:]
        m = re.search(r'MCMC: (\d+) iterations in ([\d.]+) s', r.stdout)
        secs.setdefault(rv, []).append(float(m.group(2)) / float(m.group(1)))
        for pop in ('EUR', 'EAS'):
            est[(rv, seed, pop)] = np.loadtxt(os.path.join(od, 't_%s_pst_eff_a1_b0.5_phi1e-02_chr22.txt' % pop), usecols=5)
        print('fit rank_var=%s seed=%d done (%.3f s/iter)' % (rv, seed, secs[rv][-1]), flush=True)
    out = {}
    for pop in ('EUR', 'EAS'):
        base = np.mean([est[(1.0, s, pop)] for s in (1, 2, 3)], 0)
        other = np.mean([est[(1.0, s, pop)] for s in (4, 5, 6)], 0)
        out[pop] = {'noise_floor_corr_untruncated_vs_untruncated': float(np.corrcoef(base, other)[0, 1])}
        for rv in RANK_VARS:
            avg = np.mean([est[(rv, s, pop)] for s in (1, 2, 3)], 0)
            out[pop][str(rv)] = dict(corr_with_untruncated=float(np.corrcoef(base, avg)[0, 1]) if rv < 1 else 1.0,
                                     corr_with_independent_untruncated=float(np.corrcoef(other, avg)[0, 1]),
                                     seconds_per_iteration=float(np.mean(secs[rv])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('work'); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    panel = os.path.join(a.work, 'eig1kg')
    if not os.path.isfile(os.path.join(panel, 'ldeig_eas', 'ldeig_chr22.hdf5')):
        subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'build_ld_panel.py'), 'convert', '--ref_dir', REF,
                        '--pops', 'EUR,EAS', '--chroms', '22', '--out', panel], check=True)
    res = {'part_A_conditional_mean_fixed_psi': part_a(panel)}
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res['part_A_conditional_mean_fixed_psi'], indent=1), flush=True)
    res['part_B_fits'] = part_b(panel, a.work, a.threads)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res['part_B_fits'], indent=1))


if __name__ == '__main__':
    main()
