#!/usr/bin/env python
"""
PRS-CSx-improved: the PRS-CSx model with (1) Rao-Blackwellised posterior means, (2) single-precision and low-rank
(Woodbury) LD-block updates, (3) multiple chains with R-hat, optional automatic stopping and warm starts, and optional
(4) alternative cross-ancestry couplings (effect-size scale, learned scale), (5) LD shrinkage toward the identity and
(6) a configurable psi cap.
Same inputs and output file format as PRScsx.py. Requires the fast/ folder (shared parsing + kernels).

Usage (all original options accepted, same defaults):
python PRScsx_improved.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE
                          --n_gwas=GWAS_SAMPLE_SIZE --pop=POPULATION --out_dir=OUTPUT_DIR --out_name=OUTPUT_FILE_PREFIX
                          [--a --b --phi --n_iter --n_burnin --thin --chrom --meta --write_pst --seed]
                          [--precision=single|double --n_chains=1 --linalg=auto|lowrank|full --rank_var=1 --coupling=snr|effect|learned --rb=TRUE|FALSE --psi_max=1 --ld_shrink=0|auto|VALUE --n_ref=N1,N2,..]
                          [--auto_stop=FALSE|TRUE --warm_start=FALSE|TRUE --warm_burnin=N]
                          [--threads=N --n_jobs=J --ld_cache_dir=DIR --quiet]

  --coupling   snr (default): beta_jk ~ N(0, s2_k psi_j / N_k), the PRS-CSx model.
               effect: beta_jk ~ N(0, s2_k psi_j / max(N)); prior effect-size variance shared across ancestries.
               learned: beta_jk ~ N(0, s2_k psi_j tau_k / max(N)), tau_k learned (IG(1,1) prior; tau = 1 for the largest GWAS).
               All three are identical for a single ancestry. effect and learned are not recommended as defaults
               (stress test: docs/03_final_report.md, section 7).
  --rb         TRUE (default): Rao-Blackwellised posterior mean over all post-burn-in iterations. FALSE: thinned draws.
  --psi_max    cap on the local shrinkage parameters (default 1, as PRS-CSx; 'inf' disables the cap).
  --ld_shrink  shrink each reference LD block toward I: R <- (1-l)R + lI. 0 (default), a fixed l, or 'auto'
               (Schafer-Strimmer/Ledoit-Wolf optimal l per block; needs --n_ref = reference panel sample size per
               ancestry, e.g. 1000 Genomes: AFR 661, AMR 347, EAS 504, EUR 503, SAS 489).
  --precision  single (default): Cholesky/solves in float32 (2x faster, half the LD memory; relative error ~1e-7,
               far below Monte Carlo error). double: float64.
  --n_chains   number of independent MCMC chains (seeds seed, seed+1000, ...) averaged per phi (default 1);
               posterior means are averaged, variances combined by the law of total variance, R-hat is printed.
  --linalg     auto (default): per LD block, low-rank (Woodbury) update when b > 1.7 r, else full factorisation;
               lowrank: always low-rank; full: always full. The model is the same in all three.
  --rank_var   keep the largest LD eigenvalues explaining this share of the trace (1 = no truncation, default);
               below 1 the marginal effects are projected on the kept eigenvectors.
  --phi        one value, a comma-separated grid and/or 'auto' (grid shares parsing and LD).
  --auto_stop  TRUE: run the --n_chains (>= 2) chains side by side and stop once the 99th percentile of the
               between-chain R-hat is below 1.05 at 3 consecutive checks (every 200 iterations after burn-in);
               --n_iter is then the maximum. FALSE (default): fixed length.
  --warm_start TRUE: with a phi grid, run the phi values in increasing order (auto last), start each phi's chain(s)
               from the final state of the previous one and use a shorter burn-in (--warm_burnin, default
               n_burnin/2). FALSE (default; warm starts saved only 19% of the iterations, see the report).
"""
import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import getopt  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, '..', 'fast'))
from fastcsx import parse  # noqa: E402
from csximp.mcmc_improved import Solver, lw_shrinkage, mcmc_improved  # noqa: E402

DEFAULT_NREF_1KG = {'AFR': 661, 'AMR': 347, 'EAS': 504, 'EUR': 503, 'SAS': 489}


def _ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_param(argv):
    long_opts = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=', 'pop=', 'n_iter=', 'n_burnin=',
                 'thin=', 'out_dir=', 'out_name=', 'chrom=', 'meta=', 'write_pst=', 'seed=', 'help', 'threads=',
                 'n_jobs=', 'ld_cache_dir=', 'quiet', 'precision=', 'n_chains=', 'linalg=', 'rank_var=', 'coupling=', 'rb=', 'psi_max=', 'ld_shrink=', 'n_ref=', 'auto_stop=', 'warm_start=', 'warm_burnin=']
    pd = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': [None], 'n_gwas': None,
          'pop': None, 'n_iter': None, 'n_burnin': None, 'thin': 5, 'out_dir': None, 'out_name': None,
          'chrom': list(range(1, 23)), 'meta': 'FALSE', 'write_pst': 'FALSE', 'seed': None,
          'threads': min(8, _ncpu()), 'n_jobs': 1, 'ld_cache_dir': None, 'quiet': False,
          'precision': 'single', 'n_chains': 1, 'linalg': 'auto', 'rank_var': 1.0, 'coupling': 'snr', 'rb': True, 'psi_max': 1.0, 'ld_shrink': '0', 'n_ref': None,
          'auto_stop': False, 'warm_start': False, 'warm_burnin': None}
    if not argv:
        print(__doc__); sys.exit(0)
    try:
        opts, _ = getopt.getopt(argv, 'h', long_opts)
    except getopt.GetoptError as e:
        print('* Option not recognized: %s\n' % e); sys.exit(2)
    for opt, arg in opts:
        k = opt.lstrip('-')
        if opt in ('-h', '--help'):
            print(__doc__); sys.exit(0)
        elif k in ('ref_dir', 'bim_prefix', 'out_dir', 'out_name', 'ld_cache_dir'): pd[k] = arg
        elif k == 'sst_file': pd[k] = arg.split(',')
        elif k in ('a', 'b'): pd[k] = float(arg)
        elif k == 'phi': pd[k] = [None if x.strip().lower() == 'auto' else float(x) for x in arg.split(',')]
        elif k == 'n_gwas': pd[k] = list(map(int, arg.split(',')))
        elif k == 'pop': pd[k] = arg.split(',')
        elif k in ('n_iter', 'n_burnin', 'thin', 'seed'): pd[k] = int(arg)
        elif k == 'chrom': pd[k] = [int(c) for c in arg.split(',')]
        elif k in ('meta', 'write_pst'): pd[k] = arg.upper()
        elif k in ('threads', 'n_jobs', 'n_chains'): pd[k] = max(1, int(arg))
        elif k == 'quiet': pd[k] = True
        elif k == 'coupling': pd[k] = arg.lower()
        elif k == 'precision': pd[k] = arg.lower()
        elif k == 'linalg': pd[k] = arg.lower()
        elif k == 'rank_var': pd[k] = float(arg)
        elif k == 'rb': pd[k] = arg.upper() == 'TRUE'
        elif k == 'psi_max': pd[k] = float(arg)
        elif k == 'ld_shrink': pd[k] = arg.lower()
        elif k == 'n_ref': pd[k] = [int(x) for x in arg.split(',')]
        elif k in ('auto_stop', 'warm_start'): pd[k] = arg.upper() == 'TRUE'
        elif k == 'warm_burnin': pd[k] = int(arg)
    for key in ('ref_dir', 'bim_prefix', 'sst_file', 'n_gwas', 'pop', 'out_dir', 'out_name'):
        if pd[key] is None:
            print('* Please specify --%s\n' % key); sys.exit(2)
    n_pop = len(pd['pop'])
    if len(pd['sst_file']) != n_pop or len(pd['n_gwas']) != n_pop:
        print('* Length of sst_file, n_gwas and pop does not match\n'); sys.exit(2)
    if pd['n_iter'] is None or pd['n_burnin'] is None:
        pd['n_iter'] = n_pop*1000; pd['n_burnin'] = n_pop*500
    if pd['thin'] < 1 or pd['n_burnin'] < 0 or pd['n_iter'] <= pd['n_burnin']:
        print('* Need --thin >= 1 and --n_iter > --n_burnin >= 0\n'); sys.exit(2)
    if pd['precision'] not in ('single', 'double'):
        print('* --precision must be single or double\n'); sys.exit(2)
    if pd['precision'] == 'single' and not np.isfinite(pd['psi_max']):
        # without the floor, D + ridge can be ill-conditioned; the single-precision error bound no longer holds
        print('* --psi_max=inf: switching to --precision=double (single precision needs the ridge floor)')
        pd['precision'] = 'double'
    if pd['linalg'] not in ('full', 'lowrank', 'auto'):
        print('* --linalg must be full, lowrank or auto\n'); sys.exit(2)
    if not 0 < pd['rank_var'] <= 1:
        print('* --rank_var must be in (0, 1]\n'); sys.exit(2)
    if pd['coupling'] not in ('effect', 'snr', 'learned'):
        print('* --coupling must be effect, snr or learned\n'); sys.exit(2)
    if pd['auto_stop'] and pd['n_chains'] < 2:
        print('* --auto_stop=TRUE needs --n_chains >= 2\n'); sys.exit(2)
    if pd['warm_burnin'] is None:
        pd['warm_burnin'] = pd['n_burnin']//2
    if not 0 <= pd['warm_burnin'] <= pd['n_burnin']:
        print('* --warm_burnin must be between 0 and --n_burnin\n'); sys.exit(2)
    if pd['ld_shrink'] == 'auto' and pd['n_ref'] is None:
        pd['n_ref'] = [DEFAULT_NREF_1KG.get(p.upper(), 500) for p in pd['pop']]
        print('* --ld_shrink=auto without --n_ref: assuming 1000 Genomes panel sizes %s' % pd['n_ref'])
    return pd


def run_chrom(pd, chrom, ref_dict, ssts, ref, log):
    t0 = time.time()
    n_pop = len(pd['pop'])
    log('##### process chromosome %d #####' % chrom)
    vld = parse.parse_bim(pd['bim_prefix'], chrom)
    sst_dict = {pp: parse.parse_sumstats(ref_dict, vld, ssts[pp], pd['pop'][pp], pd['n_gwas'][pp], log=log) for pp in range(n_pop)}
    ld_blk, blk_size, eig = {}, {}, {}
    for pp in range(n_pop):
        if ref == 'eigen':
            blk_size[pp], eig[pp] = parse.parse_eigblk(pd['ref_dir'], sst_dict[pp], pd['pop'][pp], chrom, log=log)
            ld_blk[pp] = [None] * len(blk_size[pp])
            continue
        eig[pp] = None
        ld_blk[pp], blk_size[pp] = parse.parse_ldblk(pd['ref_dir'], sst_dict[pp], pd['pop'][pp], chrom, ref,
                                                     cache_dir=pd['ld_cache_dir'], log=log)
        if pd['ld_shrink'] not in ('0', '0.0'):
            lams = []
            for k, blk in enumerate(ld_blk[pp]):
                if blk_size[pp][k] == 0:
                    continue
                lam = lw_shrinkage(blk, pd['n_ref'][pp]) if pd['ld_shrink'] == 'auto' else float(pd['ld_shrink'])
                ld_blk[pp][k] = (1.0-lam)*blk + lam*np.eye(blk.shape[0])
                lams.append(lam)
            log('... %s LD shrinkage toward I: lambda mean %.4f (range %.4f-%.4f) ...' % (pd['pop'][pp], np.mean(lams), min(lams), max(lams)))
    snp_dict, beta_dict, frq_dict, idx_dict = parse.align_ldblk(ref_dict, vld, sst_dict, n_pop, chrom, log=log)
    if not snp_dict['SNP']:
        return []
    prepared = None
    if pd['linalg'] != 'full' or pd['rank_var'] < 1:
        from PRScsx_fast import _prepare
        prepared = []
        for pp in range(n_pop):
            beta_dict[pp], prep = _prepare(pd, ld_blk[pp], blk_size[pp], beta_dict[pp], pd['pop'][pp], chrom, log, eig=eig[pp])
            prepared.append(prep)
    solver = Solver([ld_blk[pp] for pp in range(n_pop)], [blk_size[pp] for pp in range(n_pop)], pd['threads'],
                    precision=pd['precision'], prepared=prepared)
    files = []
    state_io = {} if pd['warm_start'] else None
    phis = pd['phi']
    if state_io is not None:
        # warm starts go from strong to weak shrinkage (increasing phi), then auto: the decreasing order lost accuracy
        # in the WP6 evaluation (docs/03_final_report.md, section 7.6)
        phis = sorted(p for p in phis if p is not None) + [p for p in phis if p is None]
        log('... warm start: phi values run in the order %s ...' % ', '.join('auto' if p is None else '%g' % p for p in phis))
    try:
        for i, phi in enumerate(phis):
            # warm start: same number of kept iterations, shorter burn-in after the first phi
            n_burnin = pd['n_burnin'] if (i == 0 or state_io is None) else pd['warm_burnin']
            n_iter = pd['n_iter'] - pd['n_burnin'] + n_burnin
            files += mcmc_improved(pd['a'], pd['b'], phi, snp_dict, beta_dict, frq_dict, idx_dict, pd['n_gwas'], solver,
                                   n_iter, n_burnin, pd['thin'], pd['pop'], chrom, pd['out_dir'],
                                   pd['out_name'], pd['meta'], pd['write_pst'], pd['seed'], coupling=pd['coupling'],
                                   rb=pd['rb'], psi_max=pd['psi_max'], log=log,
                                   n_chains=pd['n_chains'], auto_stop=pd['auto_stop'], state_io=state_io)
    finally:
        solver.close()
    log('##### chromosome %d done in %.1f s #####\n' % (chrom, time.time()-t0))
    return files


def _worker(args):
    pd, chrom, ref_dict, ssts, ref = args
    lines = []
    run_chrom(pd, chrom, ref_dict, ssts, ref, lines.append)
    return lines


def main(argv=None):
    pd = parse_param(sys.argv[1:] if argv is None else argv)
    print('\n' + '\n'.join('--%s=%s' % (k, pd[k]) for k in pd) + '\n')
    ref, ref_file = parse.find_ref(pd['ref_dir'])
    if ref is None:
        print('* SNP information file not found in --ref_dir\n'); sys.exit(2)
    os.makedirs(pd['out_dir'], exist_ok=True)
    if ref == 'eigen' and pd['linalg'] == 'full':
        print('* --linalg=full is not available with an eigen LD panel (use auto or lowrank)\n'); sys.exit(2)
    if ref == 'eigen' and pd['ld_shrink'] not in ('0', '0.0'):
        print('* --ld_shrink is not available with an eigen LD panel\n'); sys.exit(2)
    t0 = time.time()
    ref_all = parse.read_ref_all(ref_file, pd['chrom'], pops=pd['pop'])
    keep = set()
    for c in pd['chrom']:
        keep.update(ref_all[c]['SNP'])
    ssts = [parse.SumStats(f, keep) for f in pd['sst_file']]

    def log(msg):
        if not pd['quiet'] or msg.startswith('#####') or 'Estimated' in msg or 'MCMC:' in msg or 'auto_stop:' in msg or 'auto_stop WARNING' in msg:
            print(msg, flush=True)

    if pd['n_jobs'] == 1 or len(pd['chrom']) == 1:
        for c in pd['chrom']:
            run_chrom(pd, c, ref_all[c], ssts, ref, log)
    else:
        import multiprocessing as mp
        ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context('spawn')
        with ctx.Pool(min(pd['n_jobs'], len(pd['chrom']))) as pool:
            for lines in pool.imap_unordered(_worker, [(pd, c, ref_all[c], ssts, ref) for c in pd['chrom']]):
                for m in lines:
                    log(m)
    print('... all done in %.1f s ...' % (time.time()-t0))


if __name__ == '__main__':
    main()
