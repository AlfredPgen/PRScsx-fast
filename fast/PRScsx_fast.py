#!/usr/bin/env python
"""
PRS-CSx-fast: an accelerated, output-identical implementation of PRS-CSx (Ruan et al. Nat Genet 2022).

For the same inputs and --seed it writes byte-identical posterior effect files to the original PRScsx.py
(verified against the original on NumPy 1.26 and the NumPy-2-fixed version on NumPy 2.x).

Usage (all original options are accepted with the same meaning and defaults):
python PRScsx_fast.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX --sst_file=SUM_STATS_FILE
                      --n_gwas=GWAS_SAMPLE_SIZE --pop=POPULATION --out_dir=OUTPUT_DIR --out_name=OUTPUT_FILE_PREFIX
                      [--a=PARAM_A --b=PARAM_B --phi=PARAM_PHI --n_iter=MCMC_ITERATIONS --n_burnin=MCMC_BURNIN
                       --thin=MCMC_THINNING_FACTOR --chrom=CHROM --meta=META_FLAG --write_pst=WRITE_POSTERIOR_SAMPLES
                       --seed=SEED]
                      [--threads=N --n_jobs=J --ld_cache_dir=DIR --quiet --strict --linalg=full|lowrank|auto --rank_var=1]

Extra options:
  --phi=1e-6,1e-4,1e-2,1   a comma-separated list runs the whole phi grid in one process, sharing parsing and LD;
                           the RNG is re-seeded for every phi, so each output equals a separate run with that phi.
                           'auto' may be included in the list (e.g. --phi=1e-4,1e-2,auto).
  --threads=N              threads for the LD-block updates within a chromosome (default: number of CPUs, max 8).
  --n_jobs=J               (chromosome, phi) pairs processed in parallel (processes; default 1). Total cores ~ J*N.
                           Every pair is seeded exactly as a separate run, so outputs do not depend on J.
  --ld_cache_dir=DIR       cache processed LD blocks (subset/flip/PSD) on disk and reuse them in later runs.
  --quiet                  only print the start/end of each chromosome.
  --linalg=full            full (default): the original's Cholesky of every LD block -> byte-identical output.
                           lowrank | auto: low-rank (Woodbury) block update, cost ~ b r^2 instead of b^3/3 (auto: per
                           block, low-rank when b > 1.7 r). Same model, prior and estimator, but different random
                           numbers (r + b normals per low-rank block): statistically equivalent, NOT byte-identical.
  --rank_var=1             keep the largest LD eigenvalues explaining this share of the trace (1 = no truncation);
                           below 1 the marginal effects are projected on the kept eigenvectors.
  --strict                 stop with an error if the start-up self-check finds that this platform is not
                           bit-compatible with the original PRS-CSx (default: print a warning and continue).
"""
import os
import sys

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # parallelism is explicit (--threads / --n_jobs)

import getopt  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastcsx import parse  # noqa: E402
from fastcsx.mcmc import BlockSolver, mcmc  # noqa: E402


def _ncpu():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_param(argv):
    long_opts = ['ref_dir=', 'bim_prefix=', 'sst_file=', 'a=', 'b=', 'phi=', 'n_gwas=', 'pop=', 'n_iter=', 'n_burnin=',
                 'thin=', 'out_dir=', 'out_name=', 'chrom=', 'meta=', 'write_pst=', 'seed=', 'help',
                 'threads=', 'n_jobs=', 'ld_cache_dir=', 'quiet', 'strict', 'linalg=', 'rank_var=']
    pd = {'ref_dir': None, 'bim_prefix': None, 'sst_file': None, 'a': 1, 'b': 0.5, 'phi': [None], 'n_gwas': None,
          'pop': None, 'n_iter': None, 'n_burnin': None, 'thin': 5, 'out_dir': None, 'out_name': None,
          'chrom': list(range(1, 23)), 'meta': 'FALSE', 'write_pst': 'FALSE', 'seed': None,
          'threads': min(8, _ncpu()), 'n_jobs': 1, 'ld_cache_dir': None, 'quiet': False, 'strict': False,
          'linalg': 'full', 'rank_var': 1.0}
    if len(argv) == 0:
        print(__doc__); sys.exit(0)
    try:
        opts, _ = getopt.getopt(argv, 'h', long_opts)
    except getopt.GetoptError as e:
        print('* Option not recognized: %s\n* Use --help for usage information.\n' % e); sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(__doc__); sys.exit(0)
        elif opt == '--ref_dir': pd['ref_dir'] = arg
        elif opt == '--bim_prefix': pd['bim_prefix'] = arg
        elif opt == '--sst_file': pd['sst_file'] = arg.split(',')
        elif opt == '--a': pd['a'] = float(arg)
        elif opt == '--b': pd['b'] = float(arg)
        elif opt == '--phi': pd['phi'] = [None if x.strip().lower() == 'auto' else float(x) for x in arg.split(',')]
        elif opt == '--n_gwas': pd['n_gwas'] = list(map(int, arg.split(',')))
        elif opt == '--pop': pd['pop'] = arg.split(',')
        elif opt == '--n_iter': pd['n_iter'] = int(arg)
        elif opt == '--n_burnin': pd['n_burnin'] = int(arg)
        elif opt == '--thin': pd['thin'] = int(arg)
        elif opt == '--out_dir': pd['out_dir'] = arg
        elif opt == '--out_name': pd['out_name'] = arg
        elif opt == '--chrom': pd['chrom'] = [int(c) for c in arg.split(',')]
        elif opt == '--meta': pd['meta'] = arg.upper()
        elif opt == '--write_pst': pd['write_pst'] = arg.upper()
        elif opt == '--seed': pd['seed'] = int(arg)
        elif opt == '--threads': pd['threads'] = max(1, int(arg))
        elif opt == '--n_jobs': pd['n_jobs'] = max(1, int(arg))
        elif opt == '--ld_cache_dir': pd['ld_cache_dir'] = arg
        elif opt == '--quiet': pd['quiet'] = True
        elif opt == '--strict': pd['strict'] = True
        elif opt == '--linalg': pd['linalg'] = arg.lower()
        elif opt == '--rank_var': pd['rank_var'] = float(arg)

    for key, msg in [('ref_dir', 'the directory to the reference panel using --ref_dir'),
                     ('bim_prefix', 'the directory and prefix of the bim file for the target dataset using --bim_prefix'),
                     ('sst_file', 'at least one summary statistics file using --sst_file'),
                     ('n_gwas', 'the sample size of the GWAS using --n_gwas'),
                     ('pop', 'the population of the GWAS sample using --pop'),
                     ('out_dir', 'the output directory using --out_dir'),
                     ('out_name', 'the prefix of the output file using --out_name')]:
        if pd[key] is None:
            print('* Please specify %s\n' % msg); sys.exit(2)
    if len(pd['sst_file']) != len(pd['n_gwas']) or len(pd['sst_file']) != len(pd['pop']):
        print('* Length of sst_file, n_gwas and pop does not match\n'); sys.exit(2)
    n_pop = len(pd['pop'])
    if pd['n_iter'] is None or pd['n_burnin'] is None:
        pd['n_iter'] = n_pop*1000; pd['n_burnin'] = n_pop*500
    if pd['thin'] < 1:
        print('* --thin must be a positive integer\n'); sys.exit(2)
    if pd['n_burnin'] < 0 or pd['n_iter'] <= pd['n_burnin']:
        print('* --n_iter must be larger than --n_burnin (and --n_burnin non-negative)\n'); sys.exit(2)
    if pd['n_iter']//pd['thin'] - pd['n_burnin']//pd['thin'] < 1:
        print('* No posterior draws would be retained: increase --n_iter or decrease --thin\n'); sys.exit(2)
    if pd['linalg'] not in ('full', 'lowrank', 'auto'):
        print('* --linalg must be full, lowrank or auto\n'); sys.exit(2)
    if not 0 < pd['rank_var'] <= 1:
        print('* --rank_var must be in (0, 1]\n'); sys.exit(2)
    for p in pd['pop']:
        if p.upper() not in parse.POPS:
            print('* Unknown population %s (allowed: %s)\n' % (p, ','.join(parse.POPS))); sys.exit(2)
    return pd


def run_chrom(pd, chrom, ref_dict, ssts, ref, log):
    t0 = time.time()
    n_pop = len(pd['pop'])
    log('##### process chromosome %d #####' % chrom)
    log('... %d SNPs on chromosome %d read from the reference SNP information file ...' % (len(ref_dict['SNP']), chrom))
    vld = parse.parse_bim(pd['bim_prefix'], chrom)
    log('... %d SNPs on chromosome %d read from %s ...' % (len(vld['SNP']), chrom, pd['bim_prefix'] + '.bim'))
    sst_dict = {pp: parse.parse_sumstats(ref_dict, vld, ssts[pp], pd['pop'][pp], pd['n_gwas'][pp], log=log) for pp in range(n_pop)}
    ld_blk, blk_size, eig = {}, {}, {}
    for pp in range(n_pop):
        if ref == 'eigen':
            blk_size[pp], eig[pp] = parse.parse_eigblk(pd['ref_dir'], sst_dict[pp], pd['pop'][pp], chrom, log=log)
            ld_blk[pp] = [None] * len(blk_size[pp])
        else:
            ld_blk[pp], blk_size[pp] = parse.parse_ldblk(pd['ref_dir'], sst_dict[pp], pd['pop'][pp], chrom, ref,
                                                         cache_dir=pd['ld_cache_dir'], log=log)
            eig[pp] = None
    snp_dict, beta_dict, frq_dict, idx_dict = parse.align_ldblk(ref_dict, vld, sst_dict, n_pop, chrom, log=log)
    if len(snp_dict['SNP']) == 0:
        log('... no SNPs left on chromosome %d, skipped ...' % chrom)
        return []
    prepared = None
    if pd['linalg'] != 'full' or pd['rank_var'] < 1:
        prepared = []
        for pp in range(n_pop):
            beta_dict[pp], prep = _prepare(pd, ld_blk[pp], blk_size[pp], beta_dict[pp], pd['pop'][pp], chrom, log, eig=eig[pp])
            prepared.append(prep)
    solvers = BlockSolver([ld_blk[pp] for pp in range(n_pop)], [blk_size[pp] for pp in range(n_pop)], pd['threads'],
                          prepared=prepared)
    files = []
    try:
        for phi in pd['phi']:
            files += mcmc(pd['a'], pd['b'], phi, snp_dict, beta_dict, frq_dict, idx_dict, pd['n_gwas'], ld_blk, blk_size,
                          pd['n_iter'], pd['n_burnin'], pd['thin'], pd['pop'], chrom, pd['out_dir'], pd['out_name'],
                          pd['meta'], pd['write_pst'], pd['seed'], log=log, solvers=solvers)
    finally:
        solvers.close()
    log('##### chromosome %d done in %.1f s #####\n' % (chrom, time.time()-t0))
    return files


LOWRANK_NOTICE = ('... --linalg=%s / --rank_var=%g: same model, prior and estimator as PRS-CSx, but the random numbers '
                  'differ from the original\'s (r + b normals per low-rank block): output is statistically equivalent, '
                  'not byte-identical ...')


def _prepare(pd, ld_blk, blk_size, bhat, pop, chrom, log, eig=None):
    """Eigen-decompose LD blocks (cached next to the LD cache; or given by an eigen panel), truncate, choose
    full/low-rank per block."""
    import numpy as np
    from fastcsx.lowrank import prepare_blocks
    key = parse.LD_CACHE_KEYS.get((pop.upper(), chrom)) if pd['ld_cache_dir'] else None
    eigfile = key[:-4] + '_eig.npz' if key else None
    if eig is None and eigfile and os.path.isfile(eigfile):
        z = np.load(eigfile)
        eig = [(z['v%d' % k], z['l%d' % k]) if ('v%d' % k) in z else None for k in range(len(blk_size))]
    prep, bhat_out, eig_list = prepare_blocks(ld_blk, blk_size, bhat, pd['linalg'], pd['rank_var'], eig=eig)
    if eigfile and eig is None and eig_list is not None:
        tmp = eigfile + '.tmp.npz'
        np.savez(tmp, **{('%s%d' % (t, k)): (x[0] if t == 'v' else x[1]) for k, x in enumerate(eig_list) if x is not None for t in 'vl'})
        os.replace(tmp, eigfile)
    n_lr = sum(1 for D, V, lam in prep if V is not None)
    r_lr = sum(V.shape[1] for D, V, lam in prep if V is not None)
    b_lr = sum(V.shape[0] for D, V, lam in prep if V is not None)
    log('... %s: %d of %d LD blocks use the low-rank update (%d SNPs, total rank %d) ...' % (pop, n_lr, len(prep), b_lr, r_lr))
    return bhat_out, prep


def _worker(args):
    pd, chrom, ref_dict, ssts, ref = args
    if pd['seed'] is None:
        import numpy as np
        np.random.seed(None)            # forked workers would otherwise share the parent's random state
    lines = []
    files = run_chrom(pd, chrom, ref_dict, ssts, ref, lines.append)
    return chrom, files, lines


def platform_check(strict):
    from fastcsx.selfcheck import self_check
    gig_ok, blk_ok, secs = self_check()
    if gig_ok and blk_ok:
        print('... platform self-check: bit-compatible with the original PRS-CSx (%.1f s) ...' % secs)
        return
    what = ', '.join(x for x, ok in (('GIG sampler', gig_ok), ('LD-block update', blk_ok)) if not ok)
    msg = ('this platform is NOT bit-compatible with the original PRS-CSx (%s differs). Results are statistically '
           'equivalent to the original but not identical.' % what)
    if strict:
        print('* ERROR (--strict): ' + msg + '\n'); sys.exit(3)
    print('* WARNING: ' + msg)


def main(argv=None):
    pd = parse_param(sys.argv[1:] if argv is None else argv)
    print('\n' + '\n'.join('--%s=%s' % (k, pd[k]) for k in pd) + '\n')
    n_pop = len(pd['pop'])
    print('*** %d discovery population%s detected ***\n' % (n_pop, '' if n_pop == 1 else 's'))
    ref, ref_file = parse.find_ref(pd['ref_dir'])
    if ref is None:
        print('* SNP information file not found: --ref_dir must contain snpinfo_mult_1kg_hm3 or snpinfo_mult_ukbb_hm3\n')
        sys.exit(2)
    os.makedirs(pd['out_dir'], exist_ok=True)
    if ref == 'eigen' and pd['linalg'] == 'full':
        pd['linalg'] = 'auto'               # an eigen panel has no b x b matrices: full factorisation is not available
        print('... eigen LD panel: --linalg=full is not available, using --linalg=auto ...')
    platform_check(pd['strict'])
    if pd['linalg'] != 'full' or pd['rank_var'] < 1:
        print(LOWRANK_NOTICE % (pd['linalg'], pd['rank_var']))
    t0 = time.time()
    print('... parse reference file: %s ...' % ref_file)
    ref_all = parse.read_ref_all(ref_file, pd['chrom'], pops=pd['pop'])
    keep = set()
    for c in pd['chrom']:
        keep.update(ref_all[c]['SNP'])
    ssts = [parse.SumStats(f, keep) for f in pd['sst_file']]
    print('... inputs read in %.1f s ...\n' % (time.time()-t0))

    quiet = pd['quiet']

    def log(msg):
        if not quiet or msg.startswith('#####') or 'MCMC:' in msg:
            print(msg, flush=True)

    pairs = [(c, phi) for c in pd['chrom'] for phi in pd['phi']]
    if pd['n_jobs'] == 1 or len(pairs) == 1:
        for c in pd['chrom']:
            run_chrom(pd, c, ref_all[c], ssts, ref, log)
    else:
        # (chromosome, phi) pairs in parallel; each pair is re-seeded exactly as a separate run would be
        import multiprocessing as mp
        ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() else mp.get_context('spawn')
        jobs = [(dict(pd, phi=[phi]), c, ref_all[c], ssts, ref) for c, phi in pairs]
        with ctx.Pool(min(pd['n_jobs'], len(jobs))) as pool:
            for chrom, files, lines in pool.imap_unordered(_worker, jobs):
                for m in lines:
                    log(m)
    print('... all done in %.1f s ...' % (time.time()-t0))


if __name__ == '__main__':
    main()
