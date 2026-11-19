#!/usr/bin/env python
"""
Create the benchmark job queues (run with jobs.py). Paths are set for this sandbox; edit the constants below.

Queue q_runtime  (run with --slots 1 on an otherwise idle machine)
  example data    : original (unmodified, NumPy 1.26), fixed (NumPy 2.x), fast, improved; 1 and 4 threads; 3 repeats
  probe chr22     : all HapMap3 SNPs of chr22, EUR+EAS, default MCMC length (original measured separately: 1,186 s)
  scaling         : simulated 3-ancestry data on chr22 | 21-22 | 19-22 | 15-22, n_iter 20 and 80 (setup + per-iteration cost)
Queue q_accuracy (run with --slots 4; every job single-threaded)
  8 datasets      : chr22 <= 35 Mb, EUR/EAS/AFR N=300k/100k/30k, rho=0.8; (h2, pi) in {(.5,.001),(.5,.01),(.5,.1),(.2,.01)} x 2 reps
  fits            : prscsx (= fast, exact original) and improved (default), phi in {1e-6,1e-4,1e-2,1,auto}, meta, seed=rep
  ablations       : rbonly, cplonly, improved n_chains=2 on (.5,.01) and (.5,.1) reps 1-2
  MCMC seeds      : (.5,.01) rep 1 with MCMC seeds 11, 12 for prscsx and improved
"""
import json
import os

PROJ = '/work/proj'
BENCH = '/work/bench'
SIM = '/work/sim'
REF1KG = '/work/ref/1kg'
PY = 'python3'
PY_ORIG = '/work/venv_ref/bin/python'
ORIG_DIR = '/work/fixcheck/prscsx_orig'          # unmodified original code (git master)
FIXED_DIR = PROJ
FAST = PROJ + '/fast/PRScsx_fast.py'
IMP = PROJ + '/improved/PRScsx_improved.py'
TD = FIXED_DIR + '/test_data'
ENV1 = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
ENV4 = {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
PHIS = ['1e-6', '1e-4', '1e-2', '1', 'auto']
SCEN = [('h0.5_pi0.001', 0.5, 0.001), ('h0.5_pi0.01', 0.5, 0.01), ('h0.5_pi0.1', 0.5, 0.1), ('h0.2_pi0.01', 0.2, 0.01)]


def job(q, jid, cmd, out_dir, env=None, timeout=4 * 3600, priority=0, **meta):
    os.makedirs(os.path.join(q, 'jobs'), exist_ok=True)
    d = dict(id=jid, cmd=cmd, out_dir=out_dir, env=env or {}, timeout=timeout, priority=priority, meta=meta)
    json.dump(d, open(os.path.join(q, 'jobs', jid + '.json'), 'w'), indent=1)


def runtime_queue():
    q = BENCH + '/q_runtime'
    ex = ['--ref_dir=' + REF1KG, '--bim_prefix=%s/test' % TD, '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (TD, TD),
          '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--out_name=test', '--out_dir={OUT}']
    for rep in (1, 2, 3):
        s = '--seed=%d' % rep
        job(q, 'ex_r%d_a_original_t1' % rep, [PY_ORIG, ORIG_DIR + '/PRScsx.py'] + ex + [s], BENCH + '/runtime/ex/original_t1_r%d' % rep, ENV1, priority=1)
        job(q, 'ex_r%d_b_fixed_t1' % rep, [PY, FIXED_DIR + '/PRScsx.py'] + ex + [s], BENCH + '/runtime/ex/fixed_t1_r%d' % rep, ENV1, priority=1)
        for t in (1, 4):
            job(q, 'ex_r%d_c_fast_t%d' % (rep, t), [PY, FAST] + ex + [s, '--threads=%d' % t], BENCH + '/runtime/ex/fast_t%d_r%d' % (t, rep), ENV1, priority=1)
            job(q, 'ex_r%d_d_improved_t%d' % (rep, t), [PY, IMP] + ex + [s, '--threads=%d' % t], BENCH + '/runtime/ex/improved_t%d_r%d' % (t, rep), ENV1, priority=1)
    pr = ['--ref_dir=' + REF1KG, '--bim_prefix=/work/probe_chr22/target', '--sst_file=/work/probe_chr22/EUR_sumstats.txt,/work/probe_chr22/EAS_sumstats.txt',
          '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=1', '--out_name=probe', '--out_dir={OUT}']
    for m, scr in (('fast', FAST), ('improved', IMP)):
        for t in (1, 4):
            job(q, 'probe_%s_t%d' % (m, t), [PY, scr] + pr + ['--threads=%d' % t], BENCH + '/runtime/probe/%s_t%d' % (m, t), ENV1, priority=2)
    sizes = [('chr22', '22'), ('chr21_22', '21,22'), ('chr19_20_21_22', '19,20,21,22'), ('chr15_16_17_18_19_20_21_22', '15,16,17,18,19,20,21,22')]
    for tag, cs in sizes:
        d = SIM + '/scale_' + tag
        base = ['--ref_dir=%s/ref_chr15_22' % SIM, '--bim_prefix=%s/target' % d,
                '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt,%s/AFR_sumstats.txt' % (d, d, d),
                '--n_gwas=300000,100000,30000', '--pop=EUR,EAS,AFR', '--chrom=' + cs, '--phi=1e-2', '--seed=1',
                '--out_name=scale', '--out_dir={OUT}']
        for it in (20, 80):
            mc = ['--n_iter=%d' % it, '--n_burnin=%d' % (it // 2)]
            job(q, 'scale_%s_it%d_a_original_t4' % (tag, it), [PY_ORIG, ORIG_DIR + '/PRScsx.py'] + base + mc,
                BENCH + '/runtime/scale/%s/original_t4_it%d' % (tag, it), ENV4, priority=3, timeout=3 * 3600)
            job(q, 'scale_%s_it%d_b_fast_t4' % (tag, it), [PY, FAST] + base + mc + ['--threads=4'],
                BENCH + '/runtime/scale/%s/fast_t4_it%d' % (tag, it), ENV1, priority=3)
            job(q, 'scale_%s_it%d_c_improved_t4' % (tag, it), [PY, IMP] + base + mc + ['--threads=4'],
                BENCH + '/runtime/scale/%s/improved_t4_it%d' % (tag, it), ENV1, priority=3)


def make_datasets():
    """Simulate the 8 accuracy datasets up front (skips existing ones)."""
    import subprocess
    for sname, h2, pi in SCEN:
        for rep in (1, 2):
            d = SIM + '/acc/%s_r%d' % (sname, rep)
            if os.path.isfile(os.path.join(d, 'truth.npz')):
                continue
            subprocess.run([PY, PROJ + '/benchmarks/sim/simulate.py', 'make-data', '--ref_dir', REF1KG, '--chroms', '22',
                            '--pops', 'EUR,EAS,AFR', '--n_gwas', '300000,100000,30000', '--h2', str(h2), '--pi', str(pi),
                            '--rho', '0.8', '--max_bp', '35000000', '--seed', str(100 + rep * 10 + [x[0] for x in SCEN].index(sname)),
                            '--out', d], check=True, env=dict(os.environ, **ENV1))


def accuracy_queue():
    q = BENCH + '/q_accuracy'
    methods = {
        'prscsx': (FAST, []),
        'improved': (IMP, []),
        'rbonly': (IMP, ['--coupling=snr', '--rb=TRUE']),
        'cplonly': (IMP, ['--coupling=effect', '--rb=FALSE']),
        'improved2ch': (IMP, ['--n_chains=2']),
        'ldshrink': (IMP, ['--ld_shrink=auto', '--n_ref=500,500,650']),
    }
    for sname, h2, pi in SCEN:
        for rep in (1, 2):
            dname = '%s_r%d' % (sname, rep)
            d = SIM + '/acc/' + dname
            fit_methods = ['prscsx', 'improved']
            if sname in ('h0.5_pi0.01', 'h0.5_pi0.1'):
                fit_methods += ['rbonly', 'cplonly']   # n_chains=2 is evaluated post hoc from the seed replicates
            if sname == 'h0.5_pi0.01':
                fit_methods += ['ldshrink']
            seeds = [rep]
            for m in fit_methods:
                for seed in seeds + ([11, 12] if (dname == 'h0.5_pi0.01_r1' and m in ('prscsx', 'improved')) else []):
                    for phi in PHIS:
                        scr, extra = methods[m]
                        out = BENCH + '/accuracy/%s/%s_s%d/phi%s' % (dname, m, seed, phi)
                        cmd = [PY, scr, '--ref_dir=%s/ref_chr22' % SIM, '--bim_prefix=%s/target' % d,
                               '--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt,%s/AFR_sumstats.txt' % (d, d, d),
                               '--n_gwas=300000,100000,30000', '--pop=EUR,EAS,AFR', '--chrom=22', '--phi=' + phi,
                               '--meta=TRUE', '--seed=%d' % seed, '--threads=1', '--quiet', '--out_name=sim', '--out_dir={OUT}'] + extra
                        pri = 1 if seed == rep and m in ('prscsx', 'improved') else (3 if m == 'ldshrink' else 2)
                        job(q, 'a%d_%s_%s_s%d_phi%s' % (pri, dname, m, seed, phi), cmd, out, ENV1, priority=pri,
                            dataset=dname, method=m, seed=seed, phi=phi)


if __name__ == '__main__':
    import sys
    if 'make-data' in sys.argv:
        make_datasets(); sys.exit(0)
    if 'accuracy-only' not in sys.argv:
        runtime_queue()
    accuracy_queue()
    for qq in ('q_runtime', 'q_accuracy'):
        print(qq, len(os.listdir(os.path.join(BENCH, qq, 'jobs'))), 'jobs')
