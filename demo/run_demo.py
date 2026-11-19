#!/usr/bin/env python
"""
Out-of-the-box demo: runs the PRS-CSx Gibbs sampler three ways on the repository's example summary statistics, with a
small SYNTHETIC LD reference panel generated on the fly, so nothing has to be downloaded.

  python demo/run_demo.py [OUT_DIR]          (default OUT_DIR: demo_output/ next to this folder)

Steps:
  1. tests/make_synth_ref.py writes a synthetic 1000 Genomes-format panel for chr22 (EUR, EAS, AFR; 7 LD blocks).
  2. PRScsx.py   - the original PRS-CSx (NumPy-2 fixed), EUR + EAS, phi = 1e-2, seed 1
  3. fast/PRScsx_fast.py       - same command; its files must be byte-identical to step 2
  4. improved/PRScsx_improved.py - same command (default settings), then with 2 chains and --auto_stop
The synthetic panel's LD is NOT real LD: the effect sizes are for testing the software only. For real analyses
download the reference panels (get_reference.sh) and pass that folder as --ref_dir.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TD = os.path.join(ROOT, 'test_data')


def run(name, cmd, out):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    with open(os.path.join(out, name + '.log'), 'w') as f:
        f.write(r.stdout + r.stderr)
    if r.returncode != 0:
        print(r.stdout[-3000:] + r.stderr[-3000:])
        sys.exit('* %s failed (exit code %d); log: %s' % (name, r.returncode, os.path.join(out, name + '.log')))
    print('  %-28s ok  (%.1f s)' % (name, dt))
    return r.stdout


def main():
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'demo_output'))
    os.makedirs(out, exist_ok=True)
    ref = os.path.join(out, 'synthetic_ref')
    py = sys.executable
    print('1. synthetic reference panel')
    run('make_synth_ref', [py, os.path.join(ROOT, 'tests', 'make_synth_ref.py'), ref], out)
    common = ['--ref_dir=' + ref, '--bim_prefix=' + os.path.join(TD, 'test'),
              '--sst_file=%s,%s' % (os.path.join(TD, 'EUR_sumstats.txt'), os.path.join(TD, 'EAS_sumstats.txt')),
              '--n_gwas=200000,100000', '--pop=EUR,EAS', '--chrom=22', '--phi=1e-2', '--seed=1', '--out_name=demo']
    runs = [('original_prscsx', [py, os.path.join(ROOT, 'PRScsx.py')]),
            ('fast', [py, os.path.join(ROOT, 'fast', 'PRScsx_fast.py'), '--threads=2']),
            ('improved', [py, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--threads=2']),
            ('improved_2chains_autostop', [py, os.path.join(ROOT, 'improved', 'PRScsx_improved.py'), '--threads=2',
                                           '--n_chains=2', '--auto_stop=TRUE'])]
    print('2-4. Gibbs samplers (EUR + EAS, chr22, phi = 1e-2, seed 1; the first numba run compiles, ~30 s)')
    for name, cmd in runs:
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        run(name, cmd + common + ['--out_dir=' + d], out)

    import numpy as np
    files = ['demo_%s_pst_eff_a1_b0.5_phi1e-02_chr22.txt' % p for p in ('EUR', 'EAS')]
    print('\nresults (output files in %s):' % out)
    ok = True
    for f in files:
        same = open(os.path.join(out, 'original_prscsx', f), 'rb').read() == open(os.path.join(out, 'fast', f), 'rb').read()
        ok &= same
        print('  %s: fast vs original byte-identical: %s' % (f, 'YES' if same else 'NO'))
        b0 = np.loadtxt(os.path.join(out, 'original_prscsx', f), usecols=5)
        for name in ('improved', 'improved_2chains_autostop'):
            b1 = np.loadtxt(os.path.join(out, name, f), usecols=5)
            print('     %-26s correlation with the original: %.3f (%d SNPs)' % (name, np.corrcoef(b0, b1)[0, 1], b0.size))
    stop = [l for l in open(os.path.join(out, 'improved_2chains_autostop.log')) if 'auto_stop' in l or 'R-hat' in l]
    print('  ' + '  '.join(stop[-3:]).rstrip())
    print('\nDemo %s. For real data: sh get_reference.sh REF_DIR 1kg EUR,EAS  and use --ref_dir=REF_DIR.'
          % ('finished' if ok else 'finished, BUT fast was not byte-identical to the original on this machine'))


if __name__ == '__main__':
    main()
