#!/usr/bin/env python
"""
Re-run every baseline configuration with a (fixed) PRS-CSx and compare each output file against the
ORIGINAL program's baseline outputs (produced on NumPy 1.26 by run_baseline_example.sh).

Usage:
  python verify_fixed_vs_original.py --py PYTHON --repo FIXED_REPO --baseline /work/baseline \
         --out OUT_DIR --label LABEL [--ref_dir /work/ref/1kg]
Writes OUT_DIR/LABEL_verification.json and .md
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_outputs import compare  # noqa: E402

CONFIGS = {  # baseline run name -> extra PRScsx args (mirrors run_baseline_example.sh)
    'phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1'],
    'phi1e-2_seed2_t1': ['--phi=1e-2', '--seed=2'],
    'phi1e-2_seed3_t1': ['--phi=1e-2', '--seed=3'],
    'phi1e-2_seed4_t1': ['--phi=1e-2', '--seed=4'],
    'phi1e-2_seed5_t1': ['--phi=1e-2', '--seed=5'],
    'meta_phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1', '--meta=TRUE'],
    'metapst_phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1', '--meta=TRUE', '--write_pst=TRUE'],
    'phiauto_meta_seed1_t1': ['--seed=1', '--meta=TRUE'],
    'eur_only_phi1e-2_seed1_t1': ['--phi=1e-2', '--seed=1'],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--ref_dir', default='/work/ref/1kg')
    a = ap.parse_args()
    td = os.path.join(a.repo, 'test_data')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    ver = subprocess.run([a.py, '-c', 'import numpy,scipy,h5py,sys;print(sys.version.split()[0],numpy.__version__,scipy.__version__,h5py.__version__)'],
                         capture_output=True, text=True).stdout.split()
    results = {'label': a.label, 'python': ver[0], 'numpy': ver[1], 'scipy': ver[2], 'h5py': ver[3], 'runs': {}}
    for name, extra in CONFIGS.items():
        od = os.path.join(a.out, a.label, name)
        os.makedirs(od, exist_ok=True)
        if name.startswith('eur_only'):
            sst = ['--sst_file=%s/EUR_sumstats.txt' % td, '--n_gwas=200000', '--pop=EUR']
        else:
            sst = ['--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (td, td), '--n_gwas=200000,100000', '--pop=EUR,EAS']
        cmd = [a.py, os.path.join(a.repo, 'PRScsx.py'), '--ref_dir=' + a.ref_dir, '--bim_prefix=%s/test' % td,
               '--chrom=22', '--out_dir=' + od, '--out_name=test'] + sst + extra
        t0 = time.perf_counter()
        with open(os.path.join(od, 'log.txt'), 'w') as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800).returncode
        wall = time.perf_counter() - t0
        rec = {'exit_code': rc, 'wall_s': round(wall, 2), 'files': {}}
        base_files = sorted(glob.glob(os.path.join(a.baseline, name, '*_pst_eff_*.txt')))
        for bf in base_files:
            nf = os.path.join(od, os.path.basename(bf))
            if not os.path.exists(nf):
                rec['files'][os.path.basename(bf)] = {'missing': True}
                continue
            rec['files'][os.path.basename(bf)] = compare(bf, nf)
        extra_files = set(map(os.path.basename, glob.glob(os.path.join(od, '*_pst_eff_*.txt')))) - set(map(os.path.basename, base_files))
        rec['unexpected_files'] = sorted(extra_files)
        results['runs'][name] = rec
        n_id = sum(1 for v in rec['files'].values() if v.get('identical_text'))
        print('%-28s exit=%d wall=%6.1fs  files byte-identical to original: %d/%d  max_abs=%s'
              % (name, rc, wall, n_id, len(base_files), max((v.get('max_abs', float('nan')) for v in rec['files'].values()), default=None)), flush=True)
    json.dump(results, open(os.path.join(a.out, '%s_verification.json' % a.label), 'w'), indent=1)

    md = ['### %s (Python %s, NumPy %s, SciPy %s, h5py %s)' % (a.label, results['python'], results['numpy'], results['scipy'], results['h5py']), '',
          '| configuration | exit | wall (s) | output file | byte-identical | max abs diff | Pearson r |', '|---|---|---|---|---|---|---|']
    for name, rec in results['runs'].items():
        for fn, c in rec['files'].items():
            if c.get('missing'):
                md.append('| %s | %d | %.1f | %s | MISSING | | |' % (name, rec['exit_code'], rec['wall_s'], fn))
            else:
                md.append('| %s | %d | %.1f | %s | %s | %.3g | %.12f |' % (name, rec['exit_code'], rec['wall_s'], fn, c['identical_text'], c['max_abs'], c['pearson_r']))
    open(os.path.join(a.out, '%s_verification.md' % a.label), 'w').write('\n'.join(md) + '\n')


if __name__ == '__main__':
    main()
