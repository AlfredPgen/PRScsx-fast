#!/usr/bin/env python
"""
Run PRScsx_fast.py on every baseline configuration of the example data and compare each output file with the
ORIGINAL PRS-CSx outputs (byte identity, max/mean abs difference, correlation).

Usage: python verify_fast_vs_original.py --py PYTHON --fast /work/proj/fast --baseline /work/baseline
                                         --out OUT --threads 1,4 [--ref_dir /work/ref/1kg]
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
from verify_fixed_vs_original import CONFIGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--py', required=True); ap.add_argument('--fast', required=True)
    ap.add_argument('--baseline', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--threads', default='1,4'); ap.add_argument('--ref_dir', default='/work/ref/1kg')
    ap.add_argument('--repo', default='/work/PRScsx')
    a = ap.parse_args()
    td = os.path.join(a.repo, 'test_data')
    res = {}
    for thr in [int(x) for x in a.threads.split(',')]:
        for name, extra in CONFIGS.items():
            od = os.path.join(a.out, 't%d' % thr, name); os.makedirs(od, exist_ok=True)
            if name.startswith('eur_only'):
                sst = ['--sst_file=%s/EUR_sumstats.txt' % td, '--n_gwas=200000', '--pop=EUR']
            else:
                sst = ['--sst_file=%s/EUR_sumstats.txt,%s/EAS_sumstats.txt' % (td, td), '--n_gwas=200000,100000', '--pop=EUR,EAS']
            cmd = [a.py, os.path.join(a.fast, 'PRScsx_fast.py'), '--ref_dir=' + a.ref_dir, '--bim_prefix=%s/test' % td,
                   '--chrom=22', '--out_dir=' + od, '--out_name=test', '--threads=%d' % thr] + sst + extra
            t0 = time.perf_counter()
            with open(os.path.join(od, 'log.txt'), 'w') as lf:
                rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1200).returncode
            wall = time.perf_counter()-t0
            files = {}
            for bf in sorted(glob.glob(os.path.join(a.baseline, name, '*_pst_eff_*.txt'))):
                nf = os.path.join(od, os.path.basename(bf))
                files[os.path.basename(bf)] = compare(bf, nf) if os.path.exists(nf) else {'missing': True}
            res['t%d/%s' % (thr, name)] = {'exit_code': rc, 'wall_s': round(wall, 2), 'files': files}
            nid = sum(1 for v in files.values() if v.get('identical_text'))
            print('threads=%d %-28s exit=%d wall=%5.1fs byte-identical %d/%d' % (thr, name, rc, wall, nid, len(files)), flush=True)
    json.dump(res, open(os.path.join(a.out, 'verification.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
