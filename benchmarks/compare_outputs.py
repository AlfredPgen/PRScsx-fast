#!/usr/bin/env python
"""
Compare PRS-CSx posterior effect-size output files (6 columns: CHR SNP BP A1 A2 BETA, or META files).

Usage as a module:  compare(file_a, file_b) -> dict with n, max_abs, mean_abs, rel_max, pearson_r, identical_text
Usage as CLI:       python compare_outputs.py FILE_A FILE_B
"""
import sys
import numpy as np


def load(fn):
    snps, rows = [], []
    with open(fn) as f:
        for line in f:
            ll = line.split()
            snps.append((ll[1], ll[3], ll[4]))
            rows.append([float(x) for x in ll[5:]])
    return snps, np.array(rows)


def compare(fa, fb):
    sa, ba = load(fa)
    sb, bb = load(fb)
    if sa != sb:
        raise ValueError('SNP/allele lists differ between %s and %s' % (fa, fb))
    ba = ba.ravel(); bb = bb.ravel()
    d = np.abs(ba - bb)
    with open(fa) as f1, open(fb) as f2:
        identical = f1.read() == f2.read()
    sd = np.std(ba)
    return {
        'n': int(ba.size),
        'identical_text': identical,
        'max_abs': float(d.max()),
        'mean_abs': float(d.mean()),
        'max_abs_over_sd': float(d.max() / sd) if sd > 0 else float('nan'),
        'pearson_r': float(np.corrcoef(ba, bb)[0, 1]) if ba.size > 1 else float('nan'),
    }


if __name__ == '__main__':
    r = compare(sys.argv[1], sys.argv[2])
    print(r)
