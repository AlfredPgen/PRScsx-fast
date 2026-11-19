#!/usr/bin/env python
"""
Build a PRS-CSx LD reference panel in eigen form (docs/04_eigen_panel_format.md) from PLINK genotypes, for any SNP set.

  build : one ancestry, one chromosome
    python build_ld_panel.py build --bfile GENO_PREFIX --pop EAS --chrom 22 --out PANEL_DIR
           [--blocks_from REF_1KG_DIR | --blocks BLOCKS_FILE | --block_snps N] [--rank_var 0.999] [--maf 0.01]
           [--max_missing 0.05] [--max_n N --seed S]
  merge : combine the per-ancestry SNP lists of all built chromosomes into PANEL_DIR/snpinfo_mult_eigen
    python build_ld_panel.py merge --out PANEL_DIR
  convert : turn an existing PRS-CS(x) panel (1000 Genomes / UKBB, b x b blocks) into eigen form
    python build_ld_panel.py convert --ref_dir REF_1KG_DIR --pops EUR,EAS --chroms 1-22 --out PANEL_DIR [--rank_var 1]
    (blocks are PSD-projected exactly as PRS-CSx does, then eigen-decomposed; SNP info is copied unchanged)

Block boundaries (one of):
  --blocks_from DIR   the blocks of the existing 1000 Genomes panel of this ancestry (default if given): block k covers
                      positions [first position of block k, first position of block k+1)
  --blocks FILE       lines "CHR START END" (START inclusive, END exclusive)
  --block_snps N      consecutive windows of N SNPs (only for testing; LD blocks should follow recombination)
Per block: SNPs with MAF >= --maf, missing rate <= --max_missing, not strand-ambiguous; genotypes standardised
(missing -> mean); if n < b the thin SVD of X/sqrt(n), else the eigendecomposition of X'X/n; eigenvalues <= 1e-13 *
largest dropped (exact zeros), then the largest eigenvalues explaining --rank_var of the trace are kept.
Genotypes are read block by block: memory ~ n x (largest block) + the block's eigenvectors.
"""
import argparse
import glob
import os
import sys
import time

import h5py
import numpy as np

POPS = ('AFR', 'AMR', 'EAS', 'EUR', 'SAS')
COMP = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
EIG_TOL = 1e-13


# --------------------------------------------------------------------------------------------- PLINK .bed I/O
_LUT = np.array([2.0, np.nan, 1.0, 0.0])           # 2-bit code -> count of A1 (bim column 5): 00 hom A1, 01 missing, 10 het, 11 hom A2


class Bed:
    def __init__(self, prefix):
        self.prefix = prefix
        self.fam = [l.split() for l in open(prefix + '.fam')]
        self.bim = [l.split() for l in open(prefix + '.bim')]
        self.n = len(self.fam)
        self.bps = (self.n + 3) // 4
        self.fh = open(prefix + '.bed', 'rb')
        magic = self.fh.read(3)
        if magic != b'\x6c\x1b\x01':
            raise ValueError('%s.bed is not a SNP-major PLINK 1 .bed file' % prefix)
        codes = np.arange(256, dtype=np.uint8)
        self._dec = np.stack([(codes >> (2 * k)) & 3 for k in range(4)], axis=1)   # byte -> 4 codes

    def read(self, snp_idx, people=None):
        """Genotypes (A1 counts, NaN missing) of the given bim rows: n x len(snp_idx). Reads only those SNPs."""
        out = np.empty((self.n if people is None else len(people), len(snp_idx)))
        for j, i in enumerate(snp_idx):
            self.fh.seek(3 + i * self.bps)
            raw = np.frombuffer(self.fh.read(self.bps), dtype=np.uint8)
            g = _LUT[self._dec[raw].ravel()[:self.n]]
            out[:, j] = g if people is None else g[people]
        return out


def write_bed(prefix, G, snps, chrom=1, bps=None, a1=None, a2=None):
    """Write an n x m matrix of A1 counts (0/1/2, NaN missing) as PLINK .bed/.bim/.fam (used by tests)."""
    n, m = G.shape
    bps = bps if bps is not None else np.arange(1, m + 1) * 1000
    a1 = a1 if a1 is not None else ['A'] * m
    a2 = a2 if a2 is not None else ['G'] * m
    with open(prefix + '.fam', 'w') as f:
        for i in range(n):
            f.write('F%d I%d 0 0 0 -9\n' % (i, i))
    with open(prefix + '.bim', 'w') as f:
        for j in range(m):
            f.write('%d\t%s\t0\t%d\t%s\t%s\n' % (chrom, snps[j], bps[j], a1[j], a2[j]))
    code = {2.0: 0, 1.0: 2, 0.0: 3}
    nb = (n + 3) // 4
    with open(prefix + '.bed', 'wb') as f:
        f.write(b'\x6c\x1b\x01')
        for j in range(m):
            c = np.array([1 if np.isnan(x) else code[float(x)] for x in G[:, j]], dtype=np.uint8)
            c = np.concatenate([c, np.zeros(4 * nb - n, np.uint8)]).reshape(nb, 4)
            f.write((c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).astype(np.uint8).tobytes())


# --------------------------------------------------------------------------------------------- blocks
def blocks_from_1kg(ref_dir, pop, chrom):
    """[(start, end)] from the 1000 Genomes panel of this ancestry (positions of each block's first SNP)."""
    bp = {}
    with open(os.path.join(ref_dir, 'snpinfo_mult_1kg_hm3')) as f:
        next(f)
        for line in f:
            ll = line.split()
            if int(ll[0]) == chrom:
                bp[ll[1]] = int(ll[2])
    starts = []
    with h5py.File(os.path.join(ref_dir, 'ldblk_1kg_%s' % pop.lower(), 'ldblk_1kg_chr%d.hdf5' % chrom), 'r') as h:
        for k in range(1, len(h) + 1):
            s = [bp[x.decode()] for x in h['blk_%d' % k]['snplist'][:] if x.decode() in bp]
            if s:
                starts.append(min(s))
    starts = sorted(starts)
    return [(0 if k == 0 else starts[k], starts[k + 1] if k + 1 < len(starts) else np.inf) for k in range(len(starts))]


def blocks_from_file(fn, chrom):
    out = []
    for line in open(fn):
        ll = line.split()
        if not ll or not ll[0].lstrip('chr').isdigit():
            continue
        if int(ll[0].lstrip('chr')) == chrom:
            out.append((int(ll[1]), int(ll[2])))
    return sorted(out)


# --------------------------------------------------------------------------------------------- eigen
def block_eigen(X, rank_var):
    """X: n x b standardised genotypes. Returns (V b x r float32, lam r float64, trace)."""
    n, b = X.shape
    if n < b:
        _, s, Vt = np.linalg.svd(X / np.sqrt(n), full_matrices=False)
        lam = s ** 2; V = Vt.T
    else:
        w, Q = np.linalg.eigh(X.T @ X / n)
        lam = w[::-1]; V = Q[:, ::-1]
    trace = float(np.sum(X * X) / n)
    keep = lam > EIG_TOL * lam[0] if lam.size and lam[0] > 0 else np.zeros(lam.size, bool)
    V, lam = V[:, keep], lam[keep]
    if rank_var < 1 and lam.size:
        k = int(np.searchsorted(np.cumsum(lam) / trace, rank_var) + 1)
        V, lam = V[:, :k], lam[:k]
    return np.ascontiguousarray(V, dtype=np.float32), lam.astype(np.float64), trace


def build(a):
    t0 = time.time()
    bed = Bed(a.bfile)
    rows = [i for i, r in enumerate(bed.bim) if r[0].lstrip('chr').isdigit() and int(r[0].lstrip('chr')) == a.chrom]
    rows.sort(key=lambda i: (int(bed.bim[i][3]), bed.bim[i][1]))
    if a.blocks_from:
        blocks = blocks_from_1kg(a.blocks_from, a.pop, a.chrom)
    elif a.blocks:
        blocks = blocks_from_file(a.blocks, a.chrom)
    elif a.block_snps:
        pos = [int(bed.bim[i][3]) for i in rows]
        blocks = [(pos[k], pos[min(k + a.block_snps, len(pos)) - 1] + 1) for k in range(0, len(pos), a.block_snps)]
    else:
        sys.exit('* give --blocks_from, --blocks or --block_snps')
    people = None
    if a.max_n and a.max_n < bed.n:
        people = np.sort(np.random.default_rng(a.seed).choice(bed.n, a.max_n, replace=False))
    n_used = bed.n if people is None else len(people)
    os.makedirs(os.path.join(a.out, 'ldeig_%s' % a.pop.lower()), exist_ok=True)
    os.makedirs(os.path.join(a.out, 'fragments'), exist_ok=True)
    fn = os.path.join(a.out, 'ldeig_%s' % a.pop.lower(), 'ldeig_chr%d.hdf5' % a.chrom)
    frag = []
    n_snp = n_rank = 0
    with h5py.File(fn + '.tmp', 'w') as h:
        h.attrs['format'] = 'prscsx-eigen-1'; h.attrs['pop'] = a.pop.upper(); h.attrs['chrom'] = a.chrom
        h.attrs['maf'] = a.maf; h.attrs['n_ref'] = n_used; h.attrs['rank_var'] = a.rank_var
        k = 0
        pos = np.array([int(bed.bim[i][3]) for i in rows], dtype=np.float64)     # rows are sorted by position
        for start, end in blocks:
            lo, hi = np.searchsorted(pos, start, 'left'), np.searchsorted(pos, end, 'left')
            idx = [i for i in rows[lo:hi]
                   if not ({bed.bim[i][4], bed.bim[i][5]} in ({'A', 'T'}, {'C', 'G'}))
                   and bed.bim[i][4] in COMP and bed.bim[i][5] in COMP]
            G = bed.read(idx, people) if idx else np.zeros((n_used, 0))
            miss = np.isnan(G).mean(0) if idx else np.zeros(0)
            frq = np.nanmean(G, 0) / 2 if idx else np.zeros(0)
            ok = (miss <= a.max_missing) & (np.minimum(frq, 1 - frq) >= a.maf) if idx else np.zeros(0, bool)
            idx = [i for i, o in zip(idx, ok) if o]
            G = G[:, ok] if len(ok) else G
            frq = frq[ok] if len(ok) else frq
            k += 1
            g = h.create_group('blk_%d' % k)
            if not idx:
                g.create_dataset('snplist', data=np.array([], dtype='S1')); g.create_dataset('evec', data=np.zeros((0, 0), np.float32))
                g.create_dataset('eval', data=np.zeros(0)); g.attrs['n_ref'] = n_used; g.attrs['rank_var'] = a.rank_var; g.attrs['trace'] = 0.0
                continue
            mu = np.nanmean(G, 0); G = np.where(np.isnan(G), mu, G)
            X = (G - mu) / G.std(0)
            V, lam, trace = block_eigen(X, a.rank_var)
            g.create_dataset('snplist', data=np.array([bed.bim[i][1].encode() for i in idx]))
            g.create_dataset('evec', data=V); g.create_dataset('eval', data=lam)
            g.attrs['n_ref'] = n_used; g.attrs['rank_var'] = a.rank_var; g.attrs['trace'] = trace
            frag += [(bed.bim[i][1], int(bed.bim[i][3]), bed.bim[i][4], bed.bim[i][5], f) for i, f in zip(idx, frq)]
            n_snp += len(idx); n_rank += V.shape[1]
            if a.verbose:
                print('  blk_%d: %d SNPs, rank %d' % (k, len(idx), V.shape[1]), flush=True)
    os.replace(fn + '.tmp', fn)
    with open(os.path.join(a.out, 'fragments', 'snpinfo_%s_chr%d.txt' % (a.pop.upper(), a.chrom)), 'w') as f:
        for s, bp, a1, a2, fr in frag:
            f.write('%d\t%s\t%d\t%s\t%s\t%.6f\n' % (a.chrom, s, bp, a1, a2, fr))
    print('build: %s chr%d: %d blocks, %d SNPs, total rank %d, n=%d, %.1f s, %.1f MB' % (
        a.pop.upper(), a.chrom, k, n_snp, n_rank, n_used, time.time() - t0, os.path.getsize(fn) / 1e6))


def merge(a):
    """Union of all fragments -> snpinfo_mult_eigen (alleles of the first ancestry that has the SNP, in POPS order)."""
    snps = {}
    for pop in POPS:
        for fn in sorted(glob.glob(os.path.join(a.out, 'fragments', 'snpinfo_%s_chr*.txt' % pop))):
            for line in open(fn):
                c, s, bp, a1, a2, fr = line.split()
                key = (int(c), s)
                rec = snps.setdefault(key, {'BP': int(bp), 'A1': a1, 'A2': a2, 'FRQ': {}, 'FLP': {}})
                if (a1, a2) == (rec['A1'], rec['A2']) or (COMP[a1], COMP[a2]) == (rec['A1'], rec['A2']):
                    rec['FRQ'][pop] = float(fr); rec['FLP'][pop] = 1
                elif (a2, a1) == (rec['A1'], rec['A2']) or (COMP[a2], COMP[a1]) == (rec['A1'], rec['A2']):
                    rec['FRQ'][pop] = 1 - float(fr); rec['FLP'][pop] = -1
                else:
                    print('* %s %s: alleles %s/%s incompatible with %s/%s; dropped for %s' % (c, s, a1, a2, rec['A1'], rec['A2'], pop))
    with open(os.path.join(a.out, 'snpinfo_mult_eigen'), 'w') as f:
        f.write('CHR\tSNP\tBP\tA1\tA2\t' + '\t'.join('FRQ_' + p for p in POPS) + '\t' + '\t'.join('FLP_' + p for p in POPS) + '\n')
        for (c, s), r in sorted(snps.items(), key=lambda kv: (kv[0][0], kv[1]['BP'], kv[0][1])):
            f.write('%d\t%s\t%d\t%s\t%s\t' % (c, s, r['BP'], r['A1'], r['A2']) + '\t'.join('%.6f' % r['FRQ'].get(p, 0.0) for p in POPS)
                    + '\t' + '\t'.join('%d' % r['FLP'].get(p, 0) for p in POPS) + '\n')
    print('merge: %d SNPs written to %s' % (len(snps), os.path.join(a.out, 'snpinfo_mult_eigen')))


NREF_1KG = {'AFR': 661, 'AMR': 347, 'EAS': 504, 'EUR': 503, 'SAS': 489}


def convert(a):
    from scipy import linalg
    import shutil
    t0 = time.time()
    ref, info = ('1kg', 'snpinfo_mult_1kg_hm3') if os.path.isfile(os.path.join(a.ref_dir, 'snpinfo_mult_1kg_hm3')) else ('ukbb', 'snpinfo_mult_ukbb_hm3')
    os.makedirs(a.out, exist_ok=True)
    shutil.copyfile(os.path.join(a.ref_dir, info), os.path.join(a.out, 'snpinfo_mult_eigen'))
    chroms = []
    for part in a.chroms.split(','):
        lo, _, hi = part.partition('-')
        chroms += list(range(int(lo), int(hi or lo) + 1))
    for pop in a.pops.split(','):
        od = os.path.join(a.out, 'ldeig_%s' % pop.lower()); os.makedirs(od, exist_ok=True)
        for c in chroms:
            src = os.path.join(a.ref_dir, 'ldblk_%s_%s' % (ref, pop.lower()), 'ldblk_%s_chr%d.hdf5' % (ref, c))
            n_rank = n_snp = 0
            with h5py.File(src, 'r') as hs, h5py.File(os.path.join(od, 'ldeig_chr%d.hdf5.tmp' % c), 'w') as h:
                h.attrs['format'] = 'prscsx-eigen-1'; h.attrs['pop'] = pop.upper(); h.attrs['chrom'] = c
                h.attrs['maf'] = -1.0; h.attrs['n_ref'] = NREF_1KG.get(pop.upper(), -1) if ref == '1kg' else -1
                h.attrs['rank_var'] = a.rank_var; h.attrs['converted_from'] = src
                for k in range(1, len(hs) + 1):
                    R = np.array(hs['blk_%d' % k]['ldblk'], dtype=np.float64)
                    sl = hs['blk_%d' % k]['snplist'][:]
                    g = h.create_group('blk_%d' % k)
                    g.attrs['n_ref'] = h.attrs['n_ref']; g.attrs['rank_var'] = a.rank_var
                    if R.size == 0:
                        g.create_dataset('snplist', data=np.array([], dtype='S1')); g.create_dataset('evec', data=np.zeros((0, 0), np.float32))
                        g.create_dataset('eval', data=np.zeros(0)); g.attrs['trace'] = 0.0
                        continue
                    try:
                        _, sv, v = linalg.svd(R)
                    except np.linalg.LinAlgError:
                        _, sv, v = linalg.svd(R, lapack_driver='gesvd')
                    D = (R + np.dot(v.T, np.dot(np.diag(sv), v))) / 2           # PRS-CSx's PSD projection
                    w, Q = np.linalg.eigh(D)
                    w = w[::-1]; Q = Q[:, ::-1]
                    keep = w > EIG_TOL * w[0]
                    Q, w = Q[:, keep], w[keep]
                    trace = float(np.trace(D))
                    if a.rank_var < 1:
                        kk = int(np.searchsorted(np.cumsum(w) / trace, a.rank_var) + 1)
                        Q, w = Q[:, :kk], w[:kk]
                    g.create_dataset('snplist', data=sl); g.create_dataset('evec', data=np.ascontiguousarray(Q, dtype=np.float32))
                    g.create_dataset('eval', data=w); g.attrs['trace'] = trace
                    n_rank += w.size; n_snp += R.shape[0]
            os.replace(os.path.join(od, 'ldeig_chr%d.hdf5.tmp' % c), os.path.join(od, 'ldeig_chr%d.hdf5' % c))
            print('convert: %s chr%d: %d SNPs, total rank %d (%.1f s)' % (pop.upper(), c, n_snp, n_rank, time.time() - t0), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--bfile', required=True); b.add_argument('--pop', required=True); b.add_argument('--chrom', type=int, required=True)
    b.add_argument('--out', required=True); b.add_argument('--blocks_from'); b.add_argument('--blocks'); b.add_argument('--block_snps', type=int)
    b.add_argument('--rank_var', type=float, default=0.999); b.add_argument('--maf', type=float, default=0.01)
    b.add_argument('--max_missing', type=float, default=0.05); b.add_argument('--max_n', type=int); b.add_argument('--seed', type=int, default=1)
    b.add_argument('--verbose', action='store_true')
    m = sub.add_parser('merge'); m.add_argument('--out', required=True)
    c = sub.add_parser('convert')
    c.add_argument('--ref_dir', required=True); c.add_argument('--pops', required=True); c.add_argument('--chroms', required=True)
    c.add_argument('--out', required=True); c.add_argument('--rank_var', type=float, default=1.0)
    a = ap.parse_args(argv)
    {'build': build, 'merge': merge, 'convert': convert}[a.cmd](a)


if __name__ == '__main__':
    main()
