"""
Input parsing for PRS-CSx-fast. Produces exactly the same data structures (values and order) as the
original parse_genet.py, but in linear time:

* list membership tests (`snp in list`, `list.index`) replaced by sets/dicts (the original is O(p^2));
* the sumstats file is read once per run (not twice per chromosome) and kept only for SNPs present in
  the reference; per-chromosome results are identical because only SNPs in the reference can match;
* the reference SNP-info file is read once per run;
* p-value -> z conversion is vectorised (same ndtri ufunc, same elementwise operations);
* only LD blocks that contain at least one retained SNP are read from the HDF5 file;
* optional on-disk cache of the processed (subset, sign-flipped, PSD-projected) LD blocks.
"""
import hashlib
import os

import h5py
import numpy as np
from scipy import linalg
from scipy.stats import norm

ATGC = ('A', 'T', 'G', 'C')
MAPPING = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
POPS = ('AFR', 'AMR', 'EAS', 'EUR', 'SAS')
LD_CACHE_KEYS = {}        # (pop, chrom) -> path of the LD cache file (used to store eigen-decompositions next to it)


def find_ref(ref_dir):
    for ref, fn in (('1kg', 'snpinfo_mult_1kg_hm3'), ('ukbb', 'snpinfo_mult_ukbb_hm3'), ('eigen', 'snpinfo_mult_eigen')):
        if os.path.isfile(os.path.join(ref_dir, fn)):
            return ref, os.path.join(ref_dir, fn)
    return None, None


def read_ref_all(ref_file, chroms, pops=POPS):
    """{chrom: ref_dict} for the requested chromosomes (one pass over the file). Only the FRQ/FLP columns of the
    requested populations are kept (memory)."""
    chroms = set(int(c) for c in chroms)
    pops = [p.upper() for p in pops]
    cols = [(p, 5 + POPS.index(p), 10 + POPS.index(p)) for p in pops]

    def empty():
        return {'CHR': [], 'SNP': [], 'BP': [], 'A1': [], 'A2': [], **{'FRQ_' + p: [] for p in pops},
                **{'FLP_' + p: [] for p in pops}}
    out = {}
    with open(ref_file) as ff:
        next(ff)
        for line in ff:
            ll = line.split()
            c = int(ll[0])
            if c not in chroms:
                continue
            d = out.get(c)
            if d is None:
                d = out[c] = empty()
            d['CHR'].append(c); d['SNP'].append(ll[1]); d['BP'].append(int(ll[2]))
            d['A1'].append(ll[3]); d['A2'].append(ll[4])
            for p, jf, jl in cols:
                d['FRQ_' + p].append(float(ll[jf]))
                d['FLP_' + p].append(int(ll[jl]))
    for c in chroms:
        out.setdefault(c, empty())
    return out


def parse_bim(bim_prefix, chrom):
    vld = {'SNP': [], 'A1': [], 'A2': []}
    with open(bim_prefix + '.bim') as ff:
        for line in ff:
            ll = line.split()
            code = ll[0][3:] if ll[0].lower().startswith('chr') else ll[0]
            if code.isdigit() and int(code) == chrom:
                vld['SNP'].append(ll[1]); vld['A1'].append(ll[4]); vld['A2'].append(ll[5])
    return vld


class SumStats:
    """One pass over a sumstats file; keeps ATGC lines whose SNP is in `keep_snps`. Numeric columns are stored as
    float64 (bitwise the values float() gives the original); a value that cannot be parsed raises the original's
    ValueError only if that line is actually used (the original only parses matching lines)."""

    def __init__(self, sst_file, keep_snps):
        self.file = sst_file
        self.n_atgc = 0
        snp, a1, a2, v3, v4 = [], [], [], [], []
        self.bad = {}
        nan = float('nan')
        with open(sst_file) as ff:
            header = [c.upper() for c in next(ff).split()]
            for line in ff:
                ll = line.split()
                if ll[1] in ATGC and ll[2] in ATGC:
                    self.n_atgc += 1
                    if ll[0] in keep_snps:
                        try:
                            x3 = float(ll[3]); x4 = float(ll[4])
                        except ValueError:
                            self.bad[len(snp)] = (ll[3], ll[4]); x3 = x4 = nan
                        snp.append(ll[0]); a1.append(ll[1]); a2.append(ll[2]); v3.append(x3); v4.append(x4)
        self.header = header
        self.snp, self.a1, self.a2 = snp, a1, a2
        self.v3 = np.array(v3, dtype=np.float64); self.v4 = np.array(v4, dtype=np.float64)


def _std_effects(sst, rows, n_subj):
    """beta_std for the given row indices, identical to the original per-line formulas."""
    n_sqrt = np.sqrt(n_subj)
    hdr = sst.header
    for i in rows:
        if i in sst.bad:                                   # same error as the original's float(ll[3]) / float(ll[4])
            float(sst.bad[i][0]); float(sst.bad[i][1])
    rows = np.asarray(rows, dtype=np.int64)
    v3 = sst.v3[rows]
    v4 = sst.v4[rows]
    if 'BETA' in hdr:
        beta = v3
    elif 'OR' in hdr:
        # per-element np.log on Python floats, as in the original (vectorised SIMD log is not
        # guaranteed to be bit-identical to the scalar path)
        beta = np.array([np.log(float(x)) for x in v3.tolist()], dtype=np.float64)
    if 'SE' in hdr:
        if np.any(v4 == 0):
            bad = [sst.snp[int(rows[i])] for i in np.flatnonzero(v4 == 0)[:5]]
            # the original raises ZeroDivisionError here; fail with an informative message instead of producing inf
            raise ValueError('SE = 0 in %s for %d SNP(s), e.g. %s: remove these lines or use the P-value format'
                             % (sst.file, int(np.sum(v4 == 0)), ', '.join(bad)))
        return beta/v4/n_sqrt
    elif 'P' in hdr:
        p = np.maximum(v4, 1e-323)
        return np.sign(beta)*np.abs(norm.ppf(p/2.0))/n_sqrt


def parse_sumstats(ref_dict, vld_dict, sst, pop, n_subj, log=print):
    """Same output as parse_genet.parse_sumstats (sst: a SumStats object)."""
    log('... parse %s sumstats file: %s ...' % (pop.upper(), sst.file))
    log('... %d SNPs read from %s ...' % (sst.n_atgc, sst.file))
    frq_col = ref_dict['FRQ_' + pop.upper()]
    idx = [ii for ii, f in enumerate(frq_col) if f > 0]
    ref_snps = set(ref_dict['SNP'][ii] for ii in idx)
    snp_ref = [ref_dict['SNP'][ii] for ii in idx]
    a1_ref = [ref_dict['A1'][ii] for ii in idx]
    a2_ref = [ref_dict['A2'][ii] for ii in idx]
    M = MAPPING
    vld_snp = set(zip(vld_dict['SNP'], vld_dict['A1'], vld_dict['A2']))
    ref_snp = set(zip(snp_ref, a1_ref, a2_ref)) | set(zip(snp_ref, a2_ref, a1_ref)) | \
        set(zip(snp_ref, [M[a] for a in a1_ref], [M[a] for a in a2_ref])) | \
        set(zip(snp_ref, [M[a] for a in a2_ref], [M[a] for a in a1_ref]))
    rows = [i for i, s in enumerate(sst.snp) if s in ref_snps]
    s_snp = [sst.snp[i] for i in rows]; s_a1 = [sst.a1[i] for i in rows]; s_a2 = [sst.a2[i] for i in rows]
    sst_snp = set(zip(s_snp, s_a1, s_a2)) | set(zip(s_snp, s_a2, s_a1)) | \
        set(zip(s_snp, [M[a] for a in s_a1], [M[a] for a in s_a2])) | \
        set(zip(s_snp, [M[a] for a in s_a2], [M[a] for a in s_a1]))
    comm_snp = vld_snp & ref_snp & sst_snp
    log('... %d common SNPs in the %s reference, %s sumstats, and validation set ...' % (len(comm_snp), pop.upper(), pop.upper()))

    # which rows match (same / swapped orientation); later rows overwrite earlier ones (dict.update)
    sel, sign = [], []
    for i, s, a1, a2 in zip(rows, s_snp, s_a1, s_a2):
        if (s, a1, a2) in comm_snp or (s, M[a1], M[a2]) in comm_snp:
            sel.append(i); sign.append(1.0)
        elif (s, a2, a1) in comm_snp or (s, M[a2], M[a1]) in comm_snp:
            sel.append(i); sign.append(-1.0)
    sst_eff = {}
    if sel:
        bstd = _std_effects(sst, sel, n_subj)
        for i, sg, bv in zip(sel, sign, bstd.tolist()):
            sst_eff[sst.snp[i]] = bv if sg > 0 else -1*bv

    out = {'SNP': [], 'FRQ': [], 'BETA': [], 'FLP': []}
    frq_p = ref_dict['FRQ_' + pop.upper()]; flp_p = ref_dict['FLP_' + pop.upper()]
    for ii, snp in enumerate(ref_dict['SNP']):
        if snp in sst_eff:
            out['SNP'].append(snp); out['BETA'].append(sst_eff[snp])
            a1 = ref_dict['A1'][ii]; a2 = ref_dict['A2'][ii]
            if (snp, a1, a2) in comm_snp or (snp, M[a1], M[a2]) in comm_snp:
                out['FRQ'].append(frq_p[ii]); out['FLP'].append(flp_p[ii])
            elif (snp, a2, a1) in comm_snp or (snp, M[a2], M[a1]) in comm_snp:
                out['FRQ'].append(1-frq_p[ii]); out['FLP'].append(-1*flp_p[ii])
    return out


def _psd(blk, log=print):
    try:
        _, s, v = linalg.svd(blk)                        # LAPACK gesdd, exactly as the original
    except np.linalg.LinAlgError:
        # gesdd can fail to converge on strongly rank-deficient blocks (seen on simulated panels with n_ref << block
        # size); the original PRS-CSx then stops with "SVD did not converge". Fall back to the slower, more robust gesvd.
        log('... WARNING: SVD (gesdd) did not converge for an LD block of %d SNPs; using gesvd instead ...' % blk.shape[0])
        _, s, v = linalg.svd(blk, lapack_driver='gesvd')
    h = np.dot(v.T, np.dot(np.diag(s), v))
    return (blk+h)/2


def parse_ldblk(ldblk_dir, sst_dict, pop, chrom, ref, cache_dir=None, log=print):
    """Same output as parse_genet.parse_ldblk: list of processed blocks (np.array([]) if empty) + sizes."""
    log('... parse %s reference LD on chromosome %d ...' % (pop.upper(), chrom))
    chr_name = os.path.join(ldblk_dir, 'ldblk_%s_%s' % (ref, pop.lower()), 'ldblk_%s_chr%d.hdf5' % (ref, chrom))
    keep = set(sst_dict['SNP'])
    with h5py.File(chr_name, 'r') as hdf:
        n_blk = len(hdf)
        snp_blk = [[bb.decode('UTF-8') for bb in hdf['blk_%d' % b]['snplist'][:]] for b in range(1, n_blk+1)]
        sel = [[ii for ii, s in enumerate(sb) if s in keep] for sb in snp_blk]
        blk_size = [len(x) for x in sel]

        key = None
        if cache_dir:
            st = os.stat(chr_name)
            hh = hashlib.sha1(('%s|%d|%d|%s|%d' % (chr_name, st.st_size, int(st.st_mtime), pop, chrom)).encode())
            hh.update(('\n'.join('%s %d' % (s, f) for s, f in zip(sst_dict['SNP'], sst_dict['FLP']))).encode())
            key = os.path.join(cache_dir, 'ldcache_%s_chr%d_%s.npz' % (pop.lower(), chrom, hh.hexdigest()[:20]))
            LD_CACHE_KEYS[(pop.upper(), chrom)] = key
            if os.path.isfile(key):
                z = np.load(key)
                ld = [z['b%d' % k] if blk_size[k] > 0 else np.array([]) for k in range(n_blk)]
                log('... loaded processed LD from cache %s ...' % key)
                return ld, blk_size

        ld = []
        mm = 0
        for b in range(n_blk):
            idx = sel[b]
            if idx:
                raw = np.array(hdf['blk_%d' % (b+1)]['ldblk'])
                flip = sst_dict['FLP'][mm:mm+len(idx)]
                blk = raw[np.ix_(idx, idx)]*np.outer(flip, flip)
                ld.append(_psd(blk, log))
                mm += len(idx)
            else:
                ld.append(np.array([]))
    if key:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = key + '.tmp.npz'
        np.savez(tmp, **{'b%d' % k: ld[k] for k in range(n_blk) if blk_size[k] > 0})
        os.replace(tmp, key)
    return ld, blk_size


def align_ldblk(ref_dict, vld_dict, sst_dict, n_pop, chrom, log=print):
    """Same output as parse_genet.align_ldblk."""
    log('... align reference LD on chromosome %d across populations ...' % chrom)
    sst_sets = [set(sst_dict[pp]['SNP']) for pp in range(n_pop)]
    vld_first = {}
    for i, s in enumerate(vld_dict['SNP']):
        vld_first.setdefault(s, i)
    snp_dict = {'CHR': [], 'SNP': [], 'BP': [], 'A1': [], 'A2': []}
    for ii, snp in enumerate(ref_dict['SNP']):
        for pp in range(n_pop):
            if snp in sst_sets[pp]:
                snp_dict['SNP'].append(snp); snp_dict['CHR'].append(ref_dict['CHR'][ii]); snp_dict['BP'].append(ref_dict['BP'][ii])
                i = vld_first[snp]
                snp_dict['A1'].append(vld_dict['A1'][i]); snp_dict['A2'].append(vld_dict['A2'][i])
                break
    log('... %d valid SNPs across populations ...' % len(snp_dict['SNP']))
    beta_dict, frq_dict, idx_dict = {}, {}, {}
    for pp in range(n_pop):
        beta_dict[pp] = np.array(sst_dict[pp]['BETA'], ndmin=2).T
        frq_dict[pp] = np.array(sst_dict[pp]['FRQ'], ndmin=2).T
        idx_dict[pp] = [ii for ii, snp in enumerate(snp_dict['SNP']) if snp in sst_sets[pp]]
    return snp_dict, beta_dict, frq_dict, idx_dict


def parse_eigblk(ref_dir, sst_dict, pop, chrom, log=print):
    """
    Eigen panel (docs/04_eigen_panel_format.md): per block, the rows of V for the retained SNPs, sign-flipped to the
    target alleles and re-orthonormalised (thin SVD of V_sub diag(lam)^1/2), so that V' diag(lam') V'^T is exactly the
    LD matrix of the retained SNPs. Returns (blk_size, eig) with eig[k] = (V', lam') or None for empty blocks.
    """
    log('... parse %s eigen LD panel on chromosome %d ...' % (pop.upper(), chrom))
    fn = os.path.join(ref_dir, 'ldeig_%s' % pop.lower(), 'ldeig_chr%d.hdf5' % chrom)
    keep = set(sst_dict['SNP'])
    blk_size, eig = [], []
    mm = 0
    with h5py.File(fn, 'r') as h:
        n_blk = sum(1 for k in h.keys() if k.startswith('blk_'))
        for b in range(1, n_blk + 1):
            g = h['blk_%d' % b]
            snps = [x.decode() for x in g['snplist'][:]]
            idx = [i for i, s in enumerate(snps) if s in keep]
            blk_size.append(len(idx))
            if not idx:
                eig.append(None)
                continue
            V = np.asarray(g['evec'][:], dtype=np.float64)[idx]
            lam = np.asarray(g['eval'][:], dtype=np.float64)
            flip = np.array(sst_dict['FLP'][mm:mm + len(idx)], dtype=np.float64)
            U, S, _ = np.linalg.svd(V * flip[:, None] * np.sqrt(lam)[None, :], full_matrices=False)
            l2 = S ** 2
            k = l2 > 1e-13 * l2[0] if l2.size and l2[0] > 0 else np.zeros(l2.size, bool)
            eig.append((np.ascontiguousarray(U[:, k]), l2[k]))
            mm += len(idx)
    return blk_size, eig
