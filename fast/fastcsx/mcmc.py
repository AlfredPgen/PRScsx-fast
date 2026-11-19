"""
Exact, accelerated PRS-CSx Gibbs sampler.

Reproduces mcmc_gtb.mcmc (of the NumPy-2-fixed PRS-CSx) bit for bit for a given seed:
identical random-number stream, identical floating-point operations, identical output files.
Speed comes from
  * numba GIG sampler with random-stream replay (fastcsx.gig),
  * GIL-free LAPACK block updates (fastcsx.lapack) run concurrently over LD blocks,
  * no per-iteration Python overhead (no b x b np.diag, no check_finite, no Python-level sum()).
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy import random

from .gig import sample_psi
from .lapack import block_update, new_workspace
from .lowrank import lowrank_update


def seqsum(x):
    """Left-to-right float64 sum, identical to Python's builtin sum() over a (p,1) numpy array."""
    x = np.asarray(x).ravel()
    if x.size == 0:
        return 0.0
    return float(np.cumsum(x)[-1])


class BlockSolver:
    """
    Block layout of all populations + a thread pool with per-thread LAPACK workspaces.

    All LD blocks of all populations are updated concurrently. This is exact because, given psi and the
    previous sigma_k, the populations' beta updates are independent, and every random number they need can be
    drawn beforehand *in the original order*: for population k, randn(p_k) (== its per-block randn calls)
    followed by standard_gamma((n_k+p_k)/2) (numpy's legacy gamma(shape, scale) is scale*standard_gamma(shape),
    and the number of uniforms/normals it consumes does not depend on scale; verified bit for bit).
    """

    def __init__(self, ld_blk, blk_size, n_threads=1, prepared=None):
        """prepared[pp]: optional list (per non-empty block) of (D, V, lam) from lowrank.prepare_blocks; a block with
        V is not None uses the low-rank update. prepared=None (default) is the exact original path."""
        n_pop = len(ld_blk)
        self.blocks = []                  # (pp, start, size, ld, V, sqrt(lam), offset of e1)  in original order
        self.p, self.r_tot = [], []
        for pp in range(n_pop):
            mm = 0; roff = 0; kk = 0
            prep = None if prepared is None else prepared[pp]
            for ld, b in zip(ld_blk[pp], blk_size[pp]):
                if b == 0:
                    continue
                if prep is not None:
                    D, V, lam = prep[kk]
                else:
                    D, V, lam = ld, None, None
                if V is None:
                    self.blocks.append((pp, mm, b, np.ascontiguousarray(D, dtype=np.float64), None, None, 0))
                else:
                    self.blocks.append((pp, mm, b, None, np.ascontiguousarray(V, dtype=np.float64), np.sqrt(lam), roff))
                    roff += V.shape[1]
                mm += b; kk += 1
            self.p.append(mm); self.r_tot.append(roff)
        self.n_pop = n_pop
        self.max_b = max((x[2] for x in self.blocks if x[4] is None), default=1)
        self.max_br = max((x[2] * x[4].shape[1] for x in self.blocks if x[4] is not None), default=0)
        self.max_b_lr = max((x[2] for x in self.blocks if x[4] is not None), default=0)
        self.n_threads = max(1, int(n_threads))
        self._local = threading.local()
        self.pool = ThreadPoolExecutor(self.n_threads) if self.n_threads > 1 else None
        # largest blocks first for better load balance; results are re-ordered before summation
        self.order = sorted(range(len(self.blocks)), key=lambda k: -self.blocks[k][2])

    def _ws(self):
        ws = getattr(self._local, 'ws', None)
        if ws is None:
            ws = self._local.ws = new_workspace(self.max_b)
            if self.max_br:
                ws['U'] = np.empty(self.max_br); ws['mean'] = np.empty(self.max_b_lr)
        return ws

    def _one(self, k, beta, bhat, invpsi, z, s, e1=None):
        pp, st, b, ld, V, sq, ro = self.blocks[k]
        ws = self._ws()
        if V is not None:                         # low-rank update (not the original's random-number stream)
            r = V.shape[1]
            return lowrank_update(V, sq, invpsi[pp][st:st+b], bhat[pp][st:st+b, 0], e1[pp][ro:ro+r], z[pp][st:st+b],
                                  s[pp], beta[pp][st:st+b, 0], ws['mean'][:b], ws['U'][:b*r].reshape(b, r))
        D = ws['dinvt'][:b*b].reshape(b, b)
        F = ws['F'][:b*b].reshape(b, b)
        info = block_update(ld, invpsi[pp][st:st+b], bhat[pp][st:st+b, 0], z[pp][st:st+b], s[pp], D, F,
                            beta[pp][st:st+b, 0], ws['chars'], ws['ints'])
        if info != 0:
            raise np.linalg.LinAlgError('LAPACK error %d in LD block starting at SNP %d (population %d)' % (info, st, pp))
        bb = beta[pp][st:st+b]
        return np.dot(np.dot(bb.T, D), bb)[0, 0]

    def update(self, beta, bhat, psi_pp, sigma, n, gamma_shape):
        """
        In-place beta update of all populations. Returns (quad, G): quad[pp] = sum_k beta_k'(D_k+Psi_k^-1)beta_k
        (summed in block order) and G[pp] = the standard-gamma variate for population pp's sigma update.
        """
        z, e1, G = [], [], []
        for pp in range(self.n_pop):              # original random-number order (e1 only exists for low-rank blocks)
            z.append(random.randn(self.p[pp]))
            e1.append(random.randn(self.r_tot[pp]) if self.r_tot[pp] > 0 else None)
            G.append(random.standard_gamma(gamma_shape[pp]))
        invpsi = [1.0/psi_pp[pp][:, 0] for pp in range(self.n_pop)]
        s = [float(np.sqrt(sigma[pp]/n[pp])) for pp in range(self.n_pop)]
        nb = len(self.blocks)
        if self.pool is None:
            q = [self._one(k, beta, bhat, invpsi, z, s, e1) for k in range(nb)]
        else:
            futs = {k: self.pool.submit(self._one, k, beta, bhat, invpsi, z, s, e1) for k in self.order}
            q = [futs[k].result() for k in range(nb)]
        quad = [0.0]*self.n_pop
        for k in range(nb):                       # same summation order as the original
            quad[self.blocks[k][0]] += q[k]
        return quad, G

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def mcmc(a, b, phi, snp_dict, beta_mrg, frq_dict, idx_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin,
         pop, chrom, out_dir, out_name, meta, write_pst, seed, n_threads=1, log=print, solvers=None):
    log('... MCMC ...')
    if seed is not None:
        random.seed(seed)

    n_pst = n_iter//thin - n_burnin//thin
    n_pop = len(pop)
    p_tot = len(snp_dict['SNP'])

    own = solvers is None
    if own:
        solvers = BlockSolver([ld_blk[pp] for pp in range(n_pop)], [blk_size[pp] for pp in range(n_pop)], n_threads)
    p, het, idx = {}, {}, {}
    for pp in range(n_pop):
        p[pp] = len(beta_mrg[pp])
        het[pp] = np.sqrt(2.0*frq_dict[pp]*(1.0-frq_dict[pp]))
        idx[pp] = np.asarray(idx_dict[pp], dtype=np.int64)
        assert solvers.p[pp] == p[pp]
    gamma_shape = [(n[pp]+p[pp])/2.0 for pp in range(n_pop)]

    n_grp = np.zeros((p_tot, 1))
    for pp in range(n_pop):
        n_grp[idx[pp]] += 1
    gig_p = a-0.5*n_grp[:, 0]

    beta = {pp: np.zeros((p[pp], 1)) for pp in range(n_pop)}
    sigma = {pp: 1.0 for pp in range(n_pop)}
    psi = np.ones((p_tot, 1))
    if phi is None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False
    keep_pst = (write_pst == 'TRUE') and (meta == 'TRUE')
    if keep_pst:
        beta_pst = {pp: np.zeros((p[pp], n_pst)) for pp in range(n_pop)}
    beta_est = {pp: np.zeros((p[pp], 1)) for pp in range(n_pop)}
    beta_sq_est = {pp: np.zeros((p[pp], 1)) for pp in range(n_pop)}
    sigma_est = {pp: 0.0 for pp in range(n_pop)}
    psi_est = np.zeros((p_tot, 1))
    phi_est = 0.0

    qq = 0
    import time as _time
    _t0 = _time.perf_counter()
    for itr in range(1, n_iter+1):
        if itr % 100 == 0:
            log('--- iter-' + str(itr) + ' ---')

        psi_pp = [psi[idx[pp]] for pp in range(n_pop)]
        quad, G = solvers.update(beta, beta_mrg, psi_pp, sigma, n, gamma_shape)
        for pp in range(n_pop):
            err = max(n[pp]/2.0*(1.0-2.0*seqsum(beta[pp]*beta_mrg[pp])+quad[pp]),
                      n[pp]/2.0*seqsum(beta[pp]**2/psi_pp[pp]))
            sigma[pp] = 1.0/((1.0/err)*G[pp])     # == 1.0/random.gamma(shape, 1.0/err), drawn in solvers.update

        delta = random.gamma(a+b, 1.0/(psi+phi))

        xx = np.zeros((p_tot, 1))
        for pp in range(n_pop):
            xx[idx[pp]] += n[pp]*beta[pp]**2/sigma[pp]

        psi[:, 0] = sample_psi(gig_p, 2.0*delta[:, 0], xx[:, 0])
        psi[psi > 1] = 1.0

        if phi_updt:
            w = random.gamma(1.0, 1.0/(phi+1.0))
            phi = random.gamma(p_tot*b+0.5, 1.0/(seqsum(delta)+w))

        if (itr > n_burnin) and (itr % thin == 0):
            for pp in range(n_pop):
                beta_est[pp] = beta_est[pp]+beta[pp]/n_pst
                beta_sq_est[pp] = beta_sq_est[pp]+beta[pp]**2/n_pst
                sigma_est[pp] = sigma_est[pp]+sigma[pp]/n_pst
            psi_est = psi_est+psi/n_pst
            phi_est = phi_est+phi/n_pst
            if keep_pst:
                for pp in range(n_pop):
                    beta_pst[pp][:, [qq]] = beta[pp]
                qq += 1

    _dt = _time.perf_counter() - _t0
    log('... MCMC: %d iterations in %.1f s (%.4f s/iteration) ...' % (n_iter, _dt, _dt / n_iter))
    if own:
        solvers.close()

    for pp in range(n_pop):
        beta_est[pp] /= het[pp]
        beta_sq_est[pp] /= het[pp]**2
    if keep_pst:
        for pp in range(n_pop):
            beta_pst[pp] /= het[pp]

    if meta == 'TRUE':
        vv = np.zeros((p_tot, 1)); zz = np.zeros((p_tot, 1))
        for pp in range(n_pop):
            vv[idx[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)
            zz[idx[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)*beta_est[pp]
        mu = zz/vv
        if write_pst == 'TRUE':
            vv = np.zeros((p_tot, 1)); zz = np.zeros((p_tot, n_pst))
            for pp in range(n_pop):
                vv[idx[pp]] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)
                zz[idx[pp], :] += 1.0/(beta_sq_est[pp]-beta_est[pp]**2)*beta_pst[pp]
            mu_pst = zz/vv

    files = []
    for pp in range(n_pop):
        if phi_updt:
            eff_file = out_dir + '/' + '%s_%s_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, pop[pp], a, b, chrom)
        else:
            eff_file = out_dir + '/' + '%s_%s_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, pop[pp], a, b, phi, chrom)
        with open(eff_file, 'w') as ff:
            for ii, bv in zip(idx_dict[pp], beta_est[pp][:, 0]):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp_dict['SNP'][ii], snp_dict['BP'][ii],
                                                     snp_dict['A1'][ii], snp_dict['A2'][ii], bv))
        files.append(eff_file)

    if meta == 'TRUE':
        if phi_updt:
            eff_file = out_dir + '/' + '%s_META_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (out_name, a, b, chrom)
        else:
            eff_file = out_dir + '/' + '%s_META_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (out_name, a, b, phi, chrom)
        with open(eff_file, 'w') as ff:
            if write_pst == 'TRUE':
                fmt = '%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n'
                for snp, bp, a1, a2, row in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu_pst):
                    ff.write(fmt % (chrom, snp, bp, a1, a2, *row))
            else:
                for snp, bp, a1, a2, bv in zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'], mu[:, 0]):
                    ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, bv))
        files.append(eff_file)

    if phi_updt:
        log('... Estimated global shrinkage parameter: %1.2e ...' % phi_est)
    log('... Done ...')
    return files
