"""
PRS-CSx-improved Gibbs sampler.

Model (K ancestries, SNP j, ancestry k; standardised genotypes/phenotypes, reference LD D_k):
    bhat_k | beta_k, s2_k ~ "N(D_k beta_k, s2_k D_k / N_k)"           (PRS-CS summary-statistic likelihood)
    beta_jk | psi_j, s2_k ~ N(0, s2_k psi_j / Nbar_k)                   (coupling scale, see below)
    psi_j ~ Gamma(a, delta_j),  delta_j ~ Gamma(b, phi),  p(s2_k) ∝ 1/s2_k,  phi fixed or sqrt(phi) ~ C+(0,1)
Coupling scale (default 'snr', chosen by the WP4 stress test):
    coupling='snr'    : Nbar_k = N_k       (original PRS-CSx: shared signal-to-noise ratio)
    coupling='effect' : Nbar_k = max_k N_k (shared effect-size variance; identical to PRS-CSx for one ancestry)
    coupling='learned': Nbar_k = max_k N_k / tau_k, tau_k ~ IG(1, 1) learned, tau = 1 for the largest GWAS
                        (tau_k | . ~ IG(1 + p_k/2, 1 + Nmax/(2 s2_k) sum_j beta_jk^2/psi_j); derivation in improved/README.md)
Full conditionals (kappa_k = Nbar_k / N_k):
    beta_k | . ~ N( (D_k + R_k)^-1 bhat_k , (s2_k/N_k)(D_k + R_k)^-1 )
    s2_k | .   ~ IG( (N_k+p_k)/2 , max( N_k/2 [1 - 2 beta'bhat + beta'(D + R_k) beta] , N_k/2 beta' R_k beta ) ),  R_k = diag(ridge_k)
    delta_j    ~ Gamma(a+b, psi_j + phi)
    psi_j | .  ~ GIG( a - K_j/2 , 2 delta_j , sum_k Nbar_k beta_jk^2 / s2_k ),  then psi_j <- min(psi_j, psi_max*max_k kappa_k)
Robustness floor (PRS-CSx clips psi_j <= 1, i.e. prior variance <= sampling variance s2_k/N_k): applied per ancestry,
    ridge_jk = kappa_k / psi_eff_jk = max(kappa_k/psi_j, 1/psi_max)     (psi_eff_jk = min(psi_j, psi_max*kappa_k))
so the floor is the same as in PRS-CSx in every ancestry, and coupling='snr' is exactly the PRS-CSx model.
    phi (auto) : w ~ Gamma(1, phi+1);  phi ~ Gamma(M b + 1/2, sum_j delta_j + w)
Estimator:
    rb=True : posterior mean = average over ALL post-burn-in iterations of E[beta | psi, s2, data] = (D + kappa Psi^-1)^-1 bhat
              (Rao-Blackwellised; the conditional mean is a by-product of the Cholesky factor already computed)
    rb=False: average of thinned draws (original estimator)
    Posterior variances for --meta use all post-burn-in draws (unthinned).
Chains: ChainState runs one chain with its own random streams; several chains are averaged (law of total variance for
the variances), optionally interleaved with automatic stopping on the between-chain R-hat (AutoStopRule), and a
phi grid can be warm-started from the previous phi's final state.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from fastcsx.gig import sample_psi
from fastcsx.lowrank import lowrank_update
from .kernel import block_update_rb, block_update_rb32, new_workspace, new_workspace32


def lw_shrinkage(R, n_ref):
    """Schafer-Strimmer / Ledoit-Wolf optimal shrinkage intensity of a sample correlation matrix toward I."""
    b = R.shape[0]
    if b < 2:
        return 0.0
    off = ~np.eye(b, dtype=bool)
    r2 = R[off]**2
    num = np.sum((1.0-r2)**2)/(n_ref-1.0)
    den = np.sum(r2)
    return float(min(1.0, max(0.0, num/den))) if den > 0 else 0.0


class Solver:
    def __init__(self, ld_blk, blk_size, n_threads=1, precision='single', prepared=None):
        """prepared[pp]: optional list (per non-empty block) of (D, V, lam) from fastcsx.lowrank.prepare_blocks;
        blocks with V use the low-rank (Woodbury) update."""
        self.single = precision == 'single'
        dt = np.float32 if self.single else np.float64
        self.dt = dt
        self.blocks = []
        self.p, self.r_tot = [], []
        for pp in range(len(ld_blk)):
            mm = 0; roff = 0; kk = 0
            prep = None if prepared is None else prepared[pp]
            for ld, b in zip(ld_blk[pp], blk_size[pp]):
                if b == 0:
                    continue
                D, V, lam = prep[kk] if prep is not None else (ld, None, None)
                if V is None:
                    self.blocks.append((pp, mm, b, np.ascontiguousarray(D, dtype=dt), None, None, 0))
                else:
                    self.blocks.append((pp, mm, b, None, np.ascontiguousarray(V, dtype=dt), np.sqrt(lam).astype(dt), roff))
                    roff += V.shape[1]
                mm += b; kk += 1
            self.p.append(mm); self.r_tot.append(roff)
        self.n_pop = len(ld_blk)
        self.max_b = max((x[2] for x in self.blocks if x[4] is None), default=1)
        self.max_br = max((x[2] * x[4].shape[1] for x in self.blocks if x[4] is not None), default=0)
        self._local = threading.local()
        self.pool = ThreadPoolExecutor(n_threads) if n_threads > 1 else None
        self.order = sorted(range(len(self.blocks)), key=lambda k: -self.blocks[k][2])

    def _ws(self):
        ws = getattr(self._local, 'ws', None)
        if ws is None:
            ws = self._local.ws = (new_workspace32 if self.single else new_workspace)(self.max_b)
            if self.max_br:
                ws['U'] = np.empty(self.max_br, dtype=self.dt)
        return ws

    def _one(self, k, beta, mean, bhat, ridge, z, s, e1=None):
        pp, st, b, ld, V, sq, ro = self.blocks[k]
        ws = self._ws()
        if V is not None:
            r = V.shape[1]
            return lowrank_update(V, sq, ridge[pp][st:st+b], bhat[pp][st:st+b, 0], e1[pp][ro:ro+r], z[pp][st:st+b], s[pp],
                                  beta[pp][st:st+b, 0], mean[pp][st:st+b, 0], ws['U'][:b*r].reshape(b, r))
        fn = block_update_rb32 if self.single else block_update_rb
        info, q = fn(ld, ridge[pp][st:st+b], bhat[pp][st:st+b, 0], z[pp][st:st+b], s[pp],
                     ws['F'][:b*b].reshape(b, b), ws['rhs'][:2*b].reshape(2, b),
                     beta[pp][st:st+b, 0], mean[pp][st:st+b, 0], ws['chars'], ws['ints'])
        if info != 0:
            raise np.linalg.LinAlgError('LAPACK error %d in LD block at SNP %d (population %d)' % (info, st, pp))
        return q

    def update(self, rng, beta, mean, bhat, ridge, sigma, n):
        z = [rng.standard_normal(self.p[pp]) for pp in range(self.n_pop)]
        e1 = [rng.standard_normal(self.r_tot[pp]) if self.r_tot[pp] else None for pp in range(self.n_pop)]
        s = [float(np.sqrt(sigma[pp]/n[pp])) for pp in range(self.n_pop)]
        if self.pool is None:
            q = [self._one(k, beta, mean, bhat, ridge, z, s, e1) for k in range(len(self.blocks))]
        else:
            futs = {k: self.pool.submit(self._one, k, beta, mean, bhat, ridge, z, s, e1) for k in self.order}
            q = [futs[k].result() for k in range(len(self.blocks))]
        quad = [0.0]*self.n_pop
        for k, v in enumerate(q):
            quad[self.blocks[k][0]] += v
        return quad

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


AUTO_STOP_EVERY = 200      # --auto_stop: check every 200 iterations after burn-in ...
AUTO_STOP_RHAT = 1.05      # ... whether the 99th percentile (over SNPs and ancestries) of the between-chain R-hat
AUTO_STOP_CHECKS = 3       # ... is below 1.05, and stop after 3 consecutive passing checks (n_iter = maximum)


def mcmc_improved(a, b, phi, snp_dict, beta_mrg, frq_dict, idx_dict, n, solver, n_iter, n_burnin, thin, pop, chrom,
                  out_dir, out_name, meta, write_pst, seed, coupling='snr', rb=True, psi_max=1.0, log=print,
                  n_chains=1, auto_stop=False, state_io=None, _fix_sigma2=None, _trace=None):
    """Runs n_chains independent chains (seeds seed, seed+1000, ...), combines and writes the output files.
    auto_stop: run the chains interleaved and stop once converged (see AUTO_STOP_*); n_iter is then the maximum.
    state_io: dict for warm starts over a phi grid; state_io['states'] (if present) are the chains' initial states and
    are replaced by the final states."""
    init = (state_io or {}).get('states') or [None]*n_chains
    chains = []
    for c in range(n_chains):
        sd = None if seed is None else seed + 1000*c
        chains.append(ChainState(a, b, phi, beta_mrg, frq_dict, idx_dict, n, solver, n_iter, n_burnin, thin, pop,
                                 len(snp_dict['SNP']), sd, coupling, rb, psi_max, meta, write_pst, _fix_sigma2, _trace,
                                 init=init[c] if c < len(init) else None))
    log('... MCMC (improved: coupling=%s, rao-blackwell=%s, psi_max=%g%s%s) ...'
        % (coupling, rb, psi_max, ', chains=%d' % n_chains if n_chains > 1 else '',
           ', warm start' if init[0] is not None else ''))
    t0 = time.perf_counter()
    if not auto_stop:
        for c, ch in enumerate(chains):
            if n_chains > 1:
                log('... chain %d/%d (seed %s) ...' % (c+1, n_chains, None if seed is None else seed + 1000*c))
            t1 = time.perf_counter()
            for itr in range(1, n_iter+1):
                if itr % 500 == 0:
                    log('--- iter-%d ---' % itr)
                ch.step()
            dt = time.perf_counter() - t1
            log('... MCMC: %d iterations in %.1f s (%.4f s/iteration) ...' % (n_iter, dt, dt / n_iter))
    else:
        rule = AutoStopRule(n_burnin)
        itr = 0
        for itr in range(1, n_iter+1):
            if itr % 500 == 0:
                log('--- iter-%d ---' % itr)
            for ch in chains:
                ch.step()
            r99 = rule.check(itr, chains)
            if r99 is not None:
                log('... auto_stop check at iteration %d: 99th percentile R-hat %.4f (%d/%d passing) ...'
                    % (itr, r99, rule.stable, AUTO_STOP_CHECKS))
                if rule.converged:
                    break
        dt = time.perf_counter() - t0
        if rule.converged:
            log('... auto_stop: converged, stopped at iteration %d of at most %d ...' % (itr, n_iter))
        else:
            log('... auto_stop WARNING: R-hat criterion not met after %d iterations (the maximum, --n_iter); '
                'consider a larger --n_iter ...' % n_iter)
        log('... MCMC: %d iterations x %d chains in %.1f s (%.4f s/iteration/chain) ...'
            % (itr, n_chains, dt, dt / (itr * n_chains)))
    res = [ch.result(log) for ch in chains]
    if state_io is not None:
        state_io['states'] = [ch.export_state() for ch in chains]
    n_pop = len(pop)
    comb = {'beta_est': {}, 'var_est': {}, 'phi_updt': res[0]['phi_updt'], 'phi': res[0]['phi'], 'n_pst': res[0]['n_pst'],
            'n_keep': res[0]['n_keep']}
    for pp in range(n_pop):
        means = np.array([r['beta_est'][pp] for r in res])          # (C, p, 1)
        within = np.array([r['var_est'][pp] for r in res])
        comb['beta_est'][pp] = means.mean(0)
        comb['var_est'][pp] = within.mean(0) + (means.var(0) if n_chains > 1 else 0.0)   # law of total variance
        if n_chains > 1:
            rhat = _rhat(means[:, :, 0], within[:, :, 0], res[0]['n_keep'])
            big = np.abs(comb['beta_est'][pp][:, 0]) >= np.quantile(np.abs(comb['beta_est'][pp][:, 0]), 0.9)
            log('... %s R-hat over chains: median %.3f, 99th pct %.3f (top-10%% |beta| SNPs: median %.3f) ...'
                % (pop[pp], np.median(rhat), np.quantile(rhat, 0.99), np.median(rhat[big])))
    if res[0].get('beta_pst') is not None:
        comb['beta_pst'] = {pp: np.concatenate([r['beta_pst'][pp] for r in res], axis=1) for pp in range(n_pop)}
        comb['n_pst'] = comb['beta_pst'][0].shape[1]
    if comb['phi_updt']:
        log('... Estimated global shrinkage parameter: %1.2e ...' % float(np.mean([r['phi_est'] for r in res])))
    comb['tau_est'] = np.mean([r['tau_est'] for r in res], axis=0)
    return _write(a, b, comb, snp_dict, idx_dict, pop, chrom, out_dir, out_name, meta, write_pst, log)


class AutoStopRule:
    """--auto_stop: after burn-in, every AUTO_STOP_EVERY iterations compute the 99th percentile (over SNPs, worst
    ancestry) of the between-chain R-hat; converged after AUTO_STOP_CHECKS consecutive values below AUTO_STOP_RHAT."""

    def __init__(self, n_burnin, every=AUTO_STOP_EVERY, tol=AUTO_STOP_RHAT, checks=AUTO_STOP_CHECKS):
        self.n_burnin, self.every, self.tol, self.checks = n_burnin, every, tol, checks
        self.stable, self.converged, self.history = 0, False, []

    def check(self, itr, chains):
        """Call after every iteration; returns the 99th-percentile R-hat at check iterations, else None."""
        if itr <= self.n_burnin or (itr - self.n_burnin) % self.every != 0:
            return None
        r99 = float(max(np.quantile(between_chain_rhat(chains, pp), 0.99) for pp in range(chains[0].n_pop)))
        self.stable = self.stable + 1 if r99 < self.tol else 0
        self.converged = self.stable >= self.checks
        self.history.append((itr, r99))
        return r99


def _rhat(means, within, nk):
    """Between-chain R-hat per SNP (Gelman & Rubin 1992, without splitting): means (C, p) = each chain's estimate
    (the reported posterior mean: Rao-Blackwellised with --rb=TRUE), within (C, p) = each chain's posterior variance."""
    W = np.maximum(within.mean(0), 1e-300)
    B = means.var(0, ddof=1)
    return np.sqrt(((nk-1.0)/nk*W + B)/W)


def between_chain_rhat(chains, pp):
    """R-hat of ancestry pp from the running sums of interleaved chains (same quantity as printed at the end)."""
    nk = chains[0].n_keep
    means = np.array([ch.running_mean(pp) for ch in chains])
    within = np.array([ch.running_var(pp) for ch in chains])
    return _rhat(means, within, nk)


def _run_chain(a, b, phi, snp_dict, beta_mrg, frq_dict, idx_dict, n, solver, n_iter, n_burnin, thin, pop, seed,
               coupling, rb, psi_max, log, meta, write_pst, _fix_sigma2=None, _trace=None, init=None):
    """One chain run to n_iter (used by the benchmarks); returns the per-chain result dict."""
    ch = ChainState(a, b, phi, beta_mrg, frq_dict, idx_dict, n, solver, n_iter, n_burnin, thin, pop,
                    len(snp_dict['SNP']), seed, coupling, rb, psi_max, meta, write_pst, _fix_sigma2, _trace, init=init)
    for _ in range(n_iter):
        ch.step()
    r = ch.result(log)
    r['state'] = ch.export_state()
    return r


class ChainState:
    """One chain of the improved sampler. step() runs one Gibbs iteration. Each chain owns its random streams
    (numpy Generator for the normal/gamma draws, a legacy RandomState for the exact GIG sampler; for a given seed the
    latter is the same stream as np.random.seed(seed)), so chains can be interleaved without changing their draws.
    _fix_sigma2 (list) keeps the residual variances fixed (test hook for the numerical check of the tau update);
    _trace (dict) collects the tau draws when given. init: state of a previous chain (warm start)."""

    def __init__(self, a, b, phi, beta_mrg, frq_dict, idx_dict, n, solver, n_iter, n_burnin, thin, pop, p_tot, seed,
                 coupling, rb, psi_max, meta, write_pst, _fix_sigma2=None, _trace=None, init=None):
        self.pop = list(pop)
        n_pop = len(pop)
        self.a, self.b, self.n, self.solver, self.n_burnin, self.thin = a, b, n, solver, n_burnin, thin
        self.n_pop, self.p_tot, self.coupling, self.rb, self.psi_max = n_pop, p_tot, coupling, rb, psi_max
        self.beta_mrg, self._fix_sigma2, self._trace = beta_mrg, _fix_sigma2, _trace
        self.rng = np.random.default_rng(seed)
        self.rs = np.random.RandomState(seed)      # seed None: fresh entropy from the OS
        # cross-ancestry scale: beta_jk ~ N(0, s2_k psi_j tau_k / Nbar), Nbar = max_k N_k, kappa_k = Nbar/N_k
        #   effect : tau_k = 1;  snr (PRS-CSx) : tau_k = kappa_k;  learned : tau_k ~ IG(1, 1) a priori, tau = 1 for the
        #   ancestry with the largest GWAS (reference), updated each iteration
        self.nbar_all = float(max(n))
        self.kappa = [self.nbar_all/n[pp] for pp in range(n_pop)]
        self.ref_pop = int(np.argmax(n))
        self.tau = list(self.kappa) if coupling == 'snr' else [1.0]*n_pop
        self.floor = 1.0/psi_max                   # 0 if psi_max = inf
        self.p, self.het, self.idx = {}, {}, {}
        for pp in range(n_pop):
            self.p[pp] = len(beta_mrg[pp])
            self.het[pp] = np.sqrt(2.0*frq_dict[pp]*(1.0-frq_dict[pp]))
            self.idx[pp] = np.asarray(idx_dict[pp], dtype=np.int64)
        n_grp = np.zeros(p_tot)
        for pp in range(n_pop):
            n_grp[self.idx[pp]] += 1
        self.gig_p = a-0.5*n_grp
        self.beta = {pp: np.zeros((self.p[pp], 1)) for pp in range(n_pop)}
        self.mean = {pp: np.zeros((self.p[pp], 1)) for pp in range(n_pop)}
        self.sigma = {pp: 1.0 for pp in range(n_pop)}
        self.psi = np.ones(p_tot)
        self.phi_updt = phi is None
        self.phi = 1.0 if self.phi_updt else phi
        if init is not None:                       # warm start (phi grid): continue from a previous chain's state
            self.psi = init['psi'].copy()
            self.sigma = dict(init['sigma'])
            self.beta = {pp: init['beta'][pp].copy() for pp in range(n_pop)}
            if coupling == 'learned':
                self.tau = list(init['tau'])
            if self.phi_updt:
                self.phi = init['phi']
        self.keep_pst = (write_pst == 'TRUE') and (meta == 'TRUE')
        n_pst_max = n_iter//thin - n_burnin//thin
        if self.keep_pst:
            self.beta_pst = {pp: np.zeros((self.p[pp], n_pst_max)) for pp in range(n_pop)}
        self.est = {pp: np.zeros((self.p[pp], 1)) for pp in range(n_pop)}      # sum of estimates (RB means or draws)
        self.s1 = {pp: np.zeros((self.p[pp], 1)) for pp in range(n_pop)}       # sum of draws
        self.s2 = {pp: np.zeros((self.p[pp], 1)) for pp in range(n_pop)}       # sum of squared draws
        self.phi_sum = 0.0
        self.sig_sum = np.zeros(n_pop)
        self.tau_sum = np.zeros(n_pop)
        self.itr = 0
        self.n_keep = 0        # post-burn-in iterations so far
        self.n_thin = 0        # thinned draws so far

    def step(self):
        self.itr += 1
        itr, n_pop, n, rng, psi = self.itr, self.n_pop, self.n, self.rng, self.psi
        idx, beta, mean, sigma, tau, kappa = self.idx, self.beta, self.mean, self.sigma, self.tau, self.kappa
        a, b = self.a, self.b
        # numpy's Generator draws are used here; reproducible for a given --seed, but not the original's stream
        random_gamma = rng.gamma
        # ancestry-specific floor (PRS-CSx's psi<=1 rule expressed per ancestry): prior variance of beta_jk never
        # exceeds psi_max times its own sampling variance, i.e. ridge_jk = max(kappa_k/psi_j, 1/psi_max)
        kt = [kappa[pp]/tau[pp] for pp in range(n_pop)]      # = 1 exactly under snr (PRS-CSx)
        ridge = [np.maximum(kt[pp]/psi[idx[pp]], self.floor) for pp in range(n_pop)]
        quad = self.solver.update(rng, beta, mean, self.beta_mrg, ridge, sigma, n)
        for pp in range(n_pop):
            bb = beta[pp][:, 0]
            prior_q = float(np.dot(bb*bb, ridge[pp]))          # = beta' R_k beta
            err = max(n[pp]/2.0*(1.0-2.0*float(np.dot(bb, self.beta_mrg[pp][:, 0]))+quad[pp]), n[pp]/2.0*prior_q)
            if self._fix_sigma2 is None:
                sigma[pp] = 1.0/random_gamma((n[pp]+self.p[pp])/2.0, 1.0/err)
            else:
                sigma[pp] = float(self._fix_sigma2[pp])
        if self.coupling == 'learned':
            # tau_k | . ~ IG(1 + p_k/2, 1 + Nbar/(2 s2_k) sum_j beta_jk^2/psi_j)   (reference ancestry: tau = 1)
            for pp in range(n_pop):
                if pp == self.ref_pop:
                    continue
                bb = beta[pp][:, 0]
                rate = 1.0 + self.nbar_all/(2.0*sigma[pp])*float(np.dot(bb*bb, 1.0/psi[idx[pp]]))
                tau[pp] = 1.0/random_gamma(1.0+self.p[pp]/2.0, 1.0/rate)
            if self._trace is not None:
                self._trace.setdefault('tau', []).append(list(tau))

        delta = random_gamma(a+b, 1.0/(psi+self.phi))
        xx = np.zeros(self.p_tot)
        for pp in range(n_pop):
            xx[idx[pp]] += (self.nbar_all/tau[pp])*beta[pp][:, 0]**2/sigma[pp]
        # GIG draws via the exact Devroye sampler (consumes this chain's legacy stream)
        psi = sample_psi(self.gig_p, 2.0*delta, xx, rs=self.rs)
        psi_cap = self.psi_max*max(kappa[pp]/tau[pp] for pp in range(n_pop))
        if np.isfinite(psi_cap):
            psi = np.minimum(psi, psi_cap)
        self.psi = psi

        if self.phi_updt:
            w = random_gamma(1.0, 1.0/(self.phi+1.0))
            self.phi = random_gamma(self.p_tot*b+0.5, 1.0/(float(np.sum(delta))+w))

        if itr > self.n_burnin:
            self.n_keep += 1
            for pp in range(n_pop):
                if self.rb:
                    self.est[pp] += mean[pp]
                self.s1[pp] += beta[pp]
                self.s2[pp] += beta[pp]**2
                self.sig_sum[pp] += sigma[pp]
                self.tau_sum[pp] += tau[pp]
            self.phi_sum += self.phi
            if itr % self.thin == 0:
                if not self.rb:
                    for pp in range(n_pop):
                        self.est[pp] += beta[pp]
                if self.keep_pst:
                    for pp in range(n_pop):
                        self.beta_pst[pp][:, [self.n_thin]] = beta[pp]
                self.n_thin += 1

    def running_mean(self, pp):
        """Current estimate of ancestry pp (standardised scale): RB mean (rb=True) or mean of all kept draws."""
        return (self.est[pp] if self.rb else self.s1[pp])[:, 0]/self.n_keep

    def running_var(self, pp):
        ex = self.s1[pp][:, 0]/self.n_keep
        return np.maximum(self.s2[pp][:, 0]/self.n_keep-ex**2, 1e-30)

    def export_state(self):
        return {'psi': self.psi.copy(), 'sigma': dict(self.sigma), 'tau': list(self.tau), 'phi': self.phi,
                'beta': {pp: self.beta[pp].copy() for pp in range(self.n_pop)}}

    def result(self, log=print):
        n_pop, het = self.n_pop, self.het
        n_keep, n_pst = self.n_keep, self.n_thin
        beta_est, var_est = {}, {}
        for pp in range(n_pop):
            m = self.est[pp]/(n_keep if self.rb else n_pst)
            ex2 = self.s2[pp]/n_keep; ex = self.s1[pp]/n_keep
            v = np.maximum(ex2-ex**2, 1e-30)
            beta_est[pp] = m/het[pp]
            var_est[pp] = v/het[pp]**2
        beta_pst = None
        if self.keep_pst:
            beta_pst = {pp: self.beta_pst[pp][:, :n_pst]/het[pp] for pp in range(n_pop)}
        sig_est = self.sig_sum/n_keep
        log('... Estimated residual variances: %s ...' % ', '.join('%s=%.4f' % (self.pop[pp], sig_est[pp]) for pp in range(n_pop)))
        tau_est = self.tau_sum/n_keep
        if self.coupling == 'learned':
            log('... Estimated cross-ancestry scales tau (posterior mean; effect coupling = 1, PRS-CSx coupling = Nmax/N_k): %s ...'
                % ', '.join('%s=%.3f (snr value %.3f)' % (self.pop[pp], tau_est[pp], self.kappa[pp]) for pp in range(n_pop)))
        return {'tau_est': tau_est, 'beta_est': beta_est, 'var_est': var_est, 'beta_pst': beta_pst,
                'phi_updt': self.phi_updt, 'phi': self.phi, 'phi_est': self.phi_sum/n_keep, 'n_pst': n_pst,
                'n_keep': n_keep, 'sig_est': sig_est}


def _write(a, b, res, snp_dict, idx_dict, pop, chrom, out_dir, out_name, meta, write_pst, log):
    n_pop = len(pop)
    p_tot = len(snp_dict['SNP'])
    idx = {pp: np.asarray(idx_dict[pp], dtype=np.int64) for pp in range(n_pop)}
    beta_est, var_est = res['beta_est'], res['var_est']
    keep_pst = (write_pst == 'TRUE') and (meta == 'TRUE') and res.get('beta_pst') is not None
    if meta == 'TRUE':
        vv = np.zeros((p_tot, 1)); zz = np.zeros((p_tot, 1))
        for pp in range(n_pop):
            vv[idx[pp]] += 1.0/var_est[pp]
            zz[idx[pp]] += beta_est[pp]/var_est[pp]
        mu = zz/vv
        if keep_pst:
            n_pst = res['beta_pst'][0].shape[1]
            zp = np.zeros((p_tot, n_pst))
            for pp in range(n_pop):
                zp[idx[pp], :] += res['beta_pst'][pp]/var_est[pp]
            mu_pst = zp/vv
    phitag = 'phiauto' if res['phi_updt'] else 'phi%1.0e' % res['phi']
    files = []
    for pp in range(n_pop):
        fn = out_dir + '/' + '%s_%s_pst_eff_a%d_b%.1f_%s_chr%d.txt' % (out_name, pop[pp], a, b, phitag, chrom)
        with open(fn, 'w') as ff:
            for ii, bv in zip(idx_dict[pp], beta_est[pp][:, 0]):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp_dict['SNP'][ii], snp_dict['BP'][ii],
                                                     snp_dict['A1'][ii], snp_dict['A2'][ii], bv))
        files.append(fn)
    if meta == 'TRUE':
        fn = out_dir + '/' + '%s_META_pst_eff_a%d_b%.1f_%s_chr%d.txt' % (out_name, a, b, phitag, chrom)
        with open(fn, 'w') as ff:
            for jj, (snp, bp, a1, a2) in enumerate(zip(snp_dict['SNP'], snp_dict['BP'], snp_dict['A1'], snp_dict['A2'])):
                if keep_pst:
                    ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *mu_pst[jj]))
                else:
                    ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, mu[jj, 0]))
        files.append(fn)
    log('... Done ...')
    return files
