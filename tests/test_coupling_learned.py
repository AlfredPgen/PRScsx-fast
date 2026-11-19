"""
WP4 step 1: the conditional updates of --coupling=learned are checked against the joint density.

Toy problem: 3 SNPs, 2 ancestries, LD = identity, residual variances fixed (test hook), phi fixed, no psi cap.
Then beta integrates out analytically, bhat_jk | psi_j, tau_k ~ N(0, s2/N_k + s2 psi_j tau_k / Nbar), and
    p(tau_2 | bhat) ∝ IG(tau_2; 1, 1) * prod_j  ∫ TPB(psi_j; a, b, phi) N(bhat_j1; 0, v_j1(psi_j)) N(bhat_j2; 0, v_j2(psi_j, tau_2)) dpsi_j
with TPB(psi) = Γ(a+b)/(Γ(a)Γ(b)) phi^b psi^(a-1) (psi+phi)^-(a+b) (delta_j integrated out). E[tau_2 | bhat] is computed
by nested 1-D numerical integration and compared with the long-run mean of tau_2 from the Gibbs sampler (within 2%).
"""
import os
import sys

import numpy as np
from scipy import integrate
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, os.path.join(ROOT, 'fast'))
sys.path.insert(0, os.path.join(ROOT, 'improved'))
from csximp.mcmc_improved import Solver, mcmc_improved  # noqa: E402

A, B, PHI = 1.0, 0.5, 1e-2
N = [100000, 20000]
S2 = [1.0, 1.0]
BHAT = np.array([[0.020, -0.015, 0.003], [0.035, -0.030, 0.006]])     # ancestry 2: larger effects -> tau_2 > 1


def numerical_tau_mean():
    nbar = max(N)
    logc = gammaln(A + B) - gammaln(A) - gammaln(B) + B * np.log(PHI)

    def log_norm(x, v):
        return -0.5 * np.log(2 * np.pi * v) - 0.5 * x * x / v

    def snp_integral(j, tau):
        def f(u):                                    # u = log psi
            psi = np.exp(u)
            lp = logc + (A - 1) * u - (A + B) * np.log(psi + PHI) + u            # + u: Jacobian dpsi = psi du
            v1 = S2[0] / N[0] + S2[0] * psi / nbar
            v2 = S2[1] / N[1] + S2[1] * psi * tau / nbar
            return np.exp(lp + log_norm(BHAT[0, j], v1) + log_norm(BHAT[1, j], v2) - 20.0)
        return integrate.quad(f, -40, 20, limit=400, points=[-10, -5, 0, 5])[0]

    def post(lt):                                    # unnormalised posterior density of log tau
        tau = np.exp(lt)
        lp = -np.log(tau) * 2 - 1.0 / tau + lt       # IG(1, 1) density * Jacobian (tau)
        val = np.exp(lp)
        for j in range(3):
            val *= snp_integral(j, tau)
        return val
    num = integrate.quad(lambda lt: np.exp(lt) * post(lt), -8, 10, limit=200)[0]
    den = integrate.quad(post, -8, 10, limit=200)[0]
    return num / den


def gibbs_tau_mean(n_iter, seed):
    p = 3
    snp = {'CHR': [22] * p, 'SNP': ['rs%d' % i for i in range(p)], 'BP': list(range(p)), 'A1': ['A'] * p, 'A2': ['G'] * p}
    bt = {k: BHAT[k].reshape(p, 1).copy() for k in range(2)}
    fr = {k: np.full((p, 1), 0.5) for k in range(2)}
    ix = {k: list(range(p)) for k in range(2)}
    ld = [[np.array([[1.0]])] * p for _ in range(2)]
    solver = Solver(ld, [[1] * p, [1] * p], 1, precision='double')
    trace = {}
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mcmc_improved(A, B, PHI, snp, bt, fr, ix, N, solver, n_iter, n_iter // 10, 5, ['EUR', 'AFR'], 22, d, 't', 'FALSE',
                      'FALSE', seed, coupling='learned', rb=True, psi_max=np.inf, log=lambda *x: None,
                      _fix_sigma2=S2, _trace=trace)
    tau2 = np.array([t[1] for t in trace['tau']])[n_iter // 10:]
    return tau2.mean()


def test_learned_tau_matches_numerical_integration():
    """tau_2 has a very heavy-tailed posterior here (exact mean 5.92, sd 16.1: inverse-gamma-like with shape 2.5), so
    the Monte Carlo error of a long-run mean is large; 4 chains x 200,000 iterations give a standard error of about
    0.55%, enough to resolve the 2% criterion. Takes about 4 minutes."""
    import json
    exact = numerical_tau_mean()
    chains = [gibbs_tau_mean(200000, s) for s in (1, 2, 3, 4)]
    est = float(np.mean(chains))
    out = os.path.join(ROOT, 'benchmarks', 'results_v2', 'wp4_step1_tau_numerical_check.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(dict(exact_posterior_mean_tau2=exact, gibbs_chain_means=chains, gibbs_mean=est, relative_error=est / exact - 1,
                   iterations_per_chain=200000, burn_in=20000), open(out, 'w'), indent=1)
    assert abs(est / exact - 1) < 0.02, (est, exact)
