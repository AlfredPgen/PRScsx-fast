# PRS-CSx-improved

`PRScsx_improved.py` fits **the PRS-CSx model** with a better estimator of the posterior mean and faster linear algebra,
and offers optional extensions (more chains with automatic stopping, warm starts, other cross-ancestry couplings,
LD shrinkage, LD panels in eigen form). Inputs and output files are exactly the original's (same formats, names and
SNP sets), so it is a drop-in replacement. Outputs are *not* numerically identical to PRS-CSx; use `../fast` for that.

**Defaults changed on 19 November 2026** (after the stress test of WP4): `--coupling` is now `snr`, the PRS-CSx model
(it was `effect`), and `--linalg` is `auto`. With the defaults, improved/ and PRS-CSx target the same posterior; they
differ in the estimator (Rao–Blackwellised means), the precision of the factorisation and the random numbers.
In a pre-registered benchmark of 720 simulated datasets it did **not** predict demonstrably better than PRS-CSx
(see "Evidence"); its demonstrated benefits are speed, lower Monte Carlo noise and convergence diagnostics.

**Files needed to run it:** the folders `improved/` and `fast/` side by side in the same parent folder (improved/
imports the shared parsing, GIG and LAPACK code from `../fast`), and Python ≥ 3.9 with numpy, scipy, h5py and numba
(`pip install -r ../requirements.txt`). Nothing else from this repository is required. Checked on 19 Nov 2026: only these
two folders, unzipped into an empty directory, run in a freshly created virtual environment.

```
python PRScsx_improved.py --ref_dir=REF --bim_prefix=TARGET --sst_file=EUR.txt,EAS.txt,AFR.txt \
    --n_gwas=300000,100000,30000 --pop=EUR,EAS,AFR --out_dir=OUT --out_name=PREFIX \
    --phi=1e-6,1e-4,1e-2,1,auto --meta=TRUE --seed=1 [--threads=8 --n_jobs=2 --n_chains=2]
```
All original options are accepted with the same defaults. New options:

| option | default | meaning |
|---|---|---|
| `--coupling` | `snr` | `snr`: PRS-CSx's β_jk ~ N(0, σ²_k ψ_j / N_k). `effect`: β_jk ~ N(0, σ²_k ψ_j / N̄), N̄ = max_k N_k. `learned`: β_jk ~ N(0, σ²_k ψ_j τ_k / N̄) with τ_k learned. All three are identical for one ancestry. `effect` and `learned` are not recommended as defaults (see Evidence). |
| `--rb` | `TRUE` | Rao–Blackwellised posterior mean over all post-burn-in iterations. `FALSE` gives PRS-CSx's average of thinned draws. |
| `--precision` | `single` | float32 Cholesky and solves (2× faster, half the LD memory). Switches to `double` automatically if `--psi_max=inf`. |
| `--linalg` | `auto` | `auto`: per LD block, the low-rank (Woodbury) update when the block has more SNPs than 1.7 × its LD rank, else the full factorisation. `lowrank`, `full`: always that one. Same model in all three. |
| `--rank_var` | 1 | keep the largest LD eigenvalues explaining this share of each block's trace (1 = none dropped). See `../docs/05_building_a_dense_panel.md` for the measured effect (on 1000 Genomes chr22, 0.999 was indistinguishable from 1). |
| `--n_chains` | 1 | independent chains per φ (seeds s, s+1000, ...), averaged; the between-chain R̂ is printed. |
| `--auto_stop` | `FALSE` | `TRUE` (needs `--n_chains` ≥ 2): run the chains side by side and stop once converged by an R̂ rule; `--n_iter` is the maximum. |
| `--warm_start`, `--warm_burnin` | `FALSE`, n_burnin/2 | run the φ grid in increasing order, each φ starting from the previous φ's final state, with a shorter burn-in. Saves at most 19% of the iterations; not recommended (see §5). |
| `--psi_max` | 1 | ridge floor, PRS-CSx's ψ ≤ 1 rule applied per ancestry (see below). `inf` disables it. |
| `--ld_shrink`, `--n_ref` | 0 | optional shrinkage of each reference LD block toward I. Experimental; it reduced accuracy in the v1 benchmark. |
| `--threads`, `--n_jobs`, `--ld_cache_dir`, φ grid | | as in `../fast` |

LD panels in eigen form (`../docs/04_eigen_panel_format.md`, built with `../tools/build_ld_panel.py`) are detected
from `--ref_dir`; `--linalg` and eigen panels work the same way in `../fast` and here. With an eigen panel,
`--linalg=full` and `--ld_shrink` are not available.

## What is different from PRS-CSx, and why it is correct
Notation follows PRS-CSx. The likelihood is β̂_k | β_k ~ "N(D_k β_k, σ²_k D_k / N_k)". The prior is
β_jk | ψ_j ~ N(0, σ²_k ψ_j / N_k) (default coupling), ψ_j ~ Ga(a, δ_j), δ_j ~ Ga(b, φ). Write R_k = diag(ridge_jk).

### 1. Rao–Blackwellised posterior means (`--rb=TRUE`)
Given θ = (ψ, σ², δ, φ), β_k is Gaussian, and its conditional mean m_k(θ) = (D_k + R_k)⁻¹ β̂_k is available exactly. It costs
one extra triangular solve, because the Cholesky factor U (UᵀU = D_k + R_k) is already computed for drawing β_k:
m = U⁻¹U⁻ᵀβ̂ and β = U⁻¹(U⁻ᵀβ̂ + √(σ²/N) z). The estimator is

  β̄_RB = (1/T) Σ_t m(θ⁽ᵗ⁾)  over all T post-burn-in iterations,

instead of the average of every 5th draw β⁽ᵗ⁾.

- **Same target.** E[m(θ)] = E[E[β | θ]] = E[β], so it estimates the same posterior mean.
- **Less noise.** Each term has variance Var(β) − E[Var(β | θ)] ≤ Var(β). Using every iteration instead of every 5th also
  removes the efficiency loss of thinning (MacEachern & Berliner 1994).
- **Measured** on the example data, same model, 200 retained draws: seed-to-seed correlation of the posterior means rose
  from 0.963 to 0.984 (EUR) and from 0.937 to 0.972 (EAS).
- **Limit.** RB does not remove the noise from slow mixing of the ψ_j themselves. That is what `--n_chains` addresses.

### 2. Single-precision factorisation (`--precision=single`)
With the floor, every eigenvalue of D_k + R_k is ≥ 1 (D_k is PSD), so cond(D_k + R_k) ≤ λ_max(D_k) + max ridge, and
Cholesky is backward stable. On real LD blocks of 300–1,200 SNPs, the float32 solve differed from float64 by 1.1e-7 to 2.1e-7
(relative), several orders of magnitude below the Monte Carlo error of the posterior means (≥ 1e-2 relative).
`spotrf` is 2.07× faster than `dpotrf` at b = 700 (measured), and LD storage halves.

### 3. Low-rank block updates (`--linalg=auto|lowrank`)
A reference LD block from n people has rank r ≤ min(b, n − 1); for the 1000 Genomes panels r ≤ 503 while blocks can have
more SNPs. Write D = VΛVᵀ (V: b × r) and A = D + R (R diagonal, positive), s = √(σ²/N). The update needs the mean
m = A⁻¹β̂ and a draw β = A⁻¹(β̂ + s·η) with η = VΛ^{1/2}e₁ + R^{1/2}e₂ (e₁ ~ N(0, I_r), e₂ ~ N(0, I_b)); then
Cov(η) = A, so β ~ N(m, s²A⁻¹) exactly as in the full update. A⁻¹ is applied with the Woodbury identity:
U = R^{-1/2}VΛ^{1/2}, K = I_r + UᵀU (r × r, eigenvalues ≥ 1, so well conditioned even in single precision),
A⁻¹x = R^{-1/2}(y − U K⁻¹Uᵀy) with y = R^{-1/2}x (`../fast/fastcsx/lowrank.py`).
Cost per block per iteration is O(b r²) instead of O(b³/3).
- **Exactness of the formula:** conditional mean and covariance equal the full update to < 1e-9 (double) and < 1e-4
  (single) on 40 blocks, 20 of them real 1000 Genomes blocks (`../tests/test_lowrank.py`).
- **Same posterior:** on real 1000 Genomes chr22 LD, a low-rank run agrees with a full run as closely as two full runs with
  different seeds agree with each other (EUR 0.9645 vs 0.9630, EAS 0.9370 vs 0.9356).
- **Speed** (1 thread, r = 499): 3.5× at b = 2,000, 16× at b = 5,000, 55× at b = 10,000 (single precision 7×, 33×, 72×).
  `auto` uses it when b > 1.7 r, the measured break-even.

### 4. Multiple chains (`--n_chains`) and automatic stopping (`--auto_stop`)
K independent chains with posterior means m_c and variances v_c are combined as m̄ = mean_c m_c and
v = mean_c v_c + var_c m_c (law of total variance; used for `--meta` weights). The Monte Carlo variance of m̄ falls as 1/K,
and the between-chain R̂ (Gelman & Rubin 1992; not the split version) is printed:
R̂ = √(((T−1)/T · W + B)/W) per SNP, W = mean within-chain posterior variance, B = variance of the chains' posterior-mean
estimates (Rao–Blackwellised with `--rb=TRUE`), T = kept iterations.

With `--auto_stop=TRUE` the chains run side by side. After burn-in, every 200 iterations the program takes the 99th
percentile of R̂ over the SNPs of the worst ancestry, and stops after 3 consecutive checks below 1.05. Each chain owns
its random streams, so a stopped run equals the first T iterations of a fixed-length run with the same seeds (tested).
Caveat: stopping on a diagnostic can stop when chains agree by chance; requiring 3 consecutive checks and using the
99th percentile make this less likely, not impossible. With the default lengths the earliest stop is 600 iterations
after burn-in.

Measured (`../benchmarks/wp6_warm_autostop.py`, 120 simulated datasets × 3 φ = 360 runs with 2 chains, maximum 3,000
iterations): every run stopped early, median at iteration 2,100 (the earliest possible), so the cost was 0.73 of two
fixed-length chains (1.47 of one chain). Tuned r²_g, auto-stopped − fixed-length 2 chains: −0.0009 [95% CI −0.0022,
+0.0004]. In these scenarios two chains did not predict measurably better than one (+0.0010 [−0.0007, +0.0027]); use
two chains for the R̂ check, not for accuracy.

### 5. Warm starts over a φ grid (`--warm_start=TRUE`)
The fixed φ values run in increasing order (then `auto`). From the second φ on, the chain starts from the final state
(β, ψ, σ², τ, φ) of the previous φ and uses a shorter burn-in (`--warm_burnin`, default n_burnin/2) with the same
number of kept iterations. The target distribution does not change (the chain converges to the posterior from any
start); the bet is that the previous φ's state is closer to the new posterior than the default start. With the default
lengths the saving is at most 1 − (1 + 3 × 0.75)/4 = 19% of the iterations for a 4-value grid (1.23×).

Measured (`../benchmarks/wp6_warm_autostop.py`, 30 simulated datasets × 3 target ancestries, grid 1e-6 … 1): speed-up
1.23× (iterations) / 1.25× (time). In increasing φ order, tuned r²_g warm − cold +0.0015 [95% CI −0.0030, +0.0060]: no
detectable loss. In decreasing order (the first version) it lost −0.0097 [−0.0149, −0.0044]: from a weakly shrunk
state, half the burn-in was not enough to reach the strongly shrunk posterior. The program therefore sorts the grid.
The speed-up stays below the 1.3× that would justify the option, so it is off by default.

### 6. Other cross-ancestry couplings (`--coupling=effect|learned`; options, not defaults)
**`effect`.** PRS-CSx's β_jk ~ N(0, σ²_k ψ_j / N_k) makes the prior variance of a standardised effect proportional
to 1/N_k: a 10× smaller GWAS gets a 10× wider prior, and for a single SNP (D = 1) the posterior mean is β̂ ψ/(1 + ψ),
the same shrinkage factor whatever the precision of β̂. `effect` uses a common scale, β_jk ~ N(0, σ²_k ψ_j / N̄),
N̄ = max_k N_k, so that the β update is (D_k + κ_k Ψ⁻¹)⁻¹ β̂_k with κ_k = N̄/N_k ≥ 1 (a noisier GWAS is shrunk more)
and ψ_j | · ~ GIG(a − K_j/2, 2δ_j, Σ_k N̄ β²_jk / σ²_k). PRS-CSx's floor (prior variance ≤ sampling variance) is applied
per ancestry: ridge_jk = max(κ_k/ψ_j, 1), ψ_j ≤ κ_max. (Keeping ψ ≤ 1 globally instead over-shrank the small GWAS in a
pilot: AFR r²_g 0.424 → 0.389.) Its assumption is that standardised effects have the same variance in every
ancestry. The stress test shows what happens when that is wrong (Evidence).

**`learned`.** β_jk | ψ_j, σ²_k, τ_k ~ N(0, σ²_k ψ_j τ_k / N̄), τ = 1 for the ancestry with the largest GWAS
(the reference), and τ_k ~ inverse-gamma(1, 1) a priori for the others. `effect` is τ_k = 1, `snr` is τ_k = κ_k.
- *τ_k.* Only the prior on τ_k and the p_k effect densities of ancestry k involve τ_k:
  p(τ_k | ·) ∝ τ_k⁻² e^{−1/τ_k} · Π_j τ_k^{−1/2} exp(−N̄ β²_jk / (2σ²_k ψ_j τ_k))
  = τ_k^{−(1 + p_k/2) − 1} exp(−[1 + (N̄/(2σ²_k)) Σ_j β²_jk/ψ_j] / τ_k),
  the kernel of inverse-gamma(1 + p_k/2, 1 + (N̄/(2σ²_k)) Σ_j β²_jk / ψ_j).
- *ψ_j.* ψ_j ~ gamma(a, rate δ_j) times the K_j effect densities containing ψ_j gives
  ψ_j^{a − K_j/2 − 1} exp(−[2δ_j ψ_j + Σ_k N̄ β²_jk/(σ²_k τ_k) / ψ_j] / 2), i.e.
  GIG(a − K_j/2, 2δ_j, Σ_k N̄ β²_jk / (σ²_k τ_k)). With τ_k = κ_k this is exactly PRS-CSx's update.
- *β_k.* Precision (N_k/σ²_k)(D + R_k) with R_k = diag(κ_k/(τ_k ψ_j)); the floor becomes
  ridge_jk = max(κ_k/(τ_k ψ_j), 1/psi_max) and ψ_j ≤ psi_max · max_k(κ_k/τ_k). As in PRS-CSx, floor and cap are
  robustness heuristics applied after the exact draws.
- *Numerical check* (`../tests/test_coupling_learned.py`): 3 SNPs, 2 ancestries, identity LD, fixed σ², fixed φ, no
  cap. β then integrates out analytically, and E[τ₂ | data] is a 1-D integral with one inner 1-D integral over ψ_j per
  SNP. Exact E[τ₂] = 5.9243 (posterior SD 16.1); the Gibbs sampler (4 chains × 200,000 iterations) gives 5.9244, a
  relative error of 0.002% (criterion 2%).
- The estimated τ (posterior mean) is printed next to the value PRS-CSx's coupling would imply (κ_k).

## Evidence
All numbers come from simulations; nothing here was run on real phenotypes (protocol for that:
`../docs/06_real_data_evaluation.md`). Full report: `../docs/03_final_report.md`.

**v1 benchmark (8 datasets, real 1000 Genomes LD; `../benchmarks/results/`), with the old default `effect`.** Tuned r²_g
improved − PRS-CSx: EUR +0.015, EAS +0.005, AFR +0.008 (7/8 datasets better, Wilcoxon p = 0.039). Small, mostly
within replicate noise.

**WP4 coupling stress test (`../benchmarks/results_v2/wp4_coupling_stress/summary.md`).** 5 scenarios × 20 replicates,
synthetic LD, N = 300k / 100k / 30k, oracle tuning; r²_g difference from `snr` (SE):

| scenario | target | effect − snr | learned − snr |
|---|---|---|---|
| equal effect variance, ρ = 0.8 | EUR / EAS / AFR | +0.012 (0.005) / +0.004 (0.008) / +0.013 (0.010) | −0.000 / −0.004 / +0.006 |
| AFR effect variance ×3 | EUR / EAS / AFR | +0.002 / +0.002 / +0.006 | +0.008 / −0.006 / +0.000 |
| AFR effect variance ×10 | EUR / EAS / AFR | **−0.008 (0.003)** / **−0.014 (0.006)** / −0.003 | +0.006 / −0.010 (0.006) / −0.009 (0.008) |
| AFR effect variance ÷3 | EUR / EAS / AFR | +0.005 / −0.000 / +0.000 | +0.014 (0.005) / +0.002 / +0.008 |
| ρ = 0.4 | EUR / EAS / AFR | +0.010 / +0.014 / +0.012 | +0.013 / +0.014 / +0.007 |

`effect` helps when its assumption (equal effect variance) holds and hurts when the smallest GWAS's ancestry has much
larger effects. `learned` was meant to adapt, but it was more than 0.005 below `snr` in three cells. By the rule fixed
in advance, the default became `snr`, and neither alternative coupling is recommended as a default.

**WP5 accuracy benchmark v2 (pre-registered; `../benchmarks/PLAN_accuracy_v2.md`, results
`../benchmarks/results_v2/wp5_accuracy/summary.md`).** 36 scenarios × 20 replicates, tuned r²_g difference from PRS-CSx
(pooled, 95% CI):

| method | EUR | EAS | AFR | verdict (rule: mean > 0.01, CI > 0, no scenario < −0.01) |
|---|---|---|---|---|
| default (`snr`) | +0.0022 [+0.0005, +0.0039] | +0.0030 [+0.0006, +0.0055] | +0.0008 [−0.0020, +0.0035] | no demonstrated difference |
| default, 2 chains | +0.0050 [+0.0034, +0.0065] | +0.0051 [+0.0029, +0.0074] | +0.0050 [+0.0024, +0.0076] | no demonstrated difference |
| `effect` | +0.0062 | +0.0075 | +0.0069 | no demonstrated difference |
| `learned` | +0.0062 | +0.0044 | +0.0065 | no demonstrated difference |

So improved/ is **not** shown to predict better than PRS-CSx. The average differences are small and mostly positive,
but below the threshold fixed in advance, and every variant is worse on average in some scenarios (default: down to
−0.027 for AFR in one scenario). Details and WP6 (warm start, automatic stopping): `../docs/03_final_report.md`, §7.

## Recommendations
- Use the defaults. `--n_chains=2` gives an R̂ diagnostic; `--auto_stop=TRUE` with 2 chains stops when R̂ allows.
- With a validation sample, combine the ancestry-specific scores with `../tools/combine_prs.py` (optionally `--stack`).
- Keep `--psi_max=1`, `--ld_shrink=0` and `--warm_start=FALSE`.
