# PRS-CSx: what it does and how it does it

Source studied: `https://github.com/getian107/PRScsx`, commit `bdf5f30` (2026-10-08). Every file was read
in full: `PRScsx.py` (CLI driver), `parse_genet.py` (I/O and alignment), `mcmc_gtb.py` (Gibbs sampler),
`gigrnd.py` (GIG random variates), `README.md`, `test_data/`.
References: Ge et al. *Nat Commun* 2019 (PRS-CS); Ruan et al. *Nat Genet* 2022 (PRS-CSx);
Devroye *Stat Comput* 2014 (GIG sampler); Armagan, Dunson & Clyde 2011 (three-parameter beta / gamma-gamma prior).

## 1. Purpose

PRS-CSx turns GWAS summary statistics from K ancestries (e.g. EUR, EAS, AFR, ...) into posterior
mean SNP effect sizes, one set per ancestry, for computing polygenic scores. It
needs only summary statistics plus an external LD reference panel for each ancestry (1000 Genomes or UKBB,
HapMap3 SNPs, pre-split into approximately independent LD blocks). The ancestries are linked through a shared
*continuous-shrinkage* (CS) prior. A SNP that looks important in one ancestry is shrunk less in the other(s),
but the effect sizes themselves are not forced to be equal.

Two ways to use the outputs (README):
1. **Tuned:** run on a grid of φ values (e.g. 1e-6, 1e-4, 1e-2, 1). For each φ, compute one PRS per ancestry,
   standardise them, fit a linear combination in a validation sample, pick the best φ, then evaluate in an
   independent test sample.
2. **auto + meta:** learn φ from the data (`--phi` omitted) and output an inverse-variance-weighted
   meta-analysis of the ancestry-specific posterior means (`--meta=TRUE`). No validation data is needed.

## 2. Data pipeline (`parse_genet.py`, per chromosome)

| Step | Function | What happens |
|---|---|---|
| 1 | `parse_ref` | Read `snpinfo_mult_{1kg,ukbb}_hm3` for this chromosome: SNP, BP, A1/A2, allele frequency `FRQ_<POP>` and allele-coding flag `FLP_<POP>` for each of the 5 ancestries. |
| 2 | `parse_bim` | Read target `.bim`, which defines the SNPs available in the target data and the output A1/A2 orientation. |
| 3 | `parse_sumstats` | Keep A/C/G/T SNPs. Intersect target, reference (`FRQ_<POP> > 0`) and sumstats SNPs, allowing allele swaps and strand flips. Convert each SNP to a **standardised marginal effect** β̂ = z/√N, with z = BETA/SE (or log(OR)/SE), or z = sign(BETA)·|Φ⁻¹(P/2)| with P floored at 1e-323. Align signs to the `.bim` A1 allele. |
| 4 | `parse_ldblk` | Load every LD block of the chromosome from `ldblk_*_chr*.hdf5` (block SNP order equals snpinfo order, checked). Subset to retained SNPs, apply sign flips `R ∘ (f fᵀ)`, then make the block PSD with Higham's nearest-PSD formula: `H = V S Vᵀ` from the SVD, `R ← (R + H)/2`. |
| 5 | `align_ldblk` | Build the SNP union across ancestries, the per-ancestry index maps `idx_dict[k]`, and the β̂ and frequency vectors. |

The reference panel contains **no strand-ambiguous (A/T, C/G) SNPs** (0 of 1,297,431, checked), so the
strand-flip matching cannot mis-align alleles with the shipped panels.

## 3. Statistical model

Notation: ancestry k = 1..K with GWAS size N_k. SNP j belongs to the union of M SNPs on the chromosome; S_k is
the set of SNPs available in ancestry k, and K_j = #{k : j ∈ S_k}. D_k is the LD (correlation) matrix of ancestry
k, which is block-diagonal over that ancestry's LD blocks. Block boundaries differ between ancestries
(1,703 EUR vs 1,445 EAS blocks genome-wide).

**Likelihood (summary-statistic form).** With standardised genotypes and phenotype (yᵀy/N = 1),
y_k = X_k β_k + ε_k with ε ~ N(0, σ_k² I). The Gaussian log-likelihood depends on the data only through
β̂_k = X_kᵀy_k/N_k and D_k = X_kᵀX_k/N_k:

  log p(β̂_k | β_k, σ_k²) = −(N_k/2) log σ_k² − N_k/(2σ_k²) · (1 − 2 β_kᵀβ̂_k + β_kᵀ D_k β_k) + const.

The in-sample D_k is replaced by the external reference LD.

**Coupled continuous-shrinkage prior.** This is the core of PRS-CSx:

  β_jk | ψ_j, σ_k² ~ N(0, σ_k² ψ_j / N_k)     (independent over k given ψ_j)
  ψ_j | δ_j ~ Gamma(a, rate δ_j),  δ_j | φ ~ Gamma(b, rate φ)   (shared across ancestries)
  p(σ_k²) ∝ 1/σ_k²
  φ fixed (`--phi`), or "auto": φ | w ~ Gamma(1/2, w), w ~ Gamma(1/2, 1), i.e. √φ ~ half-Cauchy(0, 1)

Integrating out δ_j gives p(ψ_j) ∝ ψ_j^{a−1} (ψ_j + φ)^{−(a+b)}: ψ_j/φ is beta-prime(a, b), the three-parameter
beta / gamma-gamma family. The defaults a = 1, b = 1/2 give the Strawderman–Berger prior: substantial mass near
zero (strong shrinkage of null SNPs) and Cauchy-like tails (b = 1/2), so large effects are shrunk little.
a = b = 1/2 would be the horseshoe. φ sets the overall sparsity/scale: small φ means more shrinkage, suited to
sparse traits; larger φ suits polygenic traits.

**What "coupling" means here.**
- Only the local scale ψ_j is shared. Given ψ_j, the β_jk are independent across ancestries, so effects can differ
  in magnitude and even sign. Marginally they are dependent in *scale*: a large |β_j1| pushes ψ_j up, which
  relaxes shrinkage of β_j2. This is how a well-powered GWAS (e.g. EUR) borrows strength for an under-powered one.
- The prior variance is σ_k²ψ_j/N_k, so the shared quantity is a per-SNP **signal-to-noise ratio** (prior variance
  divided by the sampling variance σ_k²/N_k), not a shared effect-size variance. For the same ψ_j, the
  ancestry with the smaller N_k gets a larger prior variance on β_jk. This is inherited from PRS-CS and is a
  modelling choice worth revisiting (see §7).

## 4. Gibbs sampler (`mcmc_gtb.py`)

Each iteration updates, in order (systematic scan):

1. **β_k for each ancestry, block by block** (blocks are independent given ψ, σ_k²):
   β_k | · ~ N( (D_k + Ψ_k⁻¹)⁻¹ β̂_k ,  (σ_k²/N_k)(D_k + Ψ_k⁻¹)⁻¹ ),  Ψ_k = diag(ψ_j, j ∈ S_k).
   Code: Cholesky Uᵀ U = D + Ψ⁻¹, then β = U⁻¹(U⁻ᵀ β̂ + √(σ²/N) z) with z ~ N(0, I), using two triangular solves.
   This has exactly the stated mean and covariance. It also accumulates quad = Σ_blocks βᵀ(D + Ψ⁻¹)β.
2. **σ_k²:** σ_k² | · ~ Inv-Gamma( (N_k + p_k)/2 ,  (N_k/2)·[1 − 2βᵀβ̂ + βᵀ(D + Ψ⁻¹)β] ).
   The rate combines the likelihood residual term and the prior term βᵀΨ⁻¹β; p_k is the number of SNPs in
   ancestry k on this chromosome. Code floors the rate at (N_k/2)·βᵀΨ⁻¹β, because 1 − 2βᵀβ̂ + βᵀDβ (a residual
   sum of squares under in-sample LD) can go negative when the reference LD mismatches the GWAS LD.
3. **δ_j:** δ_j | · ~ Gamma(a + b, rate ψ_j + φ), vectorised.
4. **ψ_j:** combining ψ_j^{a−1}e^{−δ_jψ_j} with ∏_{k∋j} ψ_j^{−1/2} exp(−N_k β_jk²/(2σ_k²ψ_j)) gives
   ψ_j | · ~ GIG( a − K_j/2 ,  2δ_j ,  Σ_{k∋j} N_k β_jk²/σ_k² ),  where GIG(p, a, b) ∝ x^{p−1} e^{−(a x + b/x)/2}.
   This is where the ancestries are coupled. **Then `psi[psi>1] = 1`** (see §6).
5. **φ (auto only):** w ~ Gamma(1, rate φ + 1);  φ ~ Gamma(M·b + 1/2, rate Σ_j δ_j + w).

Defaults: `n_iter` = 1000·K, `n_burnin` = 500·K, `thin` = 5. For K = 2 that is 2,000 iterations and 200 retained
draws. The output is the average of the retained β draws, converted to the per-allele scale with
β/√(2f(1−f)), where f is the **reference-panel** frequency of that ancestry.
`--meta=TRUE` adds μ_j = Σ_k v_jk⁻¹ β̄_jk / Σ_k v_jk⁻¹ with v_jk = E[β²] − E[β]² (per-allele posterior variance).
`--write_pst=TRUE` with meta also writes all meta-combined draws.
The RNG (`numpy.random` legacy MT19937) is **re-seeded with the same seed for each chromosome**, so chromosomes
are independent jobs and can be run in parallel with identical results.

**GIG sampler (`gigrnd.py`).** This is Devroye's (2014) method for the two-parameter GIG(λ, ω),
ω = √(ab). For λ < 0 it uses the reciprocal identity GIG(−λ, ω) = 1/GIG(λ, ω). It samples X = log(x) from the
log-concave density exp ψ(x), ψ(x) = −α(cosh x − 1) − λ(eˣ − x − 1), α = √(ω² + λ²) − λ, by rejection from a
three-piece envelope (flat centre, exponential tails, cut points t and s chosen from −ψ(±1)). The result is
mapped back with eˣ(λ/ω + √(1 + λ²/ω²)) and rescaled by √(b/a). Each attempt uses 3 uniforms. On the
example data it averaged 1.17 attempts per draw (7.0M uniforms for 2.0M draws, measured with cProfile).

**Cost per iteration:** Σ_k Σ_blocks O(b³/3) for the Cholesky factorisations, plus O(b²) for the solves and the
quadratic form, plus M **interpreted Python calls** to `gigrnd`. Genome-wide 1KG panels have
~1.1M SNPs in ~1,450–1,700 blocks (median ~650 SNPs, max ~1,800).

## 5. Correctness check of the derivations

I re-derived each full conditional above from the stated model and checked it against the code line by line:
β-update mean/covariance via the Cholesky factor, σ² shape/rate, δ shape/rate (NumPy uses *scale*, so
`1/(psi+phi)` is correct), GIG arguments `(a − 0.5·n_grp, 2δ, Σ N β²/σ²)`, and φ/w. All match. The only departures
from an exact Gibbs sampler for the stated model are the ψ ≤ 1 clipping and the σ² rate floor, both deliberate
robustness heuristics (§6).

## 6. Issues and observations (verified where marked)

| # | Finding | Type | Evidence |
|---|---|---|---|
| 1 | **Hangs silently on NumPy ≥ 2.x.** `gigrnd` calls `float()` on shape-(1,) arrays (`n_grp[jj]`, `delta[jj]`, `xx[jj]`). NumPy 2.4 raises `TypeError` ("only 0-dimensional arrays can be converted to Python scalars"); the bare `except: continue` in `mcmc_gtb.py` L104–110 swallows it and retries forever. | Compatibility bug | **Verified:** NumPy 2.4.6 run killed after 180 s with no `iter-100` line; direct call reproduces the TypeError. Same command finishes in ~40 s on NumPy 1.26.4. |
| 2 | The bare `except:` retry loop never terminates if the error is deterministic. It also catches `KeyboardInterrupt`, so Ctrl-C can be swallowed while it is inside `gigrnd`. | Robustness | Code reading (Python semantics of bare `except`). |
| 3 | **O(p²) bookkeeping:** `snp in sst_dict['SNP']` (list) in `parse_ldblk`/`align_ldblk`, `vld_dict['SNP'].index(snp)`, `jj in idx_dict[pp]` (list) in `mcmc`. Negligible at 1,000 SNPs, but quadratic in SNPs per chromosome. | Performance | Code reading; full-chromosome timing in §8. |
| 4 | The GIG step is one interpreted Python call per SNP per iteration, with 3 scalar `numpy.random` calls per attempt. | Performance | **Profiled:** ~54% of runtime on the example. |
| 5 | Wrapper overhead: `np.diag(1/psi)` allocates a b×b matrix per block per iteration; SciPy `check_finite`; Python `sum()` over (p,1) arrays. | Performance | **Profiled:** builtin `sum` 2.4 s, `asarray_chkfinite` 1.7 s (under cProfile). |
| 6 | **ψ_j is clipped at 1.** The prior variance is therefore ≤ σ²/N and the ridge term in D + Ψ⁻¹ is ≥ 1. For an isolated SNP (D = 1) the posterior mean is β̂ψ/(1+ψ) ≤ **β̂/2**, so strong signals are always shrunk by at least half. This guards against ill-conditioning and LD-reference mismatch, but it also (a) compresses the relative weights of large-effect SNPs and (b) is a clipped, not truncated, GIG draw, so the chain does not target the stated posterior exactly. | Modelling choice | Algebra + numeric check. Whether relaxing it helps prediction is an empirical question for Task 2. |
| 7 | If `n_burnin` is not a multiple of `thin`, the number of retained draws can differ from `n_pst = int((n_iter−n_burnin)/thin)`. The mean is then mis-normalised, and with `--write_pst` an out-of-range column write can occur. Defaults are unaffected. | Edge-case bug | **Verified:** IndexError crash, §8.3. |
| 8 | Monte Carlo error at default chain length is **large**: different seeds give posterior means correlated at only r ≈ 0.96 (EUR) and 0.94 (EAS) on the example. | Statistical | **Measured**, §8. |
| 9 | Output file name formats `a` with `%d` (a = 0.5 becomes `a0`). | Cosmetic | Code reading. |
| 10 | Example data: `EUR_sumstats.txt` and `EAS_sumstats.txt` are **byte-identical**, and there are no genotypes or phenotypes, so the example tests the plumbing but not accuracy. | Data | md5sum. |
| 11 | All LD blocks of a chromosome are read and kept in memory (including blocks with no retained SNPs); the SVD-based PSD projection is O(b³) per block, once per run. | Performance / memory | Code reading. |

## 7. Implications for the next tasks (plan, not results)

- **Fast (exact-match) version.** Replicate the RNG stream exactly so outputs match bit-for-bit or to round-off:
  the legacy MT19937 sequence is identical for one `randn(n)` call or several smaller ones, and a JIT GIG sampler
  can consume pre-drawn uniforms with the stream re-synchronised afterwards. Exactness is required, not
  optional: from finding 8, any implementation that only matches *in distribution* would differ from the original
  at the r ≈ 0.95 level, which cannot be told apart from a bug. Other targets: O(p) hashing for parsing, direct
  LAPACK calls, no b×b diagonal allocation, parallelism over blocks, ancestries and chromosomes (chromosomes are
  independently seeded), and reuse of parsed LD across φ values.
- **Improved version.** Candidates, each to be justified and benchmarked: a principled alternative to the ψ ≤ 1 clip;
  coupling on the effect-size scale instead of the N-scaled SNR scale; LD-mismatch-aware regularisation;
  multi-chain/longer-chain or Rao-Blackwellised posterior means to cut Monte Carlo error; data-driven φ
  selection without validation data; and ensembling across φ.

## 8. Baseline: original PRS-CSx, measured

**Environment.** Python 3.11.18, NumPy 1.26.4, SciPy 1.11.4, h5py 3.10.0 (README-recommended), OpenBLAS.
4-core Intel Xeon @ 2.30 GHz, 15 GB RAM. 1000 Genomes EUR and EAS LD panels and `snpinfo_mult_1kg_hm3`
were downloaded from the Broad mirror listed in the README. Repo code is unmodified.
All timings are single runs unless stated; BLAS threads = 1 unless stated.

### 8.1 Example dataset (README command: chr22, EUR N = 200k + EAS N = 100k, φ = 1e-2)
- 1,000 SNPs in the union; EUR 1,000, EAS 901 (99 test SNPs have frequency 0 in the 1KG EAS panel).
  They fall in 4 non-empty LD blocks per ancestry (EUR block sizes 135/176/511/178; EAS 93/272/374/162).
- README command as written (no seed, default 4 BLAS threads): **44.4 s wall**, 78.3 s CPU, **242 MB peak RSS**.
- φ = 1e-2, seeds 1–5, 1 thread: **wall 40.6 ± 1.3 s** (range 39.0–42.4), peak RSS 235–241 MB.
- 4 BLAS threads instead of 1: 38.9 s wall but 68.7 s CPU. **No speed-up**; the extra threads only burn CPU.
- Single ancestry (EUR only, 1,000 iterations): 15.9 s. `--meta=TRUE`: 37.2 s. `phi=auto + meta`: 40.5 s,
  estimated φ = 1.19e-4. `--meta --write_pst`: META file has 205 columns (5 + 200 draws).
- Stage breakdown (seed 1): parse_ref 0.43 s, parse_sumstats 0.17 s, parse_ldblk 1.73 s, align 0.20 s,
  **MCMC 34.7 s (93%)**, about 17 ms per iteration.
- cProfile (example): `gigrnd` ~54% (2.0M interpreted calls, 7.0M scalar uniforms, 1.17 attempts per draw),
  Cholesky ~17%, Python and wrapper overhead in the `mcmc` loop ~25%.

**Determinism (fixed seed): byte-identical outputs** across reruns, across 1 vs 4 BLAS threads, with or without
`--meta` (ancestry-specific files), and under cProfile.

**Monte Carlo variability across seeds 1–5 (10 pairs):**

| pop | n SNPs | Pearson r median [min, max] | max abs diff (median) | mean abs diff (median) | median abs(posterior mean) |
|---|---|---|---|---|---|
| EUR | 1000 | 0.963 [0.958, 0.968] | 2.7e-3 | 1.4e-4 | 1.7e-4 |
| EAS | 901 | 0.937 [0.924, 0.951] | 3.0e-3 | 1.9e-4 | 2.6e-4 |

Implication: a re-implementation that is correct only in distribution would match the original only to
r ≈ 0.94–0.96. The fast version must therefore reproduce the RNG stream to be verifiably "matching". Seed-to-seed
spread is the correct yardstick for any non-exact component.

### 8.2 Runtime probe at realistic scale (all HapMap3 SNPs of chr22)
Synthetic sumstats for every chr22 HapMap3 SNP were simulated from the real 1KG LD of each ancestry
(`benchmarks/make_fullchr_probe.py`: β̂ = Rβ + N(0, R/N), 1% causal, cross-ancestry ρ = 0.8). This is a
**timing probe only**; the accuracy benchmark comes later.
- 17,007 SNPs in the union (EUR 16,464; EAS 14,968), 24 EUR / 20 EAS blocks (median ~700 SNPs).
- **Total 1,186 s (19.8 min), single thread, peak RSS 449 MB.** parse_ref 0.47 s, parse_sumstats 0.15 s,
  parse_ldblk 13.6 s (incl. SVD 8.8 s), align 5.0 s (quadratic), **MCMC 1,166 s (98.4%) = 0.58 s per iteration**.
- cProfile on 200 iterations: **Cholesky is now the largest cost** (62 s of 152 s profiled MCMC),
  then `gigrnd` (49 s), loop body (23 s), `check_finite` (9 s), triangular solves (9 s).
  At realistic block sizes the O(b³) factorisation dominates, so the GIG JIT alone is not enough;
  parallelism across blocks/ancestries/chromosomes and leaner LAPACK calls are needed too.

### 8.3 Verified failures of the original
- NumPy 2.4.6: hangs in the first MCMC iteration (finding 1). Killed after 180 s; the same run takes about 40 s on NumPy 1.26.
- `--n_iter=1000 --n_burnin=502 --thin=5 --meta=TRUE --write_pst=TRUE`: crashes with
  `IndexError: index 99 is out of bounds for axis 1 with size 99` (finding 7). Without `--write_pst`, the same
  settings silently divide the posterior-mean sum by 99 instead of 100 (a 1% scale error).
