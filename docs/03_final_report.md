# PRS-CSx: fixed, fast and improved. Final report

All numbers below were produced by code in this repository on a 4-core Intel Xeon @ 2.30 GHz (15 GB RAM); part of the
follow-up work (§7) ran on a 4-core Xeon @ 2.10 GHz after the sandbox moved, and speed is only compared within one machine. Raw results:
`benchmarks/results/`. Method and code summary: `docs/01_PRScsx_method_and_code_summary.md`. Fix notes:
`docs/02_fix_numpy2_and_robustness.md`.

## 0. Summary

| deliverable | result |
|---|---|
| **Fixed original** (root `PRScsx.py`) | The NumPy ≥ 2 hang came from `float()` of shape-(1,) arrays inside a bare `except: continue`. Fixed along with 3 more NumPy-2 crash sites and 4 other errors. Outputs are byte-identical to the original (20/20 files on NumPy 1.26 and on NumPy 2.4). |
| **Fast** (`fast/`), Task 1 | **Byte-identical** to the original for the same seed: 40/40 files on the example (9 configurations × 1 and 4 threads), all files of a full-chromosome run, and a 3-ancestry simulated fit. Speed-up 2.9× (1 thread) and 3.8× (4 threads) on the example; 1.7× and 5.5× on all HapMap3 SNPs of chr22; 5.1–5.3× per iteration at 19k–250k SNPs (4 cores each). |
| **Improved** (`improved/`), Task 2 | The PRS-CSx model (default since 19 Nov 2026, after the coupling stress test of §7.4) with Rao–Blackwellised posterior means, single-precision and low-rank block updates, chain averaging with R̂, automatic stopping; optional effect-scale and learned couplings. 2.5× (1 thread) and 8.6× (4 threads) faster than the original on chr22 with the earlier defaults (2.7× and 7.9× derived for the current defaults, §7.6). Accuracy: the v1 benchmark (8 datasets, old default `effect`) showed small gains (tuned r²_g +0.005 to +0.015), mostly not significant. The pre-registered v2 benchmark (§7.5): no method passed the claim rule in any ancestry (pooled gains of +0.001 to +0.008 in tuned r²_g, below the pre-set 0.01; some scenarios worse by up to 0.027), so better prediction is **not** demonstrated. |
| **Benchmarks** (`benchmarks/`) | example data; 3-ancestry simulation on real 1KG LD with realistic reference mismatch; runtime, memory, scaling, accuracy, ablations, MCMC-seed variability |
| **Follow-up (§7)** | low-rank LD updates (up to 55× per large block), LD panels in eigen form with a builder for your own genotypes (`--rank_var 0.999` default), a learned coupling (not adopted as default), warm starts (1.23×, off), automatic stopping (0.73× the cost of 2 chains), a real-data protocol (not run) |
| **Tools / tests** | `tools/combine_prs.py` (the README's validation step), `tools/compare_prs.py`, `tools/build_ld_panel.py`; pytest suites for fixed, fast, improved, low-rank, eigen panels, learned coupling and tools |

## 1. What PRS-CSx does (short)
Per chromosome and LD block, PRS-CSx fits the summary-statistic model β̂_k ~ N(D_k β_k, σ²_k D_k/N_k) with a continuous-shrinkage
prior β_jk ~ N(0, σ²_k ψ_j/N_k), ψ_j ~ Ga(a, δ_j), δ_j ~ Ga(b, φ). The local scale ψ_j is **shared across ancestries**:
that shared scale is the cross-ancestry coupling. The Gibbs sampler draws β_k blockwise by Cholesky, σ²_k from an inverse
gamma, δ_j from a gamma, and ψ_j from GIG(a − K_j/2, 2δ_j, Σ_k N_k β²_jk/σ²_k), which is then clipped at 1. φ is either
fixed or auto (half-Cauchy). Details, derivations and a line-by-line check against the code are in doc 01.

## 2. Making it run
The NumPy ≥ 2 hang and 7 other defects are documented in doc 02. A negative control shows the regression tests detect
them: run against the original code, 6 tests fail and 2 hang on NumPy 1.26, and 5 fail and 12 hang on NumPy 2.4.

Additional robustness finding (§5.2): the original's PSD projection uses LAPACK `gesdd`, which can fail to converge
("SVD did not converge") on strongly rank-deficient LD blocks. Whether it does depends on the LAPACK build and on the
number of BLAS threads.

## 3. Task 1: PRS-CSx-fast

### 3.1 How
Profiling showed two costs. On the example, the interpreted GIG sampler took about 54% of the time. At realistic block sizes,
the O(b³) Cholesky factorisations dominate (about 50%).
1. **GIG sampler** in numba, replaying the legacy MT19937 stream exactly. Uniforms are pre-drawn with `random_sample(L)` and
   the generator is rewound to the number actually consumed. Python exceptions are emulated. `pow(x,2)` is computed by libm
   pow (with a run-time exponent), because LLVM's `x*x` rewrite differs from glibc pow in 0.09% of inputs. Verified
   bit-identical on 200,000 draws.
2. **Block updates** call the very same LAPACK routines (`dpotrf`/`dtrtrs` from SciPy's OpenBLAS) from a nogil numba kernel.
   SciPy's f2py wrappers hold the GIL (measured 0.90× "speed-up" with 4 threads). All LD blocks of all ancestries run on
   a thread pool. Exactness: random numbers are drawn beforehand in the original order (`randn(p_k)` equals the
   per-block calls, and `gamma(k,s) = s·standard_gamma(k)` bitwise), and block results are summed in the original order.
3. **Linear-time bookkeeping:** the original's `snp in list` and `list.index` checks are O(p²). Sumstats and SNP info are read
   once per run; only the needed LD blocks and ancestry columns are kept.
4. **Workflow:** φ grid in one process (reseeded per φ, so identical to separate runs), chromosome processes (`--n_jobs`),
   LD cache.

### 3.2 Exactness (fast vs original, same seed)
| test | files | byte-identical | max abs diff |
|---|---|---|---|
| example, 9 configurations (seeds 1–5, meta, meta + write_pst, auto + meta, single ancestry) × threads 1 and 4 | 40 | 40 | 0 |
| example, runtime benchmark (3 seeds × threads 1 and 4) | 12 | 12 | 0 |
| all HapMap3 SNPs of chr22 (17,007 SNPs, EUR+EAS, 2,000 iterations), threads 1 and 4 | 4 | 4 | 0 |
| simulated 3-ancestry dataset (EUR/EAS/AFR, 3,000 iterations, meta), original on NumPy 1.26 / SciPy 1.11 vs fast on NumPy 2.4 / SciPy 1.17 | 4 | 4 | 0 |
| components: GIG (200k draws + RNG position), LAPACK kernel on real LD blocks, sequential sums, parser on BETA/OR × SE/P inputs with flips, swaps and duplicates | unit tests | all pass | 0 |

The task asks for a Monte Carlo tolerance. Because the outputs are bit-identical, the tolerance is zero. For context: two
*seeds* of the original agree only to r ≈ 0.96 (EUR) and 0.94 (EAS) on the example. A re-implementation that was exact
only in distribution could therefore not be told apart from a buggy one, which is why exact stream replay was required.

### 3.3 Runtime and memory
Example data (2,000 iterations; mean of 3 runs; 1 BLAS thread for the original):

| version | wall (s) | speed-up | peak RSS (MB) |
|---|---|---|---|
| original (NumPy 1.26) | 40.7 ± 1.9 | 1.00× | 238 |
| fixed (NumPy 2.4) | 28.0 ± 0.5 | 1.45× | 241 |
| fast, 1 thread | 14.0 ± 0.5 | 2.90× | 251 |
| fast, 1 thread, NumPy 1.26 (same environment as the original) | 12.3 / 12.2 / 14.2 | ≈ 3.1× | — |
| fast, 4 threads | 10.7 ± 0.3 | 3.81× | 279 |
| improved, 1 thread | 10.7 ± 0.3 | 3.81× | 252 |
| improved, 4 threads | 8.3 ± 0.3 | 4.88× | 269 |

All HapMap3 SNPs of chr22 (17,007 SNPs, EUR+EAS, default 2,000 iterations):

| version | wall (s) | speed-up | peak RSS (MB) | output vs original |
|---|---|---|---|---|
| original, 1 thread | 1,186 | 1.00× | 449 | — |
| fast, 1 thread | 701 | 1.69× | 561 | identical |
| fast, 4 threads | 214 | 5.54× | 682 | identical |
| improved, 1 thread | 471 | 2.52× | 556 | r = 0.970 (different model) |
| improved, 4 threads | 138 | 8.59× | 599 | r = 0.970 |

Scaling: simulated EUR+EAS+AFR, 4 cores for every method (the original via BLAS threads). Setup and per-iteration cost
come from runs of 20 and 80 iterations. **The full-run column is an extrapolation** to the default 3,000 iterations.

| SNPs (chromosomes) | original s/iter | fast s/iter | improved s/iter | original full run (h, extrap.) | fast | improved | peak RSS orig / fast / impr (MB) |
|---|---|---|---|---|---|---|---|
| 18,928 (22) | 0.634 | 0.125 (5.1×) | 0.075 (8.5×) | 0.53 | 0.11 | 0.07 | 497 / 709 / 710 |
| 36,992 (21–22) | 1.267 | 0.250 (5.1×) | 0.142 (8.9×) | 1.07 | 0.22 | 0.13 | 517 / 748 / 734 |
| 95,490 (19–22) | 3.201 | 0.600 (5.3×) | 0.426 (7.5×) | 2.71 | 0.52 | 0.38 | 889 / 1,166 / 1,233 |
| 249,925 (15–22) | 8.546 | 1.651 (5.2×) | 1.268 (6.7×) | 7.27 | 1.43 | 1.11 | 983 / 1,292 / 1,367 |

Notes:
- Cost grows linearly with the number of SNPs. Extrapolated to the 1.2M-SNP genome (3 ancestries, 4 cores): original about 35 h, fast about 7 h.
- **Memory** (peak RSS column = final code, 4 threads, re-measured; runtime columns from the timing runs): fast and improved
  use about 200–300 MB more than the original (+42% on chr22, +31% on chr15–22). Contributors are per-thread LAPACK
  workspaces, the numba runtime, and inputs held for all requested chromosomes. The parser keeps only the requested
  ancestries' columns and numeric arrays, which saved about 110 MB at 250k SNPs. Use fewer `--threads`, or one chromosome
  per job, when memory is tight.
- **Exact reproduction keeps the double-precision O(b³) Cholesky of every block in every iteration.** The single-thread
  speed-up is therefore bounded (1.7× at chromosome scale); the large gains come from parallelism.

## 4. Task 2: PRS-CSx-improved
### 4.1 What was changed, and why (details and derivations: `improved/README.md`)
1. **Rao–Blackwellised posterior means.** The average of E[β | ψ, σ², data] = (D + R)⁻¹β̂ over every post-burn-in iteration
   replaces the average of every 5th draw. It costs one extra triangular solve with the Cholesky factor that is already
   computed, has the same expectation, and removes the conditional variance of β given θ.
2. **Effect-size-scale cross-ancestry coupling.** β_jk ~ N(0, σ²_k ψ_j / N̄) with N̄ = max N_k, instead of σ²_k ψ_j / N_k.
   PRS-CSx's choice makes a SNP's prior variance proportional to 1/N_k and gives every ancestry the same shrinkage factor
   ψ/(1+ψ), however noisy its GWAS. Under a shared effect distribution, the Bayes-consistent shrinkage grows with 1/N_k.
   PRS-CSx's ψ ≤ 1 rule (prior variance ≤ sampling variance) is kept *per ancestry* (ridge_jk = max(κ_k/ψ_j, 1)).
   The pilot showed that keeping ψ ≤ 1 globally over-shrinks small GWAS.
3. **Single-precision factorisation.** The floor guarantees eigenvalues ≥ 1. Measured relative error is 1–2·10⁻⁷, against
   Monte Carlo error ≥ 10⁻²; it is 2.1× faster at b = 700.
4. **Optional:** multiple chains (`--n_chains`, law-of-total-variance combination, between-chain R̂), LD shrinkage (`--ld_shrink`; evaluated below
   and *not* recommended), and stacking over φ × ancestry in validation (`tools/combine_prs.py --stack`).

### 4.2 Accuracy benchmark
Simulation: real 1KG LD of EUR, EAS and AFR on chr22 ≤ 35 Mb (≈ 8.2k–8.8k SNPs per ancestry, about 40 LD blocks).
Reference panels are re-estimated from 500 / 500 / 650 simulated people, so there is realistic LD mismatch.
GWAS N = 300k / 100k / 30k; ρ = 0.8; 2% SNPs missing per GWAS; per-SNP heritability of a genome-wide trait.
4 scenarios × 2 replicates = 8 datasets (each replicate has new effects, new GWAS noise and a new MCMC seed).
Both methods use φ ∈ {1e-6, 1e-4, 1e-2, 1} plus auto, META, and the default chain length (3,000 iterations).
**"PRS-CSx" fits were produced with fast.** The unmodified original on NumPy 1.26 gives byte-identical files on these data:
checked on one complete fit, all 4 output files (EUR/EAS/AFR/META) identical.
Accuracy is exact (population level): r²_g = R²/h², the squared correlation of the PRS with the true genetic value.

**Tuned PRS (README procedure: φ grid + linear combination of the ancestry-specific PRS in validation, n = 3,000):**

| scenario (h², π) | target | PRS-CSx r²_g | improved r²_g | PRS-CSx R² / r | improved R² / r |
|---|---|---|---|---|---|
| 0.2, 1% | EUR | 0.524 | **0.573** | 0.105 / 0.324 | 0.115 / 0.338 |
| 0.2, 1% | EAS | 0.400 | 0.396 | 0.080 / 0.283 | 0.079 / 0.282 |
| 0.2, 1% | AFR | 0.211 | 0.211 | 0.042 / 0.205 | 0.042 / 0.206 |
| 0.5, 0.1% | EUR | 0.920 | 0.931 | 0.460 / 0.678 | 0.465 / 0.682 |
| 0.5, 0.1% | EAS | 0.906 | **0.934** | 0.453 / 0.673 | 0.467 / 0.683 |
| 0.5, 0.1% | AFR | 0.781 | 0.787 | 0.390 / 0.625 | 0.394 / 0.627 |
| 0.5, 1% | EUR | 0.748 | 0.748 | 0.374 / 0.612 | 0.374 / 0.612 |
| 0.5, 1% | EAS | 0.680 | 0.678 | 0.340 / 0.583 | 0.339 / 0.582 |
| 0.5, 1% | AFR | 0.455 | **0.474** | 0.228 / 0.477 | 0.237 / 0.487 |
| 0.5, 10% | EUR | 0.552 | 0.552 | 0.276 / 0.525 | 0.276 / 0.526 |
| 0.5, 10% | EAS | 0.409 | 0.408 | 0.204 / 0.452 | 0.204 / 0.451 |
| 0.5, 10% | AFR | 0.230 | 0.237 | 0.115 / 0.339 | 0.119 / 0.344 |

(Means over 2 replicates. R² = r²_g·h² is the phenotypic R² of a genome-wide PRS with this per-SNP accuracy; r = √R².)

**Paired differences, improved − PRS-CSx, over the 8 datasets** (mean ± SE; number of datasets where improved is better;
Wilcoxon signed-rank p). Figure: `benchmarks/results/accuracy_paired_diff.png`.

| metric | EUR | EAS | AFR |
|---|---|---|---|
| tuned (README procedure) | +0.015 ± 0.009 (6/8, p = 0.15) | +0.005 ± 0.007 (4/8, p = 0.74) | **+0.008 ± 0.004 (7/8, p = 0.039)** |
| tuned, oracle φ and weights | +0.012 ± 0.006 (6/8) | +0.008 ± 0.006 (6/8) | +0.008 ± 0.004 (7/8) |
| auto + META (no validation data) | +0.009 ± 0.009 (5/8, p = 0.38) | +0.013 ± 0.007 (6/8, p = 0.11) | +0.016 ± 0.010 (6/8, p = 0.20) |
| improved + stacking vs PRS-CSx README | +0.015 (6/8, p = 0.08) | +0.009 (5/8, p = 0.20) | **+0.013 (7/8, t-test p = 0.038)** |

Relative gains: about 2–4% for tuned PRS (EUR +2.9%, AFR +2.3%; +3.8% for AFR with stacking) and up to +8.8% for AFR
in auto mode. **The direction is consistent, but most differences are within the noise of 8 replicates.** Only the AFR
comparisons reach nominal p < 0.05, without multiple-testing correction. Improved was never much worse on a tuned metric
(worst −0.014); in auto mode the worst single dataset was −0.043 (AFR).

**Where the gain comes from (ablations, h² = 0.5, π = 1%, 2 datasets; tuned r²_g EUR / EAS / AFR):**
PRS-CSx 0.748 / 0.680 / 0.455; RB only 0.740 / 0.665 / 0.447; coupling only 0.745 / 0.677 / 0.472; both (improved)
0.748 / 0.678 / 0.474. **The AFR gain comes from the effect-size-scale coupling.** RB alone did *not* improve prediction
measurably here. Its differences are within the MCMC-seed noise below; its demonstrated benefit is lower Monte Carlo
noise of the posterior means (example data: seed-to-seed r 0.963 → 0.984 EUR, 0.937 → 0.972 EAS).

**Monte Carlo variability (same dataset, MCMC seeds 1/11/12; SD of r²_g):** tuned PRS-CSx 0.004 / 0.008 / 0.004
(EUR/EAS/AFR) and improved 0.006 / 0.005 / 0.011. Auto + META varies more: PRS-CSx 0.022 / 0.014 / 0.022, improved
0.012 / 0.005 / 0.006 (auto mode is less seed-sensitive under the improved model). These SDs are the same order as the
method differences, which is why paired replicates were needed.

**Averaging independent chains** (post hoc from the 3 seeds; `--n_chains`): tuned r²_g single-chain mean → 2-chain → 3-chain
averages: PRS-CSx EUR 0.770 → 0.773 → 0.776, AFR 0.435 → 0.442 → 0.444; improved EUR 0.780 → 0.788 → 0.786, AFR
0.443 → 0.454 → 0.450. Averaging chains helps both methods, by about as much as the model change. **Recommendation: run
2–3 chains** (`--n_chains=2`). With single precision, 2 improved chains (≈ 940 s on chr22, 1 thread) still cost less than
one original chain (1,186 s).

**Stacking over φ × ancestry** (NNLS in validation, `tools/combine_prs.py --stack`) vs selecting one φ (README):
PRS-CSx EUR +0.005 (5/8), EAS +0.0055 (8/8), AFR +0.006 (7/8); improved +0.000 / +0.004 / +0.005. A small but consistent gain at no MCMC cost.

**LD shrinkage toward I** (Schäfer–Strimmer λ from n_ref; one dataset, on top of improved): EUR unchanged, EAS −0.012,
AFR −0.028 (tuned) and −0.031 (auto). That is beyond the seed SD. **Not beneficial; off by default.** A plausible reason is that
PRS-CSx's ridge floor already regularises the LD, and extra shrinkage distorts true correlations.

**Estimation error** ‖w − β‖²/‖β‖² of the ancestry-specific posterior (best φ): improved is lower for EUR and EAS in 7 of 8
scenario × ancestry cells (e.g. h² = 0.2: EUR 0.884 → 0.844), but slightly higher for the AFR-specific posterior (e.g. 0.938
→ 0.959). This is consistent with stronger, Bayes-consistent shrinkage of the noisier GWAS. Prediction *in* AFR still
improves through the cross-ancestry combination.

### 4.3 Runtime of improved
See §3.3: 2.5× (1 thread) and 8.6× (4 threads) faster than the original on chr22 (earlier defaults; current defaults: §7.6); 6.7–8.9× per iteration in the
3-ancestry scaling runs; about 1.3–1.7× faster than fast.

### 4.4 Bottom line for Task 2
*Revised on 19 November 2026; the original v1 conclusion said improved was "at least as accurate in every summary metric"
with the `effect` coupling.* The v1 gains were small and mostly within the noise of 8 replicates. The follow-up work (§7)
changed the picture: the `effect` coupling helps when standardised effect variances are equal across ancestries and
loses accuracy when they are not (§7.4), so the default is now PRS-CSx's coupling. The pre-registered benchmark of §7.5
decides what may be claimed about prediction: no method passed the claim rule in any ancestry (pooled gains of +0.001 to +0.008 in tuned r²_g, below the pre-set 0.01; some scenarios worse by up to 0.027), so better prediction is **not** demonstrated.
What improved/ demonstrably offers: the same model as PRS-CSx, faster (about 8× vs the original on chr22 with 4 threads),
with lower Monte Carlo noise in the posterior means (RB), an R̂ diagnostic and automatic stopping with 2 chains.
Stacking over φ × ancestry (`tools/combine_prs.py --stack`) gave a small, consistent gain in v1 for both methods.

## 5. Robustness findings and failures (honest log)
1. **NumPy ≥ 2 hang** (§2). Also: my own negative-control test hung the shell, because the original's bare `except` swallows
   the timeout exception. The sandbox had to be restarted; afterwards such tests were run in the background with SIGKILL timeouts.
2. **LAPACK `gesdd` non-convergence.** The original's PSD projection (`scipy.linalg.svd`, default `gesdd`) failed with "SVD did
   not converge" on simulated LD blocks with n_ref ≈ 500 ≪ block size. 2 of about 1,300 blocks failed on the first draw with
   SciPy 1.17 and 1 thread; 2 different blocks failed with SciPy 1.11; and 1 more failed with SciPy 1.17 at 4 BLAS threads.
   Convergence therefore depends on the LAPACK build *and* the thread count. The original crashes on such blocks. Fast and
   improved fall back to `gesvd` with a warning (tested on a saved failing block). Benchmark panels were redrawn where needed
   so that the unmodified original could run on them.
3. **SE = 0 in the sumstats:** the original crashes with `ZeroDivisionError`. An early version of fast silently produced `inf`
   (found by the parser test); it now stops with a clear error naming the SNP.
4. **Pilot of the improved model:** effect-scale coupling combined with PRS-CSx's ψ ≤ 1 cap over-shrank the small-N ancestry
   (AFR 0.424 → 0.389). Fixed by the per-ancestry floor (`improved/README.md`).
5. **Monte Carlo noise in accuracy is large.** The same data with three MCMC seeds (600-iteration pilot) gave AFR r²_g from
   0.360 to 0.424. Comparisons therefore use replicates and paired differences (§4).
6. **Sandbox restarts** killed long jobs twice. All benchmarks run from a restart-safe job queue. Two runtime jobs that ran
   during a contaminated window (old data and CPU contention) were rerun on an idle machine.

## 6. Running on real data
See `README.md`, `fast/README.md`, `improved/README.md`. Inputs and outputs are the original formats. `tools/combine_prs.py`
implements the validation step (combine ancestry-specific PRS per φ, select φ, apply to test), and `tools/compare_prs.py`
compares methods on a test set with paired bootstrap intervals. Step-by-step protocol for UKB / BBJ / TWB / KoGES summary
statistics and a CKB target: `docs/06_real_data_evaluation.md`. Building an ancestry-matched LD panel: `docs/05`.

## 7. Follow-up work after the external review (WP1–WP7, 18–19 November 2026)
Status table with dates: `docs/07_status_next_steps.md`. Raw results: `benchmarks/results_v2/`. The review files the
instructions referred to were not available; the synthetic panel generator and the coupling stress test were
re-implemented from their description. The 1000 Genomes panel (EUR, EAS, AFR) was available for every test that needs it.

### 7.1 WP1: harder exactness tests and run-time safety for fast/
- **Synthetic reference panel** with the awkward cases of real panels (`tests/make_synth_ref.py`): 1,000 test SNPs plus 20
  panel-only SNPs in 7 blocks, including an empty block, a block present only in the panel, a 350-SNP block larger than the
  200-person panel, 10% allele flips, 5% A1/A2 swaps and 3% of SNPs missing per ancestry. fast is byte-identical to the
  original in all 4 configurations (`tests/test_fast_synthetic.py`).
- **Start-up self-check** (0.8 s): 2,000 GIG draws against the verbatim original sampler and one 60 × 60 block update
  against SciPy. A mismatch prints a warning (output then statistically equivalent, not byte-identical); `--strict` stops
  with exit code 3. The random state is restored, so the check never changes outputs.
- **`--n_jobs` over (chromosome, φ) pairs**, each seeded as a separate run: a 2-value φ grid with `--n_jobs=2` gave the same
  4 files as two separate runs.

### 7.2 WP2: low-rank LD-block updates (`--linalg`, `--rank_var`; fast and improved)
Woodbury form with U = R^{-1/2}VΛ^{1/2}, K = I + UᵀU (derivation in `improved/README.md` §3). Tests (`tests/test_lowrank.py`):
- T2a: conditional mean and covariance equal the full update on 40 blocks (20 real 1000 Genomes blocks): double < 1e-9,
  single < 1e-4.
- T2b: the draws have the right distribution.
- T2c, same posterior on real 1000 Genomes LD: correlation of posterior means full–full 0.9630 vs full–low-rank 0.9645
  (EUR), 0.9356 vs 0.9370 (EAS).
- T2d, speed (1 thread, rank 499): 3.5× at b = 2,000, 16× at 5,000, 55× at 10,000 (single precision 7×, 33×, 72×). The measured
  break-even b/r ≈ 1.6–1.8 gives `auto`'s threshold of 1.7.
- Eigenvalues ≤ 1e-13 × λ_max are treated as zeros: on 1000 Genomes chr22 the exact zeros lie below 1e-15 · λ_max and no
  eigenvalue lies between 1e-15 and 1e-12, whereas a 1e-8 cut (as first suggested) would drop 766 genuine eigenvalues.
- fast keeps `--linalg=full` as its default (byte-identical); improved now defaults to `auto`.

### 7.3 WP3: LD panels in eigen form, panel builder, truncation
- Format (`docs/04_eigen_panel_format.md`), builder/merger/converter (`tools/build_ld_panel.py`, own PLINK .bed reader),
  reading in both programs (row subset, allele flips, re-orthonormalisation). Tests: `tests/test_eigen_panel.py`.
- Built panel vs dense panel (same people, 3 blocks, 5,800 SNPs): LD reproduced to 3.4e-8; correlation of posterior means
  dense–dense 0.9004 vs dense–eigen 0.8978 (difference 0.0026, criterion 0.005); 78 s vs 122 s per fit.
- Truncation (`--rank_var`), chr22 HapMap3, real 1000 Genomes LD: 0.999 is within the Monte Carlo noise floor of no
  truncation (EUR 0.9976 vs floor 0.9979; EAS 0.9966 vs 0.9968) and 1.7× faster; 0.995 is measurably worse. **0.999 is the
  builder's default.** Details and build cost: `docs/05_building_a_dense_panel.md`.

### 7.4 WP4: a learned cross-ancestry scale, and a stress test of the couplings
- `--coupling=learned`: β_jk ~ N(0, σ²_k ψ_j τ_k/N̄), τ_k ~ IG(1, 1), τ = 1 for the largest GWAS; exact conditional
  τ_k | · ~ IG(1 + p_k/2, 1 + N̄/(2σ²_k) Σ_j β²_jk/ψ_j) (derivation in `improved/README.md` §6). Numerical check against the
  exact posterior on a small model: E[τ₂] 5.9244 (Gibbs) vs 5.9243 (exact), relative error 0.002%.
- Stress test (`benchmarks/coupling_stress.py`; 5 scenarios × 20 replicates; synthetic LD with 2,000 SNPs, N = 300k / 100k /
  30k; oracle tuning over φ ∈ {1e-4, 1e-2, 1}): table in `improved/README.md` ("Evidence"). `effect` gains up to +0.013
  when its equal-variance assumption holds and loses up to −0.014 (EAS, SE 0.006) when the AFR effect variance is 10×
  larger. `learned` is not uniformly safe: minimum learned − best(snr, effect) = −0.012, minimum learned − snr = −0.0096.
- **Decision by the pre-set rule: the default coupling is `snr` (PRS-CSx's model).** Neither alternative coupling is
  recommended as a default. This supersedes the v1 recommendation of `effect` (§4).
- Not done: the suggested repetition of the stress test on real LD with 20 replicates (estimated at about 37 hours of
  compute here).

### 7.5 WP5: accuracy benchmark v2 (pre-registered)
Plan fixed before the run: `benchmarks/PLAN_accuracy_v2.md` (36 scenarios: h² ∈ {0.2, 0.5} × π ∈ {0.1%, 1%, 10%} ×
ρ ∈ {0.4, 0.8} × AFR effect-variance multiplier ∈ {1/3, 1, 3}; 20 replicates each; synthetic LD, 2,000 SNPs; N = 300k /
100k / 30k; φ ∈ {1e-6, 1e-4, 1e-2}; φ and combination weights tuned in a validation sample of 3,000). Rule: "better"
needs a pooled mean gain in tuned r²_g above 0.01, a 95% CI excluding 0, and no scenario mean below −0.01.
All 720 datasets ran (`benchmarks/results_v2/wp5_accuracy/summary.md`).

| method − PRS-CSx (tuned r²_g, 720 datasets) | EUR | EAS | AFR | worst scenario mean | verdict |
|---|---|---|---|---|---|
| `snr` (default: PRS-CSx model, RB, single precision) | +0.0022 [+0.0005, +0.0039] | +0.0030 [+0.0006, +0.0055] | +0.0008 [−0.0020, +0.0035] | −0.027 (AFR) | no demonstrated difference |
| `effect` | +0.0062 [+0.0043, +0.0082] | +0.0075 [+0.0048, +0.0102] | +0.0069 [+0.0041, +0.0097] | −0.016 (AFR) | no demonstrated difference |
| `learned` | +0.0062 [+0.0040, +0.0083] | +0.0044 [+0.0015, +0.0073] | +0.0065 [+0.0035, +0.0095] | −0.016 (AFR) | no demonstrated difference |
| default with 2 chains | +0.0050 [+0.0034, +0.0065] | +0.0051 [+0.0029, +0.0074] | +0.0050 [+0.0024, +0.0076] | −0.017 (AFR) | no demonstrated difference |

- **Verdict by the rule: no demonstrated difference for any method in any target ancestry; the general claim "improved
  predicts better than PRS-CSx" is not supported.**
- The pooled means are small and mostly positive, and most CIs exclude 0: with 720 paired datasets, gains of 0.002–0.008
  in r²_g are detectable, but they are below the 0.01 that the plan required, and every method has scenarios where it is
  worse on average (by up to 0.027 for the default in AFR, h² = 0.2, π = 1%, ρ = 0.4, AFR ×3).
- Mean tuned r²_g over all datasets: PRS-CSx 0.638 / 0.546 / 0.485 (EUR / EAS / AFR); improved default 0.640 / 0.549 /
  0.486; with 2 chains 0.643 / 0.551 / 0.490.
- In this range of scenarios (AFR multiplier ≤ 3), `effect` was on average slightly better than `snr` (by +0.004 to +0.011
  for multipliers 1/3 and 1, +0.001 to +0.003 for 3). The default stays `snr` as decided by WP4's rule, which weighed the
  losses of `effect` when the multiplier is 10.
- Run time per dataset (3 φ, 1 core): PRS-CSx 42.4 s, improved 41.2 s, 2 chains 82.3 s. With blocks of only 100 SNPs, the
  factorisation is a small part of the cost, so single precision gains little here (on real chromosomes see §3.3).

### 7.6 WP6: speed and convergence options of improved/
**Warm starts over a φ grid (`--warm_start`).** `benchmarks/wp6_warm_autostop.py`, 30 datasets of WP5's ρ = 0.8, multiplier-1
scenarios (5 replicates × 6), grid of 4 φ, burn-in halved after the first φ, same kept iterations, coupling snr:
- Speed-up 1.23× in iterations and 1.23–1.25× in measured time, below the 1.3× that the instructions set for keeping the
  option. With half the burn-in this ceiling is built in: 1 − (1 + 3 × 0.75)/4 = 19% fewer iterations.
- Accuracy, tuned r²_g (90 paired comparisons): grid in **decreasing** φ order (the first version), warm − cold
  −0.0097 [95% CI −0.0149, −0.0044], against −0.0021 [−0.0077, +0.0035] between two cold runs with different seeds (the
  Monte Carlo noise reference): a real loss, largest at φ = 1e-4 (−0.0097). Grid in **increasing** order (run afterwards to
  see whether the order matters): +0.0015 [−0.0030, +0.0060], no detectable loss.
- Consequence: `--warm_start` now runs the grid in increasing order, and it stays off by default because the speed
  criterion is not met.

**Automatic stopping (`--auto_stop`, 2 chains).** 120 datasets (20 replicates × 6 scenarios) × 3 φ = 360 runs; maximum
3,000 iterations, burn-in 1,500:
- All 360 runs met the R̂ rule before the maximum; median stop at iteration 2,100 (the earliest possible: burn-in + 3 × 200),
  mean 2,199, range 2,100–2,900.
- Cost: 0.73 of two fixed-length chains, 1.47 of one chain.
- Accuracy (tuned r²_g, validation procedure; 360 paired comparisons): auto-stopped 2 chains − fixed 2 chains −0.0009
  [95% CI −0.0022, +0.0004]; auto-stopped 2 chains − 1 fixed chain +0.0001 [−0.0017, +0.0020]; fixed 2 chains − 1 chain
  +0.0010 [−0.0007, +0.0027].
- Verdict: `--auto_stop` saves 27% of the cost of two chains without a detectable loss, but in these scenarios two chains
  did not predict measurably better than one. It is useful when two chains are wanted for the R̂ diagnostic; it is not a
  speed-up over the default single chain. Off by default.

**Speed of the current defaults (re-measured).** During the follow-up the sandbox moved from a 2.30 GHz to a 2.10 GHz
Xeon, so speed is only compared within one machine. On the 2.10 GHz machine (chr22, all HapMap3 SNPs, EUR + EAS, 2,000
iterations, `benchmarks/results_v2/speed_chr22/`): fast 919 s (1 thread) and 258 s (4 threads); improved with the current
defaults 567 s and 180 s (mean of the `auto` and `full` runs, 183 s and 176 s; with the 1000 Genomes panel no chr22 block
is large enough for the low-rank update, so `auto` = `full` here). Improved is 1.62× (1 thread) and 1.43× (4 threads) faster
than fast. Combined with the original/fast ratios measured earlier on one machine (1.69× and 5.54×), improved is about
2.7× (1 thread) and 7.9× (4 threads) faster than the original. These are derived; the original was not rerun on the new
machine. (With the earlier defaults the measured values were 2.5× and 8.6×.)

**Profile of a low-rank chromosome run (6.3).** `benchmarks/wp6_profile.py`, improved/ with `--linalg=lowrank`, all
HapMap3 SNPs, EUR + EAS (`benchmarks/results_v2/wp6_profile_chr22.json`, `..._chr1.json`). Share of the time inside the
Gibbs iterations:

| run | s/iteration | LD-block updates | GIG draws | everything else |
|---|---|---|---|---|
| chr22 (≈ 17k SNPs), 1 thread | 0.286 | 98.0% | 1.6% | 0.4% |
| chr22, 4 threads | 0.080 | 92.9% | 5.7% | 1.4% |
| chr1 (≈ 93k SNPs), 4 threads | 0.417 | 92.7% | 6.0% | 1.3% |

The remaining Python/NumPy code takes 0.4–1.4% of the time, far below the 15% that would justify parallelising it with
numba `prange`; it was left as it is. No GPU was used (none available).

### 7.7 WP7: documentation
README files of fast/, improved/ and tools/ updated (self-check, `--strict`, `--n_jobs` semantics, `--linalg`, `--rank_var`,
eigen panels, timing line, what "same answer" means for low-rank runs, new options); "split R̂" corrected to
between-chain R̂; new documents 04 (eigen format), 05 (building panels), 06 (real-data protocol), 07 (status). Claims of
better prediction that the new benchmarks do not support were removed (§0, §4.4).
