# PRS-CSx-fast

A drop-in, accelerated re-implementation of PRS-CSx. For the same inputs and `--seed`, its posterior effect files are
byte-identical to the original `PRScsx.py`, both the unmodified original on NumPy 1.26 and the NumPy-2-fixed version.

## Requirements
Python ≥ 3.9, numpy, scipy, h5py, **numba** (`pip install numba`). Tested with Python 3.11, NumPy 2.4.6 / SciPy 1.17.1 /
numba 0.68.1, and NumPy 1.26.4 / SciPy 1.11.4 / numba 0.68.1.

## Usage
It takes the same command line as the original. All original options are accepted, with the same meaning and defaults.
```
python PRScsx_fast.py --ref_dir=PATH_TO_REFERENCE --bim_prefix=VALIDATION_BIM_PREFIX \
    --sst_file=EUR_sumstats.txt,EAS_sumstats.txt --n_gwas=200000,100000 --pop=EUR,EAS \
    --out_dir=OUTPUT_DIR --out_name=PREFIX [--phi=1e-2 --chrom=22 --meta=TRUE --seed=1 ...]
```
Extra options:

| option | meaning |
|---|---|
| `--threads=N` | threads for the LD-block updates (default: number of CPUs, max 8). Results do not depend on N. |
| `--n_jobs=J` | (chromosome, φ) pairs processed in parallel processes (default 1). Each pair is seeded exactly as a separate run, so the files do not depend on J (verified: a 2-value φ grid with `--n_jobs=2` gives the same 4 files as two separate runs). Uses about J×N cores. Without `--seed`, every worker draws a fresh random seed from the operating system. |
| `--phi=1e-6,1e-4,1e-2,1,auto` | runs the whole φ grid in one process, sharing parsing and LD. The RNG is re-seeded per φ, so each file equals a separate run. |
| `--ld_cache_dir=DIR` | caches processed LD blocks (subset, sign flip, PSD projection) and, with `--linalg` ≠ full, their eigendecompositions; reused in later runs with the same SNP set. |
| `--quiet` | prints only per-chromosome progress and the MCMC timing line. |
| `--strict` | stop (exit code 3) if the start-up self-check fails, instead of printing a warning (see below). |
| `--linalg=full` | `full` (default): the original's Cholesky factorisation of every LD block, **byte-identical output**. `lowrank` / `auto`: the low-rank (Woodbury) block update, which costs about b·r² instead of b³/3 for a block of b SNPs whose LD matrix has rank r (`auto`: low-rank only for blocks with b > 1.7 r, the measured break-even). Not byte-identical, see below. |
| `--rank_var=1` | keep the largest LD eigenvalues that explain this share of each block's trace (1 = no truncation, default). Below 1, the marginal effects are projected on the kept eigenvectors. Not byte-identical. |

Every run prints one line per φ, e.g. `... MCMC: 1000 iterations in 41.3 s (0.0413 s/iteration) ...`, also with `--quiet`.

Inputs (reference panels, `.bim`, sumstats in BETA/OR + SE/P format) and outputs (`<prefix>_<POP>_pst_eff_a1_b0.5_phi<φ>_chr<c>.txt`,
`..._META_...`) are exactly those of the original. See the original README for their description.

Whole genome on a 16-core node, e.g. 4 chromosomes at a time with 4 threads each:
```
python PRScsx_fast.py ... --phi=1e-6,1e-4,1e-2,1 --n_jobs=4 --threads=4 --seed=1
```

## Why it is faster, and why it is still exact
| Component | Original | Fast | Exactness argument |
|---|---|---|---|
| GIG draws for ψ (one per SNP per iteration) | interpreted Python, 3 scalar RNG calls per attempt | numba port of the same Devroye algorithm (`fastcsx/gig.py`) | Same operations and order, same C maths library as CPython's `math` (glibc on Linux; also verified on Windows 11, review of 18 Nov 2026). `pow(x,2)` is called with a run-time exponent because LLVM would turn it into `x*x`, which differs from glibc `pow` in 0.09% of inputs (measured). Python exceptions are emulated. Uniforms are pre-drawn with `random_sample(L)` (same stream as L `random()` calls) and the generator is rewound to exactly the number consumed. Verified on 200,000 draws with extreme parameters. |
| β update per LD block | `scipy.linalg.cholesky` + 2× `solve_triangular` (f2py wrappers hold the GIL; `check_finite` overhead) | the same LAPACK routines (`dpotrf('U')`, `dtrtrs`) from the same OpenBLAS via `scipy.linalg.cython_lapack`, called from a nogil numba kernel (`fastcsx/lapack.py`) | Same routine, arguments and memory layout (Fortran copy of D + Ψ⁻¹, as f2py makes). Verified bit-identical on real LD blocks. |
| Parallelism | none (BLAS threads did not help: 38.9 s vs 40.6 s on the example) | all LD blocks of all ancestries run concurrently on a thread pool | Given ψ and the previous σ², ancestries are independent. All random numbers are drawn beforehand in the original order: `randn(p_k)` equals the per-block `randn` calls, and `gamma(k, s)` equals `s*standard_gamma(k)` bitwise. Block results are summed in the original order. |
| Bookkeeping | `snp in list`, `list.index`, `jj in idx_dict` (O(p²)); sumstats read twice per chromosome | sets/dicts; sumstats and SNP info read once per run; only LD blocks with retained SNPs are read | Same SNPs, order, alleles, flips; duplicate SNP lines resolved as in the original (last wins). |
| Python `sum()` over (p,1) arrays | sequential | `np.cumsum(x)[-1]` (sequential) | Bitwise equal (tested). |

**Differences from the original (only where the original fails):**
- If LAPACK `gesdd` does not converge in the LD PSD projection (observed on strongly rank-deficient blocks; depends on
  LAPACK build and BLAS thread count), the original stops with "SVD did not converge". fast falls back to `gesvd` and
  prints a warning.
- SE = 0 in a sumstats file: the original raises `ZeroDivisionError`; fast stops with a message naming the SNP.
- The NumPy ≥ 2 incompatibilities and the other errors fixed in the root `PRScsx.py` (see `docs/02_fix_numpy2_and_robustness.md`) are absent.

**Memory.** Peak RSS is about 200–300 MB above the original: per-thread LAPACK workspaces, the numba runtime, and inputs
for all requested chromosomes. At 4 threads: 709 vs 497 MB on chr22; 1,292 vs 983 MB on chr15–22 (3 ancestries). Use fewer
`--threads`, or one chromosome per job, if memory is tight.

**Limits.** At realistic block sizes the run time is dominated by the O(b³) Cholesky factorisations. Exact
reproduction requires the same double-precision factorisation every iteration, so the single-thread speed-up is bounded:
about 3× on the example and less at full chromosome scale. The large gains come from parallelism over blocks and
ancestries. For faster, non-identical inference see `../improved` (single-precision factorisation, Rao–Blackwellised
estimates).

## Start-up self-check
Bit identity depends on numba, LLVM, the C maths library and the LAPACK build of the machine. At every start
(0.8 s), fast compares 2,000 GIG draws over a wide range of parameters with the verbatim original sampler
(`fastcsx/gigrnd_reference.py`) and one 60×60 LD-block update with SciPy's routines. If anything differs, it prints a
warning saying that the output will be statistically equivalent but not byte-identical; with `--strict` it stops with
exit code 3 instead. The check restores the random state, so it does not change the output.

## `--linalg=lowrank|auto`, `--rank_var` and eigen-form LD panels: what "same answer" means
With `--linalg=full` (the default) the output is byte-identical to the original. With `lowrank`, `auto` or
`--rank_var < 1` it is not: the model, the prior, the Gibbs updates and the estimator are the same, but the β draw
of a low-rank block uses a different (exact) formula and consumes different random numbers. The output then differs
from the original exactly as the output of a run with another seed does. This was tested (`tests/test_lowrank.py`): on
real 1000 Genomes LD, the correlation of posterior means between two original runs with different seeds and between an
original run and a low-rank run is the same (EUR 0.9630 vs 0.9645, EAS 0.9356 vs 0.9370). `--rank_var < 1` also changes the
model slightly (truncated LD); see `../docs/05_building_a_dense_panel.md` for the measured effect. The program prints a
notice when one of these options is used.

LD panels in eigen form (`../docs/04_eigen_panel_format.md`, built with `../tools/build_ld_panel.py`) are detected
automatically from `--ref_dir`. They contain no b × b matrices, so fast switches to `--linalg=auto` and prints the notice.

## Verification (see `../benchmarks` and `../tests/test_fast.py`)
- Example data: 9 configurations × {1, 4} threads, 40 output files, all byte-identical to the original.
- Same environment as the original (NumPy 1.26.4 / SciPy 1.11.4, 1 thread), seeds 1–3: byte-identical, 12.2–14.2 s
  vs 40.7 s for the original (the speed-up does not come from the NumPy version). The very first run in a new
  environment took 38.5 s wall time (13 s CPU), a one-off wait not spent computing.
- All HapMap3 SNPs of chr22 (17,007 SNPs, 2 ancestries, 2,000 iterations): byte-identical to the original at 1 and 4 threads.
  Original 1,186 s; fast 701 s (1 thread, 1.7×) and 214 s (4 threads, 5.5×).
- Scaling (3 ancestries, 4 cores each, original via BLAS threads): 5.1–5.3× faster per iteration from 19k to 250k SNPs.
- Example data: 40.7 s → 14.0 s (1 thread) and 10.7 s (4 threads).
- Synthetic reference panel with awkward cases (`tests/make_synth_ref.py`: an empty block, a block present only in the
  panel, a 350-SNP block larger than the 200-person panel, 10% allele flips, 5% A1/A2 swaps, 3% SNPs missing per
  ancestry), 4 configurations: byte-identical (`tests/test_fast_synthetic.py`).
- Low-rank update (`tests/test_lowrank.py`): same conditional mean and variance as the full update on 40 blocks (20 real
  1000 Genomes blocks; double precision < 1e-9, single < 1e-4); speed at r = 499 (1 thread): 3.5× at b = 2,000, 16× at
  b = 5,000, 55× at b = 10,000 (`../benchmarks/results_v2/wp2_T2d_lowrank_speed.json`).
