# PRScsx-fast

A fork of [PRS-CSx](https://github.com/getian107/PRScsx) (Ruan et al., *Nat Genet* 2022) with three things added:

| | what | output compared with the original |
|---|---|---|
| `PRScsx.py` (root) | the original program with the hang under NumPy 2 and a few related errors fixed | byte-identical |
| `fast/PRScsx_fast.py` | accelerated implementation: compiled GIG sampler, LAPACK block updates on threads, linear-time parsing, a whole phi grid in one run, parallelism over (chromosome, phi), LD cache | **byte-identical** for the same `--seed` |
| `improved/PRScsx_improved.py` | the same model with Rao-Blackwellised posterior means, single-precision and low-rank LD-block updates, several chains with R-hat, optional automatic stopping | same file formats, different numbers |
| `tools/` | `build_ld_panel.py` (LD panels in eigen form from your own PLINK genotypes), `combine_prs.py`, `compare_prs.py` | |

The original README, with the input formats and the links to the LD reference panels, is
[`README_PRScsx.md`](README_PRScsx.md). All original options work unchanged in all three programs.

## What to expect

Measured on chromosome 22, all HapMap3 SNPs, EUR + EAS, 2,000 iterations (details in
[`docs/03_final_report.md`](docs/03_final_report.md)):

| | 1 thread | 4 threads |
|---|---|---|
| `fast` vs original | 1.7x faster | 5.5x faster |
| `improved` vs original | about 2.7x faster | about 7.9x faster |

* **`fast` gives the same answer.** Posterior effect files are byte-identical to the original for
  the same seed. A 0.8 s self-check at start-up confirms this on each machine (`--strict` stops if
  it fails).
* **`improved` is faster, not more accurate.** In a pre-registered simulation benchmark (720
  datasets) its prediction accuracy was not demonstrably different from PRS-CSx: pooled gains of
  0.001 to 0.003 in r2 with the default settings (up to 0.008 with the experimental couplings),
  below the 0.01 threshold set in advance, with some scenarios slightly worse. Its benefits are speed, lower Monte Carlo noise and convergence diagnostics. The
  alternative couplings (`--coupling=effect|learned`) are experimental and are not the default.
* **Low-rank updates** (`--linalg=lowrank|auto`) are up to 55x faster per large LD block when the
  rank of the block is small (about 500 for the 1000 Genomes panels). With a panel built from many
  thousands of people the rank approaches the block size and the gain disappears. Whether dense
  SNP sets are practical therefore depends on the rank of real dense LD blocks, which has not been
  measured here.
* Everything above is from simulations and the 1000 Genomes LD panels. No real phenotypes were
  analysed. A protocol for a real-data evaluation is in
  [`docs/06_real_data_evaluation.md`](docs/06_real_data_evaluation.md).

## Install

```
pip install -r requirements.txt     # numpy, scipy, h5py, numba; Python >= 3.9
python demo/run_demo.py             # no download: synthetic LD panel + example data; checks fast == original
sh get_reference.sh ~/prscsx_ref 1kg EUR,EAS    # the real LD panels (4 to 6 GB per ancestry)
```

Tested with Python 3.11 and NumPy 2.4 / SciPy 1.17, and with NumPy 1.26 / SciPy 1.11.

## Use

```
# same answer as PRS-CSx, faster (extra options: --threads, --n_jobs, phi grid, --ld_cache_dir, --strict)
python fast/PRScsx_fast.py --ref_dir=REF --bim_prefix=TARGET --sst_file=EUR.txt,AFR.txt --n_gwas=300000,30000 \
    --pop=EUR,AFR --out_dir=OUT --out_name=trait --phi=1e-6,1e-4,1e-2,1 --seed=1 --n_jobs=4 --threads=4

# improved (extra options: --n_chains, --auto_stop, --linalg, --rank_var, --rb, --precision, ...)
python improved/PRScsx_improved.py --ref_dir=REF --bim_prefix=TARGET --sst_file=EUR.txt,AFR.txt --n_gwas=300000,30000 \
    --pop=EUR,AFR --out_dir=OUT --out_name=trait --phi=1e-6,1e-4,1e-2,1,auto --meta=TRUE --seed=1

# combine the per-ancestry scores in a validation sample, evaluate in a test sample, compare methods
python tools/combine_prs.py --scores_val val_scores.tsv --scores_test test_scores.tsv --pheno pheno.txt --covar covar.txt \
    --write_test_scores --out trait_improved
python tools/compare_prs.py --pheno pheno.txt --covar covar.txt \
    --scores original=trait_original.test_scores.tsv improved=trait_improved.test_scores.tsv
```

Options of each program: [`fast/README.md`](fast/README.md), [`improved/README.md`](improved/README.md),
[`tools/README.md`](tools/README.md). Building an LD panel from your own genotypes:
[`docs/05_building_a_dense_panel.md`](docs/05_building_a_dense_panel.md).

## Tests

```
python -m pytest tests/                                   # 52 tests, no download needed
PRSCSX_REF_DIR=/path/to/1kg_ref python -m pytest tests/   # 71 tests with the 1000 Genomes panels
```

The tests compare `fast` with stored outputs of the original and with the original run on a
synthetic panel that has the awkward cases of real panels (empty blocks, panel-only blocks, blocks
larger than the panel, allele flips and swaps, missing SNPs).

## Documents

| | |
|---|---|
| `docs/01_PRScsx_method_and_code_summary.md` | what PRS-CSx does and how the code is organised |
| `docs/02_fix_numpy2_and_robustness.md` | the NumPy 2 fix (also as `prscsx_numpy2_fix.patch`) |
| `docs/03_final_report.md` | methods, exactness checks, runtime and accuracy benchmarks |
| `docs/04_eigen_panel_format.md`, `docs/05_building_a_dense_panel.md` | LD panels in eigen form |
| `docs/06_real_data_evaluation.md` | protocol for evaluating on real data |
| `docs/07_status_next_steps.md` | what was and was not verified |
| `benchmarks/` | benchmark scripts and result summaries |

The documents sometimes refer to raw per-run outputs (`baseline/`, sub-folders of
`fix_verification/`, `benchmarks/**/runs/`, `benchmarks/results/job_records/`). Those are not in the
repository; the summaries and tables computed from them are.

## Licence and citation

MIT, as the original (see `LICENSE`). If you use this software, cite PRS-CSx:

Ruan Y, Lin YF, Feng YCA, et al. Improving polygenic prediction in ancestrally diverse populations.
*Nature Genetics* 54:573-580 (2022).
