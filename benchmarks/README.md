# Benchmarks

All numbers in the final report were produced by the scripts in this folder, on a 4-core Intel Xeon @ 2.30 GHz with
15 GB RAM. Results are in `results/` (CSV, Markdown and PNG).

## 1. Example data (repo `test_data/`)
- `run_baseline_example.sh` runs the original on the example (seeds 1–5, meta, auto-φ, single ancestry, threads).
- `compare_outputs.py` compares two output files (byte identity, max/mean |Δ|, Pearson r).
- `verify_fixed_vs_original.py` and `verify_fast_vs_original.py` rerun every baseline configuration with the fixed and fast versions and compare all files.
- `stage_timing_original.py` gives a per-stage timing of the original. `run_timed.py` records wall/CPU/peak RSS of one run.

## 2. Simulation (`sim/`)
`sim/simulate.py make-ref` builds a reference panel and `make-data` builds GWAS summary statistics, a target `.bim` and the truth. Everything
is written in the original PRS-CSx formats, so all methods (including the unmodified original) run on it unchanged.

| design element | choice | why |
|---|---|---|
| true LD | real 1KG LD blocks of EUR / EAS / AFR (same block boundaries as PRS-CSx), PSD-projected, ridge ε = 0.03, unit diagonal | realistic, ancestry-specific LD; 1KG blocks are rank-deficient (≈500 people), real population LD is full rank |
| reference LD used by the methods | sample correlation of n_ref = 500 / 500 / 650 individuals drawn from N(0, R_pop) | realistic LD-reference mismatch (finite reference panel, like 1KG) |
| effects | causal status shared across ancestries (prob. π); effects N(0, C), cross-ancestry correlation ρ = 0.8; scaled so βᵀR_kβ = h²·M_region/M_genome | per-SNP heritability of a genome-wide trait with heritability h², so N, φ and polygenicity are realistic for a sub-region |
| summary statistics | β̂_k = R_k β_k + N(0, R_k/N_k), written per allele (BETA, SE); 2% of SNPs missing per ancestry | exact sampling distribution of marginal GWAS estimates |
| sample sizes | EUR 300k, EAS 100k, AFR 30k | typical current imbalance |

`sim/evaluate.py` computes population-level (exact) accuracy from the true effects and true LD:
- r²_g = (wᵀRβ)² / (wᵀRw · βᵀRβ) = R²/h² for every ancestry-specific PRS applied in every target ancestry;
- **tuned (oracle):** best φ with population-optimal weights for combining the ancestry-specific PRSs;
- **tuned (validation):** φ and weights chosen in a simulated validation sample of 3,000 people, with selection noise
  matching a genome-wide PRS (the README's recommended procedure);
- **auto + META:** no tuning;
- **estimation error** ‖w − β‖²/‖β‖² on each ancestry's standardised scale.

## 3. Job queues (`plan_benchmarks.py`, `jobs.py`)
The sandbox can restart at any time, so every fit is a job. Output goes to a temporary folder that is renamed only on
success, and `jobs.py run` skips finished jobs when relaunched. Each finished job stores wall time, CPU time and peak RSS
(`os.wait4`).
- **runtime:** the example (original on NumPy 1.26, fixed on NumPy 2.4, fast and improved with 1 and 4 threads, 3 repeats); the
  full chr22 probe; scaling over chr22 → chr15–22 (20 and 80 iterations; full-length time extrapolated and labelled as such).
- **accuracy:** chr22 ≤ 35 Mb (≈8.3k–8.8k SNPs per ancestry), 4 scenarios (h², π) ∈ {(0.5, 0.001), (0.5, 0.01), (0.5, 0.1),
  (0.2, 0.01)} × 2 replicates. PRS-CSx (via fast, identical to the original) and improved use φ ∈ {1e-6, 1e-4, 1e-2, 1, auto}
  with META. Ablations (RB only, coupling only, 2 chains, LD shrinkage) on selected scenarios, and repeated MCMC seeds on one dataset.

`summarize_results.py runtime|accuracy` writes the tables and figures in `results/`.

## Reproduce
```
python sim/simulate.py make-ref --ref_dir REF --chroms 22 --pops EUR,EAS,AFR --n_ref 500,500,650 --seed 11 --out SIM/ref_chr22
python plan_benchmarks.py make-data          # 8 accuracy datasets
python plan_benchmarks.py                    # writes both queues
python jobs.py run --queue BENCH/q_runtime --slots 1      # idle machine
python jobs.py run --queue BENCH/q_accuracy --slots 4
python summarize_results.py runtime; python summarize_results.py accuracy
```
(Paths are set at the top of `plan_benchmarks.py`.)

## 4. Follow-up benchmarks (WP1–WP6, 18–19 November 2026; results in `results_v2/`)
`results/` (v1) is kept unchanged. Every v2 script writes one JSON per dataset or measurement and skips finished ones when
relaunched (the sandbox restarted four times during these runs, and moved from a 2.30 GHz to a 2.10 GHz Xeon;
speed is only compared within one machine). `speed_chr22/`: chr22 timings of fast and improved (current defaults) on the
2.10 GHz machine.

| script | WP | what | result file(s) |
|---|---|---|---|
| `lowrank_speed.py` | 2 | low-rank vs full block update, speed and break-even (T2d) | `wp2_T2d_lowrank_speed.json` |
| (`../tests/test_lowrank.py`) | 2 | same posterior as the full update on real 1000 Genomes LD (T2c) | `wp2_T2c_same_posterior_real.json` |
| `wp3_eigen_panel_check.py` | 3 | panel built by `tools/build_ld_panel.py` vs a dense panel of the same people | `wp3_step4_eigen_vs_dense.json` |
| `wp3_truncation_study.py` | 3 | effect of `--rank_var` (deterministic part A; fits part B) on chr22 | `wp3_step5_truncation.json` |
| `wp3_builder_timing.py`, `wp3_builder_scaling.py` | 3 | build cost per block; read / XᵀX / eigh at n = 20,000 and 76,000 | `wp3_step6_builder_timing.json`, `wp3_step6_builder_scaling.json` |
| (`../tests/test_coupling_learned.py`) | 4 | `--coupling=learned` vs the exact posterior of τ | `wp4_step1_tau_numerical_check.json` |
| `coupling_stress.py` | 4 | couplings snr / effect / learned, 5 scenarios × 20 replicates | `wp4_coupling_stress/summary.md` |
| `accuracy_v2.py` (plan: `PLAN_accuracy_v2.md`) | 5 | pre-registered accuracy benchmark, 36 scenarios × 20 replicates | `wp5_accuracy/summary.md` |
| `wp6_warm_autostop.py` | 6 | `--warm_start` (both φ orders) and `--auto_stop` on WP5 datasets | `wp6/wp6_summary.md` |
| `wp6_profile.py` | 6 | time shares in a low-rank chromosome run | `wp6_profile_chr22.json`, `wp6_profile_chr1.json` |

`synthbench.py` is the in-process engine of WP4–WP6: synthetic AR(1) LD per ancestry (EUR 0.95, EAS 0.93, AFR 0.85 between
neighbouring SNPs), true LD from 10,000 simulated people, reference LD from 500 / 500 / 650 others, GWAS N = 300k / 100k / 30k,
exact accuracy from the true effects and LD, and the validation procedure of `sim/evaluate.py`.
