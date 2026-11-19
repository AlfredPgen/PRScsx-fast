# Accuracy benchmark v2: plan (fixed before any result)

Written on 18 November 2026, before the benchmark was run. The rule below decides what may be said about `improved/`.
If the plan turns out to be wrong, the report says so, but the verdict is the one this plan gives.

## Question
Does `improved/` (with each coupling, and with 2 chains) predict better than PRS-CSx in each target ancestry?

## Data (synthetic, one new dataset per replicate)
Engine: `benchmarks/synthbench.py`. Driver: `benchmarks/accuracy_v2.py`.
- Three ancestries EUR, EAS, AFR; GWAS sizes 300,000 / 100,000 / 30,000.
- 2,000 SNPs in 20 LD blocks of 100 SNPs. LD in each block is generated from a latent Gaussian AR(1) process, with
  correlation 0.95 (EUR), 0.93 (EAS) and 0.85 (AFR) between neighbouring SNPs, and allele frequencies drawn per ancestry.
  The true LD is estimated from 10,000 simulated people, and the reference LD given to the methods from 500 / 500 / 650
  other people. That gives LD mismatch of the size a 1000 Genomes panel brings.
- Causal SNPs are shared across ancestries (share π). Standardised effects ~ N(0, C) with C_kl = ρ, C_kk = 1, times the
  effect-variance multiplier for the smallest GWAS (AFR). Effects are scaled to per-SNP heritability h² · M / 1.1·10⁶
  (in EUR; the other ancestries use the same scale). GWAS: β̂_k = R_k β_k + N(0, R_k/N_k).
- Why synthetic LD and 2,000 SNPs: one fit on half of chromosome 22 with real LD takes about 10 minutes. The design
  below needs 10,800 chain fits, which is only feasible at about 10 seconds per fit. The earlier benchmark on real LD
  (`benchmarks/results/`) remains as supporting evidence.

## Scenarios and replicates
h² ∈ {0.2, 0.5} × share of causal SNPs π ∈ {0.1%, 1%, 10%} × ρ ∈ {0.4, 0.8} × AFR effect-variance multiplier
∈ {1/3, 1, 3} = 36 scenarios, **20 replicates** each (720 datasets).

## Methods (all with PRS-CSx's default chain length for 3 ancestries: 3,000 iterations, 1,500 burn-in)
The table below lists the methods.

| label | what |
|---|---|
| `prscsx` | the exact PRS-CSx sampler of `fast/` (identical to the original) |
| `snr` | `improved/` with `--coupling=snr` (the PRS-CSx model; RB means, single precision) |
| `effect` | `improved/` with `--coupling=effect` |
| `learned` | `improved/` with `--coupling=learned` |
| `default_2ch` | `improved/` with the coupling chosen as default by WP4, 2 chains (seed s and s + 1000). Its first chain is the single-chain run of that coupling with seed s, so only the second chain is extra work. |

φ grid {1e-6, 1e-4, 1e-2} for every method. φ = 1 is left out: in the v1 benchmark it was never the selected value
(0 of 60 selections), and leaving it out saves a quarter of the compute.

## Primary metric
Tuned r²_g (= R²/h², exact, from the true effects and true LD) for each target ancestry, by the README procedure. For each φ,
the three ancestry-specific PRSs are combined by OLS in a validation sample of **3,000** people. The φ with the largest
validation R² is selected. The validation sample has genome-equivalent selection noise, as in `benchmarks/sim/evaluate.py`.

## Rule for a claim
For each comparison (each `improved/` method against `prscsx`) and each target ancestry:
- d_i = r²_g(method) − r²_g(prscsx) for the 720 datasets (same data, same validation draw).
- **"Better"** may be claimed only if all three hold:
  1. the pooled mean of d_i is above 0.01;
  2. its 95% confidence interval (mean ± 1.96 · SD/√720) excludes zero;
  3. no single scenario has a mean d below −0.01.
- "Worse" is claimed symmetrically: pooled mean below −0.01, CI excludes zero, and no scenario has a mean gain above 0.01.
- Otherwise the verdict is "no demonstrated difference".
- A general claim that `improved` (its default) predicts better than PRS-CSx needs "better" in all three target ancestries.

## Reported, but not part of the rule
Oracle-tuned r²_g; the selected φ; per-scenario tables of means and paired differences with SEs; run time.

## Appendix (filled in before the run starts)
- Default coupling chosen by WP4 (used by `default_2ch`): **`snr`** (filled in on 19 November 2026, 00:35 UTC, before
  WP5 started). WP4's rule (`benchmarks/results_v2/wp4_coupling_stress/summary.md`, 5 scenarios × 20 replicates):
  `learned` was more than 0.005 below the better of `snr`/`effect` in some cells (minimum −0.0124) and more than 0.005
  below `snr` in three cells (afr_x10 EAS −0.0096, afr_x10 AFR −0.0088, afr_x3 EAS −0.0064), so the rule gives `snr` as the
  default. `effect` and `learned` stay available as options and are still compared in WP5.
- Not part of this plan's rule: WP6 (`benchmarks/wp6_warm_autostop.py`) re-uses the datasets of the 6 scenarios with
  ρ = 0.8 and multiplier 1 (same seeds) to evaluate `--warm_start` and `--auto_stop`.
