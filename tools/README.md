# tools

| tool | what |
|---|---|
| `combine_prs.py` | the PRS-CSx README's validation step: combine ancestry-specific scores per φ, pick φ, apply to test |
| `compare_prs.py` | compare several methods' test scores on the same people: incremental R² with paired bootstrap intervals |
| `build_ld_panel.py` | build an LD reference panel in eigen form from PLINK genotypes, or convert an existing panel (`../docs/04_eigen_panel_format.md`, `../docs/05_building_a_dense_panel.md`) |

## combine_prs.py

The PRS-CSx README recommends this procedure when a validation sample exists: for each φ, compute one PRS per
discovery ancestry, standardise the scores, learn a linear combination in validation, pick the best φ, and evaluate in an
independent test sample. The original repository ships no tool for it; this script implements it.

### Steps
1. Run PRS-CSx / `fast/PRScsx_fast.py` / `improved/PRScsx_improved.py` over a φ grid. Concatenate the per-chromosome files
   for each (φ, ancestry).
2. Score validation and test genotypes with PLINK, e.g.
   `plink2 --bfile val --score trait_EUR_phi1e-04_all.txt 2 4 6 cols=+scoresums --out val_EUR_phi1e-04`.
3. List the score files:
```
phi    pop  file
1e-4   EUR  val_EUR_phi1e-04.sscore
1e-4   AFR  val_AFR_phi1e-04.sscore
1e-2   EUR  val_EUR_phi1e-02.sscore
...
```
4. Run
```
python combine_prs.py --scores_val val_scores.tsv --scores_test test_scores.tsv --pheno pheno.txt \
       [--covar covar.txt] [--stack] --out trait_combined
```
   - `pheno.txt`: `FID IID PHENO` (-9/NA = missing).
   - `covar.txt`: `FID IID PC1 PC2 ...` (optional).
   - Score files can be PLINK 2 `.sscore` (uses `SCORE1_AVG` / `SCORE1_SUM`) or PLINK 1.9 `.profile` (uses `SCORE`).

### Output
For each φ: validation R², R² increment over the covariate-only model, and weights on the standardised scores. Also
reported: the selected φ, and the test R² obtained with the frozen validation weights and standardisation. `--stack` additionally
fits one non-negative least-squares combination of all φ × ancestry scores (residualised on covariates). See the final
report for how stacking compared with single-φ selection in the simulations.
Binary traits are treated with a linear probability model (R² on the observed scale).
`--write_test_scores` also writes `<out>.test_scores.tsv` (`FID IID SCORE [STACK]`): the combined score of every
test person, for `compare_prs.py`.

## compare_prs.py
```
python compare_prs.py --pheno pheno.txt --covar covar.txt \
       --scores original=orig.test_scores.tsv improved=impr.test_scores.tsv [--column SCORE] [--n_boot 2000] --out cmp
```
For each method: incremental R² over the covariate-only model (OLS), on the people who have the phenotype, the
covariates and every method's score. Uncertainty: bootstrap over people (the same resamples for all methods),
percentile 95% intervals, and for each method versus the first one the paired difference, its interval and a two-sided
bootstrap p-value. Any score file with `IID` and the score column works (`--column SCORE1_AVG` for a PLINK 2
`.sscore`, `--column SCORE` for a PLINK 1.9 `.profile`). Example workflow: `../docs/06_real_data_evaluation.md`.
