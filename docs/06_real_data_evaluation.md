# Real-data evaluation protocol (to be run on your data)

**Status:** this is a protocol. It has **not** been run here: the sandbox has no access to UK Biobank, Biobank Japan,
Taiwan Biobank, KoGES or China Kadoorie Biobank (CKB) data. All numbers in the other reports come from simulations. The
table at the end is empty on purpose.

The aim is to compare, in a target sample of Chinese adults (CKB), the polygenic scores of
1. original PRS-CSx (or `fast/`, which writes byte-identical files),
2. `improved/` with its default settings,
3. `improved/` with `--n_chains=2`,
4. optionally `improved/` with a CKB-based LD panel (see `05_building_a_dense_panel.md`),

for height, BMI, systolic blood pressure (SBP) and LDL cholesterol, using European (UK Biobank) and East Asian (Biobank
Japan, Taiwan Biobank, KoGES) GWAS summary statistics.

## 1. Decide the rule before looking at results
Write down, before scoring the test set:
- the primary comparison: `improved` default versus original PRS-CSx, both tuned on the validation set as below;
- the metric: incremental R² over covariates in the test set;
- the claim rule, for example: "better" only if the paired 95% bootstrap interval of the difference excludes zero for
  at least 3 of the 4 traits and no trait shows a significant loss.

Keep the test set untouched until all choices (φ grid, chain settings, covariates, QC) are fixed.

## 2. Target data (CKB)
- **Samples.** Genotyped CKB participants with the phenotype. Remove anyone who is part of any training GWAS (CKB is not
  in UKB, BBJ, TWB or KoGES, but check that none of the summary statistics you use is a meta-analysis that includes CKB).
- **Split.** A random 50/50 split into *validation* (choose φ and the combination weights) and *test* (report), stratified
  by the 10 CKB regions and sex, with a fixed seed. Remove close relatives across the two halves (KING kinship > 0.0884),
  so that relatives do not link validation and test.
- **Phenotypes.** Use the same definitions as the GWAS where possible:
  - height, BMI: as measured;
  - SBP: add 15 mmHg for people on blood-pressure-lowering drugs (the usual GWAS convention);
  - LDL-C: only in the subset with lipid measurements; divide by 0.7 for statin users (or follow the convention of
    your LDL GWAS).
- **Covariates.** Age, age², sex, CKB region (10 indicators), genotyping array/batch and the first 10–20 genetic
  principal components computed within CKB. The tools fit them by OLS; nothing else needs to be residualised.

## 3. GWAS summary statistics
- **Format** (as for PRS-CSx): `SNP A1 A2 BETA SE` (or `BETA P`, `OR SE`, `OR P`), A1 = effect allele, rsIDs as in the
  reference panel's `snpinfo` file (HapMap3 SNPs, hg19). Convert chr:pos IDs to rsIDs first.
- **QC.** Remove SNPs with imputation INFO < 0.8 or MAF < 0.01, and strand-ambiguous SNPs (A/T, C/G) with MAF > 0.4.
- **Three East Asian GWAS.** BBJ, TWB and KoGES are all East Asian. Meta-analyse them first (inverse-variance fixed
  effects, e.g. METAL) and use the meta-analysis as the EAS input, with `--n_gwas` = the sum of the three sample sizes
  (or the median per-SNP N). Do **not** list EAS three times in `--pop`: the output files are named by population and
  would overwrite each other.
- **Sample sizes.** Quantitative traits: the GWAS N. PRS-CSx uses one N per population.

## 4. Fitting (all chromosomes; same φ grid for every method)
```
REF=/path/to/ldblk_1kg            # or the UKB-based panels, or an eigen panel (docs/04, docs/05)
SST=ukb_height.txt,eas_meta_height.txt
N=360000,250000                   # your sample sizes
COMMON="--ref_dir=$REF --bim_prefix=ckb_target --sst_file=$SST --n_gwas=$N --pop=EUR,EAS \
        --phi=1e-6,1e-4,1e-2,1 --meta=TRUE --seed=1"

python fast/PRScsx_fast.py         $COMMON --out_dir=out/original --out_name=height --threads=8 --n_jobs=2
python improved/PRScsx_improved.py $COMMON --out_dir=out/improved --out_name=height --threads=8 --n_jobs=2
python improved/PRScsx_improved.py $COMMON --out_dir=out/improved2 --out_name=height --threads=8 --n_jobs=2 --n_chains=2
```
`ckb_target.bim` only restricts the SNP set; use the bim of the genotype data you will score. `fast/` output is
byte-identical to the original PRS-CSx (verified at start-up; add `--strict` to make a mismatch an error), so it can
stand in for the original. Run each command for each trait.

## 5. Scoring
Concatenate the 22 chromosome files for each (method, φ, ancestry), then score both halves with PLINK 2
(columns 2, 4, 6 of the PRS-CSx output = SNP, A1, effect):
```
for f in out/improved/height_EAS_pst_eff_a1_b0.5_phi1e-04_chr*.txt; do cat $f; done > height_improved_EAS_phi1e-04.txt
plink2 --bfile ckb --keep validation.ids --score height_improved_EAS_phi1e-04.txt 2 4 6 cols=+scoresums \
       --out val_improved_EAS_phi1e-04
plink2 --bfile ckb --keep test.ids       --score height_improved_EAS_phi1e-04.txt 2 4 6 cols=+scoresums \
       --out test_improved_EAS_phi1e-04
```
(PLINK 1.9: `--score FILE 2 4 6 sum`.) Do this for every φ, both ancestries and every method; also the META files if you
want to evaluate the `--meta` score without tuning.

## 6. Tuning on validation, applying to test
For each method, list its score files (`phi pop file`, one line per φ and ancestry; see `tools/README.md`) and run
```
python tools/combine_prs.py --scores_val val_improved.tsv --scores_test test_improved.tsv \
       --pheno height.pheno --covar covar.txt --stack --write_test_scores --out height_improved
```
This picks φ and the EUR/EAS weights on the validation half (README procedure of PRS-CSx), applies them unchanged to the
test half and writes the combined test score of every person to `height_improved.test_scores.tsv` (`SCORE`, and
`STACK` for the stacked combination over all φ × ancestries).

## 7. Comparing methods on the test half
```
python tools/compare_prs.py --pheno height.pheno --covar covar.txt \
       --scores original=height_original.test_scores.tsv improved=height_improved.test_scores.tsv \
                improved2=height_improved2.test_scores.tsv --n_boot 2000 --out height_compare
```
This reports, on the same people for every method, the incremental R² with a 95% bootstrap interval, and the paired
difference from the first method (here the original) with its interval and a bootstrap p-value. With `--column STACK`
the stacked scores are compared instead.

## 8. Table to fill (test half; incremental R² over covariates)

| trait | n test | original | improved | Δ [95% CI] | improved, 2 chains | Δ [95% CI] | CKB eigen panel | Δ [95% CI] |
|---|---|---|---|---|---|---|---|---|
| height | | | | | | | | |
| BMI | | | | | | | | |
| SBP | | | | | | | | |
| LDL-C | | | | | | | | |

Also record: selected φ per method and trait, run time and peak memory per method (the programs print the MCMC time per
chromosome), and the R̂ lines printed with `--n_chains=2`.

## 9. Things that can go wrong
- **Allele or build mismatch** between summary statistics, LD panel and target: check the number of SNPs matched per
  chromosome in the log; a sharp drop points to IDs, build or allele coding.
- **Very different N between EUR and EAS.** Under `--coupling=snr` (the PRS-CSx model) the prior variance of an effect
  scales with 1/N, so the smaller GWAS gets a wider prior. Report the coupling used.
- **Ancestry mismatch inside "EAS".** The EAS GWAS are Japanese, Taiwanese and Korean, the target is Chinese, and the
  1000 Genomes EAS panel has 504 people of several East Asian groups. A CKB-based panel (docs/05) removes the LD part of
  this mismatch; compare both panels on the validation half only.
