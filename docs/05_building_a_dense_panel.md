# Building your own LD panel (CKB, UK Biobank, dense SNP sets)

`tools/build_ld_panel.py` builds an LD reference panel in eigen form (format: `04_eigen_panel_format.md`) from PLINK
genotypes. Both `fast/` and `improved/` read such a panel with `--ref_dir=PANEL_DIR`. This document gives the commands,
the measured cost and the measured effect of truncating the LD spectrum, and it separates measurements from
extrapolations.

## When it helps
- **Ancestry-matched LD.** The 1000 Genomes EAS panel has 504 people from several East Asian groups. For a Chinese
  target (e.g. China Kadoorie Biobank, CKB), a panel from a few thousand unrelated CKB participants matches the target's
  LD better and has a much smaller sampling error. The sampling SD of an LD correlation r is about (1 − r²)/√n: 0.045 at
  n = 500, 0.007 at n = 20,000 and 0.004 at n = 76,000 (theory, not measured here).
- **Other SNP sets.** The 1000 Genomes and UK Biobank panels of PRS-CSx are restricted to HapMap3 SNPs. The builder works
  for any SNP set, e.g. HapMap3 plus array SNPs, or an imputed set filtered on MAF and INFO.

## Commands
1. Prepare genotypes (PLINK 1 binary, hg19 positions as in the PRS-CSx SNP information files): unrelated people of one
   ancestry, standard QC, the SNP set you want, e.g.
   ```
   plink2 --bfile ckb --keep unrelated.ids --extract snpset.txt --maf 0.01 --geno 0.05 --hwe 1e-6 \
          --make-bed --out ckb_ld
   ```
2. Build each chromosome (independent jobs; run them in parallel). Block boundaries are taken from the existing
   1000 Genomes panel of the same ancestry:
   ```
   for c in $(seq 1 22); do
     python tools/build_ld_panel.py build --bfile ckb_ld --pop EAS --chrom $c --out CKB_PANEL \
            --blocks_from /path/to/ldblk_1kg --max_n 20000 --seed 1        # --rank_var 0.999 is the default
   done
   python tools/build_ld_panel.py merge --out CKB_PANEL
   ```
   For a second ancestry (e.g. UK Biobank EUR) build into the same `CKB_PANEL` directory with `--pop EUR`, then run
   `merge` once more. `--max_n` draws a random subset of people (seeded); see "How many people" below.
3. Use it:
   ```
   python improved/PRScsx_improved.py --ref_dir=CKB_PANEL ...            # same options as with the 1000 Genomes panel
   python fast/PRScsx_fast.py --ref_dir=CKB_PANEL ...                    # switches to --linalg=auto (not byte-identical)
   ```
An existing 1000 Genomes or UK Biobank panel can be converted with `build_ld_panel.py convert` (used for the tests below).

## What `--rank_var` costs in accuracy (measured)
`--rank_var` keeps the largest eigenvalues of each block's LD matrix that explain this share of its trace; the rest of
the spectrum is dropped and the marginal effects are projected on the kept eigenvectors. Measured on all HapMap3 SNPs of
chromosome 22 (1000 Genomes EUR 16,464 SNPs, EAS 14,968 SNPs, panel converted to eigen form), EUR + EAS GWAS of 200k / 100k
simulated from the real LD (`benchmarks/wp3_truncation_study.py`, `benchmarks/results_v2/wp3_step5_truncation.json`):

*Part A, no Monte Carlo noise:* correlation of the conditional posterior mean (D + Ψ⁻¹)⁻¹β̂ with and without truncation,
at fixed local shrinkage ψ (EAS shown; the EUR values are similar and are in the JSON file):

| rank_var | total rank kept (EAS) | ψ = 1e-4 | ψ = 1e-2 | ψ = 1 | sparse ψ (1 for the 1% largest \|β̂\| per block, 1e-4 elsewhere) |
|---|---|---|---|---|---|
| 1 | 10,732 | 1 | 1 | 1 | 1 |
| 0.999 | 6,363 | 0.99987 | 0.99979 | 0.9939 | 0.99980 |
| 0.995 | 4,877 | 0.9993 | 0.9989 | 0.968 | 0.9992 |
| 0.99 | 4,160 | 0.9986 | 0.9977 | 0.938 | 0.9985 |
| 0.95 | 2,442 | 0.992 | 0.988 | 0.757 | 0.993 |

*Part B, full fits* (`improved/` with `--coupling=snr`, the PRS-CSx model, φ = 1e-2, average of 3 seeds per setting), correlation of posterior means with the
untruncated fit. The noise floor is the correlation between two independent untruncated 3-seed averages:

| rank_var | EUR | EAS | seconds per iteration (4 threads) |
|---|---|---|---|
| 1 (noise floor) | 0.9979 | 0.9968 | 0.080 |
| 0.999 | 0.9976 | 0.9966 | 0.048 |
| 0.995 | 0.9964 | 0.9940 | 0.038 |
| 0.99 | 0.9946 | 0.9904 | 0.027 |
| 0.95 | 0.963 | 0.955 | 0.017 |

**Recommendation: `--rank_var 0.999`** (the builder's default). It is indistinguishable from no truncation (within the
noise floor) and 1.7× faster on this chromosome. 0.995 and below lose measurably (0.995: −0.0015 EUR, −0.0028 EAS below
the noise floor). These numbers are for HapMap3 SNP density and a 1000 Genomes-size panel; a denser SNP set has more
redundant SNPs per block, so the same `rank_var` keeps a smaller share of the SNPs, but this was not measured.

## Cost of building (measured, one thread, this sandbox)
`benchmarks/wp3_builder_timing.py` (simulated genotypes with strong LD between neighbouring SNPs, one block per run,
including reading the .bed file):

| people n | SNPs in block b | rank kept (0.999) | seconds (0.999 / 1) | file MB (0.999 / 1) |
|---|---|---|---|---|
| 600 | 2,000 | 591 | 0.4 / 0.3 | 4.8 / 4.8 |
| 2,000 | 2,000 | 1,660 | 1.6 / 1.5 | 13.3 / 16.0 |
| 2,000 | 6,000 | 1,966 | 7.7 / 7.7 | 47.2 / 48.0 |
| 5,000 | 2,000 | 1,944 | 2.0 / 2.0 | 15.6 / 16.0 |
| 5,000 | 6,000 | 4,475 | 76 / 80 | 107 / 120 |

Converting the 1000 Genomes chromosome 22 panels to eigen form: 36 MB per ancestry, against 70 MB for the original
b × b files (the rank of a block is at most 503 there).

## Biobank-size panels: measured parts and extrapolation
Measured single-thread cost of the three parts of a block build at biobank sample sizes
(`benchmarks/wp3_builder_scaling.py`, `benchmarks/results_v2/wp3_step6_builder_scaling.json`):

| part | n = 20,000 | n = 76,000 |
|---|---|---|
| reading + decoding the .bed file, per SNP | 0.30 ms | 1.29 ms |
| cross-product XᵀX/n, b = 700 / 2,000 / 3,000 SNPs | 0.15 / 1.2 / 2.6 s | 0.63 / 4.2 / (≈ 9.5) s |
| eigendecomposition, b = 700 / 2,000 / 3,000 / 6,000 (independent of n) | 0.07 / 1.2 / 3.9 / 30.6 s | same |

**Extrapolations (not measured end to end; assumptions stated):**
- *HapMap3 SNP set, CKB panel* (about 1.2 million SNPs in about 1,700 blocks, so b ≈ 700 on average): about 0.5 s per
  block at n = 20,000, i.e. about 15 minutes per ancestry on one core, and about 45 minutes at n = 76,000. Disk: with
  n ≫ b the rank of a block is close to b; at 4 bytes × b × r this is up to about 2 MB per block, a few GB per ancestry.
  Fitting with such a panel costs about as much as with the 1000 Genomes panel under `--linalg=full`, because the block
  sizes are the same and the rank no longer limits the cost.
- *Dense SNP set* (say 5 million SNPs in the same blocks, b ≈ 3,000 on average): about 7 s per block at n = 20,000
  (about 3.5 hours per ancestry on one core) and about 17 s at n = 76,000 (about 8 hours). Disk: r ≈ 0.75 b (the share
  measured on the simulated genotypes above) gives about 27 MB per block, about 45 GB per ancestry.
- *The real limit for dense panels is the Gibbs sampler, not the build.* Every iteration factorises or low-rank-updates
  every block. A full update of a 3,000-SNP block takes about 0.3 s (double precision, one thread; from the measured 0.089 s
  at b = 2,000 and b³ scaling), so 1,700 such blocks take about 8 minutes per iteration on one core and 3,000 iterations
  take weeks of CPU time. Truncation helps only as far as the LD spectrum allows (cost ∝ b r²), which was not measured on
  real dense genotypes here.

**How many people.** Building cost grows linearly with n (reading and XᵀX), while the LD sampling error falls as 1/√n.
Going from 20,000 to 76,000 people shrinks the sampling SD of an LD correlation from about 0.007 to 0.004 (theory); both
are far below the 0.045 of a 1000 Genomes-size panel, and the larger panel costs 3.8× more to build. Whether the step from
20,000 to 76,000 changes prediction was not measured. **Recommendation for CKB (≈ 76,000 genotyped): `--max_n 20000`** (a seeded random subset of unrelated participants), HapMap3 or a similar SNP set. For UK
Biobank EUR the same applies.


## Practical advice
- Use unrelated people of one ancestry; do not mix ancestries in one panel.
- Keep the SNP set close to what your GWAS summary statistics contain; SNPs absent from the summary statistics are dropped
  at run time anyway.
- The builder reads genotypes block by block; memory is about 3 × 8 · n · b bytes for the largest block (n = people
  used, b = SNPs in the block): 1.5 GB for n = 20,000 and b = 3,000.
- Validate a new panel before using it for a paper: compare its scores with the 1000 Genomes panel's on the validation
  half of your target data (`06_real_data_evaluation.md`).
