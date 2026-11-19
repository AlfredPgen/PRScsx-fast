# LD panels in eigen form (format specification)

The 1000 Genomes and UK Biobank panels of PRS-CS(x) store every LD block as a b × b matrix. That is feasible for
HapMap3 (about 1.2 million SNPs) but not for dense SNP sets: a block of 6,000 SNPs takes 288 MB in double precision.
An eigen panel stores each block as its eigenvectors V (b × r) and eigenvalues λ (r), with D = V diag(λ) Vᵀ. The full
matrix is never written. Both `fast/PRScsx_fast.py` and `improved/PRScsx_improved.py` read it, and detect it from the
files.

Panels are built with `tools/build_ld_panel.py` (see `docs/05_building_a_dense_panel.md`).

## Directory layout
The table below lists the files of one panel directory (the value of `--ref_dir`).

| path | content |
|---|---|
| `snpinfo_mult_eigen` | SNP information for all ancestries. Same columns and meaning as `snpinfo_mult_1kg_hm3`; any number of SNPs. |
| `ldeig_<pop>/ldeig_chr<c>.hdf5` | LD blocks of one ancestry (`pop` in lower case: afr, amr, eas, eur, sas) and one chromosome |
| `fragments/snpinfo_<POP>_chr<c>.txt` | per-ancestry SNP list written by the builder, merged into `snpinfo_mult_eigen` |

## `snpinfo_mult_eigen`
Tab-separated, with a header, one row per SNP, sorted by chromosome and position:
`CHR SNP BP A1 A2 FRQ_AFR FRQ_AMR FRQ_EAS FRQ_EUR FRQ_SAS FLP_AFR FLP_AMR FLP_EAS FLP_EUR FLP_SAS`.
- `FRQ_<POP>` is the frequency of allele `A1` in that ancestry. 0 means the SNP is not in that ancestry's panel.
- `FLP_<POP>` is +1 if the ancestry's LD was computed counting `A1`, −1 if counting `A2`, and 0 if absent.
  (These are the same conventions as the 1000 Genomes file.)
- Strand-ambiguous SNPs (A/T, C/G) are not included.

## Block files `ldeig_<pop>/ldeig_chr<c>.hdf5`
- File attributes: `format = "prscsx-eigen-1"`, `pop`, `chrom`, `maf`, `n_ref` (people used), `rank_var`.
- One group per LD block, `blk_1`, `blk_2`, ... in position order. Each group holds the datasets in the table below.

| dataset | type | shape | content |
|---|---|---|---|
| `snplist` | bytes | (b,) | SNP ids in position order. Within a chromosome they follow the order of `snpinfo_mult_eigen`. |
| `evec` | float32 | (b, r) | orthonormal eigenvectors of the block's LD matrix, ordered by decreasing eigenvalue |
| `eval` | float64 | (r,) | the r kept eigenvalues, decreasing |

- Group attributes: `n_ref` (people used for this block), `rank_var` (share of the trace kept), `trace` (trace of the
  full LD matrix = number of SNPs b; the kept share is `sum(eval) / trace`).
- An empty block has b = 0 (all three datasets have zero length).
- Eigenvalues not above 1e-13 × the largest are always dropped. They are exact zeros, e.g. when b > `n_ref`.

## How the programs use it
For the SNPs that pass matching, the program selects the rows of `evec` and applies the allele flips. It then
re-orthonormalises (thin SVD of `V_sub diag(λ)^1/2`), because a row subset of V is not orthonormal; the subset's LD
matrix is exactly `V_sub diag(λ) V_subᵀ`. The block update is the low-rank update of `fast/fastcsx/lowrank.py`.
`--linalg=full` is not available with an eigen panel. `fast` switches to `auto` and prints that its output is then
statistically equivalent to the original but not byte-identical. Under `auto`, small blocks (b ≤ 1.7 r) use the full
update on the reconstructed matrix `V diag(λ) Vᵀ`.
