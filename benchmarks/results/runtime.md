## Example data (chr22, 1,000 SNPs, EUR+EAS, 2,000 iterations, phi=1e-2)

| variant | n | wall_mean | wall_sd | cpu_mean | rss_max | speedup |
|---|---|---|---|---|---|---|
| original_t1 | 3 | 40.7 | 1.9 | 40.2 | 238 | 1.00x |
| fixed_t1 | 3 | 28.0 | 0.5 | 27.6 | 241 | 1.45x |
| fast_t1 | 3 | 14.0 | 0.5 | 13.7 | 251 | 2.90x |
| fast_t4 | 3 | 10.7 | 0.3 | 16.9 | 279 | 3.81x |
| improved_t1 | 3 | 10.7 | 0.3 | 10.3 | 252 | 3.81x |
| improved_t4 | 3 | 8.3 | 0.3 | 13.0 | 269 | 4.88x |

### Output agreement with the ORIGINAL (same seed), example data

| variant | files | identical | max_abs | mean_abs | min_r |
|---|---|---|---|---|---|
| fixed_t1 | 6 | 6 | 0 | 0 | 1.000000 |
| fast_t1 | 6 | 6 | 0 | 0 | 1.000000 |
| fast_t4 | 6 | 6 | 0 | 0 | 1.000000 |
| improved_t1 | 6 | 0 | 0.00357 | 0.000141 | 0.936252 |
| improved_t4 | 6 | 0 | 0.00357 | 0.000141 | 0.936252 |

## All HapMap3 SNPs of chr22 (17,007 SNPs, EUR+EAS, 2,000 iterations, phi=1e-2, seed 1)

| variant | wall | cpu | rss | speedup | identical | min_r |
|---|---|---|---|---|---|---|
| original_t1 | 1186.0 | 1180.6 | 449 | 1.00x |  |  |
| fast_t1 | 701.4 | 697.1 | 561 | 1.69x | 2/2 | 1.000000 |
| fast_t4 | 214.1 | 758.6 | 682 | 5.54x | 2/2 | 1.000000 |
| improved_t1 | 470.7 | 467.5 | 556 | 2.52x | 0/2 | 0.970394 |
| improved_t4 | 138.1 | 465.0 | 599 | 8.59x | 0/2 | 0.970394 |

## Scaling (simulated EUR+EAS+AFR, 4 threads/cores each; setup and per-iteration time from runs of 20 and 80 iterations;
   est_full_run_h = setup + 3,000 x per-iteration, i.e. EXTRAPOLATED to the default chain length)

| region | n_snps | method | setup_s | per_iter_s | wall_it80 | rss_mb | est_full_run_h |
|---|---|---|---|---|---|---|---|
| chr 15,16,17,18,19,20,21,22 | 249925 | original_t4 | 522.4 | 8.546 | 1206.0 | 983 | 7.27 |
| chr 15,16,17,18,19,20,21,22 | 249925 | fast_t4 | 181.6 | 1.651 | 313.7 | 1402 | 1.43 |
| chr 15,16,17,18,19,20,21,22 | 249925 | improved_t4 | 177.7 | 1.268 | 279.2 | 1468 | 1.11 |
| chr 19,20,21,22 | 95490 | original_t4 | 162.6 | 3.201 | 418.7 | 889 | 2.71 |
| chr 19,20,21,22 | 95490 | fast_t4 | 77.0 | 0.600 | 125.1 | 1231 | 0.52 |
| chr 19,20,21,22 | 95490 | improved_t4 | 74.5 | 0.426 | 108.6 | 1272 | 0.38 |
| chr 21,22 | 36992 | original_t4 | 46.2 | 1.267 | 147.6 | 517 | 1.07 |
| chr 21,22 | 36992 | fast_t4 | 29.5 | 0.250 | 49.5 | 757 | 0.22 |
| chr 21,22 | 36992 | improved_t4 | 31.2 | 0.142 | 42.5 | 746 | 0.13 |
| chr 22 | 18928 | original_t4 | 24.3 | 0.634 | 75.0 | 497 | 0.53 |
| chr 22 | 18928 | fast_t4 | 16.5 | 0.125 | 26.5 | 721 | 0.11 |
| chr 22 | 18928 | improved_t4 | 17.0 | 0.075 | 23.0 | 717 | 0.07 |

## Peak RSS (MB) of the FINAL code (parser keeps only requested ancestries/numeric columns), 4 threads, 20 iterations

| region | original_t4 | fast_t4 | improved_t4 |
|---|---|---|---|
| chr 15,16,17,18,19,20,21,22 | 983 | 1292 | 1367 |
| chr 19,20,21,22 | 889 | 1166 | 1233 |
| chr 21,22 | 517 | 748 | 734 |
| chr 22 | 497 | 709 | 710 |

