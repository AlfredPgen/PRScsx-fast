## Accuracy r2_g = R2/h2 (mean over replicates; each replicate = new data + MCMC seed)

### tuned (phi grid + validation, n=3000)

| scenario | target | prscsx | improved | rbonly | cplonly | ldshrink |
|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | 0.5240 | 0.5727 |  |  |  |
| h0.2_pi0.01 | EAS | 0.4003 | 0.3962 |  |  |  |
| h0.2_pi0.01 | AFR | 0.2108 | 0.2111 |  |  |  |
| h0.5_pi0.001 | EUR | 0.9197 | 0.9305 |  |  |  |
| h0.5_pi0.001 | EAS | 0.9058 | 0.9340 |  |  |  |
| h0.5_pi0.001 | AFR | 0.7807 | 0.7871 |  |  |  |
| h0.5_pi0.01 | EUR | 0.7480 | 0.7478 | 0.7404 | 0.7449 | 0.7825 |
| h0.5_pi0.01 | EAS | 0.6795 | 0.6776 | 0.6648 | 0.6774 | 0.6708 |
| h0.5_pi0.01 | AFR | 0.4554 | 0.4737 | 0.4469 | 0.4715 | 0.4248 |
| h0.5_pi0.1 | EUR | 0.5516 | 0.5522 |  |  |  |
| h0.5_pi0.1 | EAS | 0.4085 | 0.4075 |  |  |  |
| h0.5_pi0.1 | AFR | 0.2295 | 0.2373 |  |  |  |

### stacked over phi x ancestry (NNLS, validation n=3000)

| scenario | target | prscsx | improved | rbonly | cplonly | ldshrink |
|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | 0.5395 | 0.5710 |  |  |  |
| h0.2_pi0.01 | EAS | 0.4125 | 0.4052 |  |  |  |
| h0.2_pi0.01 | AFR | 0.2145 | 0.2150 |  |  |  |
| h0.5_pi0.001 | EUR | 0.9225 | 0.9287 |  |  |  |
| h0.5_pi0.001 | EAS | 0.9103 | 0.9347 |  |  |  |
| h0.5_pi0.001 | AFR | 0.7804 | 0.7903 |  |  |  |
| h0.5_pi0.01 | EUR | 0.7476 | 0.7515 | 0.7414 | 0.7515 | 0.7856 |
| h0.5_pi0.01 | EAS | 0.6827 | 0.6820 | 0.6751 | 0.6838 | 0.6826 |
| h0.5_pi0.01 | AFR | 0.4603 | 0.4825 | 0.4623 | 0.4799 | 0.4612 |
| h0.5_pi0.1 | EUR | 0.5532 | 0.5538 |  |  |  |
| h0.5_pi0.1 | EAS | 0.4104 | 0.4097 |  |  |  |
| h0.5_pi0.1 | AFR | 0.2439 | 0.2405 |  |  |  |

### tuned (oracle phi & weights)

| scenario | target | prscsx | improved | rbonly | cplonly | ldshrink |
|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | 0.5391 | 0.5745 |  |  |  |
| h0.2_pi0.01 | EAS | 0.4006 | 0.4028 |  |  |  |
| h0.2_pi0.01 | AFR | 0.2147 | 0.2121 |  |  |  |
| h0.5_pi0.001 | EUR | 0.9220 | 0.9313 |  |  |  |
| h0.5_pi0.001 | EAS | 0.9061 | 0.9343 |  |  |  |
| h0.5_pi0.001 | AFR | 0.7816 | 0.7890 |  |  |  |
| h0.5_pi0.01 | EUR | 0.7485 | 0.7483 | 0.7419 | 0.7466 | 0.7830 |
| h0.5_pi0.01 | EAS | 0.6797 | 0.6779 | 0.6663 | 0.6785 | 0.6710 |
| h0.5_pi0.01 | AFR | 0.4563 | 0.4760 | 0.4475 | 0.4739 | 0.4251 |
| h0.5_pi0.1 | EUR | 0.5521 | 0.5540 |  |  |  |
| h0.5_pi0.1 | EAS | 0.4094 | 0.4109 |  |  |  |
| h0.5_pi0.1 | AFR | 0.2304 | 0.2377 |  |  |  |

### auto + META (no tuning)

| scenario | target | prscsx | improved | rbonly | cplonly | ldshrink |
|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | 0.5019 | 0.5042 |  |  |  |
| h0.2_pi0.01 | EAS | 0.3673 | 0.3613 |  |  |  |
| h0.2_pi0.01 | AFR | 0.2139 | 0.2550 |  |  |  |
| h0.5_pi0.001 | EUR | 0.9008 | 0.8978 |  |  |  |
| h0.5_pi0.001 | EAS | 0.7851 | 0.7920 |  |  |  |
| h0.5_pi0.001 | AFR | 0.4974 | 0.4827 |  |  |  |
| h0.5_pi0.01 | EUR | 0.7045 | 0.7054 | 0.6881 | 0.7042 | 0.7202 |
| h0.5_pi0.01 | EAS | 0.5778 | 0.5981 | 0.5710 | 0.5958 | 0.6385 |
| h0.5_pi0.01 | AFR | 0.4512 | 0.4658 | 0.4377 | 0.4633 | 0.4277 |
| h0.5_pi0.1 | EUR | 0.4335 | 0.4690 |  |  |  |
| h0.5_pi0.1 | EAS | 0.3205 | 0.3524 |  |  |  |
| h0.5_pi0.1 | AFR | 0.1748 | 0.1995 |  |  |  |

## Paired differences vs PRS-CSx (method - prscsx, same datasets)

| metric | method | target | n_datasets | mean_diff | se | n_better | min_diff | max_diff |
|---|---|---|---|---|---|---|---|---|
| tuned (phi grid + validation, n=3000) | improved | EUR | 8 | +0.0150 | 0.0089 | 6 | -0.0088 | +0.0712 |
| tuned (phi grid + validation, n=3000) | improved | EAS | 8 | +0.0053 | 0.0066 | 4 | -0.0144 | +0.0469 |
| tuned (phi grid + validation, n=3000) | improved | AFR | 8 | +0.0082 | 0.0035 | 7 | -0.0096 | +0.0221 |
| tuned (phi grid + validation, n=3000) | rbonly | EUR | 2 | -0.0076 | 0.0046 | 0 | -0.0122 | -0.0029 |
| tuned (phi grid + validation, n=3000) | rbonly | EAS | 2 | -0.0146 | 0.0103 | 0 | -0.0249 | -0.0044 |
| tuned (phi grid + validation, n=3000) | rbonly | AFR | 2 | -0.0086 | 0.0014 | 0 | -0.0100 | -0.0071 |
| tuned (phi grid + validation, n=3000) | cplonly | EUR | 2 | -0.0031 | 0.0080 | 1 | -0.0111 | +0.0049 |
| tuned (phi grid + validation, n=3000) | cplonly | EAS | 2 | -0.0021 | 0.0140 | 1 | -0.0161 | +0.0120 |
| tuned (phi grid + validation, n=3000) | cplonly | AFR | 2 | +0.0160 | 0.0023 | 2 | +0.0137 | +0.0184 |
| tuned (phi grid + validation, n=3000) | ldshrink | EUR | 1 | +0.0087 | nan | 1 | +0.0087 | +0.0087 |
| tuned (phi grid + validation, n=3000) | ldshrink | EAS | 1 | -0.0016 | nan | 0 | -0.0016 | -0.0016 |
| tuned (phi grid + validation, n=3000) | ldshrink | AFR | 1 | -0.0059 | nan | 0 | -0.0059 | -0.0059 |
| stacked over phi x ancestry (NNLS, validation n=3000) | improved | EUR | 8 | +0.0105 | 0.0051 | 6 | -0.0054 | +0.0375 |
| stacked over phi x ancestry (NNLS, validation n=3000) | improved | EAS | 8 | +0.0039 | 0.0067 | 5 | -0.0203 | +0.0420 |
| stacked over phi x ancestry (NNLS, validation n=3000) | improved | AFR | 8 | +0.0073 | 0.0050 | 5 | -0.0160 | +0.0273 |
| stacked over phi x ancestry (NNLS, validation n=3000) | rbonly | EUR | 2 | -0.0062 | 0.0042 | 0 | -0.0104 | -0.0020 |
| stacked over phi x ancestry (NNLS, validation n=3000) | rbonly | EAS | 2 | -0.0075 | 0.0117 | 1 | -0.0193 | +0.0042 |
| stacked over phi x ancestry (NNLS, validation n=3000) | rbonly | AFR | 2 | +0.0020 | 0.0009 | 2 | +0.0010 | +0.0029 |
| stacked over phi x ancestry (NNLS, validation n=3000) | cplonly | EUR | 2 | +0.0039 | 0.0084 | 1 | -0.0045 | +0.0123 |
| stacked over phi x ancestry (NNLS, validation n=3000) | cplonly | EAS | 2 | +0.0011 | 0.0130 | 1 | -0.0119 | +0.0142 |
| stacked over phi x ancestry (NNLS, validation n=3000) | cplonly | AFR | 2 | +0.0196 | 0.0029 | 2 | +0.0167 | +0.0226 |
| stacked over phi x ancestry (NNLS, validation n=3000) | ldshrink | EUR | 1 | +0.0125 | nan | 1 | +0.0125 | +0.0125 |
| stacked over phi x ancestry (NNLS, validation n=3000) | ldshrink | EAS | 1 | +0.0066 | nan | 1 | +0.0066 | +0.0066 |
| stacked over phi x ancestry (NNLS, validation n=3000) | ldshrink | AFR | 1 | +0.0238 | nan | 1 | +0.0238 | +0.0238 |
| tuned (oracle phi & weights) | improved | EUR | 8 | +0.0116 | 0.0064 | 6 | -0.0089 | +0.0495 |
| tuned (oracle phi & weights) | improved | EAS | 8 | +0.0075 | 0.0063 | 6 | -0.0138 | +0.0470 |
| tuned (oracle phi & weights) | improved | AFR | 8 | +0.0080 | 0.0040 | 7 | -0.0147 | +0.0222 |
| tuned (oracle phi & weights) | rbonly | EUR | 2 | -0.0065 | 0.0051 | 0 | -0.0117 | -0.0014 |
| tuned (oracle phi & weights) | rbonly | EAS | 2 | -0.0135 | 0.0088 | 0 | -0.0223 | -0.0046 |
| tuned (oracle phi & weights) | rbonly | AFR | 2 | -0.0088 | 0.0007 | 0 | -0.0095 | -0.0082 |
| tuned (oracle phi & weights) | cplonly | EUR | 2 | -0.0019 | 0.0090 | 1 | -0.0109 | +0.0071 |
| tuned (oracle phi & weights) | cplonly | EAS | 2 | -0.0012 | 0.0129 | 1 | -0.0141 | +0.0117 |
| tuned (oracle phi & weights) | cplonly | AFR | 2 | +0.0175 | 0.0016 | 2 | +0.0159 | +0.0191 |
| tuned (oracle phi & weights) | ldshrink | EUR | 1 | +0.0084 | nan | 1 | +0.0084 | +0.0084 |
| tuned (oracle phi & weights) | ldshrink | EAS | 1 | -0.0018 | nan | 0 | -0.0018 | -0.0018 |
| tuned (oracle phi & weights) | ldshrink | AFR | 1 | -0.0057 | nan | 0 | -0.0057 | -0.0057 |
| auto + META (no tuning) | improved | EUR | 8 | +0.0089 | 0.0086 | 5 | -0.0162 | +0.0607 |
| auto + META (no tuning) | improved | EAS | 8 | +0.0133 | 0.0074 | 6 | -0.0093 | +0.0539 |
| auto + META (no tuning) | improved | AFR | 8 | +0.0164 | 0.0104 | 6 | -0.0429 | +0.0481 |
| auto + META (no tuning) | rbonly | EUR | 2 | -0.0164 | 0.0129 | 0 | -0.0293 | -0.0035 |
| auto + META (no tuning) | rbonly | EAS | 2 | -0.0068 | 0.0035 | 0 | -0.0104 | -0.0033 |
| auto + META (no tuning) | rbonly | AFR | 2 | -0.0135 | 0.0010 | 0 | -0.0146 | -0.0125 |
| auto + META (no tuning) | cplonly | EUR | 2 | -0.0003 | 0.0100 | 1 | -0.0103 | +0.0097 |
| auto + META (no tuning) | cplonly | EAS | 2 | +0.0180 | 0.0140 | 2 | +0.0040 | +0.0320 |
| auto + META (no tuning) | cplonly | AFR | 2 | +0.0121 | 0.0214 | 1 | -0.0093 | +0.0335 |
| auto + META (no tuning) | ldshrink | EUR | 1 | -0.0137 | nan | 0 | -0.0137 | -0.0137 |
| auto + META (no tuning) | ldshrink | EAS | 1 | +0.0004 | nan | 1 | +0.0004 | +0.0004 |
| auto + META (no tuning) | ldshrink | AFR | 1 | -0.0366 | nan | 0 | -0.0366 | -0.0366 |

## MCMC-seed variability of accuracy (dataset h0.5_pi0.01_r1, same data, different MCMC seeds)

| method | metric | target | n_seeds | mean | sd | min | max |
|---|---|---|---|---|---|---|---|
| prscsx | tuned (phi grid + validation, n=3000) | EUR | 3 | 0.7704 | 0.0040 | 0.7660 | 0.7738 |
| prscsx | tuned (phi grid + validation, n=3000) | EAS | 3 | 0.6738 | 0.0081 | 0.6665 | 0.6825 |
| prscsx | tuned (phi grid + validation, n=3000) | AFR | 3 | 0.4345 | 0.0039 | 0.4307 | 0.4386 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | EUR | 3 | 0.7713 | 0.0017 | 0.7696 | 0.7731 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | EAS | 3 | 0.6776 | 0.0076 | 0.6710 | 0.6860 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | AFR | 3 | 0.4447 | 0.0063 | 0.4374 | 0.4486 |
| prscsx | tuned (oracle phi & weights) | EUR | 3 | 0.7715 | 0.0045 | 0.7663 | 0.7746 |
| prscsx | tuned (oracle phi & weights) | EAS | 3 | 0.6743 | 0.0084 | 0.6667 | 0.6833 |
| prscsx | tuned (oracle phi & weights) | AFR | 3 | 0.4351 | 0.0040 | 0.4308 | 0.4387 |
| prscsx | auto + META (no tuning) | EUR | 3 | 0.7181 | 0.0218 | 0.6932 | 0.7339 |
| prscsx | auto + META (no tuning) | EAS | 3 | 0.6333 | 0.0135 | 0.6181 | 0.6438 |
| prscsx | auto + META (no tuning) | AFR | 3 | 0.4446 | 0.0215 | 0.4217 | 0.4643 |
| improved | tuned (phi grid + validation, n=3000) | EUR | 3 | 0.7796 | 0.0058 | 0.7730 | 0.7835 |
| improved | tuned (phi grid + validation, n=3000) | EAS | 3 | 0.6770 | 0.0053 | 0.6727 | 0.6830 |
| improved | tuned (phi grid + validation, n=3000) | AFR | 3 | 0.4428 | 0.0113 | 0.4306 | 0.4528 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | EUR | 3 | 0.7820 | 0.0064 | 0.7747 | 0.7862 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | EAS | 3 | 0.6858 | 0.0025 | 0.6838 | 0.6886 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | AFR | 3 | 0.4573 | 0.0103 | 0.4456 | 0.4647 |
| improved | tuned (oracle phi & weights) | EUR | 3 | 0.7814 | 0.0049 | 0.7759 | 0.7852 |
| improved | tuned (oracle phi & weights) | EAS | 3 | 0.6773 | 0.0052 | 0.6729 | 0.6830 |
| improved | tuned (oracle phi & weights) | AFR | 3 | 0.4432 | 0.0108 | 0.4316 | 0.4530 |
| improved | auto + META (no tuning) | EUR | 3 | 0.7147 | 0.0122 | 0.7009 | 0.7241 |
| improved | auto + META (no tuning) | EAS | 3 | 0.6396 | 0.0051 | 0.6338 | 0.6435 |
| improved | auto + META (no tuning) | AFR | 3 | 0.4590 | 0.0062 | 0.4530 | 0.4653 |

## Averaging independent chains (post hoc from MCMC seeds 1, 11, 12; dataset h0.5_pi0.01_r1)

| method | metric | target | single_chain_mean | single_chain_sd | avg_2_chains | avg_3_chains |
|---|---|---|---|---|---|---|
| prscsx | tuned (phi grid + validation, n=3000) | EUR | 0.7704 | 0.0040 | 0.7727 | 0.7762 |
| prscsx | tuned (phi grid + validation, n=3000) | EAS | 0.6738 | 0.0081 | 0.6731 | 0.6781 |
| prscsx | tuned (phi grid + validation, n=3000) | AFR | 0.4345 | 0.0039 | 0.4420 | 0.4435 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | EUR | 0.7713 | 0.0017 | 0.7717 | 0.7748 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | EAS | 0.6776 | 0.0076 | 0.6764 | 0.6822 |
| prscsx | stacked over phi x ancestry (NNLS, validation n=3000) | AFR | 0.4447 | 0.0063 | 0.4490 | 0.4537 |
| prscsx | tuned (oracle phi & weights) | EUR | 0.7715 | 0.0045 | 0.7744 | 0.7767 |
| prscsx | tuned (oracle phi & weights) | EAS | 0.6743 | 0.0084 | 0.6733 | 0.6785 |
| prscsx | tuned (oracle phi & weights) | AFR | 0.4351 | 0.0040 | 0.4421 | 0.4435 |
| prscsx | auto + META (no tuning) | EUR | 0.7181 | 0.0218 | 0.7355 | 0.7262 |
| prscsx | auto + META (no tuning) | EAS | 0.6333 | 0.0135 | 0.6441 | 0.6386 |
| prscsx | auto + META (no tuning) | AFR | 0.4446 | 0.0215 | 0.4622 | 0.4539 |
| improved | tuned (phi grid + validation, n=3000) | EUR | 0.7796 | 0.0058 | 0.7882 | 0.7861 |
| improved | tuned (phi grid + validation, n=3000) | EAS | 0.6770 | 0.0053 | 0.6808 | 0.6807 |
| improved | tuned (phi grid + validation, n=3000) | AFR | 0.4428 | 0.0113 | 0.4536 | 0.4495 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | EUR | 0.7820 | 0.0064 | 0.7908 | 0.7876 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | EAS | 0.6858 | 0.0025 | 0.6867 | 0.6868 |
| improved | stacked over phi x ancestry (NNLS, validation n=3000) | AFR | 0.4573 | 0.0103 | 0.4672 | 0.4579 |
| improved | tuned (oracle phi & weights) | EUR | 0.7814 | 0.0049 | 0.7885 | 0.7868 |
| improved | tuned (oracle phi & weights) | EAS | 0.6773 | 0.0052 | 0.6810 | 0.6810 |
| improved | tuned (oracle phi & weights) | AFR | 0.4432 | 0.0108 | 0.4550 | 0.4499 |
| improved | auto + META (no tuning) | EUR | 0.7147 | 0.0122 | 0.7247 | 0.7191 |
| improved | auto + META (no tuning) | EAS | 0.6396 | 0.0051 | 0.6458 | 0.6434 |
| improved | auto + META (no tuning) | AFR | 0.4590 | 0.0062 | 0.4607 | 0.4655 |

## Relative estimation error ||w - beta||^2/||beta||^2 of the ancestry-specific posterior (best phi; lower is better)

| scenario | ancestry | prscsx | improved | rbonly | cplonly | ldshrink |
|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | 0.8843 | 0.8443 |  |  |  |
| h0.2_pi0.01 | EAS | 0.9081 | 0.8845 |  |  |  |
| h0.2_pi0.01 | AFR | 0.9376 | 0.9591 |  |  |  |
| h0.5_pi0.001 | EUR | 0.5749 | 0.5754 |  |  |  |
| h0.5_pi0.001 | EAS | 0.6187 | 0.5776 |  |  |  |
| h0.5_pi0.001 | AFR | 0.4645 | 0.4908 |  |  |  |
| h0.5_pi0.01 | EUR | 0.7031 | 0.6917 | 0.7200 | 0.6931 | 0.6777 |
| h0.5_pi0.01 | EAS | 0.7192 | 0.7053 | 0.7626 | 0.7033 | 0.7074 |
| h0.5_pi0.01 | AFR | 0.7593 | 0.7703 | 0.7623 | 0.7768 | 0.7779 |
| h0.5_pi0.1 | EUR | 0.9108 | 0.9061 |  |  |  |
| h0.5_pi0.1 | EAS | 0.9640 | 0.9541 |  |  |  |
| h0.5_pi0.1 | AFR | 0.9835 | 0.9832 |  |  |  |

## Main table: tuned PRS (phi grid + validation n=3000, README procedure). r2_g = R2/h2; R2 = phenotypic R2 of a genome-wide PRS with this per-SNP accuracy; r = corr(PRS, phenotype); est_error at best phi

| scenario | target | method | n | r2_g | R2 | r | est_error |
|---|---|---|---|---|---|---|---|
| h0.2_pi0.01 | EUR | prscsx | 2 | 0.5240 | 0.1048 | 0.3237 | 0.8843 |
| h0.2_pi0.01 | EUR | improved | 2 | 0.5727 | 0.1145 | 0.3384 | 0.8443 |
| h0.2_pi0.01 | EAS | prscsx | 2 | 0.4003 | 0.0801 | 0.2829 | 0.9081 |
| h0.2_pi0.01 | EAS | improved | 2 | 0.3962 | 0.0792 | 0.2815 | 0.8845 |
| h0.2_pi0.01 | AFR | prscsx | 2 | 0.2108 | 0.0422 | 0.2053 | 0.9376 |
| h0.2_pi0.01 | AFR | improved | 2 | 0.2111 | 0.0422 | 0.2055 | 0.9591 |
| h0.5_pi0.001 | EUR | prscsx | 2 | 0.9197 | 0.4598 | 0.6781 | 0.5749 |
| h0.5_pi0.001 | EUR | improved | 2 | 0.9305 | 0.4652 | 0.6821 | 0.5754 |
| h0.5_pi0.001 | EAS | prscsx | 2 | 0.9058 | 0.4529 | 0.6730 | 0.6187 |
| h0.5_pi0.001 | EAS | improved | 2 | 0.9340 | 0.4670 | 0.6834 | 0.5776 |
| h0.5_pi0.001 | AFR | prscsx | 2 | 0.7807 | 0.3904 | 0.6248 | 0.4645 |
| h0.5_pi0.001 | AFR | improved | 2 | 0.7871 | 0.3935 | 0.6273 | 0.4908 |
| h0.5_pi0.01 | EUR | prscsx | 2 | 0.7480 | 0.3740 | 0.6115 | 0.7031 |
| h0.5_pi0.01 | EUR | improved | 2 | 0.7478 | 0.3739 | 0.6115 | 0.6917 |
| h0.5_pi0.01 | EAS | prscsx | 2 | 0.6795 | 0.3397 | 0.5829 | 0.7192 |
| h0.5_pi0.01 | EAS | improved | 2 | 0.6776 | 0.3388 | 0.5821 | 0.7053 |
| h0.5_pi0.01 | AFR | prscsx | 2 | 0.4554 | 0.2277 | 0.4772 | 0.7593 |
| h0.5_pi0.01 | AFR | improved | 2 | 0.4737 | 0.2369 | 0.4867 | 0.7703 |
| h0.5_pi0.1 | EUR | prscsx | 2 | 0.5516 | 0.2758 | 0.5252 | 0.9108 |
| h0.5_pi0.1 | EUR | improved | 2 | 0.5522 | 0.2761 | 0.5255 | 0.9061 |
| h0.5_pi0.1 | EAS | prscsx | 2 | 0.4085 | 0.2042 | 0.4519 | 0.9640 |
| h0.5_pi0.1 | EAS | improved | 2 | 0.4075 | 0.2037 | 0.4514 | 0.9541 |
| h0.5_pi0.1 | AFR | prscsx | 2 | 0.2295 | 0.1148 | 0.3388 | 0.9835 |
| h0.5_pi0.1 | AFR | improved | 2 | 0.2373 | 0.1186 | 0.3444 | 0.9832 |

