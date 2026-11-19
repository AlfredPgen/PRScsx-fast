# WP6: warm start and automatic stopping (improved/)

## 6.1 --warm_start, phi grid 1, 1e-2, 1e-4, 1e-6 (decreasing); 30 datasets x 3 target ancestries

Speed-up (cold / warm): iterations 1.231x, measured MCMC time 1.228x (mean over datasets).

| phi | warm - cold: mean R2/h2 diff [95% CI] | cold2 - cold (MC noise): mean diff [95% CI] | mean abs diff warm vs cold | mean abs diff cold2 vs cold |
|---|---|---|---|---|
| 1 | +0.00000 [+0.00000, +0.00000] | -0.00010 [-0.00021, +0.00000] | 0.00000 | 0.00039 |
| 1e-2 | -0.00179 [-0.00304, -0.00054] | -0.00009 [-0.00101, +0.00083] | 0.00396 | 0.00358 |
| 1e-4 | -0.00967 [-0.01485, -0.00449] | -0.00349 [-0.00921, +0.00223] | 0.01787 | 0.01996 |
| 1e-6 | -0.00729 [-0.02021, +0.00564] | -0.00280 [-0.01242, +0.00683] | 0.04011 | 0.03234 |
| tuned_validation | -0.00966 [-0.01489, -0.00444] | -0.00210 [-0.00767, +0.00348] | 0.01663 | 0.01601 |

Correlation of posterior means with the cold run (warm-started phi values, mean over datasets and ancestries): warm 0.8495, cold2 (other seed) 0.8431.

## 6.1 --warm_start, phi grid 1e-6, 1e-4, 1e-2, 1 (increasing); 30 datasets x 3 target ancestries

Speed-up (cold / warm): iterations 1.231x, measured MCMC time 1.248x (mean over datasets).

| phi | warm - cold: mean R2/h2 diff [95% CI] | cold2 - cold (MC noise): mean diff [95% CI] | mean abs diff warm vs cold | mean abs diff cold2 vs cold |
|---|---|---|---|---|
| 1 | -0.00020 [-0.00031, -0.00009] | -0.00010 [-0.00021, +0.00000] | 0.00047 | 0.00039 |
| 1e-2 | -0.00105 [-0.00192, -0.00017] | -0.00009 [-0.00101, +0.00083] | 0.00320 | 0.00358 |
| 1e-4 | -0.00067 [-0.00626, +0.00492] | -0.00349 [-0.00921, +0.00223] | 0.01884 | 0.01996 |
| 1e-6 | +0.00000 [+0.00000, +0.00000] | -0.00280 [-0.01242, +0.00683] | 0.00000 | 0.03234 |
| tuned_validation | +0.00148 [-0.00301, +0.00598] | -0.00259 [-0.00813, +0.00295] | 0.01049 | 0.01618 |

Correlation of posterior means with the cold run (warm-started phi values, mean over datasets and ancestries): warm 0.9557, cold2 (other seed) 0.9515.

## 6.2 --auto_stop (2 chains, max 3,000 iterations, burn-in 1,500; 120 datasets x 3 phi)

Converged before the maximum: 360 of 360 runs; iterations used: median 2100, mean 2199, range 2100-2900.
Cost relative to --n_chains=2 fixed length: 0.733 (iterations); relative to the default single chain: 1.466.

| comparison | metric | mean diff [95% CI] |
|---|---|---|
| auto2 - fixed2 | r2_tuned_oracle | -0.00047 [-0.00154, +0.00060] |
| auto2 - fixed2 | r2_tuned_validation | -0.00090 [-0.00219, +0.00040] |
| auto2 - fixed1 | r2_tuned_oracle | +0.00042 [-0.00114, +0.00199] |
| auto2 - fixed1 | r2_tuned_validation | +0.00012 [-0.00172, +0.00196] |
| fixed2 - fixed1 | r2_tuned_oracle | +0.00090 [-0.00065, +0.00244] |
| fixed2 - fixed1 | r2_tuned_validation | +0.00102 [-0.00068, +0.00273] |

Mean stop iteration by scenario: h0.2_pi0.001_rho0.8_m1 2153, h0.2_pi0.01_rho0.8_m1 2183, h0.2_pi0.1_rho0.8_m1 2170, h0.5_pi0.001_rho0.8_m1 2163, h0.5_pi0.01_rho0.8_m1 2220, h0.5_pi0.1_rho0.8_m1 2303

