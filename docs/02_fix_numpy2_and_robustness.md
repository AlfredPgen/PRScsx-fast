# Making PRS-CSx run: root cause, fix and verification

Fixed code: the root `PRScsx.py`, `mcmc_gtb.py` and `parse_genet.py` of this repository (drop-in replacement, same CLI and same output files).
Git branch `fix/numpy2-compat` in `/work/PRScsx`, commit `d32d6d3`; patch: `/work/proj/prscsx_numpy2_fix.patch`
(3 files, +43 / −7 lines).

## 1. The error

**Symptom.** With a current scientific Python stack (NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0) the README example
prints `... MCMC ...` and then never finishes. There is no error message, no `--- iter-100 ---` and no output files.
The same command finishes in about 40 s with NumPy 1.26.4.

**Root cause.** In `mcmc_gtb.py` the ψ update calls

    psi[jj] = gigrnd.gigrnd(a-0.5*n_grp[jj], 2.0*delta[jj], xx[jj])

`n_grp`, `delta` and `xx` are (p,1) arrays, so `n_grp[jj]` is a **shape-(1,) array**, not a scalar.
`gigrnd()` starts with `p = float(p)`. NumPy 1.25 deprecated converting arrays with ndim > 0 to scalars, and
NumPy 2.x raises `TypeError: only 0-dimensional arrays can be converted to Python scalars` instead.
The call sits inside

    while True:
        try:    psi[jj] = gigrnd.gigrnd(...)
        except: continue

The bare `except:` meant to retry rare numerical failures catches the `TypeError`, and the call fails the same way
every time, so **the loop never ends**. The bare `except:` also catches `KeyboardInterrupt` and other
interrupt exceptions, which is why the hung program is hard to stop with Ctrl-C.

**How all affected sites were found.** I ran the original on NumPy 1.26 with `-W always::DeprecationWarning`
over every code path (fixed φ, auto φ, meta, meta + write_pst, single ancestry) and collected each line that
emitted the "array with ndim > 0 to a scalar" warning. That gave **four sites**, each of which fails on NumPy ≥ 2:

| Site | Effect on NumPy ≥ 2 |
|---|---|
| `gigrnd.py:38` `float(p); float(a); float(b)` (called from `mcmc_gtb.py:106`) | **infinite loop** (above) |
| `mcmc_gtb.py:172` `'%.6e' % beta` (ancestry-specific output) | crash *after* the whole MCMC |
| `mcmc_gtb.py:186` `'%.6e' % beta` (META output without `--write_pst`) | crash after the MCMC |
| `mcmc_gtb.py:191` `'%1.2e' % phi_est` (auto φ) | crash after the MCMC |

I also checked that size-1 truth tests (`max()` in the σ² update) still work on NumPy 2.4, and that the legacy
`numpy.random` MT19937 stream (`randn`, `random`, `gamma`) is bit-identical between NumPy 1.26.4 and 2.4.6.

## 2. Changes

| # | File | Change | Effect on results |
|---|---|---|---|
| 1 | `mcmc_gtb.py` | Pass scalars to `gigrnd` (`n_grp[jj,0]`, `delta[jj,0]`, `xx[jj,0]`). **This is the fix for the hang.** | None (same float64 values) |
| 2 | `mcmc_gtb.py` | Write `beta_est[pp][:,0]` and `mu[:,0]` (scalar per row) | None (same text) |
| 3 | `mcmc_gtb.py` | `phi = random.gamma(...).item()`, so φ is a scalar and can be printed | None (same value, same RNG draw) |
| 4 | `mcmc_gtb.py` | Bare `except:` narrowed to `(ArithmeticError, ValueError)`, the numerical failures the retry exists for. Gives up with a clear `RuntimeError` after 1,000 consecutive failures for one SNP (a deterministic error, which previously meant an infinite loop). | None for any run that terminated before |
| 5 | `mcmc_gtb.py` | `n_pst = n_iter//thin − n_burnin//thin` (the true number of retained draws) instead of `int((n_iter−n_burnin)/thin)` | None when `n_burnin` is a multiple of `thin` (incl. defaults). Otherwise fixes an `IndexError` crash with `--write_pst` and a silent mis-scaling of the posterior mean (e.g. ÷99 instead of ÷100) |
| 6 | `PRScsx.py` | Validate `--thin ≥ 1`, `--n_iter > --n_burnin ≥ 0`, at least one retained draw | Previously, zero retained draws silently wrote **all-zero effect sizes** with exit code 0 (verified) |
| 7 | `PRScsx.py` | Clear message if `--ref_dir` has no `snpinfo_mult_*_hm3` | Previously `UnboundLocalError` (verified) |
| 8 | `parse_genet.py` | `.bim` chromosome codes: accept a `chr` prefix, skip non-numeric codes (X, Y, MT, ...) | Previously `ValueError: invalid literal for int(): 'X'` (verified). Numeric `.bim` files behave exactly as before |

## 3. Verification (all run, not inferred)

**Fixed code vs original outputs.** The original ran on NumPy 1.26.4. The same 9 configurations as the baseline were
rerun with the fixed code and every output file compared, BLAS threads = 1:

| Environment | Configurations | Output files byte-identical to original | max abs diff |
|---|---|---|---|
| Python 3.11.18, NumPy 1.26.4, SciPy 1.11.4, h5py 3.10.0 | 9 (seeds 1–5 φ=1e-2; meta; meta + write_pst; auto-φ + meta; EUR-only) | **20 / 20** | 0 |
| Python 3.11.18, **NumPy 2.4.6, SciPy 1.17.1, h5py 3.16.0** | same 9 | **20 / 20** | 0 |

The fixed program is therefore a drop-in replacement: on the new stack it gives exactly the original's numbers.

**README command, unchanged, on NumPy 2.4.6:** completes, exit 0, 27.8 s wall (100 s CPU, default threads),
241 MB peak RSS; writes 1,000 EUR and 901 EAS effects.

**Clean timing** (`benchmarks/timing_fixed_vs_original.sh`): idle machine, runs one at a time, interleaved,
3 repeats, example data, φ = 1e-2, seed 1, 1 BLAS thread. All 9 runs' outputs are byte-identical to the original baseline.

| Variant | wall per run (s) | mean (s) | peak RSS (MB) |
|---|---|---|---|
| original, NumPy 1.26.4 | 40.8, 39.8, 44.3 | **41.6** | 237 |
| fixed, NumPy 1.26.4 | 30.6, 31.9, 34.0 | **32.2** | 238 |
| fixed, NumPy 2.4.6 | 28.4, 30.1, 30.8 | **29.8** | 241 |

The fix is 23% faster on the *same* NumPy, so the gain comes from the code change. **Why (measured):** the
original call site builds about six tiny NumPy arrays per GIG call (`n_grp[jj]`, `0.5*…`, `a-…`, `delta[jj]`, `2.0*…`,
`xx[jj]`) and converts them with `float()`. That costs 4.61 µs per call, against 0.40 µs for the scalar version, so
about 8.4 s over the example's 2,000,000 calls, which is most of the measured 9.4 s. My earlier guess that the
deprecation warnings explained it was **wrong**: they account for only about 1.8 s.

**Edge cases, NumPy 2.4.6:**
- `--n_iter=1000 --n_burnin=502 --thin=5 --meta=TRUE --write_pst=TRUE`: the original crashed with `IndexError`; the fix
  runs and writes 100 draws (= 1000//5 − 502//5). The mean of the written draws equals the META posterior mean from a
  `--meta`-only run with the same seed: ratio median 1.000000, range [0.99974, 1.000013], consistent with 7-significant-digit output rounding.
- `.bim` with extra `X` and `chr22` lines: runs; output byte-identical to the original EUR-only baseline.
- Missing reference → clear message, exit 2. `--n_iter ≤ --n_burnin`, `--thin=0`, or zero retained draws → clear message, exit 2.

**Regression tests** (`tests/test_prscsx_fixed.py`, 19 tests): GIG scalar inputs; documented root cause; MCMC end
to end on a tiny synthetic problem; auto-φ printing; non-numerical errors propagate instead of hanging; transient
numerical errors are retried; a deterministic numerical error stops with a message; draw counts for 4
(n_iter, n_burnin, thin) settings; `.bim` parsing; CLI validation; and 3 end-to-end runs byte-compared against the
stored original outputs (need `PRSCSX_REF_DIR`). **19 / 19 pass on NumPy 1.26.4 and on NumPy 2.4.6.**

**Negative control** (`tests/run_negative_control.sh`): the same 19 tests against the **original** code,
one process per test, hard-killed with SIGKILL at the deadline:

| Original code on | PASS | FAIL | hung (SIGKILL) | What fails |
|---|---|---|---|---|
| NumPy 1.26.4 (120 s limit) | 11 | 6 | 2 | Error propagation and deterministic-error tests hang (bare `except`). Draw count (60, 12, 5) → IndexError. X-chromosome `.bim`, 3 CLI checks, missing reference. |
| NumPy 2.4.6 (60 s limit) | 2 | 5 | 12 | **Every test that runs the MCMC hangs** (the reported bug), plus the `.bim`/CLI failures. |

Tests that pass on the original are the ones that *should*: byte-identity on the default path, draw counts when
`n_burnin` divides cleanly, and the standalone GIG tests. `test_numpy2_root_cause…` documents the mechanism and
does not discriminate between old and new code.

**Re-test after the sandbox restart.** The fixed code on disk was checked to be identical to the committed
branch, then on NumPy 2.4.6: README command exit 0 with full outputs; **19/19 tests pass**
(`fix_verification/pytest_fixed_numpy2.4_after_restart.log`).

**Incident note.** The first attempt at this negative control ran in the foreground and relied on
pytest-timeout. That mechanism raises an exception inside the hung loop, which the original's bare `except:` swallowed,
so the process spun forever and blocked the shell until the sandbox was restarted. The control was then rerun safely
with SIGKILL (which cannot be caught) in the background. The incident is itself a live demonstration of change #4.

## 4. Input pitfalls that remain (code reading, not changed, not run)
- Alleles must be upper-case A/C/G/T. Lower-case alleles (e.g. METAL output) are silently dropped.
- Sumstats columns are read **by position** (`SNP A1 A2 BETA|OR SE|P`). Extra columns are allowed only after these.
- `NA` values, or SE = 0, in the sumstats raise an error.
- The reference panels contain no strand-ambiguous SNPs; custom panels that do would hit the original's
  ambiguous strand-flip matching.
