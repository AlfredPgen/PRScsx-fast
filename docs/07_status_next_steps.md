# Status of the next-steps work packages (WP1–WP7)

The work packages were received as a message on 18 November 2026 (after an external review). The review folder they
refer to (`instructions/review_2026-11-18/`, with `make_synth_ref.py`, `stress.py` and results) was **not** available in
this environment, so the synthetic-panel generator and the coupling stress test were re-implemented from the
description (`tests/make_synth_ref.py`, `benchmarks/coupling_stress.py`). All results below were produced on
18–19 November 2026 on a 4-core sandbox with the 1000 Genomes reference panel (EUR, EAS, AFR) available.

| WP | item | status | how it was verified | date | where |
|---|---|---|---|---|---|
| 1 | synthetic panel with awkward cases; fast byte-identical | done | 4 configurations byte-identical | 18 Nov | `tests/make_synth_ref.py` (re-implemented), `tests/test_fast_synthetic.py` |
| 1 | start-up self-check, `--strict` | done | tests pass; simulated mismatch gives warning / exit 3 | 18 Nov | `fast/fastcsx/selfcheck.py`, `tests/test_fast_selfcheck.py` |
| 1 | `--n_jobs` over (chromosome, φ) | done | φ grid in parallel = separate runs (4/4 files identical) | 18 Nov | `tests/test_fast_selfcheck.py` |
| 1 | bit identity on Windows | not re-verified here | reported by the review (Windows 11); no Windows machine here | — | `fast/README.md` |
| 2 | low-rank update (T2a–T2d) | done | tests pass; speed measured | 18 Nov | `fast/fastcsx/lowrank.py`, `tests/test_lowrank.py`, `results_v2/wp2_*` |
| 3 | eigen panel format, builder, converter, readers | done | tests pass; built vs dense panel within criterion | 18 Nov | `docs/04`, `tools/build_ld_panel.py`, `tests/test_eigen_panel.py`, `results_v2/wp3_step4_*` |
| 3 | truncation study → default `--rank_var 0.999` | done | parts A and B on real 1000 Genomes chr22 LD | 18 Nov | `results_v2/wp3_step5_truncation.json`, `docs/05` |
| 3 | build cost, dense/biobank panels | partly | per-block costs measured on simulated genotypes; genome-wide numbers are extrapolations (labelled); no real dense genotypes available | 18–19 Nov | `results_v2/wp3_step6_*`, `docs/05` |
| 4 | `--coupling=learned` + numerical check | done | Gibbs E[τ] vs exact: 0.002% | 18 Nov | `tests/test_coupling_learned.py`, `results_v2/wp4_step1_*` |
| 4 | coupling stress test (synthetic LD, 20 replicates) | done | rule applied → default `snr` | 19 Nov | `results_v2/wp4_coupling_stress/` |
| 4 | stress test on real LD with 20 replicates | **not done** | estimated about 37 h of compute here | — | — |
| 5 | pre-registered accuracy benchmark v2 | done (720/720); verdict: no demonstrated difference | rule fixed in advance | 19 Nov | `benchmarks/PLAN_accuracy_v2.md`, `results_v2/wp5_accuracy/` |
| 5 | real-data protocol (BBJ, TWB, KoGES, UKB → CKB) | written, **not run** | no access to the data | 19 Nov | `docs/06_real_data_evaluation.md`, `tools/compare_prs.py` |
| 6 | `--warm_start` | done, off by default | 1.23× (< 1.3×); no loss in increasing φ order, loss in decreasing order | 19 Nov | `results_v2/wp6/wp6_summary.md` |
| 6 | `--auto_stop` | done, off by default | 360 runs: 0.73 of the 2-chain cost, no detectable loss | 19 Nov | `results_v2/wp6/wp6_summary.md` |
| 6 | profile of a low-rank chromosome run; `prange` | done; `prange` not needed | other code 0.4–1.4% of time (< 15%) | 19 Nov | `results_v2/wp6_profile_*.json` |
| 6 | GPU | not done | no GPU available | — | — |
| 7 | documentation updates | done | — | 19 Nov | READMEs, `docs/03` §7, `docs/04`–`07` |
| — | full test suite | done: 71 passed with the 1000 Genomes panel (11.5 min); 52 passed, 19 skipped without it | `PRSCSX_REF_DIR=... python -m pytest tests/` | 19 Nov | `tests/` |

