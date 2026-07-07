# PegaSUS Modules Remediation Roadmap (non-LDO)

Generated 2026-07-07 by a 25-agent discovery workflow (`wf_44c94e6d-bb7`): 10 parallel module investigators → dedup/rank synthesis → 14 adversarial verifiers. 93 raw findings (6 blocker / 32 major / 55 minor) collapsed to the ranked items below. **Ranked by leverage toward the win condition: the full-range 2000–2024 national all-source compile+investigate running end-to-end on a 34 GB-RAM / 6 GB-VRAM box, correctly.**

Verification adjusted several findings — recorded honestly per item. The LDO/EFG-math completeness work is DONE and out of scope here.

## Verification outcomes on the top-14
- **REJECTED (over-stated):** (a) "DATASUS fetch has no retry" — the R script DOES retry 4× with `Sys.sleep(min(2·attempt,8))` backoff; (b) "eager combine fallback OOM" — the empty-group path is unreachable and the concat fallback rarely fires; (c) "on-success GC orphaned" — `_prune_success_ancillary` already deletes stdout/stderr/heartbeat inline on `{success,cached}` (the `storage_gc.py` module fns are orphaned but the debug-file pruning is live).
- **CONFIRMED:** datasus_combined re-materialization (I4/W7), datasus_combined throwaway never deleted, M2 float32 never built, tensor_values retained, SIDRA warm-cache absent, SIDRA S3/S4 sprawl, SIDRA UF fan-out hardcoded 6, DatasusConfig 3-layer drift, microdatasus-retirement incomplete, transient-chunk completeness-gate gap.
- **Self-verified (outside the top-14 slice):** T1.1 window-blind national cache = **REAL** (`_national_datasus_cached` keys on `out.exists()` of a years-less path).

## TIER-1 — win-condition blockers & correctness (land first)
| # | Item | Files | Sev | Eff | Status |
|---|---|---|---|---|---|
| T1.1 | **Window-blind national cache** silently truncates the study (a 2000-2024 request reuses a 2000-2020 `national/{system}__processed_events.parquet`) | `acquire/sidra_national.py:142`; national combine writer | BLOCKER | S | new, self-confirmed |
| T1.2 | Transient chunk loss → **no completeness gate**: nothing aggregates per-chunk `failed`/`timeout` into a fail-closed check (R retry exists, so lower risk than first stated) | `datasus/subprocess.py`, `pipeline.py:202` | MAJOR | M | confirmed (retry-claim rejected) |
| T1.3 | National **race-prior dropped** (`pipeline.py:475` hardcodes `race_prior_artifact=None`; `compile.py:519` requires it for planned/embedded) | pipeline/compile | MAJOR | M | to self-verify |
| T1.4 | **EFG stage workspace never cleaned** — full national tensor payload duplicated on disk every run | `efg/compile_attach.py`, `output/bundle_manager.py` | BLOCKER | S | to self-verify |
| T1.6 | **`datasus_combined` re-materialization** (26→50 GB): `combined_hash` folds in `fetched_at` → new full national copy per identical re-run; intermediate never deleted | `pipeline.py:237-257` | BLOCKER | M | CONFIRMED |
| T1.8 | **M2 float32 denominator solve** never built — national peak RAM ~2× floor (config already declares float32) | `denominators/reconstruction/*` | MAJOR | M | CONFIRMED |
| T1.9 | `_candidate_pairs` O(N²) national migration blowup (5570²/year before the guard no-ops) | `denominators/population/migration.py:108` | MAJOR | S | new |
| T1.10 | `tensor_values`/`migration_values` full national arrays retained on the result dataclass just to `len()` | `denominators/.../orchestrator.py:628` | MAJOR | S | CONFIRMED (aliased, not copied) |
| T1.11 | DatasusConfig defaults diverge across 3 layers (8/4/16 workers, 300/900 heartbeat) | `datasus/subprocess.py`, `acquire/datasus.py:69` | MAJOR | S | CONFIRMED |

## TIER-2 — high-leverage quality (scoped redesigns, big perf, correctness edges)
W7 lazy Hive views (subsumes T1.6-proper + SIDRA S3/S4 + the unwired `storage/duckdb.py`+`scan_table` SCALE-01 substrate) · SIDRA warm-cache fast path · SIDRA cache TTL/zstd (W8) · Q-tensor `_q_row` state re-classification (declared state can disagree with materialized n_eff) · GPU wiring (STDFM `prefer_cuda` hardcoded False; population G1 backend absent; VRAM reservation) · dead record-oracle path removal (`normalize/records.py` ~439 LOC, 0 callers) · two divergent DATASUS entrypoints + hardcoded `_ALL_UFS` · SIDRA UF fan-out config · schema-pinned-from-first-batch normalize risk · validation full-reads parquet for row count.

## TIER-3 — cleanup / dead-code / drift
`pirs/` residue · EFG materialization-manifest split · 3 fixture-only bundle writers · orphaned `curated_cause_report.py` (wire-or-delete) · `disease/variable_grammar.py` 2nd generator · `SpatialWeightGraph` view API orphaned · REG-07 3-loader unification · SIM cod7/geo-state drift · SINASC race decode divergence · `directly_standardized_rate` int truncation · SIDRA thousands-sep parse · reproducibility manifest hardcoded seed/version · empty-bundle `run_profile` under-requires keys · assorted stale annotations/egg-info.

## Progress log
**Landed this wave (tested, committed):**
- ✅ **T1.1** window-safe national cache (`64b20d6`) — the silent-truncation correctness blocker.
- ✅ **T1.4** EFG stage-workspace deleted after flush (`e999d3a`) — kills the per-run tensor-payload duplicate.
- ✅ **T1.9** migration candidate pairs O(N²)→O(N·k) + **T1.11** DatasusConfig default unification (`80456ce`).
- ✅ **M2 (partial)** population-problem inputs stored float32 (`ac17ac0`) — ~8 GB national RAM reclaimed, lossless (solver still f64).

**Remaining, in order:** T1.6 `datasus_combined` content-addressing (biggest storage lever) → T1.3 national race-prior wiring → T1.2 fetch-completeness gate → T1.10 result-array retention → **TIER-2 W7 lazy Hive views** (the definitive storage redesign) + **GPU wiring** (STDFM `prefer_cuda`, population G1) → TIER-3 dead-code cluster. Then materialize the national cache + run the contextual determinant study (needs a persistent background job).

Note: the full float32 SOLVE working-set (§V.4 mixed-precision, the other half of M2) and W7 are the two L-effort redesigns; everything above them is S/M and lands incrementally.
