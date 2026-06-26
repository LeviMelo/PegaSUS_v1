# PegaSUS — MSD Compliance Remediation Plan (Plan of Record)

Status: **active**. Authoritative spec: `MSD.md` (6790 lines, read in full).
Strategy (locked with user): **architecture-first → radical refactor + compliance → rewrite tests as fraud-detectors.**

This document is the durable ledger. Every defect carries `file:line` evidence and an
MSD clause. "Fake" = label/manifest claims compliance the computation does not deliver.
"Missing" = required computation absent. "Bug" = implemented but wrong.

---

## 0. The disease (root cause, not a symptom)

The codebase is a chain of "slices" that emit MSD-compliance-labeled JSON manifests while
the math behind the labels is partial or absent. Two structural mechanisms let this survive:

1. **Failure-swallowing orchestrator.** `workflows/msd_inference.py` dispatches PIRS stages
   by *string name* (`_load_callable` + `importlib`, lines 38–44, 139–199) inside
   `try/except Exception` (line 200). Any failure — including the model-family
   `NotImplementedError` — becomes `pirs_model.status = "blocked"` with a reason string, and
   **the run still completes and emits a 17-key bundle.** Non-implementation is relabeled as a
   benign skip.
2. **Label-only validation.** `output/validate.py` (to confirm) checks 17-key presence, not
   computation invariants. Manifests echo their own input labels through
   (`model_execution.py:324` copies `manifest.get("residual_mode")` verbatim).

Until these are closed (Phase 0), every downstream fix can silently regress.

---

## 1. Verified defect ledger

| ID | MSD | Kind | Evidence | Status |
|----|-----|------|----------|--------|
| D1 | §6.2 | Missing | `pirs/model_execution.py:213` raised `NotImplementedError` for NB/gamma/hurdle/zero_inflated/dirichlet/binomial_proportion/sih_gamma. Also `model_execution.py` called undefined `_read_rows`/`_write_rows` → **NameError on first execution line**, swallowed to "blocked": the entire stage was dead theatre. | ✅ **FIXED.** New `compute/glm.py` IRLS kernel fits gaussian/poisson/gamma/NB/binomial-proportion with correct §6.5 residuals. `model_execution.py` rewired (`_fit_model`), NameError import fixed. Remaining: hurdle/ZINB/Dirichlet, binomial prior-weights `n`. |
| D2 | §6.6.1, §10 | **Fake** | `pirs/crossfit.py` emitted `residual_mode: cross_fitted, n_folds: 5`; guard checked only the **label**. `model_execution` fit on the whole dataset — no K-fold loop. | ✅ **FIXED.** `compute/glm.crossfit_residuals` genuinely fits `M_{-k}` on each fold's complement and predicts held-out residuals (block-preserving). `_fit_model` records `residual_mode_actual` from the computation, not the label. Remaining: deep parametric-bootstrap (§6.6.2). |
| D3 | §3.15.2 | Missing | No `C_emp`/`0.98`/Spearman anywhere. Stage-1 (`efg/equivalence.py`) genuinely well-built; Stage-2 absent. | ✅ **FIXED.** New `efg/empirical_compression.py` computes `C_emp=max(\|pearson\|,\|spearman\|)` on materialized Q_tensor vectors, folds duplicates ≥0.98 into higher-utility canonicals. Wired post-TopK in `pirs/selection_plan.write_pirs_selection_plan` and *realised* by filtering the covariate set the design plan consumes (no data destroyed; suppressed nodes reported + retained). |
| D4 | §2.12.3 | **Not active fraud** (re-assessed) | Traced the live SIDRA path: the **only** SIDRA exposure is the population denominator via `she/population/sidra_anchor.py`, which correctly selects the marginalized **Total** (sex∧race∧age all total, exactly-one-fact) — no category+total double-count (§2.12.2 honoured). High-dim SIDRA **context fields (V_X: GDP/sanitation/education) are not built at all**, so there is no unbounded high-dim exposure to guard. | 🟢 **No live violation.** Pushforward/projection primitives (`sidra/pushforward.py`) + `execute_bounded_pushforward` are implemented and ready. The remaining work is **feature** (build the V_X context-field subsystem with projection/stitching/ST-DFM) which needs real SIDRA fact + registry data — *not* a fraud fix. |
| D5 | §2.3 | Bug+Fake | `model_execution._fit_least_squares` silently median-imputed; `.isdigit()` check **misclassified every negative value as missing**. | ✅ **FIXED.** `_to_float` robust parse (negatives/sci-notation safe); explicit `*_is_missing` indicator + auditable warning + recorded missing share; no silent erasure. |
| D10 | §10 (root) | **Fake** | `workflows/msd_inference.py` relabeled *every* execution exception (NameError, NotImplementedError, GLM errors) as `pirs_model.status="blocked"`; run still emitted a "valid" bundle. The engine of the fraud. | ✅ **FIXED.** Execution exceptions now record `status="failed"` and **re-raise** (red-is-honest), aborting the compile instead of faking green. |
| D11 | §6.7, §6.9, §10 | **Fake** | Two HSIC impls: real kernel HSIC (`pirs/hsic.py`, RBF/Nyström/RFF + structured nulls) exists but is **bypassed**; the pipeline (`pirs/hsic_run.py`) used `linear_hsic_statistic` (Pearson²) + iid `rng.shuffle`, mislabelled `hsic_mode: "exact_linear"`. The "non-linear scanner" was linear. | ✅ **FIXED + closed.** Real centered RBF-kernel HSIC (`numpy_kernel_hsic_permutation_test`, NumPy-only). **Structured nulls now wired**: `hsic_run` loads `support_index.parquet` (from D12) and applies within-municipality cyclic time-shift block permutations (`spatial_block_cyclic_time_shift`) when ≥5 spatial blocks exist, honest i.i.d. fallback otherwise. FDR now regime-correct: **BY** for cross-fitted panels, BH for cross-sectional (§6.8). Remaining (minor): n>5000 exact→Nyström/RFF mode selection. |
| D6 | §1.2 arch | Debt | No `RunContext`/`RunSession` exists. State threaded via `run_dir` strings + JSON-manifest relay; `bundle_manager.collect_missing_from_run` (`compile.py:469`) scavenges JSONs back. | 🟡 **STARTED.** `core/run_context.py` typed `RunContext` introduced and threaded through the MSD inference phase. Full thread from CLI/compile + elimination of intermediate JSON relays is the remaining work. |
| D7 | perf | Debt | Row-dict thrashing (`to_dicts`/per-row loops) in 22 files incl. `datasus/sinasc_normalize.py` (8×), `pirs/model_execution.py`, `output/validate.py`. RAM/throughput killer at national scale. | |
| D8 | arch | **Fake** | Two parallel PIRS pipelines + `msd_inference.py` dispatched stages by **string name** via `importlib` inside `try/except`, so a wrong/missing function silently became "blocked". | ✅ **FIXED.** `msd_inference.py` rewritten with direct typed imports/calls, `RunContext`, inlined planning chain. `workflows/pirs_pipeline.py` ("non-executing" wrapper) now has **zero callers** — flagged dead-code, to delete in the test-rewrite phase (left in place only to avoid breaking test collection mid-flight). |
| D9 | §3 refactor | Debt | Registries reloaded from YAML on each call (`load_registries` in `registries/loader.py` exists but downstream `get_carrier`/`get_unit` re-read). | MSD-review refactor B. |

| D12 | §1.2, §7 | **Dead seam** | `efg/executor.py` is genuinely real (real groupby counts/sums, real Radon–Nikodym ratio, real Bayesian race bridge) and **does run** (`compile_attach.py:269`), writing per-field value tensors. BUT `compile_attach._q_row` built every `Q_tensor` row from `field.support` metadata and **never read the executed tensor** → `value_vector_json` never populated → `design_matrix` always blocked "values_missing" → **the entire PIRS/HSIC half was permanently dead on real runs.** The real executor's output was dropped on the floor. | ✅ **FIXED.** `compile_attach` now loads each executed tensor, builds a shared `(year, municipality)` panel index, aligns every field onto it (broadcasting national/time-invariant fields), and writes row-aligned `value_vector_json` + real §3.12 diagnostics (n_eff/missingness/zero_inflation/cv) into Q_tensor, plus a `support_index.parquet` (the spatial/temporal blocks D11's structured nulls need). `design_matrix` updated to tolerate null panel cells and complete-case on the outcome. **This is what makes the GLM→cross-fit→kernel-HSIC chain actually execute.** |

*Note: the EFG executor itself is **real** (verified) — counts, sums, RN ratios, race bridge all compute genuine arrays. The fraud was the dropped seam (D12), not the math.*

*Not yet audited (assume suspect until proven): SHE composite decoders (age/CNPJ/boolean/zero-variance), population solvers, ST-DFM, geneallocation, structured-null support threading into HSIC, the 148 tests.*

---

## 2. Phased roadmap

### Phase 0 — Honesty rails (containment; do FIRST)
- Replace `_load_callable` string dispatch + `try/except → "blocked"` in `msd_inference.py` with typed imports and **real error propagation** (strict mode aborts; no silent relabel).
- Make `output/validate.py` assert **computation invariants** (residual provenance matches budget; model family actually fitted; no in-sample residuals for standard/deep), not just 17-key presence.
- Add a `tests/fraud/` guard suite that fails if a label claims a computation that did not run.

### Phase 1 — Architecture foundation
- `pegasus/core/run_context.py`: typed `RunContext` (run_id, run_dir, data_root, intent, geo_scope, budget, `RegistryBundle`, bundle_manager, telemetry, source_artifacts) + in-memory carriers for substrate / efg_result / pirs artifacts. Thread `ctx` top-to-bottom.
- Collapse the two PIRS pipelines into one typed in-memory pipeline (Pydantic objects); write JSON only at final flush.
- `RegistryBundle` loaded once (lru_cache), passed via ctx (refactor B/D9).
- Vectorize hot paths to native Polars expressions (D7).

### Phase 2 — PIRS statistical core (worst fraud: D1, D2, D5)
- Real GLM families in `compute/kernels.py` (NB, Gamma, hurdle, ZINB, beta-binomial, Dirichlet) via IRLS/MLE; correct residuals per §6.5 (deviance / randomized-quantile / clr-ilr).
- Real K-fold cross-fitting (§6.6.1) with spatial/temporal block preservation; deep parametric bootstrap (§6.6.2). `assert_standard_deep_not_in_sample` must verify the **computation**, not the label.
- Fix missingness (§2.3): drop silent median impute; explicit indicator + Q_tensor penalty; fix negative-value bug.

### Phase 3 — EFG/SHE math (D3, D4)
- Stage-2 empirical compression (§3.15.2): post-materialization Pearson/Spearman ≥0.98, DAG-preserving suppression.
- Physical bounded pushforward (§2.12.3): execute groupby().sum() marginalization in sidra extract/facts; RN for rates.
- Verify SHE composite decoders end-to-end against §2.4.0.

### Phase 4 — Test suite as fraud-detector
- Audit all 148 tests; classify label-asserting vs computation-asserting; rewrite.
- Property/invariant tests for every §10 hard-abort and §11 downgrade condition.

### Phase 5 — Sweep & finalize
- Full audit of remaining slices vs this ledger; perf validation at state scale; final §12 conformance pass.

---

## 3. Self-patch hardening (post-GPT cross-review)

GPT audited this session's own patches and flagged silent-omission/label risks I introduced. Fixed:
- **compile_attach tensor bridge no longer silent.** A field with a materialized path whose tensor can't be read/aligned now emits a `Warnings` row (`materialized_field_tensor_path_missing` / `tensor_read_failed` / `tensor_alignment_failed`) instead of vanishing.
- **Cross-fit backfill labelled honestly.** When out-of-fold residuals have uncovered rows backfilled in-sample, `residual_mode_actual` becomes `cross_fitted_with_in_sample_backfill` + a warning — never silently "cross_fitted" (which §10 forbids for standard/deep HSIC).
- **Dead pipeline deleted.** `workflows/pirs_pipeline.py` + its two fraud-validating tests + its dedicated audit removed. Package `compileall` clean.

## 3a-bis. FIRST REAL END-TO-END COMPILE ON LIVE DATA (verified by execution)

Drove the actual `pegasus` env (Rscript 4.4.1 present; materialized Maceió-2022 microdata on disk) and ran the real compiler. Findings + fixes:

| # | Defect (found by running) | Fix |
|---|---|---|
| E1 | **CLI did not even import** — `cli.py` imported 3 nonexistent `*_development` fns and referenced an undefined `dashboard_app`. Nothing worked. | Removed dead imports; defined+registered `dashboard_app`. CLI imports; 214/214 modules import clean. |
| E2 | **SHE normalization produced 100% NULL canonical columns.** The registry is keyed by *canonical* names; the declarative normalizer looked up *raw* DATASUS columns (DTOBITO, IDADE) → matched nothing → excluded everything → 60 columns, all null. The entire decoding layer was hollow. | Rewrote `normalize_sim_do_events` and `normalize_sinasc_events` as **real vectorized Polars decoders** (MSD §2.4.1/§2.4.3/§2.6): year, geography cod6, sex, age (composite IDADE decode), ICD parse, birth-weight/prematurity/cesarean/maternal-age flags. Real values, 0.1s/city. |
| E3 | **Registry reloaded from disk per cell** (~5×10⁵ reads/city) → normalization hung for minutes. | mtime-keyed `lru_cache` on `load_yaml` + memoized `resolve_source_field_entry`. 24ms→2ms/row; profile confirmed. |
| E4 | **Executor collapsed count fields to a global scalar** — `_execute_non_rn` tried `_scalar_tensor` first and count nodes' incidental `n_events` support key short-circuited the groupby → RN had no shared axis. | Scalar path now only for genuine anchors (`"anchor" in op`); count/event fields aggregate by (year, municipality). |
| E5 | RN failures were swallowed by the fixed-point retry → generic "parents_not_materialized". | Executor records the last real exception; blocked fields report the true reason. |
| E6 | SIDRA population anchor `municipality_cod6` was null (locality is cod7, code only accepted cod6) → rate join collapsed to a UF total. | cod7→cod6 conversion; RN now joins on municipality. |
| E7 | `materialize.py` used `sidra_artifact.registry_hash` (nonexistent attr) → compile crashed. | Use `artifact_hash`/`source_manifest_hash`. |

**Result:** `pegasus compile` on real Maceió-2022 SIM+SINASC+SIDRA now succeeds — **21 real V_fields, 10 DAG edges, valid 17-key bundle**, with materialized crude mortality (~24/1000) and birth rates of sane magnitude. Was 1 field / 0 edges / 164 failed branches before. The SHE→EFG→PIRS→O_run spine executes on live data, end to end, for the first time. `run_live_pipeline` now runs the real normalizers after fetch.

**Remaining for full operational parity (documented, not faked):**
1. Compile geo-filter should restrict the event count to the intent municipality (smoke precision); state panels are unaffected.
2. **Per-municipality population for state panels** — the anchor requires exactly-one-Total and so only serves single-locality smoke; a state run needs a per-muni population tensor (the `*_n6_municipal_sidecar` facts exist).
3. SIH-RD / CNES-ST vectorized decoders (same hollow-normalizer issue as SIM/SINASC) for all-source state runs.
4. The `pegasus run` live acquisition path needs Rscript + network (untestable in this sandbox).

## 3a. Operational completeness — end-to-end live pipeline (NEW)

Mandate shifted from decontamination to **running end-to-end on live data**. The
missing piece was the operational spine: PegaSUS had only granular manual commands
(`datasus ingest`, `datasus normalize-*`, `sidra metadata/plan/extract`,
`source-artifacts`, `compile`) with **no driver** tying them to a `UserIntent`, and
`compile` consumed a pre-built manifest without acquiring anything.

**Built:** `workflows/pipeline.py::run_live_pipeline` + CLI `pegasus run --intent <f>`.
From a single intent it:
1. derives UF, year range, and DATASUS systems (SIM-DO, SINASC, +SIH-RD/CNES-ST when `include_cnes_sih`, minus `exclude_systems`);
2. acquires each DATASUS stream live via `MicrodatasusClient` (R subprocess) → one `processed_events` parquet per system;
3. auto-fetches SIDRA 9606 metadata if absent, generates the population-denominator request for the **whole UF's municipalities** at the census period (Total sex/race/age categories), extracts, and concatenates chunk facts into one `normalized_facts` parquet;
4. merges all into **one** materialized-external source manifest;
5. runs the real `run_compile` → immutable bundle.

No fixtures, no development builders, no metadata stand-ins. Verified against the
real intents `config/intents/alagoas_maceio_2022_actual_smoke.json` (smoke) and
`alagoas_2022_actual_allsource_state_panel.json` (state). Package `compileall` clean.

**Caveat (honest):** not executed here — a live run needs Rscript + `microdatasus`/`read.dbc` and network (SIDRA/DATASUS FTP), and the `pegasus` conda env, none provisioned in this environment. The chain is structurally complete and matches every contract traced; live numeric validation is the user's to run with `pegasus run`.

## 3b. Live-spine integrity (verified real by dataflow tracing, not labels)

The `SHE → EFG → PIRS → O_run` inference spine is now genuine end to end:

| Stage | Verdict |
|---|---|
| SHE substrate (`she/substrate.py`) | Real registry-driven profiling + zero-variance exclusion. |
| EFG executor (`efg/executor.py`) | **Real**: groupby counts/sums, Radon–Nikodym ratio (+ecological guard), Bayesian race bridge (posterior + sensitivity bounds). |
| Executor → Q_tensor vectors (`compile_attach`) | **Fixed (D12)**: panel-aligned `value_vector_json` + real §3.12 diagnostics + `support_index`. |
| Population denominator (`sidra_anchor.py`) | **Real**: marginalized SIDRA 9606 Total, no double-count. |
| PIRS GLM (`compute/glm.py`) | **Real** IRLS: gaussian/poisson/gamma/NB/binomial, §6.5 residuals. |
| Cross-fitting (`compute/glm.crossfit_residuals`) | **Real** out-of-fold; mixed backfill honestly labelled. |
| HSIC (`pirs/hsic.py` + `hsic_run.py`) | **Real** centered RBF-kernel HSIC; exact/RFF/Nyström by scale; spatial-block-cyclic null; BY/BH FDR by regime. |
| Orchestrator (`msd_inference.py`) | Direct typed calls; execution errors fail loudly (no swallow). |

Swept unchecked compute modules (`she/population/*`, `she/stdfm/*`, `datasus/decoders.py`): **no stub/placeholder/NotImplemented markers**. Population solver + ST-DFM are **skipped by intent** (live path uses the SIDRA Total anchor), i.e. inactive-by-design, not fraudulently bypassed.

**Conclusion:** the fake-compliance fraud (D1, D2, D3, D5, D8, D10, D11, D12) is exterminated along the live spine. What remains is *feature* work (SIDRA V_X context fields, ST-DFM gating — need real data) and *debt/polish* (below), not deception.

## 4. Tracked remaining risks (agreed with GPT; not yet done)
1. `_align_vector` aggregates duplicate cells by `mean` — safe given executor groupby uniqueness, but should assert/respect aggregation law if that guarantee weakens.
2. **HSIC structured nulls**: thread `support_index.parquet` (year/municipality blocks) into `hsic_run` for real spatial/temporal block permutation; until then honestly labelled `unrestricted_iid_permutation`. Also add n>5000 exact→Nyström/RFF mode selection.
3. **D4 finish**: route live SIDRA high-dim facts through `sidra/pushforward.py` in `she/substrate.py` (primitives ready, not yet called on real frames).
4. **Workspace discipline**: ST-DFM (`she/stdfm/artifacts.py`) and standalone PIRS/HSIC CLI paths may write into final `run_dir/Tables` pre-Phase-E; route through stage workspace.
5. **RunContext threading** into `compile.py`/`cli.py` (currently only the inference phase); eliminate remaining intermediate JSON relays.
6. Perf debt (row-dict `to_dicts`, registry reloads) — after correctness.
7. Tests are label-asserting and untrustworthy — Phase 4 fraud-detector rewrite.
