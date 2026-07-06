# PegaSUS Refactor Master Plan

Governing refactor plan consolidating 9 parallel package analyses of `src/pegasus`
(271 files, 47.6k LOC). Cross-referenced against `PEGASUS_MSD_III.md` (§II–§V, Part III)
and `PEGASUS_OPERATIONAL_IMPLEMENTATION_PLAN.md` (REG-07, PANEL-01, SCALE-01, LDO-06,
NAT-01, STORE-02, RACE-01, POP-02). Every deletion is verified against `src/`, `tests/`,
`scripts/`, config, and dynamic-dispatch paths. Every conflict with the planned
architecture is flagged inline as **[PLAN-CONFLICT]** or **[PLAN-ALIGNED]**.

Ordering principle: reclaim first (zero-risk deletes), then behavior-preserving
decompositions gated by existing tests, then consolidation, then new builds. Nothing
that changes an MSD-III contract or public API happens without a test gate.

> **Verification stamp (post-synthesis, independent re-check).** The three load-bearing
> claims were re-verified against source after the workflow, before this plan was adopted:
> (1) `run_investigate` orphaning — `grep disease_graph|variable_meta|exposure`
> in `workflows/investigate.py` is empty; the live call is `run_ldo(panel, K, lambda1,
> lambda2, keep_variables, **ldo_kwargs)` (`investigate.py:67`) with none of the three
> threaded. **CONFIRMED.** (2) Q-tensor Kish gap — live `compile_attach.py:223` takes a
> plain `n_obs: int`; `q_tensor.py:214` feeds `_kish_effective_n(weights)` first (§3.12.3).
> **CONFIRMED.** (3) PIRS slice-zoo dead — `model_execution`/`hsic_run` appear only as
> descriptive `executor="…"` strings (`stage_plan.py:212,222`); the *live* `population_cube.build`
> and `stdfm.torch_solver` executors occupy the same field (`:192,:202`), proving the field is
> serialized metadata, never resolved to a callable. **CONFIRMED.**

---

## EXECUTION STATUS (executed 2026-07-05; baseline 377 → 345 tests passing, green at every step)

Executed test-gated, incrementally committed (`5abdcb0..HEAD`), each step independently re-verified
(git state + import smoke + full suite) before commit. Net **−4,881 LOC** in `src/`.

- **§1 Dead code — DONE.** Phase A (`49a58f1`): 14 modules, 471 LOC. Phase B: PIRS slice-zoo
  (`881141c`, 15 modules ~4,577 LOC) + EFG promotion half (`cd98941`, ~680 LOC). **Deviations
  (evidence-driven, safer than plan):** KEPT `families/spatial/schemas/nystrom/rff` (live via T0-4
  contract + test_hsic_foundation — plan said delete); KEPT `materialization_manifest` (build_efg
  imports it live — plan's "delete" was WRONG). §1b stragglers HELD (task #27).
- **§2 Megazords — ALL 5 DONE.** executor (`8a02112`), build (`ff75a03`), pipeline 1322→521
  (`5f3807b`), compile god-fn→44 LOC (`20940c1`), dag (`60971db`). Behavior-preserving; STORE-02,
  `_cell_index` centralization, the redundant-bridge-pass removal, and constant-dedup all deferred
  (behavior-changing, not code motion). `validate.py` correctly untouched.
- **§3 Tree reorg — §3c DONE, REG-07 flagged.** LDO-06 `pirs/ldo`→`ldo/` (`ff8bbf4`); data plane
  →`denominators/` (`b985a9f`); `sources/` + `measurement/` planes (`791c678`). **REG-07 deliberately
  NOT rushed:** the registries/ "two loader stacks" are actually 3 distinct decode-path loader
  CONTRACTS (generic/semantic/loader) + 15 legitimate payload accessors; unifying them is a
  return-type-contract rewrite with silent-corruption risk, not code motion → task #28 (dedicated,
  decode-validated).
- **§4 Arch gaps — the two conformance fixes DONE.** §4.1 run_investigate disease wiring
  (`5224b69`, revert-proof test — DIS-04 prior + §III.8 overlap guard now LIVE; caught two plan traps:
  variable_grammar is orphaned/real vars are σ_C counts, and a code-keyed graph is a silent no-op →
  added `DiseaseGraph.from_variable_code_sets`). §4.2 Q-tensor Kish n_eff (`bd6aead`, pinned). The
  greenfield builds (§4.3 Zika, §4.4 MR-01, §4.5 GPU, §4.6 randnla, §4.7 APC/STORE-02, §4.8 causal,
  §4.9 RACE-01, §4.1 exposure) DEFERRED per user decision (task #29) — new numerical/feature work
  needing real validation, not an autonomous refactor tail.

---

## 0. TL;DR — the four things that matter

1. **~1,340 LOC of dead code deletes today with zero breakage** (registry wrappers, dead
   infra shims, off-path modules). Another ~7,600 LOC (PIRS slice-zoo + promotion plumbing)
   deletes *with its pinning tests* as a coordinated cut.
2. **Two live spec-conformance bugs** hide behind dead-but-correct code: the live Q-tensor
   `n_eff` (`compile_attach.py:223`) drops the Kish weighting that the dead `q_tensor.py`
   implements per MSD-I §3.12.3; and `run_investigate` silently drops `disease_graph`,
   `variable_meta`, and `exposure` — orphaning four built-and-tested capabilities.
3. **Four genuine megazords** (`pipeline.py` 1322, `build.py` 1618, `dag.py` 1026,
   `executor.py` 1251) decompose cleanly along seams that already exist; `compile.py` (839)
   and `validate.py` (614, already well-factored — do NOT touch) round out the "large file" list.
4. **The registries/ scatter is a third dead already** — 10 of 31 files are codegen cruft.
   REG-07 collapses the rest; the deletable third can go before REG-07 starts.

---

## 1. CONFIRMED DEAD CODE — safe immediate wins

### 1a. Delete NOW — zero importers, zero tests, zero dynamic dispatch, zero config, zero plan conflict

Verified: grepped `src/ tests/ scripts/ config/` for static imports, dotted-path strings,
`importlib`/`getattr` dispatch. The only `importlib` dispatch in the tree
(`pipeline.py:227-230`) resolves the hardwired `_NORMALIZERS` dict; the `executor=` strings
in `stage_plan.py:192-222` are serialized into a manifest and **never resolved to callables**.

| Module | LOC | Verification |
|---|---|---|
| `registries/sidra.py` | 43 | 1 occurrence (self); test hit is a `config/` path, not an import |
| `registries/composite_decoders.py` | 34 | self only; primary YAML `composite_decoder_registry.yaml` absent (dead+broken) |
| `registries/hsic.py` | 34 | self only (distinct from live `pirs/hsic.py`) |
| `registries/output.py` | 34 | self only; `output_registry.yaml` absent (dead+broken) |
| `registries/race_axis.py` | 34 | self only |
| `registries/residuals.py` | 34 | self only |
| `registries/icd.py` | 27 | self only; live ICD backend is `disease/` (simple-icd-10), never imports this |
| `registries/manifest.py` | 20 | self only |
| `core/run_context.py` | 64 | grep `RunContext\|run_context` → self only. **[PLAN-CONFLICT]** plan line 88 lists `core/run_context` as `[KEEP+EXT]`, but it is a scaffold written and never wired; its docstring claims to replace the `run_dir`-string threading that `pipeline.py:227-230` still does. RECONCILE: either wire it or delete it — do not leave a KEEP-tagged 0-caller file. Recommend delete; re-introduce when actually adopted. |
| `source_artifacts/resolver.py` | 51 | dup of live `compile_policy.py`+`contracts.py`; not in package `__init__`; only ref is codegen script string literal |
| `storage/dataset.py` | 5 | pure re-export of `storage.parquet.scan_table`; not in `storage/__init__`; 0 importers |
| `she/stdfm/regime.py` | 23 | `stdfm_gate_for_sidra_context` never called; live path calls `sidra.regime.classify_sidra_context_regime` directly |
| `sidra/stitching.py` | 116 | **[PLAN-CONFLICT]** plan line 99 marks `[MOVED]`→`sources/sidra/` for PANEL-01. 0 importers today (grep hits are the `sidra_stitching.yaml` filename). HOLD — see §1c. |

**Subtotal deletable-now (excluding the two HOLD/RECONCILE rows `stitching.py` and, per your call, `run_context.py`): ~370 LOC across 11 files.** Including `run_context.py`+`resolver.py`: ~485 LOC.

### 1b. Delete WITH their pinning tests — dead in production, one test each

These fail a test if deleted silently; delete module **and** its test together.

| Module | LOC | Pinning test to drop |
|---|---|---|
| `registries/models.py` | 34 | `tests/unit/test_slice28x_generic_registry_wrappers.py` |
| `registries/nulls.py` | 34 | (same test — guards both; test guards only codegen REG-07 deletes anyway) |
| `output/pirs_bundle.py` | 33 | `tests/unit/test_pirs_foundation.py` |
| `output/sinasc_efg_bundle.py` | — | its sole importer test |
| `output/maternal_child_compile_attach.py` | 34 | **[CONFLICT IN ANALYSES]** — arch-gaps + verify-deadcode say it has a *live contract test* (`test_maternal_child_state_attach_contract.py`) and is the maternal-child compile boundary; workflows-output says compile.py dropped its import. VERDICT: keep as low-cost fossil; do NOT delete without confirming the contract test is retired. |
| `output/sidra_denominator_anchor.py` | ~250 | SUSPECTED-dead; superseded by FAL-POP-SV two-layer denominator. 3 `test_sidra_*` tests. Verify against FAL-POP-SV before cutting. |
| `dashboard/hsic_readonly.py` | 26 | audit-script-only; delete as a pair with `pirs/hsic_ranking.py` (§1d) |

### 1c. HOLD — dead today but PLAN-reserved forward infrastructure (do NOT delete)

Add a one-line `# <TICKET> target, unwired` marker to each so the next dead-code sweep
doesn't reap them. Per MSD-III §V.1c ("no aspirational flag over unwired code"), the marker
must name the ticket, not claim the code is active.

| Module | LOC | Plan reservation |
|---|---|---|
| `storage/duckdb.py` | 20 | SCALE-01, plan lines 497/667 — out-of-core streaming substrate |
| `storage/arrow.py` | 16 | SCALE-01, plan line 667 — cited alongside duckdb |
| `sidra/stitching.py` | 116 | PANEL-01, plan line 99/289 |
| `sidra/projection.py` | 123 | PANEL-01 (`cube_lifecycle.py`); note live `she/sidra_projection.py` is a *different* module |
| `sidra/pushforward.py` | 127 | PANEL-01; consumed only by `she/high_dimensional.py` (itself orphaned) |
| `sidra/category_maps.py` | 39 | PANEL-01; sibling of pushforward/projection cluster |
| `she/high_dimensional.py` | 114 | plan line 100/176 KEEP; the coherent orphaned cluster PANEL-01 will wire |
| `she/cnes_capacity.py` | 62 | plan line 176 KEEP — **[PLAN-CONFLICT]** its EFG bundle consumer no longer exists; dead-by-consequence, not plan-intended. Flag for reconciliation. |
| `she/sih_costs.py` | 61 | plan line 176 KEEP — same caveat; `test_slice28zc` already asserts it is unreferenced |

**~415 LOC of SIDRA transform modules + ~240 LOC she/ look dead but are PANEL-01 pre-builds.**
A naive dead-code sweep would wrongly reap them. This is the single most important
"do-not-delete" finding.

### 1d. The PIRS slice-zoo — dead in production, delete as one coordinated cut with tests

The live inference path is `compile.py:790` → (only when `execution_stage=="investigate"`) →
`run_investigate` → `run_ldo` (`pirs/ldo/*`), which reuses only `pirs/{hsic,nulls,fdr}.py`.
The slice-zoo is a closed self-referential island; its only inbound live edges are the two
inert `executor=` strings in `stage_plan.py:212,222`.

**Delete these modules (~4,930 LOC / 20 files):**
`model_execution.py`(817), `design_matrix.py`(669), `hsic_run.py`(657), `design_plan.py`(431),
`selection_plan.py`(412), `design_readiness.py`(372), `hsic_report.py`(358), `hsic_ranking.py`(353),
`run_candidates.py`(265), `schemas.py`(131), `families.py`(83), `spatial.py`(68), `design.py`(67),
`diagnostics.py`(25), `residuals.py`(19), `crossfit.py`(49), `field_selection.py`(50),
`nystrom.py`(38, superseded — `hsic.py` inlines Nyström), `rff.py`(33, superseded — `hsic.py`
inlines RFF), and `output/pirs_bundle.py`(33).

**Coordinated actions required (or the cut red-lights ~10 tests):**
1. Delete legacy slice tests: `tests/unit/test_slice16b/16c/17a/17b/18a/18c_*`,
   `tests/integration/test_slice16b/16d_*`, `test_pirs_foundation.py`, `test_pirs_temporal_lag.py`.
2. Re-home the two live assertions in `tests/contract/test_inference_baseline.py:25,28`
   (`families.family_for_outcome`, `spatial.select_spatial_effect_mode`) — move those two
   small functions into `ldo/margins.py`/`geo/spatial_graph.py`, or drop the assertions.
3. Strip the dead `executor=`/`expected_artifacts` strings for `pirs_model`/`pirs_hsic` in
   `stage_plan.py:206-224` (stage-plan machinery STAYS; only the two dangling refs go).

**[PLAN-ALIGNED]** This executes LDO-06 exactly (plan lines 173/555). The plan says
`hsic/nulls/fdr` **MOVE→`ldo/residual_scan.py`** (not delete) — that is a follow-up (§3d),
not part of this cut. KEEP `pirs/hsic.py`(558), `pirs/nulls.py`(165), `pirs/fdr.py`(47) and
the entire `pirs/ldo/` package (2,088 LOC — all live).

### 1e. The EFG promotion plumbing — legacy Slice-14/15, off-compile-path

`promotion_plan.py`(403), `promotion_apply.py`(362), `materialization_manifest.py`(209) +
their 3 `construct/efg_{materialize,promotion,apply}.py` wrappers + the `@efg_app.command`
handlers in `cli.py:566-690`. Reachable only via standalone `pegasus efg …` CLI verbs, never
from compile. `promotion_apply.q_tensor_row` hardcodes zero-tensor quarantined placeholders.
**[PLAN-ALIGNED]** plan lines 158/185 want these DEL/MERGED. SUSPECTED-dead (user-facing CLI
surface) — verify no operational runbook invokes the verbs, then delete modules + wrappers +
CLI handlers + `test_slice14b_*`/`test_slice15a/b/c_*`.

### 1f. Bytecode hygiene
Prune stale `.pyc` for already-deleted sources: `datasus/__pycache__/{normalize,sinasc_normalize,sih_normalize,cnes_normalize,declarative_normalize}.*.pyc`, `she/reconstruction/__pycache__/torch_kernels.*.pyc`. Delete `scratch_deadcode_report.html` from repo root (left by an analysis pass).

**Total reclaimable:** ~485 LOC immediate (1a) + ~4,930 (slice-zoo) + ~974 (promotion) +
misc = **~6,600–7,600 LOC** once the coordinated test cuts land.

---

## 2. MEGAZORD DECOMPOSITIONS — behavior-preserving, ordered by ascending risk

All are mechanical (no behavior change) and guardrailed by existing tests. **Do NOT decompose
`output/validate.py`(614)** — it is already a ~20-line thin orchestrator over 20 single-responsibility
`_validate_*` helpers; its LOC is inherent contract surface. Negative finding; leave it.

### 2a. `efg/executor.py` (1251) — LOWEST RISK (split by concern, all tests cover)

Three separable jobs already delimited by line ranges:

```
efg/executor/
  support.py    # l.114-410:  _support_frame, _with_year, _with_geo, _geo_scope,
                #             _apply_restrict_conditions, _add_icd_stratum
  kernels.py    # l.413-1071: _count_tensor, _functional_tensor, _sum_tensor,
                #             _compute_rn_ratio, _compute_bridge_tensor,
                #             _compute_race_bridge_tensor, 4× _sidra_*_tensor,
                #             _population_solver_tensor
  run.py        # l.1074-1251: _execute_non_rn, execute_efg_result, _execute_efg_result_impl
  __init__.py   # re-export execute_efg_result (preserve public symbol)
```

Additionally: replace the string-matched `elif op == …` dispatch chain
(`_execute_efg_result_impl` l.1159-1183) with an `operator→kernel` registry dict, parallel to
`operators.OPERATOR_REGISTRY`. Adding a materialized operator becomes a table entry, not a new `elif`.

### 2b. `efg/dag.py` (1026) — LOW RISK (extract the RN loop; kill the redundant second pass)

`dag.py` legitimately straddles tiers (holds pure `_build_efg_base` algebra + defines the shared
`EFGResult` dataclass). Keep `EFGResult` and the expansion spine in `dag.py`. Two targeted changes:
- Extract the registry-driven operator-expansion grammar (COUNT_MEASURE → σ_C strata →
  demographic strata → Ψ functionals → RN ratios → divergences → bridges) into
  `efg/expansion.py` if a clean seam exists; otherwise leave in place (it is a coherent grammar).
- **Kill the redundant ratio-construction pass.** `_build_efg_base` builds RN ratios twice:
  (1) the authoritative registry loop over `clinical_ratio_specs` (l.762-790), and (2) a second
  heuristic pass over `plan_bridge_candidates(...).candidates` (l.858-899) whose string-blob role
  matching (`core_seed._seed_role` on "death"/"birth"/"population") duplicates what the carrier
  registry already encodes. The duplicate edges are silently deduped by `_dedupe_edges`, masking
  the redundancy. FIX: gate `bridges.expand()` to only bridges the registry loop cannot express
  (capacity/cost); keep `bridges.py` as summary-only (`bridge_plan_summary`).

### 2c. `sidra/population_cube/build.py` (1618) — MEDIUM RISK (pure extraction, 8 test files cover)

The orchestrator `solve_population_tensor_from_sidra_strata` (l.974-1498, ~525 LOC) + 24 helpers.
Helpers communicate through explicit args + index dicts (no shared mutable state) → low-risk extraction.

```
population_cube/
  strata.py             # _pairs, _category_by_classification, _canonical_stratum,
                        #   _read_population_strata, _census_2000_records_from_facts
                        #   (+ AXES, AXIS_CLASSIFICATIONS constants)
  indexing.py           # _cell_index, _birth_cell_index, _census_count_arrays
                        #   ** CRITICAL: centralize the offset math `age*x*r + sex*r + race`
                        #      currently hand-inlined at build.py:147, 559, 1090, 1366-1372 —
                        #      four byte-consistency-required copies = latent divergence bug. **
  priors.py             # _resolve_geo_year_columns, _stratify_sex/_age_column,
                        #   _coalesce_race_columns, _bridge_race_stratified_counts,
                        #   _sim_death_priors, _sinasc_birth_priors, _census_race_composition_prior
  layer1.py             # _interpolate_shares, interpolate_census_composition (KEEP PUBLIC NAME —
                        #   referenced by she/reconstruction/projected_gradient.py:125 + test:188)
  closure.py            # _sidra_vital_totals, _datasus_event_totals,
                        #   _migration_residual_totals, _reanchor_closure_single_vintage (FAL-POP-SV)
  projection_envelope.py# _classify_projection_years + PROJECTION_* constants (FAL-POP-PROJ)
  flows.py              # _reconstruct_and_persist_migration_flows
  build.py (stays)      # PopulationTensorBuild, both solve_* orchestrators, emit loop (~500 LOC)
```

Load-bearing seam: keep the `del records; gc.collect()` (l.1267-1269) in the orchestrator, after
the last consumer (`interpolate_census_composition`) and before the solve — it is a deliberate
national-scale memory release. **[PLAN-ALIGNED]** anticipate the eventual `denominators/population/`
target (plan line 176) — do not deepen the `sidra/` nesting; name modules so the later move is clean.

### 2d. `workflows/pipeline.py` (1322) — MEDIUM RISK (~830 LOC is inlined SIDRA acquisition)

```
workflows/acquire/
  sidra_population.py  # _acquire_sidra_population(+_strata), _acquire_sidra_census_2000_strata,
                       #   _acquire_sidra_civil_registry_vital, _population_period_plan,
                       #   _all_census_periods, _sidra_population_localities, SIDRA_* consts (l.56-735)
  sidra_compendium.py  # _compendium_*, _plan_sidra_compendium_from_metadata,
                       #   _acquire_one_compendium_table, _acquire_sidra_compendium_context (l.738-927)
  sidra_metadata.py    # _ensure_sidra_metadata* (l.391-438)
  national.py          # _resolve_ufs, _acquire_sidra_for_uf, _acquire_national,
                       #   _combine_national_artifacts (l.1072-1185)  [NAT-01, plan line 678]
pipeline.py (stays)    # run_live_pipeline, plan_live_pipeline, _load_intent, _resolve_*,
                       #   _acquire/_normalize/_combine_datasus, _race_bridge_prior_artifact (~400 LOC)
```

**[PLAN-CONFLICT to avoid]** STORE-02 (plan line 686) wants `_combine_processed_datasus_chunks`
(l.238) rewritten to lazy `scan_parquet(glob)→sink_parquet` with the `datasus_combined/`
intermediate DROPPED. Do NOT faithfully relocate the eager `read_parquet→concat→write_parquet`
code — land STORE-02 in the same pass. Also delete the superseded
`workflows/acquire/ingest_sidra.py`(57) whose logic pipeline.py already inlines (redirect or drop
`test_sidra_extract.py`).

### 2e. `workflows/compile.py::_run_compile_impl` (l.408-820, ~410 LOC) — MEDIUM RISK

Linear pipeline of `telemetry.stage(...)` blocks with side-effect-isolated seams:

```
_resolve_compile_plan(...)              # l.416-520 (incl. race-bridge prior validation 480-520)
_hash_and_manifest_sources(...)         # l.522-556
_materialize_population_and_substrate() # l.558-601 (population_solver + she_build + efg_build)
_derive_stage_status(...)               # l.603-655 (pure, no I/O — high test value)
_serialize_run_bundle(...)              # l.657-778 — FOLD the duplicated extras-dict
                                        #   construction (722-749 vs 751-778 build near-identical
                                        #   manifest_extras/final_extras) into one helper
_flush_and_investigate(...)             # l.780-820 (+ the run_investigate wiring fix from §4.1)
```
`_run_compile_impl` becomes a ~40-line orchestrator. `_build_population_tensor_artifact`
(l.269-405) is already well-factored — leave it.

**Risk ordering to execute in:** 2a → 2b → 2c → 2e → 2d (2d last because STORE-02 couples in).

---

## 3. REGISTRY + DIRECTORY-TREE CONSOLIDATION

### 3a. The registries/ scatter (31 files, empty `__init__.py`, two parallel loader stacks)

- **Stack A** — `generic.py` (`RegistryEntry` dataclass, tuple returns): imported by
  `composite_decoders, events, functional, hsic, manifest, models, nulls, output, race_axis, residuals, sidra`.
- **Stack B** — `semantic.py`+`loader.py` (dict-based, `active_entries`): imported by
  `bridge, cnes_capacity, diagnostic_topology, icd, sih_cost`.
- The loader is **pure-YAML** (`loader.py:29` `root.rglob("*.yaml")`) — it never imports the `.py`
  wrappers. 10 of 31 files (32%) are codegen cruft from the slice28x episode.

**Step 1 (pre-REG-07, zero-risk):** delete the 8 hard-orphan wrappers from §1a +
`models.py`/`nulls.py` from §1b. Removes 10 files / ~330 LOC and the entire "orphan alias" class.

**Step 2 (REG-07 proper, plan lines 242-270):** collapse Stack A+B into one `kind`-tagged
schema at `registry/{schema,callables,validator,loader}.py`, catalog under `/config/registry/`.
`callables.py` (name→fn resolver) is already the §II.1-mandated single resolver — KEEP.
`validators.py`(224) is what REG-07 step 4 replaces. Migrate the substantive payloads
(`source_fields.py`+`provenance.py`+`quality.py`, `carrier.py`, `unit.py`, `aggregation.py`,
`population.py`, `race_bridge.py`, `demographic_axis.py`, `cnes_capacity.py`, `sih_cost.py`,
`diagnostic_topology.py`, `events.py`, `bridge.py`, `functional.py`) as catalog entries.
**[PLAN-CONFLICT]** the plan budgets for "~40 fragmented registries"; the real count is 31 with a
third already deletable — migration volume is smaller than budgeted.

**Fix the validators divergence NOW:** `validators.py:197-201` pins registry YAMLs via a
**hardcoded literal list**, bypassing the `REGISTRY_FILES` tuple in `registries/sidra.py:9`. Have
`validators.py` import `REGISTRY_FILES` instead — kills the edit-the-wrong-list trap. (If deleting
`registries/sidra.py` per §1a, relocate `REGISTRY_FILES` to the canonical registry module first.)

### 3b. The target directory tree (the plan's own §2/§3 destination — do NOT invent a competing scheme)

Four planes: **data** (sources/denominators/panel) · **inference** (ldo/interaction/causal/field) ·
**orchestration** (workflows/registry/compute/storage) · **output**. Contract-bearing internals
(`efg/legality`, `efg/executor`, `compute/glm`, `she/stdfm/torch_solver`, `acceptance/contracts`)
relocate WHOLE, never rewritten. Already-landed plan packages: `disease/`, `geo/spatial_graph.py`
(SPG-01), `causal/{orient,quasi}.py` (CAUSAL-01), `workflows/{acquire,construct,report}/`.

### 3c. Current → proposed move map (highlights; behind existing phase gates)

| Current | Proposed | Ticket |
|---|---|---|
| `sidra/population_cube/*` | `denominators/population/*` | (after §2c decomposition) |
| `sidra/{regime,projection,pushforward,stitching}.py` | `sources/sidra/*` | PANEL-01 |
| (new) `panel/cube_lifecycle.py` | orchestrates regime→project→pushforward→panel | PANEL-01 |
| `she/sidra_context.py`, `she/sidra_projection.py` | `sources/sidra/*` | plan line 176 |
| `she/reconstruction/*` | `denominators/*` | plan (MOVE, not delete) |
| `registries/*` (22 live) | `registry/{schema,callables,validator,loader}.py` + `/config/registry/` | REG-07 |
| `pirs/ldo/*` (18 files) | top-level `ldo/*` (promote) | LDO-06 |
| `pirs/{hsic,nulls,fdr}.py` | `ldo/residual_scan.py` (merge) | LDO-06 |
| `efg/race_bridge.py` | `measurement/race.py` (REWRITE, strip identity-default) | RACE-01 |

### 3d. The hsic/nulls/fdr MOVE (LDO-06 follow-up to §1d)
After the slice-zoo delete, `pirs/{hsic,nulls,fdr}.py` have a single live consumer
(`ldo/residual_scan.py`). MOVE them into `ldo/` per plan line 173/555. Mechanical.

---

## 4. ARCHITECTURE GAPS + OPTIMIZATIONS — ranked by leverage

### 4.1 [HIGHEST LEVERAGE — small effort] Wire orphaned capabilities into `run_investigate`
**VERIFIED:** `grep disease_graph|variable_meta|exposure=` in `workflows/investigate.py` → **empty**.
`run_investigate` calls `run_ldo(panel, ...)` but never threads `disease_graph`, `variable_meta`,
or `exposure`. This orphans FOUR built-and-tested capabilities on the real path:
- **DIS-04 disease L_D prior** (`pirs/ldo/disease_prior.py`) — never reaches a real run.
- **Mechanical-overlap Jaccard guard** (§III.8, `type_mechanical_overlap`) — gated on
  `variable_meta`, so it is inert on real data.
- **Disease-variable grammar** (`disease/variable_grammar.py`, §5.1) — referenced only by itself.
- **EFG-OUT-01 exposure offsets** (`efg/measured_quantity.py`) — count+exposure margin runs
  without offsets.
This is the `efg-domain-machinery-orphaned` seam from memory. Threading these through is the
single highest-value, lowest-effort fix. Do it as part of the §2e `_flush_and_investigate` extraction.

### 4.2 [HIGH — spec-conformance bug] Consolidate the Q-tensor, adopt Kish n_eff
**VERIFIED:** live `_moran_corrected_n_eff` (`compile_attach.py:223`) uses `n·(1−I)/(1+I)` with a
plain observation count and **no Kish weighting**; the dead-but-spec-correct `q_tensor.py:170,214`
uses `KishEffectiveN(weights)·1/(1+max(0,I))` per MSD-I §3.12.3. The dead module is *more correct*
than the live one. **[PLAN-ALIGNED]** SPG-01 (plan line 273) wants `q_tensor.py` rewired; the
structural-contiguity default landed in the live `compile_attach` path but `q_tensor.py` was never
reconnected. FIX: make `q_tensor.compute_q_state` the single source of truth, route
`compile_attach._q_row` through it, delete the duplicated `_vector_diagnostics`/`_moran_*`/
`_temporal_roughness`/`_spatial_entropy`/`_moran_corrected_n_eff` from `compile_attach.py` (~130 LOC).
Phase-4 LDO-01/FIELD-01 both plan to consume `q_tensor` for reliability weights — this unblocks them.
Until fixed, `q_tensor.py`'s tests give false confidence the spec formula is live when it is not.

### 4.3 [HIGH — the done-ness gate] Write the Zika/microcephaly acceptance test (§XI.4)
**VERIFIED:** `grep zika|microcephaly` across `tests/` → empty. The program's definition of done
does not exist. There is a synthetic `test_ldo_lag_recovery.py` (proof-of-life) but no real
monthly-Alagoas, no-forced-selector regression. Writing it surfaces which wired-together pieces
actually recover the real lagged edge. Depends on 4.1 landing first.

### 4.4 [HIGH — plan's designated Phase-3 build] MR-01 multiresolution shrinkage + real `field/`
**UNBUILT.** `grep multiresolution_decompose|shrinkage_prior|GMRF` → nothing. `pirs/ldo/resolution.py`
is coarse→fine *variable restriction*, not the "national+regional+state+municipal deviations shrunk
toward parents" GMRF §III.3 specifies. The `field/` package (`LatentField`, `assemble_field`,
weighted coarse→fine observation operator) does not exist; `pirs/ldo/assemble.py` scatters
same-resolution rows. Everything about sparse-variable / small-area behavior depends on this.

### 4.5 [HIGH — un-gated, ~15× measured] POP-02 remainder: GPU port + blocked build + parallel blocks
- `she/reconstruction/torch_kernels.py` was deleted; no torch path exists. Per §V.1, with the M7
  tuple hotspot fixed the dense loss/gradient algebra dominates — GPU port is justified, VRAM-trivial.
- `solve_population_tensor_blocked` (`solvers.py:134`) loops locality-blocks sequentially though
  `_locality_separable` proves independence — trivial process-pool parallelism.
- M6: `build.py` still has ~107 `for` loops + per-cell `.append`/`.to_dicts()`; the 5-race national
  ~209M-cell build exceeds 32 GB until the whole build-solve-emit is blocked per locality. Concrete
  blocker for national full-history reconstruction (FAL-POP-VAL reconstruction prong).

### 4.6 [MEDIUM — national-scale validity] `compute/randnla.py` + matrix-free LDO precision
**UNBUILT.** No `randomized_svd`/`hutchinson_logdet`/`jl_sketch`. LDO precision is CPU-dense:
`precision.py:50-55` builds a dense S×S via Python double-loop (5,571² ≈ 248 MB before whitening;
`eigh` infeasible nationally); `fit_contemporaneous_precision` uses dense `sklearn.graphical_lasso`
+ O(kept²) Python edge loop; `lowrank.py` ADMM does `np.linalg.eigh` every iteration (O(p³)/iter).
None of §V.2–V.4 Kronecker-separable/matrix-free machinery is present. Caps `S` at ~state scale.

### 4.7 [MEDIUM — standing debt] Half-built APC + STORE-02
- APC (`compute/controller.py`) `adaptive_precision_run` operates on a hand-supplied `list[Quantity]`
  with scalar `exact_cost` — correct in miniature but wired to NO real LDO dial (SVD rank, Hutchinson
  probes, CG iters, resolution depth). Wire it to the engine's actual knobs.
- STORE-02 lazy combine (folds into §2d).

### 4.8 [BUILT-BUT-ORPHANED] Causal Rung-1 auto-orientation not wired
`causal/` (269 LOC, LiNGAM/collider orientation) has zero non-test importers; `run_ldo` never calls
it; `certify.py:5-7` leaves contemporaneous edges undirected. MSD-III §IV Rung-1 is
implemented-but-orphaned. Wire into the live readout (smaller than 4.4, larger than 4.1).

### 4.9 [BUILT-BUT-PARTIAL] EFG-OUT-01 measured-quantity is sidecar, not terminal
`measured_quantity` is emitted as an RN sidecar (`executor.py:705`), not yet as the terminal object
replacing the rate. Compile still materializes rates as primary. Matches plan status; finish RACE-01
(`efg/race_bridge.py`→`measurement/race.py`, strip identity-default) in the same measurement pass.

---

## 5. SEQUENCING

### Phase A — Reclaim (do first; no gate needed beyond the existing green suite)
1. Delete §1a hard-orphans (11 files, ~370 LOC). Fix the `validators.py`→`REGISTRY_FILES` import
   in the same commit.
2. Reconcile `core/run_context.py` + `she/{cnes_capacity,sih_costs}.py` with the plan (they are
   KEEP-tagged but 0-caller) — decide wire-or-delete; do not leave the divergence unflagged.
3. Add `# <TICKET> target, unwired` markers to §1c HOLD modules.
4. Prune §1f bytecode + `scratch_deadcode_report.html`.

### Phase B — Coordinated dead-cuts (gate: full suite green before AND after each cut)
5. PIRS slice-zoo (§1d) — modules + pinning tests + `stage_plan.py` string strip, ONE commit.
6. EFG promotion plumbing (§1e) — verify no runbook uses the CLI verbs, then modules + wrappers +
   CLI handlers + slice14/15 tests.
7. §1b test-paired deletes (verify each pinning test individually).

### Phase C — Behavior-preserving decompositions (gate: existing tests are the guardrail; each is one PR)
8. `executor.py` (§2a) → `efg/executor/{support,kernels,run}.py` + operator-registry dispatch.
9. `dag.py` (§2b) — kill the redundant bridge pass; extract expansion if seam is clean.
10. `population_cube/build.py` (§2c) — centralize `_cell_index` FIRST (fixes the 4-way inline
    divergence), then extract the 7 modules.
11. `compile.py` (§2e) — extract the 6 helpers; fold the duplicated extras dict; **land the §4.1
    run_investigate wiring in `_flush_and_investigate`**.
12. `pipeline.py` (§2d) — extract SIDRA acquisition + land STORE-02 lazy combine + delete
    `ingest_sidra.py`.

### Phase D — Consolidation (gate: REG-07 needs its own migration test per source)
13. REG-07 (§3a step 2) — one typed loader, `/config/registry/` catalog, repoint `she/source_registry.py` + `cli.py`.
14. LDO-06 MOVE (§3d) — `pirs/{hsic,nulls,fdr}`→`ldo/`; promote `pirs/ldo/`→top-level `ldo/`.
15. Directory moves (§3c) behind phase gates — WHOLE relocation, never rewrite.

### Phase E — New builds (ranked §4; gate: each behind a new capability test)
16. §4.1 wiring (already landed in step 11) → 17. §4.2 Q-tensor consolidation →
18. §4.3 Zika acceptance test → 19. §4.4 MR-01+field/ → 20. §4.5 POP-02 → 21. §4.6 randnla →
22. §4.7 APC/STORE-02 → 23. §4.8 causal wiring → 24. §4.9/RACE-01 measurement pass.

### MUST NOT CHANGE (contracts / public API — relocate whole, never rewrite)
- The EFG **legality-algebra tier** (`legality, operators, declaration, align, equivalence, bridges,
  core_seed, node, lineage, failed_branch`) — verified no-polars, no-upward-import; MSD-III §II.3 /
  plan line 75 "the EFG's true value. Keep." Preserve the algebra/execution boundary in §2a/2b.
- Public symbols: `execute_efg_result`, `interpolate_census_composition`,
  `solve_population_tensor_from_sidra_{strata,anchor}`, `run_ldo`, `run_investigate`,
  `build_efg`, `validate_output_bundle`, `acceptance.contracts.*`.
- Contract-bearing internals relocate whole: `efg/executor`, `compute/glm`,
  `she/stdfm/torch_solver`, `acceptance/contracts`, `pirs/ldo/*` estimation math.
- The MSD-III §III LDO pipeline (copula margins, sparse+low-rank, lagged edges, disease prior,
  residual HSIC audit, stability selection, certification gate) — conforms today; do not regress.
- The FAL-POP population-tensor line (SV/PROJ/VER/RECON/AMC, census-exact) — solid; do not re-litigate.

### Two plan-vs-code divergences to reconcile explicitly (not silently absorb)
- `core/run_context.py`: plan KEEP+EXT vs 0 callers + docstring contradicted by live
  `pipeline.py:227-230`. Decide.
- `she/{cnes_capacity,sih_costs}.py`: plan KEEP vs their EFG-bundle consumer no longer existing
  (dead-by-consequence). `test_slice28zc` already asserts `sih_costs` is unreferenced.
