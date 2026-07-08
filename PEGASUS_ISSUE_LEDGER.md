# PegaSUS Master Issue Ledger

Cross-cutting **status tracker** for every documented issue/feature/objective across all
modules and dev cycles. Orthogonal to [`DOCS.md`](DOCS.md) (which says *which doc governs
what*): this says *what state each issue is in*. Update the Status cell — with a commit hash —
whenever an item advances, so we always know what is done, in flight, deferred, or needs
revisiting. The LDO/EFG-inference layer keeps its own detailed ledger in
[`PEGASUS_LDO_COMPLETENESS_AUDIT.md`](PEGASUS_LDO_COMPLETENESS_AUDIT.md); this file governs
everything else and indexes that one.

## Status legend

- **OPEN** — documented, unbuilt, verified genuinely absent.
- **OPEN?** — reported open by inventory but NOT yet live-verified (treat as a lead, not a fact).
- **WIP** — in progress this cycle.
- **DONE `<commit>`** — built + verified + landed.
- **VERIFIED-DONE** — audited against live code this cycle and found already-satisfied (no build).
- **LIKELY-DONE `<commit>`** — inventory + commit trail indicate done; not re-verified this cycle.
- **DEFERRED `<why>`** — deliberately not now; reason required.
- **SUPERSEDED `<by>`** — a later analysis/decision overrode the documented prescription.
- **DEFENSIBLE** — flagged by a critic but audited as correct-as-is (record why, don't "fix").

## How this ledger was built (2026-07-08 inventory)

Six read-only agents inventoried the non-LDO authority docs in parallel; every promising
"open" was then cross-checked against live code. **Headline finding: the codebase is well
ahead of its own docs.** Most doc-listed findings are already DONE or DEFENSIBLE; the agents
repeatedly flagged as "open" things that live code already satisfies (they read stale strata —
`COMPLIANCE_AND_REMEDIATION.md` is a finding-ID dictionary, not a plan; the math/roadmap
critiques predate the WP1–WP8 + O1–O19 LDO build). Trust the **Status** column here, not the
source docs, and re-verify any **OPEN?** before building.

---

## ★ Genuinely-open priority set (verified or high-confidence)

| Rank | ID | Item | Module | Status | Notes |
|---|---|---|---|---|---|
| 1 | FEAT-P3 / ROAD-W12 | **Export / materialization layer** — user-controllable datasets carrying uncertainty/code_system/projection_status | output | OPEN | no such layer exists; the biggest real architectural gap |
| 2 | SIDRA-CTX-01 | **Context (ST-DFM) never routed to the EFG** — `compile.py` sets `stdfm=skipped` | sidra/efg | OPEN? | matches the `efg-domain-machinery-orphaned` seam; verify intent before wiring |
| 3 | FEAT-P4 | **Multi-denominator declaration** (per-query default-denominator override) | denominators | OPEN | `exposure_ref` exists; no per-query override mechanism |
| 4 | DIS-06 + ZIKA-ACCPT | **Disease variable-grammar → live compile** + autonomous Zika→microcephaly acceptance gate | disease/efg | OPEN? | variable generator built (task #9); "wired into live pipeline" unverified |
| 5 | POP-02 M6 (POPT-3/4/5) | Population build: numpy-native scatter, per-block build-solve-emit, parallel blocks | denominators | OPEN | perf/memory, not correctness; the tuple round-trip (POPT-1/2) is already fixed |
| 6 | RACE-01 | RaceBridge region-conditioning of the per-source confusion matrix | measurement | OPEN? | literature matrices seeded; region-conditioning reported open |
| 7 | STOR-05 | Delete DATASUS stdout/stderr/heartbeat ancillaries on success | datasus | VERIFIED-DONE | already handled by `subprocess._cleanup_chunk_ephemera` on success |
| 8 | PERF-02 | Bounded-radius BFS in migration `hop_distances` | denominators | VERIFIED-DONE + hardened | live national path threads bounded `max_hops`; hardened the direct-call fallback (result-identical, drops O(N²) footgun) |

**Deferred by policy:** causal installment (FEAT-P5 / CAUSAL-THEME15 — Meek/faithfulness/DiD),
LDO-W11 Moran/HSIC dedup (behavior-identity check first), all LDO perf/GPU (LDO-NUM-01,
LDO-DESIGN-01/02, GPU-01…08, KS-02 — GPU gated on a measured CPU baseline), full residual
cross-fit, continuous-discovery scheduler (DISCO-01 / FAL-02, §VI.4 future feature).

---

## §LDO / EFG inference — CLOSED (detail: `PEGASUS_LDO_COMPLETENESS_AUDIT.md`)

WP1–WP8 + the O1–O19 re-verification wave landed; the later level-confounding and
CERT+lag-honesty clusters closed too. Live-verified this cycle as already-satisfied:

| ID | Item | Status |
|---|---|---|
| MATH-12 / LDO-WHITEN-04 | symmetric-normalized whitening Laplacian `L_sym` | VERIFIED-DONE (`precision.py:61`) |
| MATH-05/06 / ROAD-W1 | reliability-W in the (whitened) covariance moment | VERIFIED-DONE (`covariance.py:92,217`) |
| MATH-24 / MAJ-01 | spatial BYM varying-coefficient field | DONE `3370e67` |
| MATH-25 / MAJ-02 | disease Laplacian quadratic `(γ/2)tr(SᵀL_DS)` | DONE `f774e28` |
| MATH-02 / LDO-MARG-02 | per-muni EB baseline rate | DONE `d9f5524` |
| MATH-13 / LDO-ADMM-06 | ADMM dual-residual stopping | SUPERSEDED `empirically refuted — primal+dual → recall 0.0, reverted` |
| KS-01 | SLQ joint-logdet dormant at national S | DEFENSIBLE `sparse-Cholesky exact+cheap at S≈5570; SLQ is the S>20000 contingency` |
| KS-02 | `sparse_spd_logdet` uses splu not Cholesky | DEFERRED `correctness-neutral; 2× flops negligible at S≈5570` |
| LDO-CAUSAL-* | LiNGAM-pool / Meek / faithfulness / DiD | DEFERRED `P5 installment` |
| LDO-W11 | Moran + HSIC-kernel call-site dedup | DEFERRED `behavior-identity check required` |
| LDO-XFIT | full fold residual cross-fit | DEFERRED `marginal beyond n≥10·p gate` |

Remaining math-critique items flagged PARTIAL by the inventory (MATH-04 holdout re-gaussianize
leak; MATH-11 temporal pre-whitening vs telemetry-only; MATH-07 t-copula tail dependence;
MATH-08/23 exposure-uncertainty transport `σ_logE`) are **OPEN? — unverified leads**, low
priority, to be triaged against the LDO audit before any build.

---

## §Denominators / population tensor

Reconstruction code lives in `src/pegasus/denominators/reconstruction/`. GPU is gated last —
the verified bottleneck was a Python `tuple(float(...))` round-trip, since fixed.

| ID | Item | Status |
|---|---|---|
| POPT-1/2 | kill tuple round-trip; numpy-native storage | VERIFIED-DONE (`schema.py` `_STORE_DTYPE=np.float32`, numpy `__post_init__`) |
| POPT-7 / FAL-POP-PROJ | coverage beyond 2021–2022 anchors (projection) | LIKELY-DONE `a34cdc6` (task #18) |
| FAL-POP-SV | single-vintage census-anchored closure | LIKELY-DONE (task #16; 6579 kept as recency anchor by design) |
| FAL-POP-AMC | municipality boundary-change harmonization | LIKELY-DONE `dcee9f1`,`b6e8d08` |
| FAL-POP-RECON | census undeclared-race reconciliation | LIKELY-DONE `ae04ff5` |
| POP-02 M6 / POPT-3 | numpy-native container construction (drop `[None]*n_cells`, `_cell_index` loops) | OPEN |
| POPT-4 | parallelize the locality-separable blocked solve (ProcessPool) | OPEN |
| POPT-5 | per-block build-solve-emit (input-prep peak O(block)) | OPEN |
| POPT-6 | document age×sex/race×age joints as IPF/independence reconstructions | OPEN (small) |
| PERF-02 | bounded-radius BFS in `migration.hop_distances` | VERIFIED-DONE + fallback hardened (result-identical) |
| POPT-8 / POP-02 GPU | torch port of `loss.py`+SPG (f32-bulk/f64-reduction) | DEFERRED `gated on measured CPU baseline` |
| FEAT-P4 | per-query multi-denominator declaration | OPEN |
| ARCH-CTR-01 | unify `she/population` + `denominators/reconstruction` | OPEN? |

## §Measurement / race

| ID | Item | Status |
|---|---|---|
| RACE-01..07 | RaceBridge redesign (per-source C, literature prior, never identity, uncertainty) | LIKELY-DONE (tasks #4,#20) except… |
| RACE-01-REGION | …region-conditioning of the confusion matrix | OPEN? |
| EFG-DECL-02 | race-declaration gate when race axis absent | OPEN? |

## §Disease semantic axis

| ID | Item | Status |
|---|---|---|
| DIS-01/02/03 | concept registry, ICD/CID adapter, DiseaseGraph | LIKELY-DONE |
| DIS-04 | L_D prior in LDO precision | DONE `f774e28` (quadratic) + adaptive-ℓ1 |
| DIS-05 | shared-code overlap accounting (`mechanical_overlap`) | LIKELY-DONE (enforcement gate unverified) |
| DIS-06 | semantic-expansion variable-grammar wired into live compile | OPEN? |
| DIS-07 | build-time label embeddings (Qwen3-0.6B cached asset) | OPEN (large feature) |
| ZIKA-ACCPT | autonomous Zika→microcephaly acceptance test | OPEN? |

## §SIDRA / context

| ID | Item | Status |
|---|---|---|
| SIDRA-CTX-01 | route ST-DFM context into the EFG (`compile.py` `stdfm=skipped`) | OPEN? (priority-2) |
| SIDRA-CTX-02 | enforce high-cardinality axis bound / bounded pushforward on the legality path | OPEN? |

## §Output / schema

| ID | Item | Status |
|---|---|---|
| FEAT-P3 / ROAD-W12 | export / materialization layer | OPEN (priority-1) |
| OUT-02 | `output_schema.yaml` scaffold doesn't drive `validate.py` (real contract in `output/schemas.py`) | OPEN? |
| OUT-03 | PROFILE_NONEMPTY + empty_by_profile rule in `validate.py` | LIKELY-DONE |

## §Compute / GPU / storage

| ID | Item | Status |
|---|---|---|
| STOR-01/02/07 | raw.rds decoupled; microdatasus audit-only; manifest tensor reference-only | DONE `61746bb` + v4 bridge |
| STORE-02 | lazy `scan_parquet` views (retire re-materialization) | OPEN? (agent self-conflicted; verify) |
| STOR-03 | processed.parquet ZSTD vs SNAPPY | OPEN? |
| STOR-05 | delete stdout/stderr/heartbeat ancillaries on success | VERIFIED-DONE (`_cleanup_chunk_ephemera`) |
| STOR-06 | SIDRA cache/facts duplication | OPEN? |
| GPU-01..08, LDO-NUM-01, LDO-DESIGN-01/02 | LDO perf: batched/truncated eigh, warm-start, whitening reuse, GPU HSIC | DEFERRED `GPU gated; perf not correctness` |
| DISCO-01 / FAL-02 | continuous-discovery scheduler + incremental update | OPEN (§VI.4 future) |

## §Modularity / architecture (structure)  — inventory 2026-07-08 (agent B, file:line evidence)

**W11 consolidation cluster (task #53) — WIP this cycle.** Retire duplicated logic to single
sources of truth; the safe subset first, the numeric ones (Moran/HSIC) with behavior-identity
verification.

| ID | Item | Status |
|---|---|---|
| MOD-02 | `_clean`/`_digits`/`_stable_hash`/`_read_table` reimplemented ~8× (6 files) → `core.text`+`core.io` | WIP |
| WF-11 | `_load_intent` reimplemented 4× (pipeline/compile/race_bridge/stage_plan) → `load_user_intent` | WIP |
| WF-01 | population-mode→solver-mode mapper duplicated byte-identical | WIP |
| WF-02 | name collision `_race_bridge_prior_artifact` (producer vs finder, opposite logic) | WIP |
| LDO-B08 | Laplacian `L=D−W` hand-rolled 4× (disease_prior/lowrank/geo) → shared helper (`normalized=` flag) | WIP |
| MOD-01/EFG-03 / MATH-22 | Moran's I: q_tensor's 1-D `_moran_contiguity` vs compile_attach adjacency | **DEFENSIBLE** — verified: the LIVE path uses geography-aware `_moran_i_adjacency`; the 1-D proxy is only q_tensor's documented no-graph fallback (emits `moran_i_ordering_contiguity_proxy`). n_eff also already unified (shared `_kish_effective_n`+`_moran_corrected_n_eff`). |
| LDO-B03/B04 | HSIC kernel "fork" | **DEFENSIBLE / FALSE lead** — `residual_scan` imports hsic's PUBLIC API (`build_hsic_representation`/`hsic_pair_stat_and_null`/`hsic_mode_for_n`/`_gpu`); no duplicate kernel, clean boundary (matches O9 intentional split). W11 cluster (task #53) fully dissolved: HSIC false, Moran defensible, core.text false. |
| **Data-integrity (silent-loss — matches full-data mandate):** | | |
| T1.2 | DATASUS per-chunk fail-closed completeness gate | DONE `56f0bdc` — `blocked`→hard-fail (broken bridge); `timeout`/`failed`→explicit PARTIAL-COVERAGE (not silent); opt-in `require_complete` threaded to national |
| T1.3 | national race-prior `= None` before UF fan-out (`pipeline.py:475`) | **DEFENSIBLE-BY-DESIGN** — verified: comment "UF-independent; wired later"; `compile.py:524` re-resolves via `race_bridge_plan`. Study-adjacent (race-stratified national) → off-limits scope; re-verify when that path is exercised. |
| **★ EFG-QT — canonical §3.12 Q-state now wired (DONE `11fce28`):** | | |
| EFG-QT / DIRECT-QT-01 | `_q_row` now derives the Q-tensor `state` from the canonical `classify_q_state` on the COMPUTED diagnostics (was a zero-caller orphan; state was the ad-hoc materialize-time `field.state`). Surfaced + fixed a latent denom_fragility bug: bare COUNT fields (n_denom None) were quarantined by a `1.0` default — now count-aware (0.0 for counts, 1.0 only for a rate missing its denominator), via shared `default_denom_fragility` used by both `_q_row` and `compute_q_state`. | DONE `11fce28` — 502 tests pass; probe confirms count→verified, broken-rate→quarantined |
| EFG-QT-residual | materialize-time `field.state`/`dashboard_safe` are still set by ad-hoc literals (incl. a non-enum `"warning"`); these drive `QuarantinedFields`/dashboard gating (a declaration-soundness axis, distinct from the now-correct Q-tensor data-reliability verdict) | OPEN (lower priority — tighten the loosely-typed state model + reconcile declaration vs data-reliability gating) |
| **Megazord decomps (behavior-preserving splits):** | | |
| WF-07 | `compile.py` 1036-LOC god-module | OPEN |
| EFG-08 | `executor/kernels.py` 838 LOC, stringly-typed dispatch | OPEN (partial split done) |
| EFG-09 | `compile_attach.py` 674 LOC (orchestration+Q-recompute+serializers+attach) | OPEN |
| EFG-07 | `FieldNode.support/axes` untyped dict, 136 alias-probes | OPEN? |
| MOD-03 | layering inversion: `geo/migration_affinity` imports `denominators` | OPEN? |
| WF-08/09 | untyped compile→pipeline handoff; stage predicates derived twice | OPEN? |
| T1.6 / STORE-02 | `datasus_combined` re-materialized per run (combined_hash folds fetched_at) | OPEN? |
| T1.8 / POP-02 M2 | float32 denominator solver never built (peak RAM ~2× floor) | OPEN |
| REG-07-LOADER | registries still 21 files; Stack A/B duplication + codegen cruft (decode-path rewrite) | OPEN (task #28) |
| MOD-HELD | §1b straggler dead-code (WF-03..06/EFG-05/WF-10 likely-dead CLI/test-only) | OPEN? (task #27) |
| ARCH-REG-02 | declarative `source_routing.yaml` (routing hardcoded in normalizers) | OPEN? |
| SCOPE-01 / ARCH-PROFILE-01 | decouple DataScope × ExecutionStage axes in UserIntent | OPEN? |
| PANEL-01 | CommonPanel wired (op-plan §1c "pre-builds") | OPEN? |
