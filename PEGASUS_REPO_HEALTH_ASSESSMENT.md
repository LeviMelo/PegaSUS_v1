# PegaSUS Repo-Health Assessment (2026-07-08)

Deep architectural + repo-health review by five parallel read-only audits (registries, dead/
orphaned code, wiring/integration, performance, architectural coherence), cross-reconciled against
each other and the "completed" refactor tasks. Tracked in `PEGASUS_ISSUE_LEDGER.md`.

## Verdict

**Healthy and coherent; the remaining work is bounded and known.** Architecture cleanly mirrors
MSD-III (correct layering, one centralized output contract, one measured-quantity type, versioned
foundational assets). Dead code is ~2% and mostly marked forward-scaffolds. The core
acquire→normalize→SHE→EFG→denominators→LDO→output path is wired end-to-end. Performance is a good
baseline with no algorithmic bloat. **No blocker to a reduced-statewide live test.** The real work
is: health-registry typing/de-orphaning, a few typed-contract handoffs, one truthfulness fix, two
byte-safe perf wins, and the (gated) god-module + REG-07 refactors.

## Per-dimension summary (reconciled)

- **Architecture — SOUND.** Layering correct (SHE→EFG→LDO, no cycles); `output/schemas.py` is the
  single output contract; `measured_quantity` enforces the denominator principle. One benign
  layering note (`geo/migration_affinity`→`denominators` public type). God-modules are real debt.
- **Dead code — MINIMAL (~2%).** Only one genuinely-dead unmarked module: `efg/empirical_compression.py`
  (187 LOC). Six inert modules are marked PANEL-01/SCALE-01 scaffolds (keep). **Reconciliation:** the
  ~10 registry codegen files the architecture agent cited are already GONE (Phase A / task #22 was
  real) — the plan's §1a list was pre-deletion. **But** the wiring audit caught what the per-module
  dead scan missed: the whole `pirs/` package is orphaned (below).
- **Wiring — 90% solid.** Core path wired; STDFM/race-bridge/population conditionally wired via regime
  classification; multi-denominator + coverage-audit wiring complete. Gaps: untyped `dict[str,Any]`
  handoffs (`domain_summaries` EFG→compile; `Q_tensor` rows executor→investigate with silent 0.0
  fallback) and one truthfulness flag.
- **Registries — FRAGMENTED (health domain).** Orphaned/shadowed/stub/loose-typed data — the user's
  specific concern, confirmed. (Audit went deep on `health/`; other domains to sweep in remediation.)
- **Performance — GOOD baseline.** No bloat; prior campaign held; no GPU-gating issue. Two byte-safe
  structural wins remain.

## Prioritized remediation (tiered)

**Tier 1 — pre-live-test integrity (small, do first):**
1. `maternal_child_linkage` is hardcoded `True` in RunConfig (`compile.py:812`) regardless of whether
   SINASC/linkage ran → make it conditional on the actual run scope (truthfulness; §"report faithfully").
2. Delete `efg/empirical_compression.py` (187 LOC, zero callers) — **DONE** (520 tests still collect).
3. `pirs/` (354 LOC) is test-only-superseded: zero production callers, but 2 test files import it
   (`test_inference_baseline.py` — a contract guardrail — + `test_hsic_foundation.py`, which imports
   `pirs.nystrom/rff` where it should test the live `ldo/hsic` — a smell). **DEFERRED**: deleting the
   package needs those tests migrated to the LDO equivalents first (don't break the T0-4 guardrail).

**Tier 2 — health-registry typing/de-orphaning (the user's ask):**
4. `icd_curated_groups.yaml` (35 cause groups) orphaned — `icd_groups.py` hardcodes chapters/blocks
   → wire the registry as the source, or delete if the hardcode is canonical (decide + document).
5. `cnes_capacity` + `sih_cost` registries shadowed by hardcoded enums in their loaders (two sources of
   truth) → make the loader build from the registry, or delete the YAML + document. Pick one authority.
6. `icd_catalog.yaml` / `icd_quality_groups.yaml` 7-chapter "Macro-Slice 27A" stubs, only validator-
   referenced → reconcile against the disease package's full CID-10 adapter; clarify stub-vs-legacy.
7. Type `diagnostic_topology` + `clinical_event_definitions` (add dataclasses; retire loose `.get()`).
8. Sweep the non-health registry domains (datasus/demographic/ontology/fields/inference/spatial/sidra)
   for the same classes of issue.

**Tier 3 — typed contracts (de-risk live inference):**
9. `Q_tensor` rows executor→investigate: silent 0.0 fallback on missing n_eff/denom_fragility masks
   incomplete materialization → typed contract / column-presence guard (matters for LDO reliability weights).
10. `domain_summaries` EFG→compile: typed contract instead of `dict[str,Any].get()`.

**Tier 4 — byte-safe perf wins:**
11. Vectorize the nested lag-loop in `covariance.py` whitened-correlation (~15–25% LDO; byte-identity gate).
12. Batch-align fields in `panel.py` compile (~10–20%; byte-identity gate).

**Tier 5 — gated refactors (low-risk when started, tests cover):**
13. God-module decomposition: `compile.py` (1036), `efg/dag.py` (1026), `efg/executor.py` (1251),
    `kernels.py` (838) — mechanical extraction per REFACTOR-MASTER-PLAN §2 (Phase C reorganized the
    package but left large cores).
14. REG-07 registry consolidation (two decode stacks → one) — **deferred** (decode-validation is a
    silent-corruption risk if rushed; keep the deliberate hold).

## Deep-dive redesign synthesis (2026-07-09)

Four improvement-oriented deep-dives (RaceBridge, data-plane, registries, per-module), consolidated
into four workstreams. (Both perf/registry agents carried a classifier-unavailable flag → verify
line claims before acting; the structural conclusions are cross-checked.)

### A. RaceBridge — staged measurement-model upgrade (centerpiece)
- **Wired: YES** — both the embedded population-solver path (SIM-death / SINASC-birth race bridged
  into the denominator) and the autonomous EFG `Bridge_R` field are live. The wiring fear was unfounded.
- **Core gap: the confusion matrix `C` is an identity/synthetic placeholder.** The Bayes crosswalk
  (`W ∝ C·π_local`) + bootstrap are correct, but with identity `C` it quantifies only *sampling*
  variability — it does NOT correct the real admin↔self-declared reclassification bias. Plus:
  national-only `C` (`region_scope` infra exists, unpopulated); `local-π` heuristic instead of
  principled shrinkage; and **the bridge's computed uncertainty (CV/credible bounds) is dropped, not
  propagated into rates/LDO**.
- **Workstream:**
  - **W-RACE-1 (code-now, no new data):** propagate the bridge posterior covariance into the rate +
    LDO measurement-error term (biggest honest-uncertainty win, currently dropped); complete the
    region-conditioned prior-selection path; replace `local-π` with a census-anchored shrinkage prior.
  - **W-RACE-2 (data-acquisition):** a real region-conditioned `C_s` from published Brazilian
    misclassification studies / PNS-PNAD / record-linkage — the RACE-01-REGION deferral.
  - **W-RACE-3 (eventual):** full hierarchical latent-class Bayesian model (EM+Laplace → MCMC) if
    sensitivity warrants + linkage acquired.
  - **First step:** W-RACE-1 uncertainty propagation.

### B. Data plane — solid, micro-optimize (not a rewrite)
- The stream redesign held (batched normalizers, 8-way fetch). NOT an LDO-level headache anymore.
- **Workstream:** parallel normalize (decouple from the hardcoded 2-way cap → 3–4-way, +25–40%
  wall-clock, byte-safe) + byte-safe quick wins (1s R-poll→backoff; skip redundant warm-cache SHA256;
  SIDRA metadata content-hash cache). Out-of-core (DuckDB) only at full 2000–2024 scale.

### C. Registries — ~70% to "add data = registry edit"; operationalize transforms
- Strongly-typed + centralized loader/entry contract; record-level normalize is generic.
  **Correction:** there is NO `declarative_normalize.py` — the declarative engine is
  `records.py::normalize_record` + `callables.py::resolve_callable`.
- **Blocker to registry-only add:** vectorized batch transforms + categorical codebooks + SIDRA
  extraction policies are still per-system code.
- **Workstream:** W-REG-1 (10–15 d) operationalize vectorized transforms declaratively (a
  `vectorized_transform` op-spec in `source_fields.yaml` + an `apply_vectorized_transform` engine;
  SINASC pilot) → new DATASUS system ≈ 20-min registry edit for the common case. W-REG-2 (~4 wk)
  declarative categorical codebooks + SIDRA extraction policies. W-REG-3 health-registry typing/de-
  orphan (Tier-2 above) + REG-07 consolidation (deferred). **Feasibility: ~90% registry-only
  achievable (~6 wk); 100% impossible (computations are code) — ceiling is "registry declaration +
  minimalist code extension."**

### D. Per-module improvements — robustness/perf backlog
- Top safe wins: structured logging for silent exception-fallbacks (EFG `run.py` — honesty gap);
  mtime cache-invalidation for ICD/concept lru_caches; cache the spatial-graph Laplacian view; bundle
  first-class consistency checks; multi-reason field-exclusion records; fixed-point no-progress exit.
- **Rejected:** a silent denominator *fallback* in the query engine — contradicts no-silent-degrade
  (typed refusal is correct).

### Consolidated priority (highest-leverage first)
1. **W-RACE-1** — propagate bridge uncertainty downstream (honesty + your #1 concern; code-only).
2. **Data-plane byte-safe quick wins + parallel normalize** (the first live-test bottleneck).
3. **Per-module safe robustness fixes** (silent-exception logging, cache invalidation, spatial-view
   cache, bundle consistency).
4. **W-REG-1** — operationalize vectorized transforms (the "add data = registry edit" enabler).
5. **Health-registry typing/de-orphaning (Tier-2)**.
6. Tier-1 remainder (maternal_child flag) + Output Query Layer P3d/P3e.
7. Staged/deferred: W-RACE-2/3, W-REG-2/3, god-module decomposition, REG-07.

## Live-test go / no-go

**GO for reduced-statewide.** Recommended staging: `core_vital` scope, `validate` stage first
(exercises acquire→normalize→SHE→EFG→population denominator — all architecturally sound), then the
`investigate` (LDO) stage — the LDO is built + wired + verified this cycle. **Pre-flight:** land Tier-1
(the truthfulness flag + the two safe deletions) and ideally Tier-3 #9 (Q-tensor contract) so a live
run's provenance is honest and its reliability weights aren't silently defaulted.
