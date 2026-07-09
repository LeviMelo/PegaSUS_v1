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

## Live-test go / no-go

**GO for reduced-statewide.** Recommended staging: `core_vital` scope, `validate` stage first
(exercises acquire→normalize→SHE→EFG→population denominator — all architecturally sound), then the
`investigate` (LDO) stage — the LDO is built + wired + verified this cycle. **Pre-flight:** land Tier-1
(the truthfulness flag + the two safe deletions) and ideally Tier-3 #9 (Q-tensor contract) so a live
run's provenance is honest and its reliability weights aren't silently defaulted.
