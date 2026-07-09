# PegaSUS Output Query Layer — design spec (FEAT-P3 + FEAT-P4)

Status: **design (2026-07-08)** · closes into MSD-III §VIII (output contract) · tracked in
`PEGASUS_ISSUE_LEDGER.md`. Implements the two coupled roadmap items:

- **FEAT-P3** — a first-class, user-invokable **export/materialization layer**: ask a run bundle
  for a *derived, provenance-carrying dataset* (a rate, an age-standardized rate, a certified-edge
  table, a raw field), in a chosen format — not just the raw first-class tables.
- **FEAT-P4** — **multi-denominator declaration**: a quantity may admit several denominators
  (resident population, live births, at-risk population, facility count); the query resolves one
  (declared default, or per-query override), and the choice is recorded in provenance.

They are one subsystem: you cannot export a *rate* without resolving its *denominator*.

> **Revision (2026-07-08, after user steer) — SCOPE PRUNED.** The query layer does **not** recompute
> rates or standardized rates. Recomputing a rate here would be a *second* rate definition that can
> drift from the EFG's canonical RN-operator rate (the two-generators anti-pattern) and would drag
> denominator-CI math (Byar/Fay-Feuer/fragility-widening) into a user-facing reader for little
> epidemiological gain. Refocused: **(a)** a `rate` is a **read of the EFG's already-materialized RN
> rate field** — never a computation; **(b)** the layer's real value is a **self-describing hypotheses
> export** — the edge table enriched with each variable's label/carrier/unit/ICD-context from the EFG
> `VariableDictionary` (the richer bridge from PegaSUS's orchestration to the user's own analysis;
> **built + tested**); **(c)** multi-denominator (FEAT-P4) is a compile-time *declaration* of which
> rate fields to materialize + a query-time *selection* among them, never a query-time recompute.
> Age-standardization stays the existing `age_standardization` engine (compile-time or user-side
> utility). This keeps the fragile LDO/denominator math untouched by the export path. §2/§8 updated.

## 1. Why this is not already covered

The bundle (`OUTPUT_BUNDLE_FILES`, 17 keys) materializes the **raw** typed outputs: counts as EFG
fields, rates as RN-operator fields (numerator÷denominator *parents*), the population tensor, and
`Hypotheses.parquet` (typed LinkRecords). Three gaps:

1. **No query surface.** A consumer wanting "C25 ASR by UF-year with 95% CI" must hand-join the
   materialized field tensors + the population tensor + call `age_standardization` by hand. That
   join/standardize logic is exactly what a study script re-implements — it belongs in the engine.
2. **Age-standardization is orphaned from the output.** `measurement.age_standardization`
   (`standardize_grouped`, Fay-Feuer CI) is complete but library-only; a directly-standardized rate
   is not a *requestable* output, so every study re-wires it (and can re-wire it wrong).
3. **Denominators are single-valued.** An RN field's denominator is `parent_ids[1]` — exactly one.
   A quantity that is legitimately normalizable several ways (deaths per resident-population vs per
   at-risk-population; births per women-15-49 vs per resident-population) cannot express that, so
   the choice is baked at compile time instead of being a query-time, provenance-recorded decision.

## 2. Object model

A **QueryableQuantity** is anything the bundle can serve, typed by what it needs:

| kind | source in bundle | denominator? | uncertainty |
|---|---|---|---|
| `raw_field` | materialized field tensor (V_fields) | — | field Q-state (n_eff, fragility) |
| `count` | extensive EFG field (numerator) | — | Poisson √count |
| `rate` | count ÷ resolved denominator | **required** (FEAT-P4) | count Poisson ⊕ denominator state-tensor fragility → gamma/Byar CI |
| `standardized_rate` | age-specific counts ÷ age-specific person-time | **required** + a reference population | Fay-Feuer gamma CI (existing engine) |
| `edge` | `Hypotheses.parquet` LinkRecords | — | carried `uncertainty`, `stability`, `fdr_qvalue` |

`raw_field`/`count`/`edge` read what already exists. `rate`/`standardized_rate` are *computed* by
the layer from count + a resolved denominator (never pre-materialized at every denominator choice).

## 3. Multi-denominator declaration + resolution (FEAT-P4)

**Declaration** — a new registry `config/registries/health/denominators.yaml`. Each entry:
```yaml
<quantity_key>:                       # e.g. "mortality_all_cause", "incidence_c25"
  admissible:
    - id: resident_population         # a denominator quantity key (population tensor field / RN denom)
      default: true                   # exactly one default per quantity (validator-enforced)
      strata: [age, sex, race]        # the axes on which this denominator is defined
      offset_semantics: log_exposure  # matches the LDO count-with-exposure margin
    - id: women_15_49                 # e.g. for fertility rates
      strata: [age, race]
    - id: at_risk_population          # e.g. a screening-eligible subpopulation
      strata: [age, sex]
  extensive: true                     # O7 gate: a rate is only legal for an extensive numerator
```
Provenance-typed like every registry (`legality_class`); a `context_derived` denominator is
guarded exactly as the disease/spatial graphs are.

**Resolution** at query time, deterministic + recorded:
1. `QuerySpec.denominator` given → use it if `admissible`; else typed error (never silently
   fall back — §V never-silently-degrade).
2. else the `default: true` entry.
3. The resolved denominator id, its strata, `offset_semantics`, and its own Q-state fragility are
   written into the dataset's **provenance manifest** (below). Two runs that pick different
   denominators are distinguishable from the manifest alone.
4. **Strata compatibility**: the numerator's strata must be a subset of (or aggregable to) the
   denominator's strata; otherwise the ecological-fallacy guard (already in the RN kernel) aggregates
   the numerator up to the denominator support and records `ecological_aggregation` in provenance.

## 4. Uncertainty + provenance propagation (never present a number as exact)

Every exported **rate** cell carries a CI, not a bare point:
- **Crude rate** `r = Y/N`: Byar's approximation gamma CI on the Poisson count `Y` (well-behaved for
  the small counts of rare causes), divided by exposure `N`. When `N` is itself reconstructed (the
  population tensor is a two-layer estimate), widen by the denominator's state-tensor fragility —
  the `denom_fragility` already computed in the RN kernel — combined in quadrature on the log-rate
  scale. This reuses the state tensor the EFG already produces; it does not invent uncertainty.
- **Standardized rate**: the existing Fay-Feuer gamma CI (`directly_standardized_rate`), unchanged.
- **Edge**: pass through `uncertainty`, `stability`, `fdr_qvalue`, `certification_status` verbatim.

The **provenance manifest** (one JSON sidecar per exported dataset) records, per the §VIII coverage
contract: the run id + bundle hash, the QuerySpec, the resolved denominator (+ its fragility), the
reference population (for standardized), the `code_system`/`projection_status` of the numerator
concept, any `ecological_aggregation`, the `n_eff`/missingness of the numerator field, and the
`sparsity_of_truth`/unsearched note if the quantity was pruned. Nothing is exported without it.

## 5. API surface

New package `src/pegasus/output/query/`:
- `spec.py` — `QuerySpec` (pydantic): `quantity`, `kind`, `denominator: str | None`, `strata:
  list[str]`, `standardize: str | None` (reference-population name), `format: {"parquet","csv"}`,
  `per: int = 100_000`, `filters: dict` (year/UF/cause selectors).
- `denominators.py` — load + validate `denominators.yaml`; `resolve_denominator(quantity, spec)`.
- `engine.py` — `materialize_query(bundle_dir, spec) -> MaterializedDataset` (frame +
  provenance dict). Pure reads over the bundle + the population tensor; calls the RN join logic for
  rates and `standardize_grouped` for standardized rates. No mutation of first-class tables.
- `export.py` — `write_dataset(dataset, out_dir)`: the frame (parquet/csv) + `<name>.provenance.json`.
- Workflow entrypoint `workflows/report/export_dataset.py::run_export(intent_or_bundle, spec)` and a
  CLI verb `pegasus export`.

`MaterializedDataset = {frame: pl.DataFrame, provenance: dict, kind: str, warnings: list[str]}`.

## 6. Enforcement + safety (reuse existing guards)

- **Denominator principle (O7).** A `rate`/`standardized_rate` on a non-extensive numerator is
  illegal (`extensive: true` gate) — mirror the existing extensive/intensive routing.
- **No silent degrade.** Missing denominator strata, an unresolvable denominator, or a
  reference-population mismatch → typed error + a recorded coverage gap, never a partial silent rate.
- **Additivity / multi-label (disease axis).** Summing counts across `multi-label` disease concepts
  for a rate numerator is illegal (`disease_concept_multilabel_nonadditive`) — reuse the aggregation
  registry gate; a σ_C partition numerator is required for an exact cause-specific rate.
- **Legality class.** A `context_derived` denominator cannot normalize a quantity tested on the same
  data partition (circularity) — reuse the §II.4.1 guard.

## 7. What it does NOT change

Additive only. The inference core (EFG, LDO, population tensor) is untouched; the layer is a
*reader* over the emitted bundle + population asset. This keeps it outside the compute envelope of a
run and lets it operate on any already-materialized bundle (including a versioned asset slice).

## 8. Phased implementation

1. **P3a — denominators registry + resolver** — **DONE** (`denominators.yaml` + `denominators.py`,
   one-default rule, admissible-override-or-refuse, registered in the validator tree; 8 tests).
2. **P3b — edge query + export vertical slice** — **DONE** (`materialize_query` serves `edge` with
   filters + a run-identity provenance manifest; `write_dataset` → parquet/csv + `<name>.provenance.json`).
3. **P3c — edge enrichment** — **DONE** (join `VariableDictionary` metadata — name/carrier/unit/
   ICD-group/topology — onto each edge's source+target → self-describing hypotheses; graceful
   degradation + warning when the dictionary is absent; provenance records `enriched`).
4. **P3d — `materialized_field` read** — expose any EFG-computed field by id (counts, the canonical
   RN **rate** fields, population denominators) as a pure read + its Q-state provenance; no recompute.
   Test: a materialized rate field round-trips with its Q-state; unknown field → typed error.
5. **P3e — CLI + workflow entrypoint** (`pegasus export` / `workflows/report/export_dataset.py`): run a
   `QuerySpec` against a bundle → dataset + sidecar. A `rate` request = a `materialized_field` read of
   the pre-computed RN rate field selected by the FEAT-P4 denominator (no query-time math).

Each phase lands with its focused proof-of-capability test; the full suite gates the batch.
**P3a–c landed** (14 tests); P3d/P3e remain (pure reads + CLI — no LDO/denominator math).

## 9. Acceptance (the §IX-style battery for this layer)

- A rate export names its denominator + a CI in the provenance; changing `--denominator` changes both
  the numbers AND the manifest (P4 is real, not cosmetic).
- A standardized-rate export reproduces `directly_standardized_rate` bit-for-bit on a fixture.
- An extensive-numerator rate with no admissible denominator refuses loudly (typed), never a bare count.
- A multi-label disease concept as a rate numerator refuses (non-additive), pointing to the partition path.
