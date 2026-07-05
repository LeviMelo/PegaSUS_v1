# PegaSUS — Operational Implementation Plan (Current Code → MSD-III)

> **The practical counterpart to MSD-III.** Where MSD-III defines *what PegaSUS should be*, this document defines *what to do to the actual codebase to get there* — file by file, test first, with pseudocode, a target module tree, and a disposition for every existing file. It is grounded in the real snapshot: **232 files, 40,113 lines** across `/src`, plus `/config`.

**Companion authority:** MSD-III is the spec; this is the build order. Where this document references a contract, MSD-III (by part/section) is the normative source. Compliance finding IDs and MSD-III work-item IDs (`T0-*`, `LDO-*`, `DIS-*`, …) are the cross-references.

---

## 0. Orientation

### 0.1 The governing engineering principle: strangler-fig, never big-bang

40k lines with real, working substrate logic must **not** be rewritten in one pass. Every phase follows the **strangler-fig pattern**:

1. **Pin** the current behavior you're about to touch with a contract test (Red that currently passes on the old code, or that encodes the *target* and currently fails).
2. **Build the new** module *alongside* the old one, behind a feature flag or a new entrypoint.
3. **Route** a thin adapter so the new module can be exercised in isolation and against the old one.
4. **Migrate** callers to the new module once its tests pass and it matches (or supersedes) the old.
5. **Delete** the old module only after nothing calls it and the contract tests are green.

This means at every commit the system *runs*. There is no "rewrite branch" that diverges for months. The PIRS slice-zoo, for instance, is not deleted up front — it is strangled: the new `ldo/` package is built and validated, then the old `pirs/` files are removed once `ldo/orchestrator.py` subsumes them.

### 0.2 How each work item is written

Every item below has: **ID** · **MSD-III ref** · **Files** (exact paths, `NEW`/`EDIT`/`DELETE`) · **Red** (the test to write first, with assertions) · **Green** (numbered steps + signatures/pseudocode) · **Refactor** · **Done**. Do items in printed order; the dependency graph (App B) is the sequencing law. One item = one PR.

### 0.3 Test layout convention (adopt in Phase 0)

```
tests/
  contract/     # behavior the code MUST honor (compliance + MSD-III contracts); never deleted to pass a build
  unit/         # per-module logic
  integration/  # cross-module flows (compile, investigate)
  synthetic/    # generated-ground-truth recovery tests (the validation battery)
  acceptance/   # the program acceptance test (Zika, no forced selectors)
  fixtures/     # tiny multi-state DATASUS/SIDRA fixtures + planted-truth generators
```

Test runner: `pytest`. Contract tests are the guardrail behind which every refactor happens; a red contract test means the *code* is wrong, never the test.

---

## 1. Current-state map (the codebase as it is)

### 1.1 Layer inventory with dispositions

| Layer | Files | Lines | Role | Disposition |
|---|---:|---:|---|---|
| `efg/` | 23 | 7,765 | legality algebra, DAG, executor, materialize, race_bridge, promotion | **KEEP core, CHANGE output** (measured-quantity objects), **EXTRACT** race_bridge → `measurement/` |
| `pirs/` | 23 | 5,770 | single-outcome GLM + HSIC scanner (the slice-zoo) | **STRANGLE → `ldo/`**; keep `hsic.py`, `nulls.py`, `crossfit.py`, `fdr.py`, `nystrom.py`, `rff.py` as residual-audit citizens |
| `she/` | 32 | 5,390 | substrate, population tensor, ST-DFM, sidra context | **KEEP + EXTEND**: substrate stays; `population/` → `denominators/`; `stdfm/` → LDO low-rank layer. *`reconstruction/` (population solver) holds its `O(n_cells)` problem as Python tuples → OOMs full-national — the §V.1 memory-representation fix is `POP-02` (§4B; M4 bounded-BFS landed).* |
| `datasus/` | 16 | 4,187 | decoders, normalizers, ICD, fetch | **KEEP decoders; CONSOLIDATE 4 normalizers → declarative; MOVE** under `sources/datasus/`. *Fetch layer: v4 bridge (single ZSTD parquet, no `raw.rds`/sidecar) + `storage_gc.py` landed (`STORE-01`, §4B); microdatasus is now a retiring fetch transport only — translation is the in-house codebook.* |
| `workflows/` | 32 | 3,655 | stage drivers + thin per-stage wrappers | **CONSOLIDATE → few entrypoints** (`build.py`, `investigate.py`) |
| `sidra/` | 17 | 2,701 | SIDRA acquire/metadata/compendium/project | **KEEP; MOVE** under `sources/sidra/`; wire context end-to-end |
| `registries/` | 30 | 2,564 | two-world registry (DATASUS + inference + SIDRA) | **REWRITE → unified `registry/`** (`kind`-tagged) |
| `output/` | 13 | 2,294 | 17-key bundle, validate, reproducibility | **KEEP + EXTEND** (link records) |
| `compute/` | 7 | 1,467 | `glm.py` (1,150), devices, memory | **KEEP; ADD `randnla.py`, `controller.py`**. *`devices.py`/`torch_backend.py` already dispatch CUDA for HSIC/STDFM; the population-tensor GPU path (`she/reconstruction/torch_kernels.py`) is orphaned (0 callers) — wire it in `POP-02` (§4B) or drop its `compute.yaml` flag.* |
| `geo/` | 11 | 995 | geography, tiny `adjacency.py`+`spatial_index.py` | **KEEP; GROW adjacency → SpatialWeightGraph** |
| `core/` | 9 | 726 | schemas, config, io, paths | **KEEP + EXTEND** (scope/stage, kinds) |
| `source_artifacts/`, `acceptance/`, `dashboard/`, `storage/` | 16 | 1,528 | contracts, acceptance harness, read-only dashboard, parquet/duckdb | **KEEP** (acceptance harness is a real asset; storage is the out-of-core layer) |

### 1.2 The three structural debts (exact files)

1. **The PIRS slice-zoo** — one conceptual operation ("select fields → build design → fit → scan → rank → report") fragmented across ~10 manifest-passing files: `pirs/{design,design_plan,design_readiness,design_matrix,field_selection,selection_plan,run_candidates,model_execution,hsic_run,hsic_ranking,hsic_report}.py`, mirrored by ~10 workflow wrappers `workflows/{pirs_design,pirs_readiness,pirs_selection,pirs_candidates,pirs_matrix,pirs_execute,hsic_execute,hsic_rank,hsic_report,stage_plan}.py`. Each boundary serializes a manifest to disk. **This is the primary target of the LDO build (Phase 4).**

2. **The two-world registry** — DATASUS world (`source_fields, carrier, unit, aggregation, events, population, cnes_capacity, sih_cost, diagnostic_topology, composite_decoders, quality, semantic, demographic_axis, race_axis, provenance, functional, generic`), a lone SIDRA stub (`sidra.py`, 45 lines), and inference registries (`hsic, nulls, models, output, residuals`). No `kind` tag; the flow/stock/field ontology is implicit in "which world." **Target of Phase 1.**

3. **Normalizer duplication (`XCUT-02`)** — `datasus/declarative_normalize.py` (464) coexists with the four hand-written vectorized normalizers `normalize.py` (434), `sinasc_normalize.py` (459), `sih_normalize.py` (407), `cnes_normalize.py` (357). Decode logic lives in two places. **Target of Phase 0 (`REFACTOR-01`).**

### 1.3 The load-bearing assets to preserve (do not rewrite)

- **`compute/glm.py` (1,150 lines)** — 11 GLM families; becomes the LDO copula-margin transforms unchanged. *Wrap, don't touch the math.*
- **`she/stdfm/torch_solver.py` (372)** — masked PyTorch factor solver with temporal/transition/spatial-Laplacian penalties; becomes the LDO low-rank layer.
- **`she/population/loss.py` (239)** — the analytic population loss; the tensor's Layer 2. *Guard with golden-value tests before any refactor.*
- **`efg/legality.py` (359), `efg/executor.py` (1,127), `efg/dag.py` (1,017), `efg/operators.py` (460)** — the legality type system and materialization; the EFG's true value. Keep; only change the *output object*.
- **`acceptance/contracts.py` (401)** — an existing acceptance framework (currently contract-drift focused). Extend it, don't replace.
- **`storage/` (parquet/duckdb/arrow)** — the out-of-core substrate the national-scale plan needs.
- **`pirs/hsic.py` (563), `nulls.py`, `crossfit.py`, `fdr.py`, `nystrom.py`, `rff.py`** — the HSIC stack; repositioned as the LDO residual-audit, reused.

---

## 2. Target module structure (the codebase as it should be)

The MSD-III codebase reorganizes around the architecture. New packages are **bold**. This is the destination the migration map (§3) routes toward.

```
pegasus/
  core/                 schemas, config, io, paths, hashing, run_context, enums   [KEEP+EXT]
  registry/             ★ UNIFIED kind-tagged registry
    schema.py           source_registry_entry, axis-type tuple, MeasuredQuantity spec
    callables.py        name→function resolver (decoders/parsers/adapters)
    validator.py        cross-registry validator (resolution, reachability, axis-type)
    loader.py           load /config catalog → typed registry objects
  sources/              ★ source adapters (plugin interface)
    base.py             SourceAdapter(source_family, realm∈{event,cube}, discover/read/schema)
    datasus/            decoders.py[KEEP] declarative_normalize.py[KEEP as sole path]
                        icd_parser.py icd_groups.py fetch/ manifests.py  [MOVED]
    sidra/              api.py metadata.py compendium.py extract.py facts.py
                        regime.py projection.py pushforward.py stitching.py  [MOVED]
  she/                  substrate.py[KEEP] zero_variance high_dimensional maternal_child  [KEEP]
  panel/                ★ CommonPanel compilation + per-cell provenance
    plan.py             derive target panel from intent axes × availability × legality
    event_lifecycle.py  event substrate → panel cells
    cube_lifecycle.py   cube → regime → project → pushforward → panel cells
    manifest.py         per-(field,cell) provenance/state
  denominators/         ★ the denominator-asset family
    population/         two_layer.py (Layer1 closed form + warm start), loss.py[MOVED],
                        solvers.py[MOVED: state_space/admm/proj_grad/block_coord]
    reconstruction/     ctr.py (CTRProblem, generalizes population + age-disaggregation), certify.py
    exposures.py        household/establishment/area denominator assets
  measurement/          ★ measurement-model layer (RaceBridge generalized)
    bridge.py           EmissionBridge(latent, sources[C_k], local_pi, uncertainty)
    race.py             per-source race matrices (SIM/SIH/SINASC), literature-informed priors
  disease/              ★ disease semantic axis
    concepts.py         multi-label concept registry (CCSR/CCIR/CCC/ICSAP), projection typing
    icd_adapter.py      simple-icd-10 wrapper (closure, NCA), CID-10 authoritative
    graph.py            DiseaseGraph → L_D, legality_class, overlap (Jaccard)
    embed_build.py      (optional, build-time) cached label embeddings
  geo/                  ★ SpatialWeightGraph (grow adjacency)
    spatial_graph.py    load_spatial_graph(id).view(binary|rowstd|symmetric|laplacian).blocks(n)
    (existing crosswalks/support KEEP)
  efg/                  legality[KEEP] executor[KEEP] dag[KEEP] operators[KEEP]
    measured_quantity.py ★ terminal output = (count, exposure, structure, provenance)  [replaces rate materialization]
  field/                ★ latent-field + multiresolution engine (THE substrate for the LDO)
    latent.py           latent field + weighted observation operator (sparse/dense unified)
    multiresolution.py  hierarchical shrinkage across space/time/disease scales
    assemble.py         panel → field tensor X ∈ R^{p×S×T} + weights W
  ldo/                  ★ the inference engine (replaces pirs/)
    records.py          LinkRecord schema
    margins.py          copula margins (wraps compute/glm; extensive→count+exposure)
    lowrank.py          low-rank latent factors (wraps she/stdfm)
    precision.py        sparse + lagged structured precision (graphical lasso + Kronecker)
    edges.py            edge readout + stability selection + overlap typing
    residual_scan.py    HSIC on residual field (wraps pirs/hsic,nulls,crossfit,fdr)
    certify.py          certification gate (holdout, path-agreement, separability)
    orchestrator.py     run_ldo(panel, intent) — the whole pipeline, in memory
  causal/               ★ escalation ladder
    orient.py           rung1: non-Gaussian (LiNGAM) + colliders
    quasi.py            rung2: ITS/DiD/negative-control (shock-triggered)
    identify.py         rung3: do-calculus (expert-invoked)
  compute/              glm.py[KEEP] devices memory random torch_backend  [KEEP]
    randnla.py          ★ randomized SVD, Hutchinson/SLQ logdet, JL sketching (bounded)
    controller.py       ★ Adaptive Precision Controller (anytime, uncertainty-scheduled)
  assets/               ★ foundational asset layer
    tiers.py            foundational vs query; the scope-invariance invariant
    build.py            data-arrival-triggered builds; incremental update
    version.py          immutable versioned artifacts + input manifests
    store.py            artifact store + query version-pinning
  validation/           ★ the standing battery
    controls.py         known-positive/known-negative control sets
    synthetic.py        planted-truth generator + recovery scoring
    holdout.py          temporal holdout; exact-vs-approx cross-check
  output/               validate.py[KEEP+EXT] bundle reproducibility table_io  [KEEP]
    link_records.py     ★ Hypotheses key = link records (retire pairwise row)
  interaction/          ★ the five verbs
    interrogate.py lens.py escalate.py steer.py inject.py
  workflows/            build.py (foundation build) investigate.py (query)  [CONSOLIDATED]
    (retire the ~20 pirs_*/hsic_*/efg_* wrappers)
  cli.py                [KEEP, re-point to consolidated workflows]
```

**Reading:** roughly half the current tree is KEEP/EXTEND (substrate, decoders, GLM, ST-DFM, EFG legality, storage, output). The rewrites are concentrated in three places — the registry (unify), PIRS (→ LDO), and the workflow wrappers (consolidate) — plus a set of genuinely new packages (`field/`, `ldo/`, `disease/`, `measurement/`, `assets/`, `validation/`, `causal/`, `interaction/`, `compute/randnla+controller`) that implement the parts of MSD-III the current code simply does not have yet.
---

## 3. The migration map (every current file → disposition)

`KEEP` = unchanged/minor · `EXT` = extend in place · `MOVE→` = relocate · `MERGE→` = fold into · `REWRITE→` = replace with new · `DEL` = delete after strangle.

**efg/** → `efg/` + `measurement/`
`legality`·`executor`·`dag`·`operators`·`node`·`lineage`·`align`·`equivalence`·`empirical_compression`·`failed_branch` **KEEP** · `materialize`·`compile_attach`·`materialization_manifest` **KEEP+EXT** (emit MeasuredQuantity) · `q_tensor`·`diagnostic_strata` **KEEP** (feed reliability weights) · `core_seed`·`core_seed_registry` **EXT** (variable generation moves to disease-axis grammar, §Phase 2) · `promotion_plan`·`promotion_apply` **KEEP** · `race_bridge` **REWRITE→`measurement/race.py`** · `bridges`·`declaration` **KEEP**

**pirs/** → `ldo/` (strangle)
`hsic`·`nulls`·`crossfit`·`fdr`·`nystrom`·`rff` **MOVE→`ldo/residual_scan.py`** (reused) · `families`·`spatial` **MERGE→`ldo/margins.py`,`geo/spatial_graph.py`** · `schemas` **MERGE→`ldo/records.py`** · `design`·`design_plan`·`design_readiness`·`design_matrix`·`field_selection`·`selection_plan`·`run_candidates`·`model_execution`·`hsic_run`·`hsic_ranking`·`hsic_report`·`diagnostics`·`residuals` **DEL** (subsumed by `ldo/` package)

**she/** → `she/` + `denominators/` + `ldo/lowrank`
`substrate`·`zero_variance`·`high_dimensional`·`maternal_child`·`maternal_child_linkage`·`demographic_tensor`·`cnes_capacity`·`sih_costs` **KEEP** · `population/*` **MOVE→`denominators/population/`** (+ add `two_layer.py`) · `stdfm/*` **MOVE→ wrapped by `ldo/lowrank.py`** (solver KEEP) · `sidra_context`·`sidra_projection` **MOVE→`sources/sidra/` + `panel/cube_lifecycle.py`**

**datasus/** → `sources/datasus/`
`decoders` **KEEP (authoritative)** · `declarative_normalize` **KEEP (sole path)** · `normalize`·`sinasc_normalize`·`sih_normalize`·`cnes_normalize` **DEL** (after `REFACTOR-01`; keep only thin batch wrappers if profiling needs them) · `icd_parser`·`icd_groups` **MOVE→`disease/`** (icd_groups becomes one partition view) · `manifests`·`subprocess`·`client_microdatasus`·`profile`·`cache`·`schema_compare`·`r_scripts/` **MOVE (KEEP)**

**registries/** → `registry/` (rewrite to unified schema)
All 30 files **REWRITE→** `registry/{schema,callables,validator,loader}.py` reading `/config` catalog. The *data* (source_fields, carrier, unit, aggregation, diagnostic_topology, etc.) is preserved but re-expressed under the one `kind`-tagged schema; the *inference* registries (hsic, nulls, models, output, residuals) become `ldo/` config. `validators.py` **REWRITE→`registry/validator.py`**.

**sidra/** → `sources/sidra/` **MOVE (KEEP)**, wire context end-to-end (`DIS`/`CTX` items).
**workflows/** → `workflows/{build,investigate}.py` — `pipeline`·`compile`·`build_efg`·`build_substrate`·`population`·`race_bridge`·`msd_inference` **MERGE→** the two entrypoints; all `pirs_*`,`hsic_*`,`efg_*`,`stage_plan`,`ingest_*` **DEL/MERGE**.
**output/** **KEEP+EXT** (`link_records.py` NEW; `validate.py` extend). **compute/** **KEEP+ADD** (`randnla`,`controller`). **geo/** **KEEP** (`adjacency`→`spatial_graph`). **core/**·**storage/**·**acceptance/**·**dashboard/**·**source_artifacts/** **KEEP+EXT**.

---

## 4. The phased implementation plan

Phases gate strictly (`N` requires `N−1` green). Near-term phases (0–4) are file-level with pseudocode; later phases (5–7) give operational shape (fully specified normatively in MSD-III).

---

### PHASE 0 — Lock the compliant codebase, then de-duplicate

*Goal: pin every landed compliance remediation with a contract test so all later refactoring happens behind a green guardrail; then remove the normalizer duplication. Nothing new is built yet.*

#### `T0-1` — Source-reality contract tests
- **MSD-III:** §II.2, X.1. **Files:** `tests/contract/test_source_reality.py` NEW; `tests/fixtures/datasus_multistate.py` NEW.
- **Red — write these assertions against a fixture with one record per §2.3 state per system:**
  ```python
  def test_no_admissible_column_all_null(sim_df, sih_df, sinasc_df, cnes_df): ...
  def test_age_decode_exact(sim_df):            # age_years/age_days == decode_sim_idade
  def test_cause_chain_positions(sim_df):       # cause_chain_norm has 4 ordered positions + per-position state
  def test_sinasc_emits_primitives(sinasc_df):  # birth_weight_grams/apgar_5min/gestational_weeks present; NO flags
  def test_cnes_no_bed_total(cnes_df, registry):# no capacity_total_observed; per-index carriers present
  def test_cnpj_gate(cnes_df, sih_df):          # all-zero CNPJ -> null link + ZeroCNPJShare observer
  def test_state_vocab(all_df):                 # every *_state distinguishes Missing/Invalid/Unparseable/Unknown/Valid
  ```
- **Green:** run; fix only what fails (most already pass — see X.1). If `cause_chain_positions` or CNES/CNPJ fail, complete the declarative routing for SIH/CNES in `datasus/declarative_normalize.py`.
- **Done:** all green on the multi-state fixture; tag commit `phase0-source-reality`.

#### `T0-2` — EFG legality / state-tensor / declaration contracts
- **Files:** `tests/contract/test_efg_contracts.py` NEW. **Red:**
  ```python
  def test_q_tensor_required_columns(q): # includes CV, MoranI, temporal_roughness, spatial_entropy
  def test_neff_moran_correction(q):     # n_eff == (Σw)²/Σw² · 1/(1+max(0,MoranI))
  def test_race_axis_declaration_gate(): # race-stratified operand w/o race_axis_type -> fails, not permissive
  def test_high_card_axis_bound():       # oversized SIDRA axis product -> mandatory pushforward or Δaxes=0
  ```
- **Green:** ensure the four columns are in `REQUIRED_Q_TENSOR_COLUMNS` (`output/validate.py` / `efg/q_tensor.py`); wire the high-card bound into the legality path (`efg/dag.py`→`efg/materialize.py`) *before* materialization. **Done:** green.

#### `T0-3` — RaceBridge numerics contracts (pre-redesign pin)
- **Files:** `tests/contract/test_race_bridge.py` NEW. **Red:** `W` depends on local π (perturb π → W changes); `race_bridge_cv` = SD/mean across draws (not cross-category spread); `SensitivityWidth` from inf/sup over the credible set; outputs `state ≤ fragile`. **Green:** verify current `efg/race_bridge.py` satisfies these (fix CV if it's cross-category). **Done:** green — this pins behavior before the Phase-2 redesign.

#### `T0-4` — Inference baseline + output contract (the LDO refactor guardrail)
- **Files:** `tests/contract/test_inference_baseline.py` NEW. **Red:** HSIC mode selection exact; family routing for count/proportion/simplex/skewed; spatial-mode precedence; 17 keys present; empty keys carry typed reasons. **Green:** confirm against current `pirs/` + `output/validate.py`. **Done:** green; tag `pre-ldo-baseline`. *This is the guardrail behind which Phase 4 strangles PIRS.*

#### `REFACTOR-01` — De-duplicate normalizers (resolve `XCUT-02`)
- **MSD-III:** §II.2, X.2. **Files:** EDIT `datasus/declarative_normalize.py`; DEL `datasus/{normalize,sinasc_normalize,sih_normalize,cnes_normalize}.py` (or reduce to thin batch wrappers). **Red:** `tests/contract/test_no_decode_duplication.py` — a property test asserting vectorized output (if retained) == record-level declarative output on the fixture, AND that inline `when/then` decode chains are absent from the four files. **Green:** make declarative the sole record-level normalizer for all four systems; regenerate any vectorized fast path *from* the routing registry, or drop it. **Done:** duplication test green; `T0-1` still green.

**Phase 0 exit:** a fully MSD-I-compliant codebase, every remediation pinned, duplication gone. All later work happens behind these tests.

---

### PHASE 1 — Platform foundations

*Goal: the unified registry, the SpatialWeightGraph, the CommonPanel with per-cell provenance, and the orthogonal DataScope × ExecutionStage. These are the substrate the data plane and engine sit on.*

#### `REG-07` — Unified `kind`-tagged registry + validator
- **MSD-III:** §II.1, I.4. **Files:** NEW `registry/{schema,callables,validator,loader}.py`; `/config/registry/` catalog (migrate existing YAML). DEL (after migration) the 30 `registries/*.py`.
- **Red:** `tests/contract/test_registry.py`:
  ```python
  def test_every_routing_resolves_callable(reg): ...        # no unresolved decoder/parser name
  def test_kind_and_axis_tags_present(reg):                 # every measure has kind; every axis has (entity,structure,role)
  def test_roundtrip_reachability(reg):                     # every admissible canonical field producible from raw/reconstruction
  def test_onboarding_cost(tmp_registry):                   # add a fixture source -> only catalog + <=1 callable changed
  def test_extensive_needs_exposure(reg):                   # every extensive measure names an exposure_ref
  ```
- **Green — pseudocode:**
  ```python
  # registry/schema.py
  @dataclass(frozen=True)
  class Measure: name:str; unit:str; kind:Literal["flow","stock","field"]; extensive:bool
                 carrier:str|None=None; exposure_ref:str|None=None
  @dataclass(frozen=True)
  class Axis: name:str; entity_type:str; coordinate_structure:Literal["metric_hier","tree_hier","flat_cat"]
              role:Literal["support","identity","stratify_join"]; categories:tuple; aggregation_law:str
  @dataclass(frozen=True)
  class SourceEntry: source_id:str; measures:tuple[Measure,...]; axes:tuple[Axis,...]
                     native_resolution:dict; kind:str; provenance:dict; adapter_callable:str
  # registry/callables.py
  def resolve_callable(name:str)->Callable: return _REGISTRY[name]     # single resolver; raises on miss
  # registry/validator.py
  def validate_registry(reg)->list[Violation]: # resolution + tags + reachability + extensive/exposure
  ```
- **Green — steps:** (1) define the schema; (2) write `callables.py` collecting existing decoder/parser functions by name; (3) migrate the DATASUS + SIDRA + inference registry *data* into `/config/registry/` under the schema (this is mechanical translation, one source at a time, each behind a test); (4) `validator.py` replacing `registries/validators.py::validate_registry_tree`; (5) `loader.py` producing typed objects; (6) point `she/source_registry.py` and callers at the new loader; (7) DEL old registries once nothing imports them.
- **Refactor:** delete orphan alias files; one canonical filename per registry. **Done:** validator green; onboarding-cost test green.

#### `SPG-01/02/03` — SpatialWeightGraph
- **MSD-III:** §II.6 (guard), I.4. **Files:** NEW `geo/spatial_graph.py`; `/config/registry/spatial_graphs.yaml`. Rewire `efg/q_tensor.py` (Moran), `compute/glm.py::fit_glm_penalized` (ICAR), `pirs/nulls.py`+`crossfit.py` (blocks).
- **Red:** `tests/contract/test_spatial_graph.py`: views (`binary|rowstd|symmetric|laplacian`) correct on a 5-node fixture; `.blocks(n)` partitions; Moran/ICAR/nulls/folds all consume the shared graph (no ad-hoc adjacency remains); a `context_derived` graph sharing provenance with a tested variable is **rejected** (the circularity abort).
- **Green — pseudocode:**
  ```python
  # geo/spatial_graph.py
  class SpatialWeightGraph:
      def __init__(self, W:sparse.csr_matrix, legality_class:str, provenance:list): ...
      def view(self, kind): return {"binary":..., "rowstd":..., "symmetric":..., "laplacian":D-W}[kind]
      def blocks(self, n): return _partition(self.W, n)     # for nulls/folds
  def load_spatial_graph(gid)->SpatialWeightGraph: ...        # from registry; default = contiguity_queen (structural)
  def assert_not_circular(graph, tested_provenance):          # §II.6 abort
      if graph.legality_class=="context_derived" and set(graph.provenance)&set(tested_provenance): raise ...
  ```
- **Green — steps:** grow `geo/adjacency.py` (27 lines) into the graph; register `contiguity_queen` as the structural default; replace each ad-hoc adjacency call site; add the guard to the legality/selection path. **Done:** all three green; old adjacency call sites gone.

#### `PANEL-01` — CommonPanel + per-cell provenance + month resolution
- **MSD-III:** §I.5, VII.2. **Files:** NEW `panel/{plan,event_lifecycle,cube_lifecycle,manifest}.py`; EDIT `workflows/compile.py` (delegate to panel). **Red:** `tests/contract/test_common_panel.py`: manifest records per-(field,cell) `state ∈ {observed,projected,reconstructed,bounded,unavailable+reason}`; **no blank cell**; panel supports `year` AND `month` from intent. **Green:** `plan.py` derives the target panel from intent axes × availability × legality; the two lifecycle files orchestrate existing regime/stitch/projection/pushforward modules; thread `month` (already decoded at ingest) through the panel grain. **Done:** green; monthly Alagoas panel materializes.

#### `SCOPE-01` — DataScope × ExecutionStage
- **Files:** EDIT `core/schemas.py` (UserIntent gains orthogonal `data_scope`, `execution_stage`), `output/validate.py`. **Red:** `tests/contract/test_scope_stage.py`: axes orthogonal; `LinkRecords` required-nonempty only at `investigate`; all keys present with typed empty reasons; default stage = `investigate`. **Green:** add axes; make `PROFILE_NONEMPTY` a function of `(scope, stage)`. **Done:** green; existing intents still validate (default applied).

**Phase 1 exit:** one registry, one spatial graph, a provenance-complete panel at month resolution, and clean scope/stage axes.

---

### PHASE 2 — Data plane

*Goal: the EFG emits count+exposure objects (not rates); the population tensor becomes the two-layer design; RaceBridge is redesigned to actually correct bias; the disease semantic axis exists.*

#### `EFG-OUT-01` — Measured-quantity objects (retire rate materialization)
- **MSD-III:** §II.3, I.2. **Files:** NEW `efg/measured_quantity.py`; EDIT `efg/materialize.py`, `efg/compile_attach.py`, `output/validate.py`. **Red:** `tests/contract/test_measured_quantity.py`: a mortality field materializes as `MeasuredQuantity(numerator_count, exposure, offset_semantics, structure, provenance, uncertainty)` — NOT a pre-divided rate; an **extensive field** (sewage-connected households) carries its exposure (total households); the 17-key bundle still validates.
- **Green — pseudocode:**
  ```python
  # efg/measured_quantity.py
  @dataclass(frozen=True)
  class MeasuredQuantity:
      numerator_count: pl.Series      # the count (deaths, connected households, ...)
      exposure: pl.Series             # population-at-risk / total households / ... (denominator)
      offset_semantics: str           # "log_exposure" -> model uses log(exposure) as offset
      structure: dict                 # axes, strata
      provenance: dict; uncertainty: dict   # from state tensor
  # materialize.py: where it used to compute count/exposure -> emit MeasuredQuantity(count, exposure, ...)
  ```
- **Green — steps:** locate the rate-division sites in `materialize.py`; replace terminal rate with the MeasuredQuantity bundle; the *rate* becomes a derived view only (for dashboards), never the modeling input. **Done:** green; the LDO (Phase 4) consumes count+exposure directly.

#### `POP-01` — Two-layer population tensor
- **MSD-III:** §II.4. **Files:** NEW `denominators/population/two_layer.py`; MOVE `she/population/*`→`denominators/population/`; add table **2093** to `/config` compendium. **Red:** `tests/contract/test_population_two_layer.py`:
  ```python
  def test_layer1_closed_form(census2010, census2022, totals):
      # interpolated composition × yearly totals; deterministic; ILR geometry
  def test_solver_warm_starts_from_layer1(solver_trace):
      # solver initial state == Layer1 (NOT uniform); no uniform-seed trap
  def test_single_census_returns_layer1(al_2015_2022):
      # with one in-window census + no flow signal, optimum == Layer1
  def test_three_census_cohort_aging(nat_2000_2010_2022):
      # ≥2 censuses -> cohort momentum captured (a 2010 bulge appears aged in 2022)
  def test_2093_registered(compendium): ...
  ```
- **Green — pseudocode:**
  ```python
  # denominators/population/two_layer.py
  def layer1_prior_mean(census_years, closure_totals, axes)->Tensor:
      shares = interpolate_ilr(census_composition(census_years))   # compositional interpolation across years
      return shares * closure_totals[:,None,None,None]             # scale to known yearly totals
  def solve_population(intent, censuses, totals, flows)->Tensor:
      x0 = layer1_prior_mean(censuses, totals, axes)               # WARM START (fixes uniform trap)
      if not has_flow_signal(flows) and n_in_window_censuses(censuses)<2:
          return x0                                                # data-poor: Layer1 is the optimum
      return run_msd_six_term_solver(x0, censuses, totals, flows)  # she/population/loss.py, warm-started
  ```
- **Green — steps:** (1) implement `layer1_prior_mean` (ILR interpolation + closure scaling); (2) modify the existing solver entry (`she/population/orchestrator.py`) to accept and start from `x0`; (3) add the data-poor short-circuit; (4) register table 2093 (select the clean age partition; roll-ups derived, never summed); (5) ingest all three censuses regardless of query window. **Refactor:** guard `loss.py` with golden-value tests before moving it. **Done:** all green; the uniform trap is structurally impossible.
- **Compute contract:** POP-01 is the *math* (two layers). The *memory/scale* of that solve — array representation, float32, locality-blocking, GPU wiring — is `POP-02` (§4B). POP-01 is largely landed (the two-layer solver + warm start exist); the national demographic tensor currently OOMs on the full window for the reasons POP-02 fixes. Do not add scale hacks here — they belong in POP-02 behind its parity tests.

#### `RACE-01..07` — RaceBridge redesign (per-source, literature-informed, never identity)
- **MSD-III:** §II.5. **Files:** NEW `measurement/{bridge,race}.py`; DEL `efg/race_bridge.py` (after strangle); `/config/registry/emission_matrices/` with `C_SIM/C_SIH/C_SINASC`. **Red:** extend `tests/contract/test_race_bridge.py`:
  ```python
  def test_identity_prohibited_as_default(bridge):   # default C != I; raises/refuses if only identity available
  def test_literature_informed_seed(C_SIM):          # off-diagonals nonzero, sourced, diagonal-dominant, rows sum 1
  def test_per_source_matrices(reg):                 # C_SIM, C_SIH, C_SINASC distinct entries
  def test_region_conditioning(bridge):              # C[k][j|region] available where declared
  def test_uncertainty_propagated(posterior):        # sensitivity_width>0; state<=fragile; dashboard-unsafe
  ```
- **Green — pseudocode:**
  ```python
  # measurement/bridge.py  (the general measurement-model pattern)
  class EmissionBridge:                              # observation operator for a source measuring a latent attribute
      def __init__(self, C:dict[str,dict[str,float]], target_pi_source, uncertainty):
          assert not is_identity(C), "identity emission is a prohibited no-op default (MSD-III §II.5)"
      def reallocate(self, admin_counts, pi_local):
          W = normalize_rows({k:{j:C[k][j]*pi_local[j] for j in J} for k in K})   # Bayesian local-pi
          out = {j: sum(admin_counts[k]*W[k][j] for k in K) for j in J}
          out = add_missing(out, pi_local); return with_uncertainty(out, draws=dirichlet(C))
  # measurement/race.py: load C_SIM/C_SIH/C_SINASC from literature-informed registry (per source, region-conditioned)
  ```
- **Green — steps:** (1) build `EmissionBridge` (generalizes the current `fixedc_dynamic_weight_bridge` math, which is *kept*); (2) populate `C_SIM/C_SIH/C_SINASC` from published Brazilian discordance patterns (a research task — seed with documented branca↔parda↔preta flows, diagonal-dominant); (3) prohibit identity default; (4) region-conditioning where data supports; (5) propagate uncertainty (already present); (6) strangle `efg/race_bridge.py`. **Done:** green — the bridge now *corrects* bias instead of no-op'ing.

#### `DIS-01/02/03` — Disease semantic axis
- **MSD-III:** §II.6, I.4. **Files:** NEW `disease/{concepts,icd_adapter,graph}.py`; `/config/registry/disease_concepts.yaml`, `disease_graphs.yaml`. **Red:** `tests/contract/test_disease_axis.py`: multi-label preserved (a code → ≥2 CCSR concepts); ICD-10-CM concept on CID-10 → `projection_status=approximate`, `state≤fragile`; `icd_nearest_common_ancestor` correct; CID-10-only code (dengue A90) → `source_system_specific`, chapter resolved, not coerced; `DiseaseGraph.laplacian()` sparse; `context_derived` disease graph rejected as same-data prior; high-Jaccard concept pair → `mechanical_overlap` (not a discovery). **Green — steps:** (1) `concepts.py` importing groupers (wrap `icd-mappings`) with provenance/projection typing; (2) `icd_adapter.py` wrapping `simple-icd-10`, CID-10 authoritative (extend `datasus/icd_groups.py`→ one partition view); (3) `graph.py` building the tree/embedding → `L_D`, with `legality_class`, overlap Jaccard. **Done:** green.

**Phase 2 exit:** the data plane speaks count+exposure, denominators are demographically valid and warm-started, race bias is actually corrected, and the disease axis (with `L_D`) exists to feed the engine.
---

### PHASE 3 — The latent-field + multiresolution engine

*Goal: the single highest-leverage new build. It is the substrate that dissolves sparse-vs-dense and hierarchy-on-every-axis, and everything in Phase 4 sits on it. It rests on established methodology (small-area estimation / INLA / SPDE / GMRF), so it is buildable, not speculative.*

#### `LF-01` — Latent field + weighted observation model
- **MSD-III:** §III.2. **Files:** NEW `field/{latent,assemble}.py`. **Depends:** `PANEL-01`, `SPG-01`.
- **Red:** `tests/synthetic/test_latent_field.py`:
  ```python
  def test_sparse_variable_informs_only_observed_cells():
      # a variable observed only at t=2018 contributes to the 2018 cross-section, ZERO to dynamics
      f = LatentField(shape=(S,T)); f.observe(cells_2018, values, weight=1.0)
      assert f.dynamic_information() == 0  # single slice -> no temporal info; nothing fabricated
  def test_dense_and_sparse_coexist():
      # one weighted likelihood handles both; only the weight pattern differs
  def test_weights_from_state_tensor(panel):
      # reliability weight per cell derived from n_eff / fragility (efg/q_tensor)
  ```
- **Green — pseudocode:**
  ```python
  # field/latent.py
  class LatentField:
      """A quantity defined on the finest lattice, observed at arbitrary resolution/sparsity via a
      weighted observation operator. Sparsity == zero weight; nothing is fabricated."""
      def __init__(self, lattice_shape): self.Z = None; self.W = zeros(lattice_shape)  # W = obs weights
      def observe(self, cells, values, weight):        # cells may be at coarse resolution
          A = observation_operator(cells, self.lattice_shape)   # maps latent -> observed (aggregation)
          self._accum(A, values, weight)               # accumulate weighted sufficient stats
  # field/assemble.py
  def assemble_field(panel_manifest, state_tensor) -> tuple[Tensor, Tensor]:
      X = zeros((p, S, T)); W = zeros((p, S, T))
      for var in panel_manifest.variables:
          for obs in var.observations:                 # each dataset's cells, at its native resolution
              A = observation_operator(obs.cells, (S,T))
              X, W = scatter_weighted(X, W, var.idx, A, obs.values,
                                      weight=reliability(state_tensor, var, obs))  # n_eff/fragility
      return X, W                                       # the LDO consumes (X, W)
  ```
- **Green — steps:** (1) `LatentField` with the observation operator (aggregation matrix from coarse cells → fine lattice); (2) reliability weights from `efg/q_tensor.py` (n_eff, fragility); (3) `assemble_field` producing `(X, W)` from the panel manifest. **Done:** synthetic tests green; sparse/dense unified.

#### `MR-01` — Multiresolution hierarchical shrinkage
- **MSD-III:** §III.3. **Files:** NEW `field/multiresolution.py`. **Depends:** `LF-01`, `SPG-01`, `DIS-03`.
- **Red:** `tests/synthetic/test_multiresolution.py`:
  ```python
  def test_rare_cell_borrows_from_parent():
      # a leaf-disease/tiny-municipality estimate is dominated by shrinkage toward its better-estimated parent
  def test_all_scales_coexist():
      # national + regional + state + municipal components estimated jointly; data decides where signal lives
  def test_strong_fine_signal_survives_shrinkage():
      # shrinkage is adaptive, not hard pooling: a strong fine-only signal still emerges
  ```
- **Green — pseudocode:**
  ```python
  # field/multiresolution.py
  def multiresolution_decompose(X, hierarchies):        # hierarchies: {axis: parent_map}
      """X = Σ_scale component_scale, with each fine level a deviation from its parent."""
      # e.g. municipal = national + region_dev + state_dev + muni_dev (per axis, nested)
      return {scale: component(X, scale) for scale in scales(hierarchies)}
  def shrinkage_prior(components, laplacians):           # L_W (space), L_D (disease), temporal
      # penalty pulls each fine component toward its parent (GMRF/hierarchical); ADAPTIVE (data overrides)
      return sum(lam[s] * quadratic(components[s], laplacians[s]) for s in components)
  ```
  Implementation note: this is a **GMRF / latent-Gaussian** model; use sparse precision from the graph Laplacians (`geo/spatial_graph`, `disease/graph`) and solve with the matrix-free machinery (Phase 4 `randnla`). The literature engine is INLA/SPDE — mirror its structure (sparse GMRF + observation operator at arbitrary resolution).
- **Done:** synthetic tests green; hierarchy and sparsity solved by one device.

**Phase 3 exit:** a coherent multiresolution latent field over space × time × disease that any variable (sparse or dense, coarse or fine) feeds into uniformly. The LDO is now buildable on top.

---

### PHASE 4 — The LDO core (strangle PIRS)

*Goal: build the `ldo/` package layer by layer on the Phase-3 field, validate each layer on synthetic ground truth, then retire the PIRS slice-zoo. Keep `T0-4` (pre-ldo-baseline) green throughout — the LDO **adds** capability; the old scan keeps running until the orchestrator subsumes it.*

Build order is strict: `records → margins → precision(K=0) → lowrank → lags → edges → certify → orchestrator`.

#### `LDO-00` — Field assembly + LinkRecord schema
- **Files:** NEW `ldo/records.py`; EDIT `output/validate.py`, `output/link_records.py`. **Red:** `tests/contract/test_link_record.py` — bundle validator accepts a `LinkRecord` in the `Hypotheses` key (additive; old row schema still accepted for now).
- **Green — pseudocode:**
  ```python
  # ldo/records.py
  @dataclass(frozen=True)
  class LinkRecord:
      source_var:str; target_var:str; lag_k:int
      edge_type:Literal["contemporaneous","lagged_directed","latent_shared","nonlinear_residual","mechanical_overlap"]
      weight:float; partial_correlation:float
      response_curve_ref:str|None; spatial_field_ref:str|None
      stability:float; uncertainty:dict; confounding_factor_refs:tuple
      projection_status:str; code_system:str; topology_role:str; overlap_jaccard:float
      certification_status:str; warnings:tuple
  ```
- **Done:** schema test green; 17-key validator green.

#### `LDO-01` — Layer 0: copula margins (wrap `compute/glm.py`)
- **MSD-III:** §III.5, I.2. **Files:** NEW `ldo/margins.py`. **Red:** `tests/synthetic/test_margins.py` — Poisson/NB/beta-binomial/Gamma column → `Z = Φ⁻¹(F_j(·))` via existing GLM CDF; sparse-count → PIT/rank fallback; **extensive** variable → binomial/Poisson-with-exposure margin (uses the MeasuredQuantity exposure from `EFG-OUT-01`).
- **Green — pseudocode:**
  ```python
  # ldo/margins.py
  def to_latent_gaussian(x, family, exposure=None):
      """Push a variable to the common Gaussian scale. Extensive -> count+exposure margin."""
      F = glm_cdf(family, x, offset=log(exposure) if exposure is not None else None)  # wraps compute/glm.py
      return norm_ppf(clip(F, 1e-6, 1-1e-6))          # PIT/rank fallback for sparse ties
  ```
  *No new family math — this wraps the 11 families already in `compute/glm.py`.*
- **Done:** synthetic margins green.

#### `LDO-02` — Layer 1: sparse + spatial + temporal precision (K=0)
- **MSD-III:** §III.4. **Files:** NEW `ldo/precision.py`; NEW `compute/randnla.py`. **Depends:** `SPG-01`, `MR-01`.
- **Red:** `tests/synthetic/test_precision_k0.py` — plant a known sparse contemporaneous graph + GMRF spatial structure; the estimator recovers the true edges (precision/recall > threshold); spatial prior uses `L_W`; runs within the compute envelope on Alagoas-monthly.
- **Green — pseudocode:**
  ```python
  # compute/randnla.py   (bounded-error primitives — Phase-4 foundation for scale)
  def randomized_svd(A, rank, oversample=10): ...     # Halko-Martinsson-Tropp; error-bounded low-rank
  def hutchinson_logdet(matvec, n, probes=32): ...    # stochastic log-det for the Gaussian likelihood
  def jl_sketch(X, target_dim): ...                   # Johnson-Lindenstrauss row compression
  # ldo/precision.py
  def fit_precision(Z, W, L_W, L_time, lam1, K=0):
      """Neighborhood-selection / graphical-lasso on the latent Z with GMRF whitening.
         Kronecker-separable: Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹ — never densified."""
      Zw = whiten(Z, L_W, L_time)                     # apply sparse GMRF whitening (matrix-free)
      # proximal-gradient graphical lasso: soft-threshold small entries -> sparse edge selection
      S = prox_graphical_lasso(sufficient_stats(Zw, W), lam1)   # ℓ1 prox
      return S
  ```
- **Green — steps:** (1) `randnla.py` primitives; (2) sparse GMRF whitening via `spatial_graph.view("laplacian")`; (3) proximal-gradient graphical lasso (soft-threshold = edge selection), matrix-free on torch; (4) sufficient-statistics streaming (`storage/duckdb`) for out-of-core. **Done:** edge recovery green on synthetic; within envelope on AL-monthly.

#### `LDO-03` — Layer 2: low-rank latent factors (wrap `she/stdfm`)
- **MSD-III:** §III.4. **Files:** NEW `ldo/lowrank.py`. **Red:** `tests/synthetic/test_lowrank.py` — plant a shared latent factor (an "epidemic wave"); variables co-moving via the factor are flagged `latent_shared`, **not** dense direct edges; `S` recovers only true direct edges net of the factor.
- **Green — pseudocode:**
  ```python
  # ldo/lowrank.py
  def fit_lowrank(Z, W, rank):
      """The shared drivers (epidemic wave). Reuses she/stdfm/torch_solver.py as the nuclear-norm step."""
      L = stdfm_solve(Z, W, rank, spatial_penalty=L_W, transition_penalty=True)   # existing solver
      return L
  def sparse_minus_lowrank(Z, W, L_W, lam1, lam_star, rank):
      # alternate: precision = S(sparse) - L(low-rank); CPW latent-variable decomposition
      L = fit_lowrank(Z, W, rank); S = fit_precision(residual(Z, L), W, L_W, ..., lam1)
      return S, L                                      # S=direct links, L=shared drivers
  ```
  *The market-factor/idiosyncratic split you know from finance — same object.*
- **Done:** confounded-by-wave variables correctly separated; synthetic green.

#### `LDO-04` — Lag extension (the Zika capability)
- **MSD-III:** §III.4. **Files:** EDIT `ldo/{precision,records}.py`, `field/assemble.py`. **Red — the core scientific test:** `tests/synthetic/test_lags.py`:
  ```python
  def test_recovers_planted_lag():
      # monthly synthetic: variable A at lag 7 drives B, + a shared wave + spatial structure
      X = generate(edges={("A","B",7):0.6}, factors=["wave_2015"], spatial=L_W)
      links = fit_ldo(X, K=9)
      e = find_edge(links, "A", "B")
      assert e.edge_type=="lagged_directed" and abs(e.lag_k-7)<=1     # discovers the delay
      assert "wave_2015" in e.confounding_factor_refs                 # wave attributed to L, not a spurious edge
      # and NO variable was hand-specified
  ```
- **Green — pseudocode:**
  ```python
  # field/assemble.py: extend the variable axis with time-shifted copies X^{(0..K)}
  def time_extend(X, K): return stack([shift(X, k) for k in range(K+1)])   # months
  # ldo/precision.py: estimate cross-lag blocks S^{(k,0)}; a nonzero = directed lag-k edge past->present
  # ldo/records.py: emit lagged_directed edges; response_curve_ref = {S^{(k,0)}}_k  (the discovered lag profile)
  ```
- **Done:** the lag test green; recovered peak lag within ±1 of truth. **This is the capability the whole redesign exists for.**

#### `LDO-05` — Edge readout + stability selection + residual HSIC + overlap
- **MSD-III:** §III.6, III.7, II.6. **Files:** NEW `ldo/{edges,residual_scan}.py`; MOVE `pirs/{hsic,nulls,crossfit,fdr,nystrom,rff}.py`→ used by `residual_scan.py`. **Red:** `tests/synthetic/test_edges.py` — edges below a stability threshold dropped/marked descriptive; HSIC on the joint-model **residual field** detects a nonlinear edge the linear backbone missed (reusing the moved HSIC stack); high-Jaccard concept pair typed `mechanical_overlap`; `n_eff<100` edges descriptive-only.
- **Green — pseudocode:**
  ```python
  # ldo/edges.py
  def stability_select(fit_fn, X, subsamples=50, threshold=0.7):
      counts = defaultdict(int)
      for _ in range(subsamples):
          for e in fit_fn(lattice_subsample(X)).edges: counts[e.key]+=1
      return [e for e in all_edges if counts[e.key]/subsamples >= threshold]  # multiplicity control
  def type_overlap(edge, membership_graph):
      if jaccard(codes(edge.source), codes(edge.target)) > TAU: edge.edge_type="mechanical_overlap"
  # ldo/residual_scan.py: HSIC (moved) on residual field / lagged-residual pairs, under the §6.8 nulls
  def residual_nonlinear_scan(residual_field, nulls): return hsic_scan(residual_field, nulls)  # reuses pirs/hsic
  ```
- **Done:** synthetic green; the HSIC stack reused unchanged, now residual-targeted.

#### `LDO-06` — Certification + orchestrator (+ the slice-zoo deletion)
- **MSD-III:** §III.8, X.2. **Files:** NEW `ldo/{certify,orchestrator}.py`; NEW `workflows/investigate.py`; **DEL** `pirs/{design,design_plan,design_readiness,design_matrix,field_selection,selection_plan,run_candidates,model_execution,hsic_run,hsic_ranking,hsic_report,diagnostics,residuals}.py` and the mirror workflow wrappers.
- **Red:** `tests/contract/test_certify.py` (holdout stability, path-agreement, latent/lag separability → per-edge `certification_status`; edge promoted without holdout/uncertainty → abort) + `tests/integration/test_orchestrator.py` (`run_ldo(panel, intent)` produces the full `LinkRecords` bundle in memory; **no inter-stage manifests**).
- **Green — pseudocode:**
  ```python
  # ldo/orchestrator.py  — the whole engine, replacing the slice-zoo, in memory
  def run_ldo(panel, intent):
      X, W   = assemble_field(panel.manifest, panel.state_tensor)          # field/assemble
      X      = time_extend(X, K=intent.max_lag)                            # LDO-04
      Z      = {v: to_latent_gaussian(X[v], family(v), exposure(v)) for v in vars}   # LDO-01
      S, L   = sparse_minus_lowrank(Z, W, L_W, lam1, lam_star, rank)       # LDO-02/03
      edges  = read_edges(S, L)                                            # LDO-05
      edges  = stability_select(lambda Xs: fit(Xs), X)                     # multiplicity
      edges += residual_nonlinear_scan(residual(Z, S, L), nulls)          # HSIC audit
      for e in edges: type_overlap(e, membership_graph)                   # overlap
      edges  = [certify(e) for e in edges]                                # LDO-06
      return LinkRecordBundle(edges)
  ```
- **Refactor (the payoff):** once `run_ldo` passes and `workflows/investigate.py` calls it, **delete** the ~13 PIRS micro-stage files and ~10 workflow wrappers. `pirs/` shrinks from 23 files to ~7 (the `ldo/`-adjacent utilities). Keep `T0-4` green throughout; flip the default path from old scan to `run_ldo` only when the acceptance test (Phase 5) passes.
- **Done:** orchestrator + certification green; PIRS slice-zoo deleted; baseline still green.

**Phase 4 exit:** one in-memory LDO engine producing certified link records, the slice-zoo gone, the manifest-passing I/O eliminated. The system can now *discover* lagged, latent-controlled, spatially-structured links.
---

### PHASE 5 — Causal ladder, adaptive controller, validation battery

*Goal: the science-and-trust layer. This is what turns the LDO's associational skeleton into typed causal claims, makes uncertainty steer compute, and proves the whole thing isn't finding noise. Detail here is structural; MSD-III §IV, §V.5, §IX are normative.*

#### `CAUSAL-01` — Rung-1 orientation (non-Gaussian + colliders)
- **Files:** NEW `causal/orient.py`. **Red:** on synthetic non-Gaussian data with a known direction A→B, orientation recovers A→B (not B→A); on a planted collider A→C←B, the v-structure is detected. **Green — pseudocode:**
  ```python
  # causal/orient.py
  def orient_edge(edge, X):
      if is_lagged(edge): return edge.with_direction("past->present")   # time already orients
      d = lingam_direction(X[edge.source], X[edge.target])   # non-Gaussian: residual-independence asymmetry
      return edge.with_direction(d, licensed_by="non_gaussian") if d else \
             collider_orientation(edge, X)                    # else v-structure from CI pattern
  ```
  *Auto-apply only where machine-checkable (non-Gaussianity is testable); the copula margins already characterize non-Gaussianity, so this is nearly free.*
- **Done:** synthetic orientation green; each oriented edge carries its licensing check.

#### `CAUSAL-02` — Rung-2 quasi-experimental (shock-triggered)
- **Files:** NEW `causal/quasi.py`. **Green:** interrupted-time-series where a *dated shock* exists (epidemic onset, policy date); difference-in-differences; negative-control outcomes. Triggered by detected structural breaks, not sprayed. Rung-3 (`identify.py`, do-calculus) is **expert-invoked only** — a thin API, no autonomous use. **Done:** ITS recovers a known break effect on synthetic.

#### `APC-01` — Adaptive Precision Controller (simple version first)
- **MSD-III:** §V.5. **Files:** NEW `compute/controller.py`. **Red:** `tests/contract/test_controller.py` — the engine is **anytime** (a valid answer + uncertainties at any stop point); when a decision-relevant edge's *numerical* uncertainty dominates, the controller escalates *that edge* to exact computation; a target-uncertainty run reports which quantities met the target and which are approximation-limited (typed). **Green — pseudocode:**
  ```python
  # compute/controller.py
  def adaptive_run(engine, targets, budget):
      state = engine.anytime_init()                      # current answer + uncertainties (statistical+numerical)
      while budget.remaining() and not targets.met(state):
          action = argmax(candidate_actions(state),      # {refine cell, +probes, +rank, exact-escalate, +bootstrap}
                          key=lambda a: expected_uncertainty_reduction(a, targets)/a.cost)
          if dominant_uncertainty(state, targets) == "numerical":
              action = exact_escalate(most_uncertain_decision_relevant_edge(state))  # self-correction
          state = engine.apply(action, state); budget.spend(action.cost)
      return state.report(typed_targets=targets)         # met vs approximation-limited, never silent
  ```
  *Start simple: a precision cascade with per-edge targets + automatic exact-escalation when numerical uncertainty dominates. RL-flavored scheduling is a later refinement.*
- **Done:** controller test green; uncertainty demonstrably *steers* compute.

#### `VAL-01..04` — The validation battery
- **MSD-III:** §IX. **Files:** NEW `validation/{controls,synthetic,holdout}.py`; `tests/synthetic/`, `tests/acceptance/`. **Red/Green:**
  - `VAL-01` known-positive controls — a registry of established links (Zika→microcephaly, sanitation→diarrheal, vaccination→decline); the engine MUST recover them.
  - `VAL-02` known-negative controls — unrelated pairs / negative-control outcomes; measure the **false-alarm rate**.
  - `VAL-03` synthetic ground truth — `synthetic.py` plants edges/lags/factors/spatial structure; score precision/recall; **this is the required test for separability adequacy and certification power** (§IX.2).
  - `VAL-04` temporal holdout + exact-vs-approx — fit ≤T, verify T+1; state-exact certifies national-approx.
  ```python
  # validation/synthetic.py
  def generate_planted(edges, lags, factors, spatial, S, T, seed) -> Field: ...  # exact ground truth
  def recovery_score(discovered, planted) -> dict:  # precision, recall, lag-error, factor-attribution
  ```
- **Done:** battery green; a certified result now carries "recovers known truths; false-alarm X%; synthetic accuracy Y%; holds out-of-sample."

**Phase 5 exit:** typed causal escalation, uncertainty-steered compute, and a standing battery that makes the output *science rather than assertion*.

---

### PHASE 6 — Foundational Asset Layer & lifecycle

*Goal: make denominators/graphs/registry/skeleton scope-invariant, versioned, build-once/serve-many. This is the fix for the whole class of "query scope amputated a global asset" bug.*

- `FAL-01` **asset tiers + scope-invariance + versioning + pinning** — NEW `assets/{tiers,version,store}.py`. **Red:** a state-scoped query **slices** the national population tensor, never rebuilds it (assert the tensor's inputs include all three censuses regardless of the query window); every artifact is immutable+versioned with an input manifest; every query records the foundation versions it consumed. **Green — pseudocode:**
  ```python
  # assets/tiers.py
  FOUNDATIONAL = {"substrate","population_tensor","exposures","spatial_graph","disease_graph","registry","skeleton"}
  def resolve_asset(name, query_scope):
      art = store.latest(name)                       # ALWAYS national/full-history; built by data-arrival, not query
      assert art.build_scope == "national_full_history", "foundational asset must never be built at query scope"
      return slice_view(art, query_scope)            # query slices; never rebuilds
  ```
- `FAL-02` **incremental update + snapshots** — new EstimaPOP/census/SIDRA/DATASUS arrival triggers a scoped rebuild of affected assets; the skeleton is a versioned snapshot.
- `DISCO-01` **continuous discovery scheduler** — NEW `assets/build.py` background process spending idle compute (via `APC-01`) on highest-value uncertainty reductions, incorporating new data, re-certifying. **This is the heaviest lift; stage it last.** Needs scheduler + artifact store + provenance DB (the `store.py` foundation from `FAL-01`).

**Phase 6 exit:** PegaSUS maintains a living, versioned foundational skeleton; queries slice it.

---

### PHASE 7 — Multi-resolution scanning, bounded exhaustiveness, national scale

*Goal: the coarse→fine discovery discipline, honest coverage, and full national throughput. Structural; MSD-III §VIII, §V, §III.3 normative.*

- `RES-01` **coarse→fine scanning** — NEW `ldo/resolution.py`: coarse pass (chapter/annual/regional) → candidate edges → fine local refit only on flagged variables/subsets. **Screen on heterogeneity/dispersion/max, not the aggregate mean** (§VIII.2), so cancellation/Simpson cases fire the drill-down.
  ```python
  # ldo/resolution.py
  def scan(field):
      coarse = run_ldo(field.at(resolution="coarse"))
      candidates = [e for e in coarse.edges if heterogeneity_screen(e) or e.certified]  # sensitivity, not mean
      fine = [refit_local(e, field.at(resolution="fine")) for e in candidates]
      audit = random_deep_probe(field, pruned=coarse.pruned)          # measure false-negative rate
      return merge(fine, audit, coverage_manifest(field, pruned=coarse.pruned))  # typed unsearched regions
  ```
- `EXH-01` **coverage manifest** — every unsearched region typed (`unsearched` + reason); the sparsity-of-truth assumption stated; random-audit false-negative estimate reported.
- `SCALE-01` **national tiling/streaming** — stream tiles via `storage/duckdb`+`arrow`; reuse `randnla` at scale; a run that can't fit **refuses** with `scale_exceeds_compute_envelope` (never silent subsample).

**Phase 7 exit:** national-monthly discovery with honest, quantified coverage, within the compute envelope.

---

## 4B. Cross-cutting: data-plane storage & compute remediations

*Surfaced during national-scale bring-up (not anticipated in the original phasing — recorded here so the work is tracked, not orphaned). These enforce MSD-III **§V.1** (compute envelope binds the tensor solve) and **§V.7** (storage lifecycle). They are cross-cutting data-plane items, sequenced opportunistically rather than phase-gated. **`SCALE-01` (Phase 7) is the LDO-scan scaling item and is distinct from `NAT-01` here — do not conflate the IDs.***

#### `NAT-01` — National acquisition + combine + national compile scope — ✅ LANDED
- **MSD-III:** §VI.1 (scope-invariance), §V.1. **Files:** `workflows/pipeline.py` (`_resolve_ufs`, `_acquire_national`, `_combine_national_artifacts`), `workflows/compile.py` (national `_intent_municipality_filter_cod6`/`_geo_scope_from_intent`), `geo/{uf,state_panel}.py` (`ALL_UF_SIGLAS`, `GeoScope.national`), `sidra/plan.py` (locality-capped chunking), `core/schemas.py` (`execution_scale="national"`). **Commits:** `abb1ad2` (+ earlier geo/schema/compile).
- **What:** national run acquires all 27 UFs and combines each `(system,role)` into one national artifact via streaming concat; compile materializes an all-municipality panel. Proven: national C25 mortality over 5,571 munis, `neoplasm_pancreas` σ_C = 11,974 (2021)/12,654 (2022), matches INCA. A describe-only reporter (`workflows/report/curated_cause_report.py`) narrates it. **Done:** national compile validates `OK`.

#### `STORE-01` — Storage lifecycle contracts V.7(1,2,4) — ✅ LANDED
- **MSD-III:** §V.7. **Files:** `datasus/r_scripts/fetch_process_microdatasus.R` (v4 bridge: drop `raw.rds`, gate microdatasus sidecar off, ZSTD), `datasus/subprocess.py` (cache-hit on `processed.parquet`+manifest, accepts legacy v3, prune ancillary on success), `datasus/storage_gc.py` (NEW — `gc_datasus_raw_sidecars`, `gc_sidra_raw_payloads`), `sidra/{extract,facts}.py` (slim dump, ZSTD facts), `she/reconstruction/schema.py` (manifest references arrays by length, not inline). **Commits:** `83914ac`, `6245664`, `61746bb`. **Tests:** `tests/unit/test_datasus_storage_gc.py`, updated bridge-contract tests.
- **Result:** `data/` 53 GB → 26 GB; per-chunk 3× serialization → 1×; `ReproducibilityManifest` 430 MB → 150 KB. **Column fidelity preserved** (all raw DBF columns kept — never pruned). **Done:** all 308 unit/contract tests green; national compile re-validated post-GC.

#### `STORE-02` — Lazy `scan_parquet` views (retire re-materialization) — ⬜ OPEN
- **MSD-III:** §V.7(3). **Files:** `workflows/pipeline.py` (`_combine_processed_datasus_chunks` → `scan_parquet(glob)`→normalize lazy→`sink_parquet`; drop `datasus_combined`), `she/substrate.py` (accept a Hive-partitioned scan), SIDRA facts likewise.
- **Red:** `tests/unit/test_lazy_combine.py` — a multi-chunk fixture compiles with **no `datasus_combined` copy on disk**; the national field set + validation are byte-for-byte what eager combine produced.
- **Green:** replace eager `pl.read_parquet`→`concat` with `pl.scan_parquet`→lazy normalize→single streamed sink; Hive-partition `normalized/datasus/<sys>/uf=…/year=…`; national "combine" becomes a lazy scan the substrate reads. **Done:** intermediate copies gone; national compile still `OK`.

#### `POP-02` — Population-tensor compute contract (§V.1) — 🟢 M1/M3/M4/M5/M7 landed; GPU + parallel + M6 open; see `FAL-POP` for the deeper scope issue
- **MSD-III:** §V.1(a,b,c), §II.4. **Depends on / extends:** `POP-01`. **Files:** `she/reconstruction/schema.py` (`PopulationTensorProblem` fields tuple→`np.ndarray`, None→nan + companion mask), `she/reconstruction/{projected_gradient,loss,sparse_admm,sparse_block_coordinate,state_space}.py` (read arrays directly; no re-copy; float32 path), `sidra/population_cube/build.py` (locality-blocked solve loop + streamed tensor sink), `she/reconstruction/torch_kernels.py` (wire into block solve), `sidra/population_cube/migration.py` (`hop_distances` bounded — **done**, commit `6bb12a7`).
- **Problem (measured):** the problem is held as ~10 `n_cells` Python **float tuples** (~32 B/elem) then copied to numpy at solve — 2-yr national (16.7M cells) ≈ 16 GB; full 25-yr (209M cells) ≈ 50–67 GB → **OOM on 32 GB**. GPU never used (`torch_kernels.py` has **zero callers**; `config/compute.yaml enabled_for: population_tensor_kernels` is aspirational).
- **Red:** `tests/contract/test_population_memory.py` — constructing a 209M-cell problem stays `< a few GB` (arrays, not tuples) and the 2-yr national denominators match the pre-change run within f32 tol (`test_population_denominator_parity`); `tests/unit/test_population_blocked_solve.py` — blocked solve peak-RSS is `O(block)` and equals the whole-national solve within tol; `tests/contract/test_solver_device.py` — telemetry records `device∈{cuda,cpu}` and CUDA==CPU within f32 tol, with graceful `CUDA_VISIBLE_DEVICES=""` fallback.
- **Green — landed:** **(M1)** `PopulationTensorProblem` stores float64 numpy arrays (None→NaN sentinel); `loss.py`/solvers read them directly (removed the per-iteration tuple→array rebuild); `build.py` constructs with arrays — commit `POP-02 M1`. **(M3)** `solve_population_tensor_blocked` solves in ~2M-cell locality-blocks (exact — the objective is locality-separable when `migration_totals is None`; each block takes the fast dense path); peak memory `O(block)` — commit `POP-02 M3`. **(M4)** bounded-BFS migration — commit `6bb12a7`. **(G3)** deleted the orphaned `she/reconstruction/torch_kernels.py` and removed `population_tensor_kernels` from `compute.yaml` (§V.1c: no aspirational flag over unwired code). Tests: `test_population_tensor_memory.py` (numpy storage + blocked==whole exactness). **(M5)** vectorized tensor emission — the output loop built one dict per cell (~13 GB for 16.7M national cells); replaced with `np.repeat`/`tile` label columns + the flat solution arrays (~1 GB) — commit `POP-02 M5`. **(M7 — the real speed fix, from an adversarial review of the GPU question)** `PopulationLossEvaluation`/`PopulationOptimizationResult` now hold **numpy arrays, not `tuple[float,...]`**: `tuple(float(x) for x in gradient)` was **~99% of a loss eval** (110 ms→22 ms at block scale, run 2× per SPG iteration) and the result tuple was ~1 GB at national scale — *this*, not the dense math, was why the solve read slow. **Verified:** 2-yr national tensor builds `OK`; **census-exact** (2022 total 203,080,756 = IBGE; marginal closure to 3.5e-14; race/sex/age match the census).

  **Remaining POP-02 items (re-prioritized after M7 + the multi-agent audit):**
  - **(M-parallel)** the blocked solve loops locality-blocks **sequentially** (`solvers.py:134`) though proven independent — parallelize across a process pool. *medium/medium.*
  - **(M6)** input-construction memory (~14.5 GB) — `interpolate_census_composition` + strata→anchor build materialize `O(n_cells)` Python lists/dicts. Replace with numpy/polars-native construction; then block the WHOLE build-solve-emit per locality so input prep is `O(block)`. *large/medium — bottleneck for full 2000–2024.*
  - **(GPU — now JUSTIFIED, not "marginal")** with M7 done the dense loss/gradient algebra dominates; a measured microbenchmark shows **~15× on GPU** at block scale, VRAM-trivial (5.4M-cell block = 236 MB). Port `loss.py` + SPG loop to torch (reuse `stdfm/torch_solver.py` dispatch), **f32-bulk / f64-reduction (§V.4)** — verified to reproduce the objective exactly — CPU fallback + device telemetry, re-add a *wired* `population_tensor_solve` flag. *Correcting the record: deleting the no-op `torch_kernels.py` stub was right, but "GPU is marginal" was wrong — it was masked by the M7 tuple hotspot.* Gate: measure a national wall-clock baseline post-M6 first. *medium/medium.*
  - **(M2)** float32 storage deferred — applied inside the GPU port's §V.4 mixed precision.

  **Done (this scope):** M1/M3/M4/M5/M7 landed; 2-yr national builds within the envelope, census-exact, 313 tests green.

#### `FAL-POP` — Population tensor as a scope-invariant foundational asset — 🟢 DATA PIPELINE LANDED; single-vintage / projection / versioning / validation open
- **MSD-III:** §VI.1 ("Building a scope-invariant asset at reduced scope is a **correctness bug**"), §II.4 (all three censuses MUST be ingested regardless of window; a query *slices*; **single-vintage closure**, **native projection beyond anchors**, **anchor-distance uncertainty** — the FAL-POP contract added to §II.4). **Files:** `workflows/{pipeline,compile}.py`, `sidra/population_cube/{build,census_2000}.py`, `she/reconstruction/*` → TDD Phase-6 `FAL-01`/`assets/`.
- **The original violation (now fixed at the data layer):** the run *rebuilt the tensor per query at the intent's year window* — the first validated national tensor covered only 2021–2022. The data pipeline below makes the tensor ingest all three census anchors and build over the full history regardless of window. `interpolate_census_composition` is *correctly* the §II.4 Layer-1 warm-start (not "naive interpolation"); the bug was the *scope*.
- **Target:** build the population tensor ONCE as a versioned national + full-history foundational asset; queries slice it. **User decisions (locked):** cohort-project between censuses (§II.4 Layer-2 process solver) · single-vintage anchoring (census-anchored, not the 6579 series that causes the 2021→2022 jump) · **native forward/backward projection beyond official anchors** (process-model extrapolation, `fragile`-typed, uncertainty grows with anchor distance — the decisive current-year-epidemiology capability).
- **Build sequence (concrete):**
  - **#1 — census-scope invariance** — ✅ LANDED (`f715dd6`): `_all_census_periods` ingests every census the strata/closure tables declare regardless of the query window (both 9606 strata and 6579/9606 closure totals); `test_population_scope_invariance.py`.
  - **#2 — register + acquire table 2093 (the 2000-census strata)** — ✅ LANDED: `pipeline.py` `SIDRA_CENSUS_2000_STRATA_*` constants + `_census_2000_strata_classifications` (race clsf 86, sex clsf 2, age = 13-bracket clean partition of clsf 58, situation clsf 1 = Total) + `_acquire_sidra_census_2000_strata` wired into `_acquire_national` and `run_live_pipeline`.
  - **#3 — CTR age disaggregation** — ✅ LANDED: `sidra/population_cube/census_2000.py` selects the **clean non-overlapping 13-bracket partition** (`CLEAN_AGE_BRACKETS_2093`; roll-ups 0-14/65+/70+/80+ excluded, §II.4 "never summed") and disaggregates each bracket to single-year via the 2010 shape, closure-preserving; `test_census_2000_disaggregation.py` (6 tests).
  - **#4 — full-history cohort-projected build** — ✅ LANDED (`f8cef86`): `_census_2000_records_from_facts` injects the 2000 anchor; the build spans the full census range with all three anchors; the §II.4 Layer-2 solver cohort-ages between them. Composed with `POP-02` (numpy/blocked/vectorized-emit).
  - **`FAL-POP-RECON` — census undeclared-race reconciliation** — ⬜ OPEN (§II.5, **the prime-directive fix**): 2093's clean partition fetched only the 5 declared races, silently dropping "Sem declaração" (2781) = **26,775 people / 0.95% of AL 2000** → the 2000 anchor summed short (2,801,085 vs enumerated 2,827,856) and was NOT computationally equal to the 2010/2022 direct-total anchors. Confirmed the entire residual is undeclared *race* (no unknown-age; the 13 brackets cover everyone). **Fix:** (a) add race **2781** to `_census_2000_strata_classifications`; (b) in `census_2000.py`, reallocate the undeclared mass per `(locality, age_bracket, sex)` across the 5 declared races by the **local declared composition** `π_local[j|age,sex]`, hierarchical fallback `(sex)→(locality)→state` where a cell is all-undeclared; NEVER drop it. Result: 2000 strata sum to the enumerated total, census-exact zero deviation. Imputation uncertainty propagated into `cell_uncertainty`. **Generalize:** apply the same reconciliation to any census anchor with an undeclared bin (check 9606 2010/2022).
  - **`FAL-POP-SV` — single-vintage re-anchoring** — ⬜ OPEN (**the anti-discontinuity contract, §II.4**): intercensal closure totals `E_{s,t}` currently still come from SIDRA 6579, which mixes IBGE's projection vintage with the census (the 2021→2022 ~5% jump). Replace with **census-anchored cohort-component closure**: project the total between census enumerations (2000→2010→2022) and *scale strata to that total*; 6579/EstimaPOP becomes a validation cross-check + recency signal, not the anchor. **File:** `sidra/population_cube` closure-total assembly (`load_combined_population_totals_frame`). This is the operative fix — full-range alone does **not** remove the jump; the vintage swap does.
  - **`FAL-POP-RECON` — census undeclared-race reconciliation** — ✅ LANDED (`ae04ff5`, §II.5): 2093 fetch now includes race 2781 ("Sem declaração"); `census_2000.reconcile_undeclared_race` reallocates it into the declared races per `(locality, sex, age_bracket)` by local composition (hierarchical fallback), never dropped. Verified: AL 2000 anchor = **2,827,870 = enumerated total** (was 2,801,085 declared-only). 3 tests.
  - **`FAL-POP-AMC` — temporal municipality-boundary harmonization (CARVE)** — 🟢 MECHANISM LANDED (`080cfee`), national genealogy data OPEN. A municipality installed after a census (AL: 2703759 Jequiá da Praia, Lei 5.675/1995 installed ~2001, absent from Censo 2000) has its residents counted inside its parents; the build interpolated its 2000 population *on top* → double-count → +0.5% AL 2000 overshoot. **User chose CARVE.** `census_2000.carve_pre_census_children` + `config/registries/spatial/municipality_genealogy.yaml`: estimate the child's census-year pop `X` (earliest-census pop × state census-ratio), subtract `X` from the **authoritative** parents (split by size), assign to the child — parents lose `X`, child gains `X`, census-year total unchanged. **Verified AL:** 2000 = **2,827,870 exact** (was 2,842,552); Jequiá 10,901→12,029, parents (São Miguel dos Campos 270860 + Coruripe 270230) shed their share, all trajectories sane. Parents are AUTHORITATIVE (IBGE territorial evolution), never inferred — confirmed necessary: both parents *grew* 2000→2010 despite losing Jequiá, so no drop-signal. **REMAINING (national task):** onboard the full IBGE "Evolução da Divisão Territorial" succession table into the genealogy registry (the registry is seeded only for AL/Jequiá; other post-census municipalities nationally are not yet carved). `census_boundary_amc_carved` telemetry.
  - **`FAL-POP-PROJ` — native projection beyond official anchors** — ⬜ OPEN (§II.4) — *the "hairy math" item; design below, build after VER.*

    **Key architectural insight (de-risks the whole item):** the Layer-2 process solver ALREADY runs cohort-component dynamics — it cohort-ages the age×sex(×race) state between two censuses, constrained by *both* endpoints. Projection is the **same operators run past the last anchor with no future census to hit** — a free-running forward integration instead of a two-point boundary-value solve. So PROJ reuses the solver's aging/birth/death/migration machinery; it does not introduce a new model. This is why it fits §II.4 rather than fighting it.

    **The cohort-component step (one projected year `t → t+1`), per municipality × sex(×race):**
    1. **Age.** Shift every single-year cohort up one (`a → a+1`); the terminal open age-group accumulates (`ω` absorbs `ω-1` and its own survivors).
    2. **Die.** Apply age-sex(-race)-specific survival `s_{a} = 1 - m_{a}` from the SIM death-rate field the tensor already carries (`_sim_death_priors`); deaths are `pop_{a}·m_{a}`.
    3. **Be born.** New age-0 cohort `= Σ_a (female pop_{a} · ASFR_{a})`, age-specific fertility from SINASC births / female population (already computed for the birth prior); split by sex via the ~1.05 SRB and by race via maternal-race composition (RaceBridge, §II.5).
    4. **Migrate.** Net migration by age-sex from the migration-residual field (`_migration_residual_totals`) where informative; **damped toward zero as the horizon grows** (migration is the least predictable component and the residual is unidentified past the last civil-registry year) — the honest default, not a silent zero.

    **The uncertainty envelope (the prime-directive discipline — the part that must not be hand-waved):** projection variance ACCUMULATES per step, `σ²(h) = σ²_anchor + Σ_{i=1}^{h} (component-rate variance)_i`, dominated by the migration term and growing monotonically in the horizon `h` (years from the nearest anchor). Type `state = fragile` for `1 ≤ h ≤ H_soft` and `state = unreliable` (dashboard-blocked) beyond `H_soft` (~5 yr default, configurable); the per-cell uncertainty widens every rate computed on a projected denominator (§V.6 propagation). Record `projection_horizon` + `H_soft` in the asset manifest (§VI.2).

    **Calibration is free and built-in:** running the forward CCPM from census A and comparing to census B *is* the intercensal balancing equation — the discrepancy is the coverage/migration residual, which both calibrates the projection rates AND is exactly what the two-layer solver already computes between censuses. So PROJ's rates are validated on 2010→2022 before being trusted to run past 2022. Backward projection (pre-2000) is the same operators inverted (un-age, restore deaths, remove births), lower priority.

    *Care item (unchanged):* no fabricated structure — only forward integration of the process with an honest, monotonically-growing uncertainty envelope. Backward extrapolation of the *closure total* is already handled crudely by `FAL-POP-SV`'s geometric extrapolation; PROJ supersedes it with the full age-structured projection and its uncertainty typing.
  - **`FAL-POP-PROJ` — native projection uncertainty envelope** — ✅ LANDED (`a34cdc6`): the projection VALUES already come from the solver's cohort-component dynamics (loss.py aging/birth/death) on the SV-extrapolated closure past the last census; PROJ adds the prime-directive discipline — `_classify_projection_years` classifies each year (census/interpolated/projected_forward/projected_backward), the tensor parquet gains `anchor_class`/`projection_horizon`/`cell_state`/`cell_uncertainty` columns with uncertainty `sqrt(base²+h·per_year²)` growing per horizon year (fragile 1≤h≤5, unreliable beyond), and the build manifest records `anchored_range`/`max_projection_horizon`/`projected_periods`. Migration damping past the last civil-registry year is a documented refinement (the uncertainty envelope already carries the honesty). 5 tests.
  - **`FAL-POP-VER` — version + store + slice-on-query** — ✅ LANDED (`1ac754b`, Phase-6 `assets/`): `PersistentAssetStore` (on-disk JSON index → build-once survives across runs); `assets/foundation.py` `population_tensor_input_identity` (build-once key over input hashes+mode+code version), `resolve_or_build_population_tensor` (reuse-or-build + version + slice), `slice_population_tensor` (lazy scan/sink to the window); `compile.py` resolves-or-builds via the lifecycle — a NATIONAL run stores the immutable foundational version + slices, a reduced-scope build stays run-local and is never mislabeled foundational (so `resolve_asset`'s scope guard holds). Query pins `asset_version`. 6 tests.
  - **`FAL-POP-VAL` — demographic-sanity validation** — 🟢 CLOSURE PRONG NATIONALLY VALIDATED (SV: −4.80%→+0.58%, all 5570 munis); RECONSTRUCTION PRONG at feasible (state) scale (§V.6 state-exact certifies national-approximate) — the full **national** full-history reconstruction solve is **memory-gated on `POP-02` M6** (per-block build-solve-emit; the full 5-race national ~209M-cell input construction exceeds 32 GB until M6 makes peak O(block)). Audit: totals vs *each* census (2000/2010/2022), closure residual, age-pyramid, sex/race per census year, single-vintage trend continuity.
- **Open (report to user as built):** joint age×sex×race denominator — the tensor emits only 1-D marginals, so cross-classified rates need a documented IPF/independence reconstruction (validation thread caveat).

---

## 5. What to do now — the concrete near-term checklist

The first stretch of work, in exact order, with the most operational detail. This is where development literally resumes.

**Week-scale, immediately actionable:**

1. **Stand up the test harness** (§0.3). Create `tests/` layout + `tests/fixtures/datasus_multistate.py` (a tiny synthetic fixture: a handful of records per system, one per §2.3 state). *Nothing else can be TDD'd without this.*
2. **`T0-1` source-reality contracts.** Write the seven assertions; run; fix only failures. This *pins* the S0 remediations and tells you precisely what (if anything) in SIH/CNES declarative routing is incomplete.
3. **`T0-2`, `T0-3`, `T0-4`** in parallel — pin EFG legality/state-tensor, RaceBridge numerics, and the inference baseline. Tag `pre-ldo-baseline`.
4. **`REFACTOR-01`** — collapse the four vectorized normalizers into the declarative path behind the property test. This is the one genuinely-open compliance finding (`XCUT-02`) and it's low-risk once `T0-1` is green.

**Then, the first architectural build (highest leverage):**

5. **`REG-07`** — the unified registry. Do it *source-by-source* (SIM first: translate its `source_fields` entry into the `kind`-tagged schema, get the validator green, repeat). This unblocks the ontology-driven treatment everything else assumes.
6. **`SPG-01`** — grow `geo/adjacency.py` into `SpatialWeightGraph`; rewire the four consumers. Small, clarifying, unblocks the field engine.
7. **`POP-01`** — the two-layer tensor. This *closes the loop* on the whole population-tensor saga: implement `layer1_prior_mean`, warm-start the existing solver, add the data-poor short-circuit, register table 2093. Guard `loss.py` with golden tests first.

**The decisive proof-of-life (do as soon as Phases 1–3 are green):**

8. **`LDO-04` on synthetic data** — before wiring the full engine to real data, run the planted-lag test (`test_lags.py`). If a monthly synthetic field with a planted lag-7 A→B edge (plus a shared wave) is recovered with the wave correctly attributed and *no variable hand-named*, the core thesis of MSD-III is demonstrated. This single green test is the earliest moment PegaSUS provably does what it was redesigned to do.

**The milestone that ends the program:**

9. **The acceptance test** (`tests/acceptance/test_zika_microcephaly.py`) — a real monthly Alagoas run, no forced selectors, recovering the lagged arbovirus→microcephaly edge peaked at 6–9 months, net of the epidemic-wave factor, with stability and uncertainty. Flip the default path from the old scan to `run_ldo` when this passes.

---

## Appendix A — Work-item dependency graph

```
Phase 0:  T0-1 ─┬─ T0-2 ── T0-3 ── T0-4 ──┬── REFACTOR-01
                └──────────────────────────┘         │
Phase 1:  REG-07 ── SPG-01/02/03 ── PANEL-01 ── SCOPE-01     (REG-07 gates SPG; PANEL needs SPG)
Phase 2:  EFG-OUT-01 ─┬─ POP-01 ─┬─ RACE-01.. ─┬─ DIS-01/02/03
                       │ (needs 2093)           │ (needs REG-07)
Phase 3:  LF-01 ── MR-01              (LF needs PANEL-01+SPG-01; MR needs LF+SPG+DIS-03)
Phase 4:  LDO-00 → LDO-01 → LDO-02 → LDO-03 → LDO-04 → LDO-05 → LDO-06
          (LDO-02 needs SPG+MR; LDO-03 needs stdfm; LDO-01 needs EFG-OUT-01)
Phase 5:  CAUSAL-01/02 · APC-01 · VAL-01..04     (all need LDO-06)
Phase 6:  FAL-01 → FAL-02 → DISCO-01             (needs the assets to build; skeleton from LDO)
Phase 7:  RES-01 · EXH-01 · SCALE-01             (need LDO-06 + APC-01 + FAL-01)
```
Law: never start an item whose dependencies aren't green. The critical path to the acceptance test is `T0 → REG-07 → SPG-01 → PANEL-01 → LF-01 → MR-01 → LDO-00..06 → VAL → acceptance`.

## Appendix B — Key pseudocode reference (the three that matter most)

**The LDO fit (the heart):**
```python
def run_ldo(panel, intent):
    X, W = assemble_field(panel.manifest, panel.state_tensor)        # sparse/dense unified (LF-01)
    X    = time_extend(X, K=intent.max_lag)                          # lags (LDO-04)
    Z    = {v: to_latent_gaussian(X[v], family(v), exposure(v)) for v in vars}   # copula (LDO-01)
    S, L = sparse_minus_lowrank(Z, W, L_W, lam1, lam_star, rank)     # direct − shared (LDO-02/03)
    edges = stability_select(lambda Xs: read_edges(fit(Xs)), X)      # multiplicity (LDO-05)
    edges += residual_nonlinear_scan(residual(Z, S, L), nulls)      # HSIC audit
    for e in edges: type_overlap(e, membership_graph); certify(e)   # overlap + gate (LDO-05/06)
    return LinkRecordBundle(edges)
```

**The two-layer denominator (closes the population saga):**
```python
def solve_population(intent, censuses, totals, flows):
    x0 = interpolate_ilr(census_composition(censuses)) * totals      # Layer 1: prior mean + warm start
    if not has_flow_signal(flows) and n_in_window(censuses) < 2:
        return x0                                                    # data-poor: Layer 1 IS the optimum
    return msd_six_term_solver(x0, censuses, totals, flows)          # Layer 2: warm-started process solver
```

**The scope-invariance guard (kills the whole bug class):**
```python
def resolve_asset(name, query_scope):
    art = store.latest(name)
    assert art.build_scope == "national_full_history"               # never built at query scope
    return slice_view(art, query_scope)                             # a query slices, never rebuilds
```

## Appendix C — Effort shape (relative, not calendar)

- **Small / mechanical:** `T0-*`, `REFACTOR-01`, `SPG-01`, `SCOPE-01`, `LDO-00/01`, `CAUSAL-01`. (Pin, wrap, translate.)
- **Medium / real design:** `REG-07` (translation volume), `PANEL-01`, `EFG-OUT-01`, `POP-01`, `RACE-01..`, `DIS-01/02/03`, `LDO-05`, `APC-01`, `VAL-01..04`, `RES-01`.
- **Large / the core builds:** `LF-01`+`MR-01` (the field engine), `LDO-02/03/04` (the estimator), `LDO-06` (orchestrator + slice-zoo deletion), `FAL-*`+`DISCO-01` (the asset/lifecycle layer).
- **Research-gated (not pure engineering):** the literature-informed race matrices (`RACE`), the known-positive control curation (`VAL-01`), and — empirically, not by code — separability adequacy and certification power (settled *by* `VAL-03`, not before).

---

*This is the build order. Start at the near-term checklist (§5): stand up `tests/`, pin the remediations, de-duplicate, then the unified registry and the two-layer tensor — and drive toward the planted-lag synthetic test (`LDO-04`) as the first proof that PegaSUS discovers what it was redesigned to discover. Every step happens behind a green contract test; the slice-zoo is strangled, never big-banged; the acceptance test (§5.9) is done-ness. Keep MSD-III open beside this document — it is the "why" for every "what" here.*
