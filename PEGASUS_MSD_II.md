# PegaSUS — Secondary MSD & Implementation TDD (MSD-II)

> **One document to converge the project.** It states what PegaSUS *is becoming*, reviews exactly where the code *is now*, specifies the new normative contracts, and gives test-first, step-by-step build instructions detailed enough to be executed by a small model that cannot infer the sequence on its own.

---

## 0. Front matter — lineage, authority, and how to use this document

### 0.1 Document lineage (read this first; it is binding)

This document has **two parents it does not replace** and **three predecessors it does replace**.

**Direct textual parents (still authoritative; MSD-II extends, never contradicts them):**
- **`MSD.md`** — the Master System Document. The 12 sections of the MSD remain the constitution. Every contract in Part 4 of this document is an *addition* (`§II.*`) that sits beside the MSD, never an overwrite. Where MSD-II appears to change a behavior (e.g. PIRS), it does so by **adding** a new section and **amending** a named MSD section via the explicit amendment list in §4.12 — the MSD text is edited there, not here.
- **`PEGASUS_COMPLIANCE_AND_REMEDIATION.md`** — the compliance report. Its finding IDs (`SHE-NORM-01`, `PIRS-SPAT-01`, …) remain the canonical handles for correctness work. Part 2 of this document is a **progress review of every one of those findings**, because MSD-II's new work must be built on a codebase that is already MSD-compliant.

**Predecessors fully superseded by this document (do not consult them for new work; their content is absorbed here):**
- `PEGASUS_ARCHITECTURE_ADDENDUM.md` (the four pillars + `ARCH-*` items).
- `PEGASUS_STATISTICAL_ENGINE_STATE_AND_GAP_ANALYSIS.md` (the statistical-ceiling diagnosis).
- `PEGASUS_VISION_ASSESSMENT_AND_NEW_PIRS_STRATEGY.md` (the Lattice Dependency Operator design).

Everything those three contained is restated, reconciled, and extended here. `ARCH-*` IDs are preserved as cross-references in Appendix A so existing commit trails remain traceable.

### 0.2 Authority order (when documents disagree)

1. `MSD.md` (constitution) — wins on any locked invariant in MSD §12.
2. **This document (MSD-II)** — wins on everything it newly specifies (the joint inference engine, the platform pillars, the build sequence).
3. `PEGASUS_COMPLIANCE_AND_REMEDIATION.md` — wins on correctness details not yet folded into MSD-II.
4. Code comments — never authoritative.

If MSD-II requires a behavior the MSD currently forbids (it does, for PIRS), the MSD amendment in §4.12 is the instrument that resolves it; apply that amendment to `MSD.md` **before** building the dependent code.

### 0.3 Audience and the "small-model contract"

Parts of this will be executed by smaller/less-capable coding agents. Therefore:

- **Every work item in Part 5 is written as Red → Green → Refactor.** Write the failing test first (Red), make it pass with the listed steps (Green), then consolidate (Refactor). Do not skip Red.
- **Do exactly what the steps say, in order.** Do not re-plan. If a step references a file or function that does not exist, that is a signal you are out of sequence — stop and check the dependency column.
- **One work item = one pull request.** Cite the work-item ID and the parent finding/ARCH ID in the commit.
- **Never delete a contract test to make a build pass.** If a contract test fails, the code is wrong, not the test.
- **If uncertain, prefer refusing over silent downgrade.** The MSD's prime directive (§12) is: never replace mathematical legality with convenience, missingness with silence, source structure with scalar fiction, or measurement-process uncertainty with false precision. This is the tie-breaker for every ambiguous decision.

### 0.4 Severity and status vocabulary

| Token | Meaning |
|---|---|
| **S0–S3** | Compliance severities, inherited from the compliance report (S0 contract breach … S3 hygiene). |
| **Remediated** | The mandated mechanism is present in the current `/src`; correctness still to be locked by a contract test. |
| **Partial** | Mechanism present but incomplete or coexisting with the defect it was meant to remove. |
| **Open** | Not yet addressed. |
| **C0/C1/C2** | Capability-gap classes for the *new* goals (C0 unrepresentable today; C1 vestigial; C2 shallow). |
| **MII-*** | New work-item IDs introduced by this document. |

---

## 1. The synthesis thesis (where this is all going)

PegaSUS has two halves. The **data half** — substrate harmonization, the typed field graph, reconstruction, spatial structure, onboarding — has been driven to a high and now largely *compliant* state (Part 2 proves this). The **inference half** — PIRS — remains, by the MSD's own locked design, a *single-outcome, contemporaneous, pairwise residual scanner*. The project's actual goal, stated by its architect, is the opposite of a pairwise scanner:

> **Automate epidemiology**: fit *one* systematic model over the spatiotemporal lattice (geo × time, multi-resolution) that **discovers all kinds of links** — contemporaneous, lagged, spatial, conditional, nonlinear — as queries against a single fitted object, with calibrated uncertainty, runnable on a personal computer (RTX 4050 / 32 GB).

MSD-II converges the whole program onto that goal with a single reframing:

> **The data half is a refinery that produces one clean multivariate spatiotemporal field; the inference half estimates one structured dependency operator over that field; every epidemiological "link" is a structured query against that operator.** The four platform pillars feed the field; the new engine (the **Lattice Dependency Operator**, LDO) consumes it; the existing GLM/HSIC/ST-DFM/spatial machinery become *components* of the LDO rather than a pairwise pipeline.

Concretely, MSD-II asserts and then specifies:
1. **The platform pillars are kept** (Registry Platform, Two Lifecycles + CommonPanel, the CTR reconstruction kernel, the SpatialWeightGraph) — they are how the clean field is built. (Part 4 §II.1–§II.5.)
2. **PIRS is re-founded** as the LDO: a sparse-plus-low-rank structured precision estimator over the time-extended, copula-margined, GMRF-spatial field, fit once, queried for all links. ST-DFM becomes its low-rank layer; the SpatialWeightGraph its spatial prior; the GLM families its copula margins; HSIC its residual nonlinear-edge detector; CTR certification its edge-certification gate. (Part 4 §II.6.)
3. **The output object changes** from the pairwise hypothesis row (MSD §8.3) to a typed **link record** carrying lag, direction, shape, spatial heterogeneity, latent-confounding, stability, and certification. (Part 4 §II.8.)
4. **Resolution becomes a first-class axis** — coarse for global discovery, fine for surgical refinement — which is simultaneously the scientific multi-resolution story and the laptop-scale compute strategy. (Part 4 §II.7.)
5. **The codebase is consolidated** from its development-slice fragmentation into domain modules, so the new engine is built on coherent foundations rather than a manifest-passing maze. (Part 3 + Part 5 refactor items.)

The acceptance test for the entire program is one regression (Appendix D): **a monthly Alagoas run with no `force_selectors` recovers a lagged arbovirus→microcephaly edge peaked at 6–9 months, net of the epidemic-wave latent factor, with a stability score and propagated uncertainty.** Today that is impossible by construction; when it passes, PegaSUS is the system its architect described.

---

## 2. State of the code — compliance & remediation progress review

**Headline:** a substantial remediation pass has landed since the compliance report was written. **Nearly every S0 and S1 finding now has an implemented mechanism in `/src`.** The premise of MSD-II — "build the new directions on a compliant codebase" — is therefore largely satisfied; what remains is (a) locking the remediations with contract tests, (b) finishing a few partials, and (c) removing the duplication the remediations left behind. Each finding below was re-verified against the current `/src` dump.

### 2.1 Per-finding status table

| Finding | Sev | Status (verified) | Evidence in current `/src` | Residual action |
|---|---|---|---|---|
| `SHE-DEC-01` | none | **Remediated** | `datasus/decoders.py` (550 ln) intact; `filter_cnpj` (l.5559), decoders dispatched by name | golden-value tests |
| `SHE-NORM-01` | S0 | **Partial** | declarative path wired: `normalize_sim_do_record`/`normalize_sinasc_record` delegate to `_registry_normalize_*` (declarative_normalize.py, l.6347/8178); nulled-column greps now empty | finish SIH/CNES declarative routing; **remove vectorized duplicates** (see §3) |
| `SHE-CNES-01` | S0 | **Remediated** | `capacity_total_observed` no longer present anywhere | contract test: no generic bed total in CNES output/registry |
| `SHE-CNES-02` | S0 | **Remediated** | `filter_cnpj` applied to `CPF_CNPJ`/`CNPJ_MAN` (l.4512–13, 4639–40) and SIH `CGC_HOSP` (l.7474) | test: zero-CNPJ → null link + observer share |
| `SHE-CNES-03` | S2 | **Verify** | confirm null≠0 preserved unless registry declares default | test on null capacity cell |
| `SHE-SINASC-01` | S0 | **Remediated** | `low_birth_weight_flag`/`prematurity_flag`/… no longer emitted by SHE | confirm primitives emitted; thresholds moved to EFG seeds |
| `SHE-POP-01` | none | **Remediated** | `she/population/loss.py` intact | golden-value tests (protect from solver refactors) |
| `SHE-POP-02` | S1 | **Remediated** | RTS smoother `solve_population_state_space_smoother`; ADMM `solve_population_admm` (`sparse_admm_split_projection`, l.30114); `sim_informed_sparse_admm_v1` active | convergence test on multi-UF synthetic; confirm `primal-dual_sparse` either built or removed from MSD §2.8.12 |
| `SIDRA-CTX-01` | S1 | **Partial** | `compile.py` profile-gates context; `stdfm_executed_count>0→success`; regime/stitch/projection/pushforward modules exist | confirm end-to-end context→EFG node materialization; acquisition still shows `skipped` in default path |
| `SIDRA-CTX-02` | S1 | **Verify** | high-card axis bound — confirm invoked on legality path before materialization | test: oversized SIDRA axis product → mandatory pushforward or `Δ_axes=0` |
| `EFG-LEG-01` | none | **Remediated** | `efg/legality.py` 8-term predicate intact | keep; add high-card term test (pairs with SIDRA-CTX-02) |
| `EFG-DECL-01` | none | **Remediated** | declaration gate blocks illegal race division | — |
| `EFG-DECL-02` | S2 | **Verify** | confirm race-stratified-but-undeclared operand now fails (not permissive) | test: missing `race_axis_type` → `race_axis_declaration_unverifiable` |
| `EFG-EXEC-01` | none | **Remediated** | `efg/executor.py` (1127 ln) real materialization, cross-join forbidden | keep |
| `EFG-CMP-01` | none | **Remediated** | `efg/empirical_compression.py` ρ≥0.98 on materialized only | keep |
| `EFG-Q-01` | S2 | **Remediated** | `moran_i`/`temporal_roughness`/`spatial_entropy` computed (l.8891–8967) + dataclass fields | confirm in `REQUIRED_Q_TENSOR_COLUMNS`; test `n_eff` reflects Moran correction |
| `EFG-REG-01` | S1 | **Partial** | depends on SHE-NORM-01 completeness | integration test: each `V_core` materializes or emits typed `FailedBranch` |
| `RACE-01` | S2 | **Remediated** | `Bridge_R_localPi_posteriorC`, `bridge_mode=localPi_posterior`, posterior-C path (l.10141–10224) | confirm CV is bootstrap-derived; partial-ID inf/sup bounds; lock with tests |
| `PIRS-HSIC-01` | none | **Remediated** | `pirs/hsic.py`,`nulls.py`,`fdr.py`,`crossfit.py` faithful | keep — becomes LDO residual layer |
| `PIRS-FAM-01` | S1 | **Remediated** | `compute/glm.py` 11 family/fit paths incl. hurdle/lognormal/two-part/student-t/softmax/beta-binomial | keep — becomes LDO copula margins |
| `PIRS-SPAT-01` | S1 | **Remediated** | `pirs/spatial.py` §6.3 selector + `fit_glm_penalized` real ICAR/GMRF penalty | keep — folds into SpatialWeightGraph + LDO prior |
| `PIRS-REG-01` | S3 | **Remediated** | `model_registry.yaml` v3.0 reconciled with `glm.py` | — |
| `OUT-01` | none | **Remediated** | `output/validate.py` 17-key contract | extend (not break) for link record |
| `OUT-02` | S3 | **Verify** | `output_schema.yaml` scaffold — confirm populated or removed | — |
| `OUT-03` | S2 | **Remediated** | `PROFILE_NONEMPTY`, `_append_empty_by_profile_warnings`, `empty_by_profile` warnings present | extend to DataScope×ExecutionStage (§II.5) |
| `REG-DEAD-01` | S3 | **Partial** | validator now tolerant of dual names (`REGISTRY_FILES` tuples list old+new) | delete true orphans; single canonical name per registry |
| `REG-VALIDATOR-01` | S3 | **Partial** | `validate_registry_tree` still referenced; needs the executable-authority validator (§II.1) | replace per `MII-REG-07` |
| `REG-STALE-LIST-01` | S3 | **Verify** | stale filename lists | hygiene sweep |
| `REG-SCAFFOLD-01` | S3 | **Verify** | scaffold `output_schema.yaml` | with OUT-02 |
| `REG-EOL-01` | S3 | **Verify** | EOL/encoding hygiene | sweep |
| `XCUT-01` | S1 | **Improving** | two-tier gap closing; production tier now honors most contracts | add the "contract→production" tests this report lists |
| `XCUT-02` | S3 | **Open** | **vectorized normalizers still coexist with declarative path** (normalize.py/sih_/sinasc_/cnes_ all present) | de-duplicate (`MII-REFACTOR-01`) |
| `XCUT-03` | S0-root | **Remediated** | SINASC/CNES layering violations removed | review-checklist lint |
| `XCUT-04` | S3 | **Partial** | metadata contradictions reduced; registry validator pending | with REG cluster |
| `XCUT-05` | S2 | **Improving** | `empty_by_profile` + typed branches reduce silent absence | complete via panel manifest (§II.2) |

### 2.2 Reading of the table

- **The S0 core is essentially closed.** Every S0 finding (`SHE-NORM-01`, `SHE-CNES-01/02`, `SHE-SINASC-01`, `XCUT-03`) has a landed mechanism. The codebase is, to a first approximation, **MSD-compliant** — exactly the precondition MSD-II requires.
- **The dominant residual risk is now *unlocked* compliance, not *absent* compliance.** Many remediations are present but not yet pinned by contract tests. MSD-II's Phase 0 (Part 5) is therefore: **write the contract tests that prove the remediations**, before adding new capability on top. This is the single most leverage-rich step left in the correctness program.
- **Two genuine debts remain:** (1) the **duplication** the fast remediations left (vectorized normalizers beside the declarative path — `XCUT-02`), and (2) the **partials** (`SHE-NORM-01` SIH/CNES routing, `SIDRA-CTX-01` end-to-end context materialization, `RACE-01` bounds/CV). Both are handled in Part 5 before the new engine work begins.

---

## 3. State of the code — architecture & engorgement assessment

**Verdict on the 40k-line concern: the size is not bloat; the *boundary structure* is.** 40,113 lines across 232 files (mean 173) is appropriate for a system of this scientific complexity, and the largest files (`glm.py` 1150, `executor.py` 1127, `dag.py` 1017) are large because their domains are large, not because they are padded. The real problem is **fragmentation along development-slice boundaries** rather than domain concepts.

### 3.1 The fragmentation signature

The code was grown in incremental "slices" ("Slice 17A", "Slice 17B", "Macro-Slice 23A" appear in docstrings), and each slice left behind a file plus a workflow wrapper plus a JSON manifest. The result, concentrated in PIRS and workflows:

- **PIRS micro-stages (≈10 files) that pass JSON manifests to each other:** `pirs/{design,design_plan,design_readiness,design_matrix,field_selection,selection_plan,run_candidates,hsic_run,hsic_ranking,hsic_report}.py`. Several are tiny (`design.py` 73 ln, `field_selection.py` 56 ln, `spatial.py` 74 ln). Conceptually this is *one* operation — "select fields, build the design, fit, scan" — split across ten files communicating through `Tables/pirs_*_manifest.json`.
- **Workflow wrappers mirroring them 1:1 (≈10 files):** `workflows/{pirs_design,pirs_selection,pirs_candidates,pirs_matrix,pirs_readiness,pirs_execute,hsic_execute,hsic_rank,hsic_report,stage_plan}.py`. Each is a thin CLI step around one PIRS micro-stage.
- **Duplicated normalizers:** the four vectorized `datasus/*_normalize.py` (≈1700 ln total) coexist with `declarative_normalize.py` (`XCUT-02`), so decoder logic lives in two places.

This costs real money: every cross-stage boundary serializes a manifest to disk and re-reads it, multiplying I/O and plumbing; a single conceptual change touches many files; and the single-outcome flow looks larger and more entrenched than it is — which will actively fight the LDO refactor, because the LDO replaces "the design/selection/candidate/scan chain" with one estimator.

### 3.2 Refactor program (de-engorgement) — targets and expected reduction

These are specified as work items in Part 5 (`MII-REFACTOR-*`). Summary of intent:

1. **Collapse the PIRS slice-zoo into a coherent `pirs/` estimator package** organized by *concept*, not slice: `margins.py` (copula), `lowrank.py` (ST-DFM-backed), `precision.py` (sparse+lagged), `edges.py` (readout+stability), `residual_scan.py` (HSIC), `certify.py`. The ten manifest-passing micro-stages become in-memory phases of one orchestrator. *Expected: ~10 files → ~6, and elimination of intermediate manifest I/O.* This is done **as part of building the LDO**, not before — the new engine is the natural home, so the refactor and the feature are the same work (the same discipline the addendum used for `SHE-NORM-01`).
2. **De-duplicate the normalizers (`MII-REFACTOR-01`, resolves `XCUT-02`):** make `declarative_normalize.py` the single record-level normalizer; regenerate any vectorized fast path *from* the registry rather than by hand; delete the inline `when/then` decode logic in the four `*_normalize.py` files (keep only thin batch wrappers).
3. **Consolidate workflow wrappers:** replace the ~10 `workflows/pirs_*`/`hsic_*` step files with one `workflows/investigate.py` that runs the LDO orchestrator. Keep sub-step entrypoints only where a genuine staging/debug need exists (per `ExecutionStage`, §II.5).
4. **Registry hygiene sweep (`REG-*` cluster):** one canonical filename per registry; delete orphan aliases; replace `validate_registry_tree` with the executable-authority validator (§II.1).

**Guardrail:** every refactor is gated behind the **contract tests written in Phase 0**. You may not consolidate a module until the behavior it implements is pinned by a test, so consolidation cannot silently change results. This is why Phase 0 (lock compliance with tests) precedes all refactors and all new features.

---

*(Part 4 — the Secondary MSD normative contracts — and Part 5 — the Implementation TDD — follow.)*
---

## 4. The Secondary MSD — normative contracts (`§II.*`)

These sections are written to be appended to `MSD.md` as a numbered block. They are normative: "MUST" means a contract test enforces it. Each section names the MSD sections it extends and the predecessor (`ARCH-*`) it absorbs.

### §II.1 Registry Platform — executable authority

*Extends MSD §2.4; absorbs `ARCH-REG-*`, `ARCH-ADAPT-*`; resolves `SHE-NORM-01`, `PIRS-REG-01`, `REG-*`.*

**Principle.** Registries are executable authority, not post-hoc metadata. The system MUST decode and route every raw field through declarations, with code referenced *by name*.

**Contracts.**
1. **`source_routing.yaml`** declares, per `(source_system, raw_field)`: a route ∈ {Decode, Parse, PreserveMark, Exclude}, the decoder/parser callable name, and the output canonical field. No raw field may be silently dropped; `Exclude` MUST be explicit.
2. **`callables.py`** (`pegasus/registries/callables.py`) is the single name→function resolver. Every `decoder`/`parser`/`adapter` string in any registry MUST resolve through `resolve_callable`, or the cross-registry validator fails.
3. **`SourceAdapter`** plugin interface (`pegasus/sources/base.py`) with fields `source_family`, `realm ∈ {event, cube}`, and methods `discover/read_raw/raw_schema`. `source_adapters.yaml` maps family→adapter. `realm` routes data into the correct lifecycle (§II.2).
4. **Cross-registry validator** (`MII-REG-07`) replaces `validate_registry_tree`. It MUST enforce: every routing entry resolves a callable; every canonical field referenced exists with consistent carrier/unit/aggregation; every concept's operators/axes are registered; every spatial graph has provenance + legality_class; **round-trip reachability** — every admissible canonical field is producible from some raw field or a declared reconstruction output.
5. **Onboarding cost** MUST match: new table = 1 manifest entry; new encoding = 1 callable + 1 routing entry; new family = 1 adapter + 1 registry entry; new concept = 1 `concepts.yaml` entry. A test onboards a fixture source and asserts only declarative files + ≤1 callable changed.

### §II.2 Two Lifecycles + CommonPanel

*Extends MSD §2.9, §2.12, §3.8.1; absorbs `ARCH-PANEL-01`; resolves `SIDRA-CTX-01`, `XCUT-05`.*

**Event lifecycle:** raw record → typed records → (decode/parse via routing) → canonical event substrate → EFG count/functional ops → panel cells.
**Cube lifecycle:** raw cube → typed cube+metadata → regime classify (§2.9) → classification projection (§2.12.2) → [CTR reconstruction if regime demands] → bounded pushforward (§2.12.3) → substrate tagging (`B_official|B_harmonized|B_reconstructed|B_latent|B_cross-sectional`) → panel cells.

**CommonPanel (`§II.2.1`).** The shared `municipality × time [× age × sex × race]` panel is a **compiled product** with its own manifest. The panel manifest MUST record, **per (field, cell)**, a provenance/state ∈ {observed, projected, reconstructed, bounded, `unavailable_on_panel`+reason}. A panel cell is **never blank** — this is the structural elimination of silent absence (`XCUT-05`). The temporal axis of the panel MUST support both `year` and `month` resolution (see §II.7); the resolution is an intent parameter, not a hardcoded grain.

### §II.3 Constrained Tensor Reconstruction (CTR) kernel

*Extends MSD §2.8, §2.10; absorbs `ARCH-CTR-*`; generalizes `SHE-POP-01/02`.*

**Unification.** The population tensor (§2.8), age-bin disaggregation, synthetic context cubes, and ST-DFM latent factors (§2.10) are one object: a constrained optimization over a non-negative latent tensor with anchors, a linear observation operator, dynamics, marginals, and penalties (2nd-difference age/time, **spatial Laplacian from §II.4**).

**Contract.** `pegasus/she/reconstruction/problem.py::CTRProblem(latent_shape, observations[A_operator, values, weight], anchors, penalties, dynamics, marginals, composition?)`. The population tensor is the canonical *instance*. The spatial penalty MUST take the SpatialWeightGraph Laplacian (§II.4).

**Certification gate (`§II.3.1`, generalizes §2.10.4).** Every CTR output passes: `holdout_mape ≤ {verified .15, fragile .35}`, `reconstruction_var ≤ {verified .25, fragile .40}`, `constraint_residual ≤ tol`, `stability ≥ .85` ⇒ status ∈ {verified, fragile, illegal_excluded}. A CTR output MAY enter substrate only as `B_reconstructed`/`B_latent`, never `B_official`; it is dashboard-unsafe by default; promotion to dashboard-safe requires `verified` + propagated uncertainty. **New §10 abort:** "reconstructed tensor promoted to verified without holdout certification or propagated uncertainty."

### §II.4 SpatialWeightGraph

*Extends MSD §6.3, §3.12; absorbs `ARCH-SPATIAL-*`; reframes `PIRS-SPAT-01`, feeds `EFG-Q-01`, CTR, HSIC nulls, and the LDO spatial prior.*

**One base graph, many views.** `spatial_graphs.yaml` declares weighted (possibly directed) base graphs with `legality_class ∈ {structural, context_derived}` and `provenance`. The API `load_spatial_graph(id).view(v)` yields `v ∈ {binary, row_standardized, symmetric, laplacian}` and `.blocks(n)` for null/fold partitioning.

**Consumers MUST use the shared graph:** ICAR/CAR precision → `symmetric`/`laplacian`; Moran's I → `row_standardized`; CTR/ST-DFM/LDO spatial penalty → `laplacian`; HSIC spatial null → `blocks()`; cross-fit folds → `blocks()`.

**Circularity guard (`§II.4.1`, critical).** A `context_derived` graph (weight depends on a substantive variable: population, GDP, flows) MUST be rejected for any test/edge whose outcome or covariate shares the graph's provenance. Default spatial structure for all inference is `structural` (contiguity). **New §10 abort:** "context-derived spatial weight shares provenance with the tested variable." This guard applies inside the LDO's spatial prior exactly as it applies to a pairwise test.

**Migration-affinity graph (`§II.4.2`, the first realized `context_derived` graph).** The population tensor's reconstructed origin→destination migration field (MSD-I §2.8.7 flow layer) induces a *functional* adjacency that is deeper than geographic contiguity: two municipalities are close to the degree that people move between them, border or not (a capital↔satellite corridor can dominate a shared border with an empty neighbour). The kernel is the mass-normalized symmetrized flow

$$
\text{affinity}(i,j)=\frac{F_{i\to j}+F_{j\to i}}{\sqrt{\mathrm{Pop}_i\,\mathrm{Pop}_j}},
$$

a per-capita interchange *propensity* (not raw volume, which would merely rank the largest cities), summed across the run's years. It is a weighted `SpatialWeightGraph` with `legality_class=context_derived` and `provenance={migration_flow, population}`, so the §II.4.1 guard forbids it from smoothing any migration- or population-derived variable (notably the population denominator itself) while allowing it for provenance-disjoint outcomes (mortality, morbidity, socioeconomic context) — where migration corridors are exactly the right diffusion structure. `SpatialWeightGraph` now carries optional edge weights (the `weight` view returns the raw kernel; `symmetric`/`row_standardized`/`laplacian` carry them, `binary` remains 0/1 presence), and the unweighted structural contiguity graph is the `_weights=None` special case — identical behaviour as before. Implementation: `geo.migration_affinity.build_migration_affinity_graph`; produced as an artifact of the population-tensor compile stage.

### §II.5 Concept grammar + DataScope/ExecutionStage

*Extends MSD §3.3, §3.10, §1.5; absorbs `ARCH-CONCEPT-*`, `ARCH-PROFILE-01`; sharpens `EFG-REG-01`.*

**Concept grammar.** A registered concept (e.g. `mortality_rate`) is *grammar*, not a fixed column. `concepts.yaml` declares `{family, numerator, denominator_family, legal_operators, required_axes, optional_stratifiers, restriction_templates, default_state}`. The EFG `concept_planner` expands a concept family against intent+availability+legality into concrete legal `V`-fields, each materializing or emitting a typed `FailedBranch` — never a silent absence. Division of labor: concept registry = grammar; planner = parser/codegen; `legality.py` = type checker; `executor.py` = backend.

**Two orthogonal axes.**
- **DataScope** (which substrate exists): `core_vital | contextual | full` (unchanged from MSD §1.5).
- **ExecutionStage** (how far the pipeline runs): `validate` (SHE+EFG legality, no materialization) | `compile` (materialize `V_fields`+`Q_tensor`, no inference) | `investigate` (full + LDO).

`investigate` is the **default**. `ModelAssociations`/`ResidualAssociations`/`Hypotheses`(→`LinkRecords`) are required-non-empty only at `investigate`. All 17 keys always exist; empty keys carry `empty_by_stage`/`empty_by_profile` reasons. **PIRS/LDO is never "optional" — it is the stage skipped only on deliberate sub-runs.**

### §II.6 The Lattice Dependency Operator (LDO) — the re-founded PIRS

*Re-founds MSD §6 (Parametric Inference & Residual Scanner → Lattice Dependency Estimator). Reuses §6.2 (families), §6.3 (spatial), §6.5–6.9 (HSIC/nulls/FDR/cross-fit), §2.10 (ST-DFM). This is the heart of MSD-II.*

#### §II.6.1 Object

Let the lattice be $(s,t)$ at a chosen resolution, $p$ field-variables $X_1..X_p$ materialized on the CommonPanel, data tensor $X\in\mathbb R^{p\times S\times T}$. The LDO is **one structured generative model of $X$** whose dependency structure is the scientific output. Three composable layers:

**Layer 0 — Copula margins (reuse §6.2 / `compute/glm.py`).** Push each variable to a latent Gaussian scale via its admissible family $F_j$ (rank/PIT transform for sparse counts):
$$Z_{j,s,t}=\Phi^{-1}\!\big(F_j(X_{j,s,t}\mid \text{offset},\text{covariates})\big).$$
The §3.12 state tensor (n_eff, fragility, missingness) supplies per-cell reliability weights $w_{j,s,t}$.

**Layer 1 — Structured precision (the links).** Model $Z$ with a separable, structured precision (separability is the tractability backbone):
$$\operatorname{Prec}(Z)\approx \Omega_{\text{var}}\otimes \Sigma_{\text{space}}^{-1}\otimes \Sigma_{\text{time}}^{-1},$$
where $\Omega_{\text{var}}$ is the variable-dependency operator over the **time-extended** variable set $\{X_j^{(0)},\dots,X_j^{(K)}\}$ (variable $j$ at lags $0..K$). Its blocks are the links: same-lag block → contemporaneous conditional links; cross-lag block $\Omega^{(k,0)}_{ij}\neq 0$ → **directed lag-$k$ link** $X_i(t-k)\to X_j(t)$. $\Sigma_{\text{space}}^{-1}=\kappa I+L_W$ is the GMRF spatial precision from §II.4; $\Sigma_{\text{time}}^{-1}$ encodes temporal smoothness/seasonality.

**Layer 2 — Low-rank latent drivers (reuse ST-DFM).** Decompose
$$\Omega_{\text{var}} = S_{\text{sparse}} - L_{\text{low-rank}},\qquad \operatorname{rank}(L)=r\ll p$$
(Chandrasekaran–Parrilo–Willsky latent-variable graphical decomposition). $L$'s factors are the ST-DFM factors; ST-DFM ceases to be a context side-channel and becomes Layer 2.

#### §II.6.2 Estimation (one penalized fit, GPU-resident)

$$\min_{S\succeq0,\,L\succeq0}\;-\ell_w\big(Z;\,S-L,\,L_W,\,\Sigma_{\text{time}}\big)+\lambda_1\|S\|_{1,\text{off}}+\lambda_*\|L\|_*+\lambda_{\text{sp}}\,\mathrm{tr}(\cdot L_W\cdot).$$
$\|S\|_{1,\text{off}}$ (graphical-lasso/neighborhood selection) → sparse direct+lagged links; $\|L\|_*$ (nuclear norm) → low-rank shared drivers; $L_W$ quadratic → spatial smoothness. Solve by proximal-gradient/ADMM on GPU. **The only genuinely new numerical code is the $\ell_1$ proximal step for $S$ and the lag-extension of the variable axis;** the low-rank + spatial-Laplacian + transition machinery already exists in the ST-DFM solver.

#### §II.6.3 Link readout and the residual nonlinear layer

After one fit, every link is a query on $\{S,L,\Sigma\}$:
- contemporaneous link: $S^{(0,0)}_{ij}\neq0$ (weight = partial correlation, sign = association direction);
- **lagged directed link**: $S^{(k,0)}_{ij}\neq0$; the profile $\{S^{(k,0)}_{ij}\}_{k=0}^K$ **is the discovered distributed-lag response curve** (the engine finds the lag; it is not told it);
- **latent-shared (confounded)**: $i,j$ load on a common factor in $L$ → flagged shared-driver, not a direct edge;
- **spatial heterogeneity** of any link: free from the GMRF (report the link's spatial field);
- **nonlinear link**: compute the joint-model residual field $e=Z-\hat Z$ and run the **existing HSIC scanner** (§6.7) on residual / lagged-residual pairs under the §6.8 nulls. HSIC is now a targeted nonlinear-edge detector on what the structured backbone cannot explain — a strict generalization of MSD §6, reusing `pirs/{hsic,nulls,crossfit,fdr}.py`.

**Multiplicity** is handled by **stability selection** (refit on lattice subsamples; keep edges recurring above a frequency threshold), giving finite-sample edge-set control without $p^2$ separate tests.

#### §II.6.4 Identifiability and certification (mandatory)

The LDO inherits and generalizes the ST-DFM disciplines (§2.10.3 multi-start stability, §2.10.4 certification) into an **LDO certification gate** (holdout edge stability, regularization-path agreement, latent/lag separability diagnostics). Directionality MUST be reported only where time (lagged edges) or an explicitly-gated structure-learning layer licenses it; contemporaneous edges are undirected by default. Consistent with MSD §1.1/§12, the LDO **produces causal hypotheses; it does not certify causality.** **New §10 abort:** "dependency edge promoted without holdout stability or propagated uncertainty."

### §II.7 Multi-resolution lattice

*New; the scientific "finer phenomena lie deeper" axis and the laptop compute strategy, unified.*

The lattice resolution is a first-class intent axis. The LDO runs **coarse → fine**:
- **Coarse pass (global):** fit at municipality × year (or region × month) where $S\cdot T$ is small; produce the global link graph and candidate edges.
- **Fine pass (surgical):** for each candidate edge, refit **only those variables** at municipality × month on the spatial subset where the coarse edge is strong.
- **Formalization:** a multiresolution GMRF / graph-wavelet basis on $L_W$ makes coarse links low-frequency graph components and fine links high-frequency — one nested model, resolution as a parameter. This is simultaneously the compute-management plan (each pass is laptop-sized; national-monthly is reached by tiling/streaming via the existing DuckDB/Arrow out-of-core layer).

### §II.8 Link-record output contract

*Replaces MSD §8.3 hypothesis row; extends §8 (17-key bundle preserved).*

The `Hypotheses` key MUST carry **typed link records**, not pairwise rows:
```
(source_var, target_var, lag_k, edge_type ∈ {contemporaneous, lagged_directed, latent_shared, nonlinear_residual},
 weight, partial_correlation, response_curve_ref, spatial_field_ref, stability,
 uncertainty, confounding_factor_refs, null_strategy, fdr_method, certification_status, warnings)
```
The 17-key bundle and its validator (`OUT-01`) are preserved; the hypotheses key's schema is enriched, not removed.

### §II.9 Deferred — higher-order context-conditioned functionals

*Absorbs `ARCH-HOF-01`; remains deferred.* Define the contract (`higher_order_operators.yaml`: dispersion functional over a legal context axis, provenance-disjoint from input, output `quarantined_descriptive`, never a denominator), but implement only after Part 5 Phases 0–3 and the MSD §3 operator-set amendment. Highest legality risk; do not build early.

### §II.10 Compute envelope (binding)

All numerics MUST honor `compute.yaml`: float32, `max_vram_fraction ≤ 0.80` (≈4.8 GB of the RTX 4050's 6 GB), PyTorch backend, CUDA for HSIC/ST-DFM/population/LDO kernels. Any operation whose dense form exceeds the envelope MUST use the structured (separable/sparse/low-rank) form and, at national-monthly scale, the multi-resolution + streaming strategy of §II.7. A run that cannot fit MUST refuse with `scale_exceeds_compute_envelope`, never silently subsample without a warning.

### §II.11 Per-cell provenance & anti-silence (restated invariant)

Every panel cell, every reconstructed value, every link edge carries an explicit state/provenance/certification. Nothing is blank; nothing is promoted without uncertainty; nothing reconstructed is labeled observed. This is the MSD §12 prime directive made operational across the new surfaces.

### §II.12 MSD amendments this document requires (apply to `MSD.md`)

1. **§1.6** Two Lifecycles + CommonPanel (per-cell provenance; year/month resolution).
2. **§2.13** Constrained Tensor Reconstruction family + certification gate + `ARCH-CTR-04` abort.
3. **§2.14 / §6.10** SpatialWeightGraph contract, views, `legality_class`, circularity abort; restate §6.3 ICAR as consuming the graph.
4. **§3.16** Concept grammar + instantiation planner.
5. **§6′ (rewrite of §6)** Lattice Dependency Estimator: families→copula margins; spatial modes→GMRF prior; HSIC/nulls/FDR→residual nonlinear-edge layer; add Layers 1–2, link readout, stability selection, LDO certification, and the new edge-promotion abort.
6. **§8.3 (replace)** typed link record.
7. **§3.17 (deferred)** higher-order operators extension point.
8. **§1.5/§9.3** add ExecutionStage orthogonal to DataScope; restate PIRS/LDO as central to `investigate`.
9. **§12 (scope)** widen from "field compiler + residual scanning" to "field compiler + joint spatiotemporal dependency discovery"; add the new aborts to §10.

---
## 5. The Implementation TDD

**How to execute a work item.** Each item has: **Lineage** (parent IDs), **Intent**, **Red** (the test you write and watch fail first), **Green** (numbered steps to make it pass), **Refactor** (cleanups, only after green), **Acceptance** (the assertion that closes the item), **Files**. Do them in the order printed. The **Depends-on** column in §5.0 is the law of sequencing — never start an item whose dependencies are not Closed.

### 5.0 Master sequencing (do not reorder)

```
Phase 0  Lock compliance + de-duplicate      → MII-T0-*, MII-REFACTOR-01
Phase 1  Platform foundations                → MII-REG-07, MII-SPG-01/02/03, MII-PANEL-01, MII-SCOPE-01
Phase 2  Finish partials + CTR kernel        → MII-CTX-01, MII-RACE-01, MII-CTR-01/02/03
Phase 3  Build the LDO (the heart)           → MII-LDO-00..06
Phase 4  Multi-resolution + scale + output   → MII-RES-01, MII-OUT-01, MII-SCALE-01
Phase 5  Deferred (gated)                    → MII-HOF-01
```

Dependency edges: Phase N requires Phase N−1 **Closed**. Within Phase 3, `LDO-00 → LDO-01 → LDO-02 → LDO-03 → LDO-04 → LDO-05 → LDO-06` strictly. `MII-SPG-*` must close before `LDO-02` (the spatial prior). `MII-CTR-01` must close before `LDO-03` (Layer 2 reuses the kernel).

---

### Phase 0 — Lock the compliant codebase with contract tests, then de-duplicate

> Rationale (Part 2): the remediations exist but are not pinned. Pin them before building on them. Every test here must **fail if the production tier regresses below the contract**.

#### `MII-T0-1` — Source-reality contract tests
- **Lineage:** `SHE-NORM-01`, `SHE-CNES-01/02/03`, `SHE-SINASC-01`, `XCUT-01/05`.
- **Intent:** prove canonical fields are populated, states preserved, no forbidden composites.
- **Red:** add `tests/contract/test_source_reality.py` asserting, on a fixture with one record per §2.3 state per system:
  1. no canonical column declared admissible in `source_fields.yaml` is 100% null;
  2. `age_years/age_days` equal `decode_sim_idade` exactly;
  3. `cause_chain_norm` has 4 ordered positions with per-position parse states; `associated_conditions_norm` non-null;
  4. SINASC emits continuous `birth_weight_grams`, `apgar_5min`, `gestational_weeks` (not flags); thresholds are NOT in SHE output;
  5. CNES output contains **no** `capacity_total_observed` and **no** generic bed total; typed per-index carriers carry `axes.capacity_index`;
  6. `filter_cnpj` applied: an all-zero CNPJ → null link + a `ZeroCNPJShare`/observer field;
  7. every `*_state` column distinguishes Missing/Invalid/Unparseable/Unknown.
- **Green:** run the suite; for any failure, complete the corresponding remediation (most are already done — fix only what the test catches). Finish SIH/CNES declarative routing if (3)/(5)/(6) fail.
- **Acceptance:** all assertions green on the multi-state fixture.
- **Files:** `tests/contract/test_source_reality.py`, possibly `datasus/declarative_normalize.py`, `config/registries/source_routing.yaml`.

#### `MII-T0-2` — EFG legality, state-tensor, declaration contract tests
- **Lineage:** `EFG-LEG-01`, `EFG-Q-01`, `EFG-DECL-02`, `SIDRA-CTX-02`.
- **Red:** `tests/contract/test_efg_contracts.py`:
  1. `REQUIRED_Q_TENSOR_COLUMNS` includes `CV, MoranI, temporal_roughness, spatial_entropy`; all are computed on a fixture;
  2. `n_eff` equals `(Σw)²/Σw² · 1/(1+max(0,MoranI))` (Moran correction wired);
  3. a race-stratified operand lacking `race_axis_type` fails with `race_axis_declaration_unverifiable` (not permissive);
  4. a SIDRA field whose raw axis product exceeds the EFG cap triggers mandatory bounded pushforward or `Δ_axes=0`.
- **Green:** add the four columns to `REQUIRED_Q_TENSOR_COLUMNS`; wire the high-card bound into the legality path (`efg/dag.py`/`materialize.py` call the pushforward planner before materialization); harden the declaration gate.
- **Acceptance:** all green.
- **Files:** `output/validate.py`, `efg/q_tensor.py`, `efg/legality.py`, `efg/dag.py`, `tests/contract/test_efg_contracts.py`.

#### `MII-T0-3` — Race-bridge numerics contract tests
- **Lineage:** `RACE-01`.
- **Red:** `tests/contract/test_race_bridge.py`: (1) `W` depends on the local self-declared composition $\pi_i$ (perturbing $\pi_i$ changes `W`); (2) `race_bridge_cv` is the SD/E across bootstrap/posterior draws of $\hat\rho$, **not** cross-category spread; (3) `SensitivityWidth` comes from inf/sup over the emission-matrix credible set, not a `value·(1±w)` heuristic; (4) `State(Bridge_R(v)) ≤ fragile` by default.
- **Green:** finish any of these not already satisfied (local-π and posterior-C are present — verify CV and bounds).
- **Acceptance:** all green.
- **Files:** `efg/race_bridge.py`, `tests/contract/test_race_bridge.py`.

#### `MII-T0-4` — Inference + output contract tests (pre-LDO baseline)
- **Lineage:** `PIRS-HSIC-01`, `PIRS-FAM-01`, `PIRS-SPAT-01`, `OUT-01/03`.
- **Red:** `tests/contract/test_inference_baseline.py`: HSIC mode selection (§6.7) exact; family routing (§6.2) for count/proportion/simplex/skewed; spatial mode precedence (§6.3); 17 keys present; `empty_by_profile` reasons on empty keys. **These tests are the guardrail the LDO refactor must keep green.**
- **Acceptance:** all green; tag this commit `pre-ldo-baseline`.
- **Files:** `tests/contract/test_inference_baseline.py`.

#### `MII-REFACTOR-01` — De-duplicate normalizers (resolve `XCUT-02`)
- **Lineage:** `XCUT-02`, `SHE-NORM-01`.
- **Intent:** one decoder source of truth.
- **Red:** `tests/contract/test_no_decode_duplication.py` asserts the inline `when/then` decode chains are gone from the four `*_normalize.py` and that the vectorized path, where retained, is generated from the registry (a property test: vectorized output == record-level declarative output on a fixture).
- **Green:** 1) make `declarative_normalize.py` the single record-level normalizer for all four systems; 2) regenerate any vectorized fast path *from* `source_routing.yaml` (or drop it and accept record-level throughput until profiled); 3) delete inline decode logic; keep `*_normalize.py` only as thin batch wrappers.
- **Acceptance:** duplication test green; `MII-T0-1` still green.
- **Files:** `datasus/declarative_normalize.py`, `datasus/{normalize,sih_normalize,sinasc_normalize,cnes_normalize}.py`.

---

### Phase 1 — Platform foundations

#### `MII-REG-07` — Executable-authority cross-registry validator + generator
- **Lineage:** `ARCH-REG-07`, `REG-VALIDATOR-01`, `REG-DEAD-01`, `REG-STALE-LIST-01`.
- **Red:** `tests/contract/test_registry_validator.py`: every routing entry resolves a callable; every referenced canonical field exists with consistent carrier/unit/aggregation; round-trip reachability (every admissible canonical field producible from a raw field or declared reconstruction); orphan/dead registries are rejected; the validator fails loudly on a seeded inconsistency.
- **Green:** 1) create `pegasus/registries/callables.py` with `resolve_callable`; 2) replace `validators.py::validate_registry_tree` with the cross-registry validator; 3) delete orphan alias stubs and dual-name tolerance; one canonical filename per registry; 4) add the generator CLI (`pegasus registry new-source/new-field/check`).
- **Acceptance:** validator green; registry tree has no orphans; `pegasus registry check` passes.
- **Files:** `registries/callables.py`, `registries/validators.py`, `config/registries/*`, `cli.py`.

#### `MII-SPG-01/02/03` — SpatialWeightGraph (registry, views/consumers, circularity guard)
- **Lineage:** `ARCH-SPATIAL-01/02/03`, `PIRS-SPAT-01`, `EFG-Q-01`.
- **Red:** `tests/contract/test_spatial_graph.py`: (01) `load_spatial_graph("contiguity_queen").view(v)` returns correct binary/row_standardized/symmetric/laplacian matrices; `.blocks(n)` partitions; (02) Moran's I, ICAR penalty, HSIC nulls, cross-fit folds all consume the shared graph (no ad-hoc adjacency remains); (03) a `context_derived` graph sharing provenance with a tested variable is rejected with the §II.4.1 abort.
- **Green:** 1) `config/registries/spatial_graphs.yaml` with `structural` contiguity_queen as default; 2) `pegasus/geo/spatial_graph.py` generalizing `adjacency.py`; 3) rewire `q_tensor.py` (Moran), `glm.py::fit_glm_penalized` (ICAR), `nulls.py`/`crossfit.py` (blocks) to the graph; 4) implement the circularity guard in the legality/selection path and add the §10 abort.
- **Acceptance:** all three green; old ad-hoc adjacency call sites removed.
- **Files:** `config/registries/spatial_graphs.yaml`, `geo/spatial_graph.py`, `efg/q_tensor.py`, `compute/glm.py`, `pirs/nulls.py`, `pirs/crossfit.py`.

#### `MII-PANEL-01` — CommonPanel compilation phase + manifest
- **Lineage:** `ARCH-PANEL-01`, `SIDRA-CTX-01`, `XCUT-05`.
- **Red:** `tests/contract/test_common_panel.py`: a compiled panel manifest records per (field, cell) a provenance/state ∈ {observed, projected, reconstructed, bounded, unavailable_on_panel+reason}; **no cell is blank**; the panel supports both `year` and `month` resolution from intent.
- **Green:** 1) `pegasus/she/panel.py` planner deriving the target panel from intent axes + legality; 2) event/cube lifecycle split `she/lifecycle/{event,cube}.py` (orchestration over existing regime/stitch/projection/pushforward modules); 3) thread `month` resolution from ingest (already decoded) through the panel.
- **Acceptance:** panel manifest test green; monthly panel materializes on the Alagoas fixture.
- **Files:** `she/panel.py`, `she/lifecycle/event.py`, `she/lifecycle/cube.py`, `workflows/compile.py`.

#### `MII-SCOPE-01` — DataScope × ExecutionStage
- **Lineage:** `ARCH-PROFILE-01`, `OUT-03`.
- **Red:** `tests/contract/test_scope_stage.py`: `UserIntent` carries orthogonal `data_scope` and `execution_stage`; bundle validation requires `LinkRecords` non-empty only at `investigate`; all 17 keys always present with typed empty reasons.
- **Green:** add the two axes to `UserIntent`; make `PROFILE_NONEMPTY` a function of `(data_scope, execution_stage)`; default stage = `investigate`.
- **Acceptance:** green; existing intents still validate (default stage applied).
- **Files:** `core/schemas.py`, `output/validate.py`, `config/intents/*` (add default), `workflows/compile.py`.

---

### Phase 2 — Finish partials + the CTR kernel

#### `MII-CTX-01` — SIDRA context end-to-end into EFG nodes
- **Lineage:** `SIDRA-CTX-01/02`.
- **Red:** `tests/integration/test_context_lifecycle.py`: in `contextual` scope, a compendium fact flows regime→stitch→projection→pushforward→`context_gradient`/`latent_context` EFG node; ST-DFM-certified factors become `latent_context` (dashboard-unsafe by default); the §10 aborts fire (cell budget, mandatory pushforward, ST-DFM proportion-without-denominator).
- **Green:** wire the cube lifecycle (`MII-PANEL-01`) so each compendium artifact becomes a substrate field with regime attached; expand `source_fields.yaml` SIDRA section to admissible context fields with latent-provenance quarantine; branch ST-DFM in `compile.py` on scope+regime and gate via `certification.py`.
- **Acceptance:** integration test green; context nodes appear in `V_fields` on the Alagoas all-source fixture.
- **Files:** `she/lifecycle/cube.py`, `config/registries/source_fields.yaml`, `workflows/compile.py`, `she/stdfm/pipeline.py`.

#### `MII-RACE-01` — Race-bridge bounds + simulation completion
- **Lineage:** `RACE-01` steps 3–4.
- **Red:** extend `MII-T0-3` with the inf/sup partial-ID bounds and full posterior simulation (Dirichlet `C`, π draws, Poisson ν) for standard/deep.
- **Green:** implement per §4.6/§4.7; wire `SensitivityWidth` to the §4.10 downgrade.
- **Acceptance:** green.
- **Files:** `efg/race_bridge.py`.

#### `MII-CTR-01/02/03` — Constrained Tensor Reconstruction kernel
- **Lineage:** `ARCH-CTR-01/02/03/04`, `SHE-POP-01/02`.
- **Red:** `tests/contract/test_ctr.py`: (01) `CTRProblem` reproduces the population tensor as an instance (golden values vs current `loss.py`); (02) age-bin disaggregation `A·n_fine=n_broad` recovered under totals+smoothness+cohort, output `B_reconstructed` dashboard-unsafe with per-cell uncertainty; (03) certification gate enforces the §II.3.1 thresholds; the §10 abort fires when a reconstructed tensor is promoted without holdout/uncertainty.
- **Green:** 1) refactor `she/population/` → `she/reconstruction/` with the population tensor as the canonical instance (keep `loss.py` math byte-for-byte; only re-wrap it — `SHE-POP-01` golden tests guard this); 2) add age-bin disaggregation as a CTR instance; 3) generalize `she/stdfm/certification.py` into the CTR gate; 4) the spatial penalty takes `Laplacian(W)` from `MII-SPG-01`.
- **Acceptance:** all green; population golden values unchanged.
- **Files:** `she/reconstruction/{problem,instances,certify}.py`, `she/population/loss.py` (rewrapped), `she/stdfm/certification.py`.

---

### Phase 3 — Build the Lattice Dependency Operator (the heart)

> This is where PegaSUS becomes the system its architect described. Build strictly in order. Each layer is independently testable. Keep `MII-T0-4` (the pre-LDO baseline) green throughout — the LDO **adds** a capability; it does not break the existing scan.

#### `MII-LDO-00` — Field assembly + link-record schema
- **Intent:** assemble the multivariate tensor $X\in\mathbb R^{p\times S\times T}$ from the CommonPanel and define the output object.
- **Red:** `tests/ldo/test_assembly.py`: given a panel manifest, build $X$ with per-cell weights $w$ from the §3.12 state tensor; define `LinkRecord` (§II.8) and assert the bundle validator accepts it in the `Hypotheses` key.
- **Green:** `pegasus/pirs/ldo/assemble.py` (panel→tensor+weights); `pegasus/pirs/ldo/records.py` (`LinkRecord`); extend `output/schemas.py`/`validate.py` to accept link records (do NOT remove the old row schema yet — additive).
- **Acceptance:** assembly + schema tests green; 17-key validator still green.
- **Files:** `pirs/ldo/assemble.py`, `pirs/ldo/records.py`, `output/schemas.py`, `output/validate.py`.

#### `MII-LDO-01` — Layer 0: copula margins
- **Intent:** push each variable to the latent Gaussian scale via its admissible family.
- **Red:** `tests/ldo/test_margins.py`: for a Poisson/NB/beta-binomial/Gamma column, $Z=\Phi^{-1}(F_j(\cdot))$ is computed via the existing `compute/glm.py` family; sparse-count columns use the rank/PIT transform; reliability weights $w$ propagate.
- **Green:** `pirs/ldo/margins.py` wrapping `compute/glm.py` family CDFs as copula transforms; PIT/rank fallback for near-zero cells.
- **Acceptance:** green; margins reuse `glm.py` (no new family code).
- **Files:** `pirs/ldo/margins.py`.

#### `MII-LDO-02` — Layer 1 spatial+temporal structured precision (no lags yet)
- **Depends:** `MII-SPG-01`.
- **Intent:** fit $\Omega_{\text{var}}\otimes\Sigma_{\text{space}}^{-1}\otimes\Sigma_{\text{time}}^{-1}$ at $K=0$ (contemporaneous only) to validate the separable backbone.
- **Red:** `tests/ldo/test_precision_k0.py`: on synthetic data with a known sparse contemporaneous graph + GMRF spatial structure, the estimator recovers the true edges (precision/recall above threshold); the spatial prior uses `Laplacian(W)`.
- **Green:** `pirs/ldo/precision.py` — neighborhood-selection/graphical-lasso on $Z$ with the GMRF whitening from `MII-SPG-01`; proximal-gradient solver on torch (GPU, float32, ≤0.80 VRAM).
- **Acceptance:** edge recovery green on synthetic; runs within envelope on Alagoas monthly.
- **Files:** `pirs/ldo/precision.py`.

#### `MII-LDO-03` — Layer 2: low-rank latent drivers (ST-DFM reuse)
- **Depends:** `MII-CTR-01`.
- **Intent:** add $\Omega_{\text{var}}=S-L$ with $L$ low-rank from the ST-DFM solver.
- **Red:** `tests/ldo/test_lowrank.py`: on synthetic data with a shared latent factor (an "epidemic wave"), variables co-moving via the factor are flagged `latent_shared`, NOT as dense direct edges; $S$ recovers only the true direct edges net of the factor.
- **Green:** `pirs/ldo/lowrank.py` calling the ST-DFM solver as the nuclear-norm/low-rank step; alternate with `precision.py`'s $\ell_1$ step (proximal/ADMM).
- **Acceptance:** confounded-by-wave variables correctly separated from direct edges.
- **Files:** `pirs/ldo/lowrank.py`, reuse `she/stdfm/torch_solver.py`.

#### `MII-LDO-04` — Lag extension (the Zika capability)
- **Intent:** extend the variable axis to lags $0..K$; fit cross-lag blocks; read off the distributed-lag response curve.
- **Red:** `tests/ldo/test_lags.py` — **the core scientific test.** On synthetic monthly data where variable A at lag 7 drives variable B (plus a shared wave + spatial structure), the estimator: (1) recovers a directed edge $A\to_k B$ peaked at $k=7$; (2) returns the lag profile as the response curve; (3) attributes the co-epidemic wave to $L$, not to a spurious edge; (4) does this with **no variable hand-specified**.
- **Green:** extend `assemble.py` to build the time-extended tensor; `precision.py` to estimate cross-lag blocks; `records.py` to emit `lagged_directed` edges with `response_curve_ref`.
- **Acceptance:** lag test green; the recovered peak lag is within ±1 of truth.
- **Files:** `pirs/ldo/{assemble,precision,records}.py`.

#### `MII-LDO-05` — Edge readout, stability selection, residual nonlinear layer
- **Intent:** turn $\{S,L,\Sigma\}$ into `LinkRecord`s; control multiplicity by stability selection; run HSIC on residuals as the nonlinear-edge detector.
- **Red:** `tests/ldo/test_edges.py`: (1) edges below a stability threshold are dropped or marked descriptive; (2) HSIC on the joint-model residual field detects a nonlinear edge the linear backbone misses (reusing `pirs/hsic.py` + §6.8 nulls + FDR); (3) `n_eff<100` edges → descriptive only.
- **Green:** `pirs/ldo/edges.py` (readout + stability selection via lattice subsampling); `pirs/ldo/residual_scan.py` wrapping the existing HSIC stack on residual/lagged-residual pairs.
- **Acceptance:** green; existing `pirs/hsic.py` reused unchanged.
- **Files:** `pirs/ldo/edges.py`, `pirs/ldo/residual_scan.py`.

#### `MII-LDO-06` — LDO certification + orchestrator (+ PIRS consolidation refactor)
- **Intent:** certify the dependency graph; one orchestrator replaces the slice-zoo.
- **Red:** `tests/ldo/test_certify.py`: holdout edge stability, regularization-path agreement, latent/lag separability diagnostics produce a per-edge `certification_status`; the §10 abort fires for an edge promoted without holdout/uncertainty. `tests/ldo/test_orchestrator.py`: one call `run_ldo(panel, intent)` produces the full `LinkRecords` bundle.
- **Green:** 1) `pirs/ldo/certify.py` generalizing CTR certification to edges; 2) `pirs/ldo/orchestrator.py::run_ldo` running margins→lowrank+precision→edges→residual_scan→certify **in memory** (no inter-stage manifests); 3) **Refactor:** retire the manifest-passing micro-stages (`design_plan/design_readiness/design_matrix/selection_plan/run_candidates/hsic_run/...`) and the mirror workflow wrappers, folding their still-needed logic into the orchestrator; one `workflows/investigate.py` entrypoint. Keep `MII-T0-4` green.
- **Acceptance:** orchestrator + certification green; PIRS file count materially reduced; baseline tests still green.
- **Files:** `pirs/ldo/{certify,orchestrator}.py`, `workflows/investigate.py`; **delete** the retired micro-stage + wrapper files.

---

### Phase 4 — Multi-resolution, scale, output

#### `MII-RES-01` — Coarse→fine multi-resolution
- **Lineage:** §II.7.
- **Red:** `tests/ldo/test_multiresolution.py`: a coarse (annual) global pass yields candidate edges; a fine (monthly) pass refits only candidate variables on the strong-signal spatial subset and sharpens the lag peak; results are consistent across resolutions.
- **Green:** `pirs/ldo/resolution.py` orchestrating the two passes; intent gains a temporal-resolution + refinement contract.
- **Acceptance:** green; fine pass stays within compute envelope.
- **Files:** `pirs/ldo/resolution.py`, `core/schemas.py`.

#### `MII-OUT-01` — Link-record output is authoritative; retire pairwise row
- **Lineage:** §II.8, `OUT-01`.
- **Red:** `tests/contract/test_link_output.py`: the `Hypotheses` key carries link records; the 17-key bundle validates; the old pairwise row schema is removed (now that the LDO is the engine).
- **Green:** make `LinkRecord` the authoritative hypotheses schema; remove the deprecated row schema and its now-dead producers.
- **Acceptance:** green; no producer emits the old row.
- **Files:** `output/schemas.py`, `output/validate.py`, `pirs/ldo/records.py`.

#### `MII-SCALE-01` — National scale via tiling/streaming
- **Lineage:** §II.7, §II.10, `SHE-POP-02`.
- **Red:** `tests/scale/test_national_envelope.py` (may be slow/gated): a national-monthly run either completes within the VRAM envelope via tiling/streaming or **refuses** with `scale_exceeds_compute_envelope` — never silently subsamples.
- **Green:** stream tiles via the existing DuckDB/Arrow out-of-core layer; reuse CTR ADMM/state-space backends for the low-rank step at scale.
- **Acceptance:** green or explicit refusal; no silent subsampling.
- **Files:** `pirs/ldo/resolution.py`, `storage/*`.

---

### Phase 5 — Deferred (gated)

#### `MII-HOF-01` — Higher-order context-conditioned functionals
- **Lineage:** §II.9, `ARCH-HOF-01`. **Gate:** Phases 0–3 Closed + MSD §3 operator-set amendment applied.
- **Red/Green/Acceptance:** per §II.9 contract (`higher_order_operators.yaml`; provenance-disjoint; `quarantined_descriptive`; never a denominator). Do not start until the gate is open.

---

## 6. Appendices

### Appendix A — ID cross-reference (predecessor → MSD-II)

| Predecessor | MSD-II home |
|---|---|
| `ARCH-REG-01..07`, `ARCH-ADAPT-01` | §II.1 / `MII-REG-07` |
| `ARCH-PANEL-01`, lifecycle split | §II.2 / `MII-PANEL-01` |
| `ARCH-CTR-01..04` | §II.3 / `MII-CTR-01/02/03` |
| `ARCH-SPATIAL-01/02/03` | §II.4 / `MII-SPG-01/02/03` |
| `ARCH-CONCEPT-01/02` | §II.5 |
| `ARCH-PROFILE-01` | §II.5 / `MII-SCOPE-01` |
| `ARCH-PIRS-EXPLAIN-01/02` | folded into LDO decision records (`MII-LDO-06` certification manifest) |
| `ARCH-HOF-01` | §II.9 / `MII-HOF-01` |
| Gap-analysis Pillars A–D | §II.6 (LDO), §II.7 (resolution), §II.8 (link record) |
| Vision/PIRS LDO design | §II.6 in full |

### Appendix B — Compliance finding → current status (compact)

Remediated: `SHE-DEC-01`, `SHE-CNES-01/02`, `SHE-SINASC-01`, `SHE-POP-01/02`, `EFG-LEG-01`, `EFG-DECL-01`, `EFG-EXEC-01`, `EFG-CMP-01`, `EFG-Q-01`, `RACE-01`, `PIRS-HSIC-01`, `PIRS-FAM-01`, `PIRS-SPAT-01`, `PIRS-REG-01`, `OUT-01/03`, `XCUT-03`.
Partial: `SHE-NORM-01` (SIH/CNES routing + de-dup), `SIDRA-CTX-01`, `EFG-REG-01`, `REG-DEAD-01/VALIDATOR-01`, `XCUT-04`.
Verify-only (likely done): `SHE-CNES-03`, `SIDRA-CTX-02`, `EFG-DECL-02`, `OUT-02`, `REG-STALE-LIST-01/SCAFFOLD-01/EOL-01`.
Open: `XCUT-02` (duplication) → `MII-REFACTOR-01`.
All are pinned by Phase 0 contract tests before any new work proceeds.

### Appendix C — Glossary (delta from MSD glossary)

- **LDO** — Lattice Dependency Operator: the structured precision operator over the multivariate spatiotemporal field; the re-founded PIRS engine.
- **CommonPanel** — the compiled geo×time[×demographic] support panel with per-cell provenance.
- **CTR** — Constrained Tensor Reconstruction: the unified solver for population/age/cube/factor reconstruction.
- **SpatialWeightGraph** — the single declared spatial structure with derived views and a legality class.
- **Copula margins** — the GLM-family CDF transforms taking variables to the latent Gaussian scale.
- **Link record** — the typed relationship output object (lag, direction, shape, spatial field, stability, certification).
- **Stability selection** — multiplicity control by keeping edges recurring across lattice subsamples.
- **DataScope / ExecutionStage** — orthogonal run axes (which substrate exists / how far the pipeline runs).

### Appendix D — The program acceptance test (the one that matters)

`tests/acceptance/test_zika_microcephaly.py`:
> A **monthly Alagoas** run, **DataScope=core_vital+SIH/SINASC**, **ExecutionStage=investigate**, **with `force_selectors` empty**, MUST produce a `LinkRecord` with `edge_type=lagged_directed`, `source≈ArbovirusHospitalAdmissions`, `target≈MicrocephalyBirths`, `lag_k` peaked in **6–9 months**, `confounding_factor_refs` including the 2015–16 epidemic-wave latent factor, a non-trivial `stability`, and propagated `uncertainty`.

When this passes, MSD-II's goal is met: PegaSUS discovered the link without being told it. Until it passes, the program is not done.

---

*End of MSD-II. Use `MII-*` IDs in commits; cite the parent finding/ARCH ID. Apply the §II.12 amendments to `MSD.md` before building the dependent code. Keep Phase 0 contract tests green at every step — they are the definition of "still compliant."*
