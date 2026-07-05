# PegaSUS — Master System Document III (MSD-III)

> **The definitive architecture, vision, and implementation plan for PegaSUS.** This document consolidates and supersedes everything developed from MSD-II onward into one coherent, self-contained specification. It is dense by design; it is meant to be the primary reference an engineer or agent works from.

---

## 0. Front matter

### 0.1 Lineage and authority

**Constitutional parents (retained, authoritative, *not* superseded):**
- **`MSD.md` (MSD-I)** — the substrate constitution. Its detailed data-plane contracts remain in force *by reference*: the SHE decoders and state taxonomy (§2.3–2.4), the population-tensor loss terms (§2.8), the EFG legality predicate (§3.8), the null/FDR regimes (§6.8). MSD-III does not re-derive these; it incorporates them and *amends* named sections where it extends them (amendment list §11.5).
- **`PEGASUS_COMPLIANCE_AND_REMEDIATION.md`** — the correctness baseline. Its finding IDs remain the canonical handles; §10.1 reviews their current status, because MSD-III's new work must sit on an MSD-I-compliant codebase.

**Superseded and absorbed by this document (do not consult for new work; their conclusions are restated here in definitive form):** MSD-II; the statistical-engine gap analysis; the vision assessment; the disease-semantic-axis note; the conceptual-foundations note; the architecture-explorations note; the closing-the-open-items note; and the mathematics teaching companion (retained only as optional pedagogy; its *content* is consolidated here).

**Authority order:** (1) MSD-I on any locked substrate invariant it specifies; (2) **MSD-III** on all architecture, the inference engine, and everything from MSD-II onward; (3) the compliance report on correctness details not yet folded in; (4) code comments never. Where MSD-III requires behavior MSD-I forbids, the §11.5 amendment is the instrument; apply it before building.

### 0.2 What PegaSUS is (definitive statement)

> **PegaSUS is an engine that automates epidemiological discovery.** It compiles Brazil's public-health and socioeconomic record (DATASUS, IBGE/SIDRA) into one clean, multi-resolution field over a space × time lattice, then estimates a single structured model of the *dependency* among all quantities in that field — recovering, as queries against one fitted object, the contemporaneous, lagged, spatial, conditional, and nonlinear relationships that constitute epidemiological knowledge, each with calibrated uncertainty, on commodity hardware. It maintains this discovered structure as a living foundational asset that humans and agents interrogate, causally escalate, lens, steer, and stress-test.

PegaSUS is **not** a causal oracle, a query-per-analysis tool, or a health-data-only system. It is a field-science engine whose "epidemiological" output is a *typed view* over a broader discovered structure.

### 0.3 Conventions and the prime directive

- **Normative language.** "MUST" denotes a contract enforced by a test. "SHOULD" is a strong default.
- **Build discipline.** All implementation follows Red → Green → Refactor (§11.1). Every work item is one pull request citing its ID and parent finding.
- **The prime directive (inherited from MSD-I §12, elevated here as the tie-breaker for every ambiguous decision):** never replace mathematical legality with convenience, missingness with silence, source structure with scalar fiction, or measurement uncertainty with false precision. But note the development-phase corollary (§0.4).
- **Development-phase corollary.** The prime directive governs *what the system emits to users*, not *whether we build the working version*. In development, "works" is the bar. We populate mechanisms with real (if imperfect, uncertainty-typed) content rather than leaving them as honest no-ops. A documented failure is still a failure.

### 0.4 The permanent limits (stated first, so nothing downstream pretends otherwise)

Three things are standing properties of the system, not gaps to be engineered away. MSD-III treats them as design principles:

1. **The information ceiling.** Reliably-discoverable epidemiology is bounded by the *information content* of the data (effective independent observations), not by compute. No hardware moves this ceiling; larger machines search more candidates but do not make them true.
2. **Association is not causation.** The discovered dependency structure is a *substrate*. Causal claims require the escalation ladder (§4), and each is only as strong as the assumptions its rung needs — reported as such, never overstated.
3. **Some misclassification is unidentified without external data.** Measurement bridges (race, cause coding) cannot recover their corrections from marginals alone. The system's duty is to make the uncertainty *explicit and propagated*, and to use the best *externally-informed* estimate — not to fabricate, and not to no-op.

---

## Part I — Ontological Foundations

All architecture derives from getting the ontology of the data right. These are normative: the engine's treatment of any quantity is determined by its ontological type.

### I.1 The three kinds of quantity: flows, stocks, fields

Every quantity PegaSUS handles is exactly one of:

- **Flow (event).** An *occurrence* attributable to an individual at a moment — a death, hospitalization, birth, notification. Mathematically a *point process*; aggregated over a cell it becomes a **count** with an *exposure* (opportunity for the event). DATASUS is overwhelmingly flows.
- **Stock (state).** A *standing level* obeying conservation: a stock is the running integral of its flows. Population is the canonical stock (changed by births/deaths/migration/aging). Stocks that are denominators or conserved MUST be modeled as processes, not interpolated (§II.4).
- **Field (context).** A *standing property of a place-time*, usually slowly varying and environmental — sewage, GDP, urbanization. Fields are the *conditions* under which flows occur. SIDRA is overwhelmingly fields.

Every registry entry MUST tag `kind ∈ {flow, stock, field}`. This tag — uninferable from data format — routes the quantity to its mathematics (flow → count+exposure; stock → conservation/reconstruction; field → typed covariate).

### I.2 Extensive vs. intensive: the denominator principle

Orthogonal to kind, and equally load-bearing:

- **Extensive** quantities scale with system size (total deaths, sewage-connected households, GDP). They are counts/sums and are **not comparable across places of different size**.
- **Intensive** quantities are size-independent ratios (mortality rate, % coverage, GDP per capita). They are comparable directly.

**The denominator principle (normative):** *every extensive quantity — flow OR field — MUST be modeled as a count with an exposure/denominator that renders it comparable and supplies the correct count-variance.* Intensive quantities enter directly. Consequences: (a) normalization lives *in the model as an offset*, never as a pre-materialized rate (§II.3); (b) extensive context fields need denominators exactly as flows do, implying a *family* of denominator assets, not just population (§II.4).

### I.3 Events vs. context; format vs. ontology

Any tabular source — DATASUS or SIDRA — is expressible in one universal cube form: `(measure, classification axes, categories, cell values)`. **Format is interchangeable; ontology is not.** After rewriting SIM as a cube, its cells are still *counts of occurrences* (flow); sewage cells are still *levels* (field). Therefore the architecture is: **one universal cube-shaped registry, plus explicit ontological tags the format cannot infer** (§II.1). This is what lets DATASUS and SIDRA share one onboarding surface without erasing their difference.

### I.4 Entities and their coordinate systems (the formalization of "axis")

An **axis is a coordinate system on an entity type**, and its *kind of coordinate structure* determines its architectural role. PegaSUS has a small, fixed set:

| Entity | Axis | Coordinate structure | Role | Induced prior |
|---|---|---|---|---|
| **Place** | geography | metric + hierarchical (adjacency, distance) | support (locates every quantity) | spatial GMRF `L_W` |
| **Time** | time | ordered/metric + hierarchical | support | temporal dynamics `Σ_time` |
| **Condition** | disease (ICD), procedure, cause | **tree**-hierarchical (nesting, no metric) | *identity*: generates & organizes the variable set | disease graph `L_D` |
| **Person** | age, sex, race | flat categorical | *stratify* + **join key between flows and stocks about people** | demographic smoothing; join constraints |
| **Facility** | CNES unit, ownership | shallow categorical | observer identity / attribute | weak grouping |

Every classification axis in the registry MUST carry `(entity_type, coordinate_structure ∈ {metric_hier, tree_hier, flat_cat}, role ∈ {support, identity, stratify_join})`. This tuple is a *decision procedure* for any new axis, and it drives the engine's treatment automatically: metric-hierarchical → smoothing/dynamics prior; tree-hierarchical → hierarchical shrinkage + variable generation; flat-categorical-on-Person → join/stratify.

### I.5 The lattice and multi-resolution

The **support lattice** is Place × Time — the domain on which every quantity lives, because every observation is located in space and time. All three hierarchical axes (space, time, disease) exist at **multiple resolutions simultaneously** (city ⊂ state ⊂ region ⊂ nation; month ⊂ quarter ⊂ year; leaf ⊂ category ⊂ block ⊂ chapter). Resolution is a first-class parameter, not a fixed grain (§III.3). The default operational grain is municipality × month, sliced from the finest available.

### I.6 What counts as epidemiology: link-type taxonomy, not source hierarchy

PegaSUS does **not** privilege health data. It discovers dependencies in the joint field of flows, stocks, and fields; each discovered edge is *labeled* by the ontological kinds it connects:

- edge touching ≥1 **health flow** → *epidemiological*;
- edge among **fields only** (sewage↔GDP) → *socioeconomic/environmental* (valid science, retained);
- edge touching the **population stock** + a health flow → *demographic-epidemiological*.

An "epidemiological query" is a *filter* over link-types; the engine never refuses to find, or discards, a valid non-health edge. Relevance is **emergent** (which quantities participate in strong, certified edges) and **query-dependent**, never hard-coded.

---

## Part II — The Data Plane

The data plane compiles raw sources into one clean, typed, multi-resolution field with per-cell provenance. It extends MSD-I's SHE/EFG rather than replacing them.

### II.1 The unified `kind`-tagged registry

One schema types every source (DATASUS, SIDRA, future). Replaces the ~40 fragmented, two-world registries with one onboarding surface.

```
source_registry_entry:
  measures:            [ {name, unit, kind: flow|stock|field, extensive: bool,
                          carrier?, exposure_ref?} ]           # exposure_ref names the denominator asset
  classification_axes: [ {name, entity_type, coordinate_structure, role,
                          categories, aggregation_law: partition|multilabel|hierarchy} ]
  native_resolution:   {geo, time}
  provenance, projection_status, refresh_cadence, adapter_callable
```

**Contracts.** Every raw field routes explicitly (decode/parse/preserve/exclude — no silent drops). Every callable resolves by name through one resolver (`registries/callables.py`). A single cross-registry validator enforces callable resolution, canonical-field consistency, round-trip reachability (every admissible field is producible), and axis-type completeness. Onboarding a new source = one entry + ≤1 callable.

### II.2 SHE — substrate harmonization (inherited, extended)

MSD-I's SHE (decoders, production normalizers, the state taxonomy) stands. MSD-III requires only: (a) the declarative record-level normalizer is the single decode path (retire the duplicated vectorized normalizers — `XCUT-02`); (b) SHE emits *primitives*, never EFG-level indicators (thresholds/flags belong downstream); (c) every canonical field carries its state (Missing/Invalid/Unparseable/Unknown/Valid). The compliance battery (§10.1) locks these.

### II.3 The EFG reconceived: legality algebra + measured-quantity objects

The EFG's *legality and provenance algebra* — the type system that knows deaths may divide population but not births, that multi-label concepts may not be summed, that axes must align — is sound and **retained in full**. One change: its **terminal output is no longer a materialized rate**. The EFG emits **measured-quantity objects**:

```
MeasuredQuantity: { numerator_count, exposure/denominator, offset_semantics,
                    structure(axes, strata), provenance, uncertainty(state tensor) }
```

Rationale (the denominator principle, I.2): normalization belongs in the model as an offset (`log E[count] = log(exposure) + effects`), where the rate is the *implied* view `exp(effects)` with correct uncertainty. Materializing rates and having the LDO reverse-engineer them is waste and loses the count-variance. This makes the EFG and the LDO speak one language: **count + exposure + structure**.

### II.4 The denominator-asset family (population tensor as first member)

**Denominators are foundational assets** (scope-invariant, national, full-history, versioned — §VI). The population tensor is the *first member*; extensive context fields need their own (total households, establishments, area), themselves fields (often census-derived).

**The population tensor — definitive two-layer design.** Population is a conserved stock; its intercensal age×sex×race structure is *determined by* vital flows (aging/births/deaths/migration), not free. Reconstruction is two layers:

- **Layer 1 (closed-form, cached):** `interpolate_census_composition` — census composition interpolated across years (ILR/compositional geometry), scaled to each year's closure total. This is the classical intercensal estimate *and* the solver's prior-mean/warm-start. Deterministic, instant.
- **Layer 2 (optimization, conditional):** the MSD-I §2.8 six-term process solver (aging/birth/death/migration/race/smooth) refines Layer 1 **only where a flow term carries signal** (≥2 censuses to cohort-age between, a death prior, births, or an observed migration residual). Absent all, the structure is underdetermined and its optimum *is* Layer 1, so the solver returns it directly.

This is not "interpolation vs. process model" — the process model is mandatory (only it yields *demographically valid* latent states, forbidding impossible trajectories); Layer 1 is its prior mean and data-poor optimum. The engine MUST warm-start from Layer 1 (the uniform-seed trap — a flat distribution has zero age-smoothness gradient — is thereby avoided). All three censuses (2000, 2010, 2022) MUST be ingested regardless of a query's time window; a query *slices*, never re-scopes, the tensor (§VI.1). SIDRA table **2093** (2000/2010 census race×sex×situation×age) MUST be added to the compendium as an anchor source; its overlapping age brackets are a CTR disaggregation instance (a clean partition selected, roll-ups treated as derived, never summed).

### II.5 The measurement-model layer (RaceBridge, generalized)

**The pattern:** two sources measure the same latent attribute through different observation processes. The latent quantity (e.g., true self-declared racial composition) is observed *directly* by one source (census) and *through a confusion matrix* by another (administrative DATASUS). The bridge is that source's **observation operator**. This generalizes beyond race to cause-coding drift, ICD-version differences, etc.

**RaceBridge — definitive design.** Administrative race (heteroidentification, per-record) ≠ self-declared race (census). A race-specific rate MUST reconcile numerator and denominator onto the self-declared axis via an emission matrix `C[k][j]`, applied as a Bayesian local-composition reallocation `W[k][j] ∝ C[k][j]·π_local[j]` (normalized), then `counts_self[j] = Σ_k n_k · W[k][j]`, with missing race reallocated by `π_local`. Requirements that fix the current failure:

1. **Never default to identity.** `C = I` is a no-op that emits the biased quantity the bridge exists to remove; it is prohibited as a default.
2. **Literature-informed prior.** `C` MUST be seeded from published Brazilian admin-vs-self-declared discordance patterns (branca↔parda↔preta flows), not fabricated and not identity. The problem is bounded (a ~5×5 row-stochastic, diagonal-dominant matrix), so an educated prior's worst-case error is small by construction.
3. **Per-source matrices.** SIM, SIH, SINASC record race through different processes → `C_SIM`, `C_SIH`, `C_SINASC`, not one global `C`.
4. **Covariate-conditioning** (region at minimum) where data supports it; frontier: `C[k][j | region, cause, …]`.
5. **Uncertainty propagated** (Dirichlet draws → `sensitivity_width`, `cv`), outputs flagged `state ≤ fragile`, dashboard-unsafe, bridge-derived. Upgradeable to linkage calibration as a one-file swap.
6. **Vocabulary/label contracts** (the `"valid"` vs `"valid_admin_race"` class of bug) locked by test.

### II.6 The disease semantic axis

ICD is the **variable-identity axis** (I.4): it indexes the *variable set*, organized as a tree, generating what gets modeled. Definitive design:

- **Disease Concept Registry** — multi-label, provenance/projection-typed `code→concept` assertions imported from multiple groupers (WHO chapter/block = exact; AHRQ CCSR/CCIR/CCC via `icd-mappings` = ICD-10-CM → `approximate` on CID-10, `state ≤ fragile`; Brazilian ICSAP/avoidable lists = exact). A code is NEVER silently forced to one concept; partition views (MSD-I `icd_curated_groups`) are *one view* over the multi-label registry.
- **ICD/CID adapter** — deterministic hierarchy (dot-normalization, closure, nearest-common-ancestor) via `simple-icd-10`, behind the callable registry. **CID-10 is authoritative for DATASUS**: WHO-absent codes (dengue A90/A91) are typed `source_system_specific`, never coerced, but resolve their chapter from the canonical 22-chapter CID-10 table.
- **DiseaseGraph** — the tree/embedding structure yielding the Laplacian `L_D` (the disease-space smoothness prior, §III.4), with a `legality_class`: structural (hierarchy, label-text, external mappings) is safe as a prior; empirical co-occurrence is `context_derived` and **refused as a same-data prior** (the circularity guard).
- **Multi-label overlap** — concept-variables sharing codes are *mechanically* correlated; the membership graph makes overlap exactly known (Jaccard); high-overlap edges are typed `mechanical_overlap`, never reported as discoveries.
- **Embeddings** (optional) — build-time only (cached vectors, ~60 MB, Qwen3-0.6B), never a runtime dependency; label-text embeddings are structural (safe prior/search); empirical embeddings are output-only/partition-separated. **No edge is ever asserted by an embedding**; similarity is a prior weight, not a finding.

### II.7 Context participation (subsumed, not a subsystem)

Context enters as **`field`-kind latent variables** through the same latent-field engine (§III.2): observed at native resolution/sparsity via weighted observation operators, shrunk across scales, extensive fields carrying denominators (I.2). Its *edges are typed by ontology*: a `field↔flow` edge is a *determinant/effect-modifier*, evaluated by the causal ladder, not a co-series. **Effect-modification** ("is arbovirus→microcephaly stronger where sanitation is poor?") is an *interaction term* (§III.4), represented explicitly. The circularity guard applies to context-derived structure. Internal classification axes (sewage by type) follow multi-resolution drill-down (§VIII), using the marginal by default. There is no separate "Context Participation subsystem" — it is these four already-specified pieces applied to field-kind variables.
---

## Part III — The Inference Engine (the Lattice Dependency Operator)

The LDO replaces MSD-I's single-outcome, contemporaneous, pairwise residual scanner. It is one structured model of the *joint dependency* among all variables over the lattice, from which every link is a query. (Conceptual grounding: a *direct link* is a nonzero of the precision matrix — the inverse covariance — which encodes conditional dependence, i.e., relationships that survive after accounting for all other variables. The LDO estimates a *structured* precision so that this is tractable and interpretable.)

### III.1 The object

Variables `X_1..X_p` (diseases × carriers × strata) over `S` places × `T` times, the tensor `X ∈ ℝ^{p×S×T}`. The LDO is a structured estimate of the precision (inverse covariance) over this collection; its nonzero entries — with lags, directions, strengths, spatial patterns, and uncertainties — are the discovered **link records** (§III.7). It is fit **once**; all links are read off it, replacing per-outcome runs.

### III.2 Latent field + observation model (the sparse/dense unification)

Every variable is a **latent quantity defined on the finest lattice**, of which each dataset provides *observations* at its native resolution/sparsity through a **weighted observation operator**. Sparsity is encoded as zero observation weight on unobserved cells; estimation is by weighted likelihood.

Consequences (normative): a variable observed once (e.g., 2018 dengue) informs the cross-section at that time and contributes **nothing** to dynamics (correct — a single slice has no dynamic information); the engine *learns less* from sparse variables in the dimensions where they lack data, and **fabricates nothing**. Sparse and dense are handled *identically* — the only difference is where weight exists. Reconstruction of a *dense* latent field (as for the population tensor) is done **only** for conserved stocks, denominators, or outcomes (I.1, II.4); other sparse variables remain latent-but-weakly-informed, filled only on demand with typed uncertainty. This is the definitive resolution of "must everything be dense": *no — state the model on dense latent fields; let sparse data inform sparsely through weights.*

### III.3 Multiresolution hierarchical shrinkage (hierarchy on every axis)

Every hierarchical axis (space, time, disease) is represented as a **sum of components across scales**, with a prior that shrinks each fine level toward its coarse parent (national + regional + state + municipal deviations; chapter + block + category + leaf deviations). This:

- handles hierarchy natively (all scales coexist; data determines where signal lives at which scale);
- **finishes off sparsity** — a rare leaf/tiny-municipality estimate is dominated by shrinkage toward its better-estimated parent (borrowing strength). Sparsity and hierarchy are solved by *one device*.

This is the established methodology of **small-area estimation / multilevel models / BYM**, with **INLA over latent-Gaussian GMRFs** and the **SPDE** approach as the computational engine that admits arbitrary, mismatched observation resolutions. The coarse→fine "scanner" is therefore a *readout of the multiresolution posterior*, not a heuristic prune: unsupported fine detail is shrunk into its parent, not dropped.

### III.4 The structural decomposition

The precision is estimated under five simultaneous structural assumptions, each reducing unknowns and storage:

1. **Sparse** — most pairs are not directly linked; a graphical-lasso (ℓ1) penalty selects the few real edges and zeros the rest.
2. **Low-rank** — a few **shared drivers** (latent factors: the epidemic wave, seasonality, reporting shocks) are separated out: `precision = sparse(direct) − low-rank(shared)` (the Chandrasekaran–Parrilo–Willsky latent-variable decomposition; identical in form to market-factor/idiosyncratic risk models). This prevents everything-riding-a-wave from being mistaken for a web of direct links.
3. **Separable (Kronecker)** — structure is described axis-by-axis (`Ω_var ⊗ Σ_space^{-1} ⊗ Σ_time^{-1}`) and combined, collapsing an intractable joint into small per-axis factors. An approximation (audited by 6).
4. **Lagged** — the variable set is extended with time-shifted copies; cross-lag precision entries are **directed** links (past→present); their profile over lags is the distributed-lag response (the engine *discovers* the delay, e.g. Zika's 6–9 months).
5. **Prior-regularized** — the spatial GMRF `L_W` (adjacency), the disease Laplacian `L_D` (hierarchy/embedding), and temporal smoothness shrink estimates across neighbors/parents. Interaction (effect-modification) terms enter here as varying-coefficient structure.

### III.5 Copula margins (non-Gaussian handling)

Counts/proportions are non-Gaussian, so each variable is transformed onto a common Gaussian scale (rank/PIT), dependence is modeled there, and mapped back. The MSD-I GLM families are the per-variable transforms (Poisson/NB/binomial/beta-binomial/lognormal/…); **extensive quantities use a count-with-exposure margin** (binomial/Poisson-offset — the denominator principle, I.2), unifying flows and extensive fields. This is Layer 0 of the LDO.

### III.6 The residual nonlinear audit (HSIC, repositioned)

After fitting, the **residual field** (what the structured model could not explain) is scanned by kernel dependence (**HSIC**, with Nyström/RFF approximations) for leftover nonlinear/threshold/seasonal structure the named terms missed. HSIC is the **residual detector, not the engine**: it flags candidates for promotion to named terms; it never *is* the model. This is the definitive resolution of the "one method for nonlinearity" fixation: unify by *architecture* (named nonlinear terms — marginal via copula, interaction via varying coefficients, temporal via lags/seasonality — plus a residual detector), not by a single test. HSIC asserts no edge by itself.

### III.7 Estimation, stability selection, link records

Estimation is penalized (pseudo-)likelihood — graphical-lasso (ℓ1) + nuclear-norm (low-rank) + GMRF/hierarchy quadratics — solved by proximal-gradient/ADMM, matrix-free, on GPU (§V). **Multiplicity** is controlled by **stability selection** (refit on lattice subsamples; keep edges recurring above a frequency threshold), which controls the edge-set error without per-pair testing. Output is the **link record**:

```
LinkRecord: { source_var, target_var, lag_k,
              edge_type ∈ {contemporaneous, lagged_directed, latent_shared,
                           nonlinear_residual, mechanical_overlap},
              weight, partial_correlation, response_curve_ref, spatial_field_ref,
              stability, uncertainty, confounding_factor_refs, projection_status,
              code_system, topology_role, overlap_jaccard, certification_status, warnings }
```

The MSD-I 17-key output bundle is preserved; the `Hypotheses` key carries link records (the pairwise-row schema is retired).

### III.8 Certification and the anti-spurious gates

Every edge passes a **certification gate** before promotion: holdout stability, regularization-path agreement, latent-vs-lag separability diagnostics, and the reliability gates (`n_eff` floors, fragility). Directionality is reported **only** where time (lagged edges) or a licensed structure-learning method (§IV) permits; contemporaneous edges are undirected by default. Standing aborts (extend MSD-I §10): context-derived spatial/disease prior sharing provenance with the tested variables; disease-concept edge without overlap accounting above the Jaccard threshold; edge promoted without holdout stability or propagated uncertainty; reconstructed tensor promoted to verified without holdout/uncertainty. Consistent with §0.4: the LDO **produces causal hypotheses; it does not certify causation.**

---

## Part IV — Causal Escalation

The LDO yields an *associational + temporal-precedence skeleton*. Causal claims are a typed **ladder** on top, each rung adding assumptions and labeled by them. Classical bivariate Granger is *subsumed* (it is the crude special case of the LDO's conditional, latent-controlled lagged edges); it is not added.

- **Rung 0 — LDO baseline.** Conditional, latent-controlled, lagged, nonlinear-audited dependence. Directed only by time.
- **Rung 1 — Orientation without experiments**, auto-applied **only where assumptions are machine-checkable**: collider/v-structure detection (PC/GES/FCI), and **non-Gaussian orientation (LiNGAM / additive-noise)** — the causal direction is identifiable from higher moments when data is non-Gaussian (which epidemiological counts are), and the copula margins already characterize this. Real extra science the LDO does not natively extract.
- **Rung 2 — Quasi-experimental leverage**, triggered by detected structure: interrupted-time-series around dated shocks (epidemic onset, policy dates), difference-in-differences, negative-control outcomes.
- **Rung 3 — Interventional identification (do-calculus)** for `E[Y | do(X)]` — **expert-invoked only** (identifiability assumptions are not machine-verifiable).

Every causal claim MUST carry its rung and assumptions; strength is explicit and never overstated. Automation is conservative: auto-apply machine-checkable rungs, escalate the rest to humans.

---

## Part V — Computation & Scale

National-monthly discovery MUST fit commodity hardware **with quantified validity**. The methods that make it fit are, mostly, *randomized algorithms with provable error bounds* — which is why validity is preservable.

### V.1 Compute envelope (binding)

Target: RTX 4050 (6 GB VRAM), i7-13700H, 32 GB RAM. All numerics honor `compute.yaml`: float32 bulk, `max_vram_fraction ≤ 0.80`, PyTorch/CUDA. A run that cannot fit MUST refuse with `scale_exceeds_compute_envelope`, never silently subsample.

**This envelope binds every heavy numeric — the population-tensor / denominator solve (§II.4) included, not just the LDO.** Three consequences are contracts, not preferences: (a) a solve's working set MUST be numpy/torch arrays in **float32 bulk** — never `O(n_cells)` Python-object containers (a national demographic tensor is ~10⁸ cells; Python float *tuples* at ~32 B/element OOM a 32 GB box before the solver runs); (b) a solve whose dense form exceeds the envelope MUST be **blocked** (the population tensor is separable across localities given per-locality closure — solve in muni-blocks so peak memory is `O(block)`, not `O(national)`) or else refuse — **never OOM**; (c) where `compute.yaml` marks a task CUDA-enabled, a **wired GPU path MUST exist** with a CPU fallback and device-level telemetry — an aspirational, unwired flag over orphaned kernels is prohibited (it is precisely how orphaned/conflicting code accrues). *`POP-02` brought the population solver into compliance: (a) numpy-array storage (was Python tuples), (b) exact locality-blocked solve (peak `O(block)`), and **(M7) numpy-array loss/solver returns** — the last was the true per-eval hotspot: `tuple(float(x) for x in gradient)` was ~99% of a loss evaluation (110 ms → 22 ms), and it, not the dense math, is why the CPU solve read slow. (c) The orphaned population GPU kernel (a 36-line no-op stub, zero callers) was correctly deleted, but the reasoning attached to it — "the CPU-blocked solve is fast, so GPU is a deferred optimization" — was a **misdiagnosis**: the CPU solve was slow because of the tuple round-trip, not because the dense math is cheap. With M7 landed the dense loss/gradient algebra now dominates, and it is an ideal CUDA workload — a measured microbenchmark shows ~15× on the dense passes at block scale, VRAM-trivial (a 5.4M-cell block = 236 MB), with float32-bulk/float64-reduction (§V.4) reproducing the objective exactly. So a GPU port of the loss + SPG loop **is** justified (`POP-02` GPU item, now un-gated) and must re-add a wired `population_tensor_solve` flag with a CPU fallback + device telemetry per (c) above. GPU is already wired+used for HSIC and STDFM.*

### V.2 Structural levers: sparsity and separability

- **Sparsity** — the spatial GMRF (~6 neighbors/row), the disease Laplacian, and the graphical-lasso precision are all mostly zeros; stored sparse (`scipy.sparse`, sparse Cholesky), never densified.
- **Kronecker separability** — the single biggest lever: operations (matvec, solve, log-determinant) factor across the small per-axis factors, collapsing a `(p·S·T)²` object (~10¹⁴) into `p²`, `S²`, `T²` pieces (~10⁶). Precedent: **GPyTorch** (GPU structured-Gaussian inference).

### V.3 Randomized numerical linear algebra (bounded)

- **Randomized SVD** (Halko–Martinsson–Tropp) for the low-rank factors — bounded error vs. the best rank-`r` approximation. (`sklearn.randomized_svd`, `torch.svd_lowrank`.)
- **Stochastic log-determinant** (Hutchinson + stochastic Lanczos quadrature) for the Gaussian likelihood — unbiased, variance controlled by probe count (as in GPyTorch).
- **Sketching** (Johnson–Lindenstrauss) for the neighborhood regressions — `(1±ε)` distortion guarantees.

### V.4 Out-of-core, mixed precision, matrix-free solvers

- **Streaming sufficient statistics** — `XᵀX`, `Xᵀy` are tiny (`p²`) regardless of row count; stream Parquet/Arrow (DuckDB) once to accumulate them; national data need never fit in RAM.
- **Matrix-free iterative solvers** — CG/MINRES with preconditioning (incomplete Cholesky, Kronecker-factor, pivoted-Cholesky); never invert.
- **Mixed precision** — float32 bulk; float64 (or Kahan summation) on reductions, log-dets, and ill-conditioned solves; **monitor condition numbers** and escalate to float64 when they exceed threshold (priors improve conditioning by construction).

### V.5 The Adaptive Precision Controller (uncertainty as compute objective)

Uncertainty MUST be an **actionable control signal**, not a report annotation. The engine runs as an **anytime** computation: always a current best answer with current uncertainties, monotonically tightened by more compute. Every approximation knob (SVD rank, Hutchinson probes, sketch size, CG iterations, resolution depth, bootstrap count) is a *dial* trading compute for precision; a scheduler sets each **per-subproblem** by expected decision-relevant uncertainty-reduction per unit cost (value-of-computation / active-inference).

The self-correcting property (the demand that uncertainty *do something*): **when the *numerical* approximation is the dominant uncertainty for a decision-relevant edge, the scheduler's best action is to escalate that edge to exact computation** — a bad approximation triggers its own correction exactly where it matters, and is left alone where it doesn't. The user/policy sets a target uncertainty for decision-relevant quantities; the system spends until the target is met or the budget exhausts, then reports which quantities met it and which remain approximation-limited (typed, never silent).

### V.6 The validity contract

Trading compute for scale without lying requires, normatively:

1. **Bounded, not heuristic** — prefer methods with error bounds (V.3); their error is a known distribution.
2. **Propagate approximation error into results** — numerical error becomes a component of each affected edge's `uncertainty` (widened error bars), consistent with the prime directive. Never present an approximated number as exact.
3. **Exact certifies approximate** — a *state*-scale exact run MUST agree (within propagated bounds) with the national approximate run where they overlap; disagreement beyond bounds rejects the approximation loudly.

### V.7 The data storage lifecycle (fetch → build footprint)

Scale is bounded by **disk** as much as by compute; a full national all-source acquisition must land in the low tens of GB, not the hundreds. On-disk redundancy is correctness-adjacent, not cosmetic: a second serialization of the same rows misleads provenance and caps scaling. The fetch→build layer is therefore governed by four contracts:

1. **One canonical serialization per acquired unit.** DATASUS retains only the raw-coded, full-fidelity `processed.parquet` (ZSTD, **all** source columns — column fidelity is a §II.2 requirement so a future registry binding never forces a re-fetch; pruning is prohibited). The R-native `raw.rds` and the microdatasus semantic sidecar are **not** persisted — neither is on the consumption path (the in-house codebook, §II.2, translates the raw-coded parquet; microdatasus is a *retiring fetch transport*, not the translator). Cache validity keys on the **consumed** artifact + its manifest, never on a dead sidecar.
2. **One raw archive per SIDRA response.** The client response cache *is* the archive; the extract dump carries provenance (request + payload hash), not a duplicated payload; failures keep their small error body.
3. **Lazy views, not re-materialization.** The per-UF `combined` → `canonical` → `national` layers MUST be lazy `scan_parquet` views over a Hive-partitioned dataset, not full copies; a single national file is materialized only when a consumer genuinely needs one, by streaming `sink_parquet`. Re-materializing the same events at each layer is prohibited.
4. **Bounded run bundles.** The `ReproducibilityManifest` and every artifact manifest reference large tensors by path/hash; they **never** inline `O(n_cells)` arrays (inlining the population/migration tensors produced a ~430 MB manifest at national scale — the same anti-pattern as V.1(a)).

Codecs are tuned (ZSTD + dictionary), not left at library defaults; debug ancillaries are pruned on success; intermediates are GC-able with the consumed artifact as the safety gate. Contracts 1, 2, and 4 are **landed** (`STORE-01`); contract 3 (the lazy-view collapse) is the open item (`STORE-02`).
---

## Part VI — The Foundational Asset Layer & Lifecycle

The deepest architectural gap resolved by MSD-III. It fixes the class of error where a query's scope silently amputates a scope-invariant asset (the "one-census" collapse).

### VI.1 Asset tiers and the scope-invariance invariant

- **Foundational assets** — *scope-invariant*: the harmonized substrate, the denominator-asset family (population tensor + household/establishment/area denominators), the spatial graph, the disease registry/graph, context cubes, and the discovered link skeleton. Built **once, at national + full-history scope**, versioned, refreshed on cadence.
- **Query artifacts** — *scope-dependent*: produced by *slicing* foundational assets.

**The invariant (normative, the contract that prevents the whole class of bug):** *a foundational asset's value never depends on any query's scope. A query selects a view over foundational assets; it MUST NOT trigger a rebuild at reduced scope.* Building a scope-invariant asset at reduced scope is a **correctness bug** (it drops out-of-window anchors like the 2000/2010 censuses and breaks national constraints like migration enclosure), not merely inefficiency.

### VI.2 Build triggers, versioning, pinning

- **Builds are triggered by data arrival, not queries.** A new EstimaPOP vintage, census, SIDRA release, or DATASUS year triggers a *foundation build* of affected assets.
- **Immutable, versioned artifacts** with input manifests: `population_tensor@v2026.1 ← {census: 2000,2010,2022; estimapop:…2025; code:<hash>; certification:verified}`.
- **Every query pins the foundation versions it consumed** → exact reproducibility from `(foundation versions, query)`.

### VI.3 Incremental update and snapshots

New data updates only affected assets, and where possible only affected regions/times (the multiresolution structure propagates fine updates upward). A new EstimaPOP year *extends* denominator series without disturbing prior years; a new *census* triggers a fuller demographic re-solve. The link skeleton is itself a **versioned, snapshotted** foundational asset; queries interrogate a snapshot.

### VI.4 The continuous discovery process

The operational end-state of the foundational-skeleton logic: a **background process** that, under the Adaptive Precision Controller (§V.5), spends available compute on the highest-value uncertainty reductions, incorporates new data, and re-certifies edges, producing new skeleton snapshots. Users *steer* it (§VII). This is a real systems layer — scheduler + artifact/version store + provenance DB — and the **heaviest engineering lift**; MSD-III specifies it as design and stages it as a later operational milestone (§XI).

### VI.5 The living skeleton

The discovered dependency structure is maintained as a living foundational asset. This reframes the project's end-state: PegaSUS does not "finish epidemiology" (impossible — §0.4 information ceiling; and association is not causation — §IV; and higher-order questions live on top). It maintains a **living associational skeleton** from which causal escalation and interpretation proceed indefinitely.

---

## Part VII — The Interaction Model

Because the skeleton is discovered proactively as a foundational asset, "query for a link" is largely obsolete. The user's role shifts up a level; the levers change.

### VII.1 The five verbs

PegaSUS is a *living scientific model* users **interrogate, escalate, lens, steer, and stress-test**:

1. **Interrogate** — traverse the discovered graph ("everything linked to microcephaly, ranked by strength and certainty"; "where is this link strongest"). Reads, not new analyses.
2. **Escalate** — invoke the causal ladder (§IV) on a subgraph ("orient this edge"; "estimate the interventional effect of sewer expansion"; "is this confounded by X"). On-demand heavy compute, governed by §V.5.
3. **Lens** — select scope/resolution ("children only"; "monthly"; "the Northeast"). A view over the skeleton; a lens finer than what's computed triggers a *local deep search* (§III.3 under §V.5 budget).
4. **Steer** — direct discovery ("search this region deeper"; "prioritize compute here"), exposing the §V.5 controller as a lever.
5. **Inject & falsify** — bring a hypothesis to be situated and stress-tested against the discovered structure.

### VII.2 DataScope × ExecutionStage, reframed

Two orthogonal axes remain: **DataScope** (`core_vital | contextual | full` — which substrate exists) and **ExecutionStage** (`validate | compile | investigate` — how far the pipeline runs, default `investigate`). MSD-III reframes them from "how much to build for this query" to "which *view/steering/escalation* the user invokes over an already-living model." All output keys always exist; empty keys carry typed reasons.

### VII.3 Querying as directing bounded compute

There is **no artificial limitation, only resource-boundedness**. A request to search an `unsearched` region is not fighting a limit — it is *allocating bounded compute*; the user's query is a *prior over where computation is valuable*. This dissolves the "end of epidemiology" tension: the skeleton can be largely built, but interrogation, escalation, and renewal by new data are inexhaustible.

---

## Part VIII — Bounded Exhaustiveness & Coverage

Absolute exhaustiveness is impossible (§0.4 information ceiling; combinatorial link space). MSD-III requires **honest bounded-exhaustiveness** instead of a false completeness claim.

### VIII.1 Why naïve coarse-screening fails

Aggregation *hides* links: **Simpson reversal** (subgroup effects reverse in aggregate), **cancellation** (opposite regional effects sum to zero), **thresholds** (effects only above a level), **sparsity dilution** (rare-code links swamped in a chapter). So a coarse *test of association* has real false negatives, and "drill only where a coarse link fires" systematically misses exactly the localized/heterogeneous/threshold/rare effects that are often most interesting.

### VIII.2 The three-part remedy (normative)

1. **Screen on sensitivity, not the aggregate mean.** Coarse passes MUST screen on statistics *designed to fire on what aggregation hides* — subgroup **heterogeneity**, **dispersion**, or **max-subgroup signal** — not the pooled mean. A cancellation case has zero mean but high heterogeneity, so a heterogeneity screen triggers the drill-down. This converts the coarse pass from a *test* (with false negatives) into a *sensitive filter* (tuned for recall).
2. **Random deep audits.** On a random sample of pruned branches, run the full fine analysis anyway, to *measure* the false-negative rate empirically and catch a bounded fraction of misses.
3. **Typed coverage manifest.** Whatever is not searched is recorded as a typed `unsearched` region (at what resolutions/conditioning-sets/functional-forms coverage was complete, and where pruned and why) — never a silent gap. Users may request deeper search of an `unsearched` region (§VII.1 steer).

The underlying **sparsity-of-truth assumption** (most fine links are null) MUST be stated in the coverage manifest, not hidden. Residual risk: a *weak* fine-only phenomenon with no coarse footprint sits near the information-detectability floor and may be undetectable by any method — an information limit (§0.4), disclosed, not a flaw.

---

## Part IX — The Validation Program

For a system claiming to *automate epidemiology*, discoveries are worthless without a *quantified* trust level. Validation is a first-class, standing part of PegaSUS: **no capability is certified until it passes its relevant prong.** The strategy is to test on cases where the answer is already known.

### IX.1 Known-positive and known-negative controls

- **Known-positive controls** — a curated set of *established* links (Zika→microcephaly; poor sanitation→diarrheal disease; vaccination→disease decline; others with DATASUS/SIDRA support). The engine MUST recover these. The floor test.
- **Known-negative controls** — pairs known unrelated, and negative-control *outcomes*. The engine MUST NOT flag these; the rate at which it does *is* the measured **false-alarm rate** — the spurious-link risk, quantified rather than assumed.

### IX.2 Synthetic ground truth

Generate data from a *known* dependency structure (planted edges, lags, latent factors, spatial structure); measure recovery (precision/recall of the edge set). The only source of exact ground truth, and the required test for the hardest items: **separability adequacy** (simulate non-separable structure — does the residual-HSIC layer catch it?) and **certification power** (simulate a latent-factor-vs-true-lag confound — does the gate distinguish them?).

### IX.3 Temporal holdout; exact-vs-approximate

- **Temporal holdout** — fit through year `T`, verify on `T+1`. Real links persist/predict; flukes evaporate.
- **Exact-vs-approximate** — state-scale exact certifies national-scale approximate (§V.6).
- **Perturbation/stability** — edges must survive resampling/subsampling/tuning perturbation (stability selection).

### IX.4 The standing battery

MSD-III makes this battery normative. A certified result is expressible as: *"recovers known truths; measured false-alarm rate X%; synthetic-recovery accuracy Y%; holds out-of-sample."* That sentence is what makes output **science rather than assertion.** The battery simultaneously (a) closes the "is it spurious?" credibility gap and (b) provides the *empirical* tests for separability, identifiability, and certification power that cannot be settled by argument.
---

## Part X — Compliance Baseline

New work sits on an MSD-I-compliant codebase. This part records where compliance stands and the refactor it enables.

### X.1 MSD-I compliance progress review

A substantial remediation pass has landed; **nearly every S0/S1 finding now has an implemented mechanism.** Verified status (re-confirm each with a Phase-0 contract test before building on it):

**Remediated:** `SHE-DEC-01`; `SHE-CNES-01` (forbidden bed total removed); `SHE-CNES-02` (CNPJ gate applied); `SHE-SINASC-01` (baked indicators removed); `SHE-POP-01/02` (loss intact; RTS smoother + ADMM landed); `EFG-LEG-01`, `EFG-DECL-01`, `EFG-EXEC-01`, `EFG-CMP-01`; `EFG-Q-01` (Moran/roughness/entropy computed); `RACE-01` (local-π posterior crosswalk — but see §II.5: the *content* was a no-op identity/fixture, now corrected); `PIRS-HSIC-01`, `PIRS-FAM-01` (11 GLM families), `PIRS-SPAT-01` (ICAR/GMRF penalty), `PIRS-REG-01`; `OUT-01/03`; `XCUT-03`.
**Partial:** `SHE-NORM-01` (declarative path wired; SIH/CNES routing + de-dup remain); `SIDRA-CTX-01` (profile-gated; end-to-end context materialization remains); `EFG-REG-01`; `REG-DEAD-01/VALIDATOR-01`; `XCUT-04`.
**Open:** `XCUT-02` (vectorized normalizers still duplicate the declarative decode path).

**Reading:** the S0 core is essentially closed; the dominant residual risk is *unlocked* compliance (remediations not yet pinned by tests). Phase 0 (§XI.2) locks them before new work.

### X.2 The refactor / de-engorgement program

The ~40k-line codebase is not bloated (mean ~173 lines/file) but is **fragmented along development-slice boundaries** (PIRS is ~10 manifest-passing micro-stage files mirrored by ~10 workflow wrappers; ~40 registries in two worlds; duplicated normalizers). Refactor targets, each gated behind Phase-0 contract tests so consolidation cannot change results:

1. **Collapse the PIRS slice-zoo** into one in-memory LDO orchestrator package (`margins / lowrank / precision / edges / residual_scan / certify / orchestrator`), retiring the manifest-passing micro-stages and their wrappers — done *as part of* building the LDO.
2. **De-duplicate normalizers** — the declarative record-level normalizer becomes the single decode path (resolves `XCUT-02`).
3. **Unify registries** — migrate the ~40 files (two worlds) into the one `kind`-tagged schema (§II.1).
4. **One `workflows/investigate.py`** entrypoint replacing the step wrappers.

---

## Part XI — Implementation Plan

### XI.1 Build discipline (the small-model contract)

Executable by smaller agents, so: every work item is **Red → Green → Refactor** (write the failing contract test first; make it pass; consolidate). One item = one PR citing its ID and parent finding. **Never delete a contract test to pass a build.** If uncertain, the prime directive and its development corollary (§0.3–0.4) are the tie-breakers. Do items in printed order; the dependency note is the law of sequencing.

### XI.2 Master sequencing

```
Phase 0   Lock compliance with contract tests + de-duplicate      (X.1, X.2)
Phase 1   Platform: unified registry, spatial graph, CommonPanel,  (§II.1, I.4,
          DataScope×ExecutionStage, denominator-asset framing       §II.4, VII.2)
Phase 2   Data plane: EFG measured-quantity output, two-layer       (§II.3, II.4,
          population tensor, RaceBridge redesign, disease axis       II.5, II.6)
Phase 3   The latent-field + multiresolution engine                 (§III.2, III.3)
Phase 4   The LDO core, layer by layer                              (§III.4–III.8)
Phase 5   Causal ladder + Adaptive Precision Controller + validation (§IV, V.5, IX)
Phase 6   Foundational Asset Layer + continuous discovery            (§VI)
Phase 7   Multi-resolution scanning, bounded-exhaustiveness, scale   (§VIII, V, III.3)
```

Phase `N` requires `N−1` closed. Rationale for order: lock correctness first; build the *data plane* the engine consumes before the engine; build the *latent-field/multiresolution substrate* (which dissolves sparse/dense) before the LDO decomposition on top; add causal/uncertainty/validation as the science layer; make it a maintained foundational asset; then push scale and exhaustiveness. The single highest-leverage new build is **Phase 3** — it dissolves the longest-standing block (sparse vs. dense, hierarchy on every axis) and rests on established methodology (small-area estimation / INLA / SPDE).

### XI.3 Work items (consolidated)

Carried forward and unified under MSD-III (predecessor `MII-*` / `MII-DIS-*` IDs remain valid cross-references):

- **Phase 0:** `T0-1` source-reality contracts; `T0-2` EFG legality/state-tensor; `T0-3` RaceBridge numerics; `T0-4` inference baseline; `REFACTOR-01` normalizer de-dup.
- **Phase 1:** `REG-07` unified `kind`-tagged registry + validator; `SPG-01/02/03` spatial graph + circularity guard; `PANEL-01` CommonPanel + per-cell provenance + month resolution; `SCOPE-01` DataScope×ExecutionStage.
- **Phase 2:** `EFG-OUT-01` measured-quantity objects; `POP-01` two-layer tensor (Layer-1 closed form + warm start + conditional solver) + table 2093; `POP-02` **tensor compute contract** (§V.1: float32 numpy/torch arrays not Python tuples; locality-blocked solve; wired GPU kernels + CPU fallback + device telemetry; refuse-not-OOM) — remediates the OOM that blocks full-national demographics; `RACE-01..07` per-source literature-informed emission matrices (§II.5); `DIS-01/02/03` concept registry + ICD adapter + DiseaseGraph.
- **Phase 3:** `LF-01` latent-field observation model (weighted, arbitrary-resolution); `MR-01` multiresolution hierarchical shrinkage on space/time/disease.
- **Phase 4:** `LDO-00` assembly + link-record schema; `LDO-01` copula margins (incl. extensive count+exposure); `LDO-02` sparse+spatial+temporal precision; `LDO-03` low-rank latent factors; `LDO-04` lag extension (the Zika capability); `LDO-05` edge readout + stability selection + residual HSIC + `DIS-05` overlap; `LDO-06` certification + orchestrator (+ slice-zoo refactor).
- **Phase 5:** `CAUSAL-01` rung-1 non-Gaussian orientation; `CAUSAL-02` quasi-experimental where shocks exist; `APC-01` adaptive precision controller (simple version); `VAL-01..04` the validation battery.
- **Phase 6:** `FAL-01` asset tiers + scope-invariance + versioning + pinning; `FAL-02` incremental update + snapshots; `DISCO-01` continuous discovery scheduler (later milestone).
- **Phase 7:** `RES-01` coarse→fine multi-resolution scanning; `EXH-01` heterogeneity screens + random audits + coverage manifest; `SCALE-01` national **LDO-scan** tiling/streaming + RandNLA. *(Note: `SCALE-01` is the inference-scan scaling item; the national **acquisition/compile** path is `NAT-01`, a data-plane item — see cross-cutting below. Do not conflate.)*
- **Cross-cutting (engineering-debt remediation — §V.1/§V.7 contracts; sequenced opportunistically, not phase-gated):** `NAT-01` national acquisition + combine + national compile scope (all 27 UFs → national municipality panel) — **landed**; `STORE-01` storage lifecycle contracts V.7(1,2,4) (single-serialization DATASUS v4 bridge, slim SIDRA, non-inlining manifests, ZSTD, GC) — **landed**; `STORE-02` V.7(3) lazy `scan_parquet` views over Hive-partitioned canonical/facts (retire the `combined`/`canonical`/`national` re-materialization) — **open**; `POP-02` the §V.1 tensor compute contract (above) — **M1/M3/M4/M5/M7 landed** (numpy-array problem storage + exact locality-blocked solve + vectorized emission + **numpy-array loss/solver returns**, the last being the true hotspot — `tuple(float)` gradient round-trips were ~99% of a loss eval); **GPU port now justified** (the dense math dominates post-M7; ~15× measured) + solve parallelization + M6 input-construction memory remain. `FAL-POP` national+full-history scope-invariant population-tensor asset (§VI.1/§II.4) — **open** (the deeper issue: the tensor is currently rebuilt per-query at the intent window, covering only the query's years — a §VI.1 correctness bug; interpolate_census_composition itself is correctly the §II.4 Layer-1 prior-mean).

### XI.4 The program acceptance test

The single regression that certifies the vision (Appendix D): a **monthly Alagoas** run, **DataScope core_vital+SIH/SINASC**, **ExecutionStage investigate**, **with no forced selectors**, MUST produce a `LinkRecord` with `edge_type=lagged_directed`, source ≈ arbovirus admissions, target ≈ microcephaly, `lag_k` peaked in **6–9 months**, `confounding_factor_refs` including the 2015–16 epidemic-wave latent factor, a non-trivial `stability`, and propagated `uncertainty` — the link **discovered without being named.** Until it passes, MSD-III's goal is not met.

---

## Part XII — Permanent Limits & Standing Principles

Restated as design principles (§0.4), so no future agent mistakes them for bugs:

1. **The information ceiling** — discoverable epidemiology is bounded by data information, not compute. Larger hardware searches more candidates; it does not make them true. *Design consequence:* the laptop bounds the *search*, not the *science*; the Adaptive Precision Controller (§V.5) spends the bounded budget where it changes conclusions.
2. **Association is not causation** — the skeleton is substrate; causal strength is only ever as good as the escalation rung's assumptions, typed honestly (§IV). *Design consequence:* every causal claim carries its rung; no autonomous Rung-3.
3. **Some misclassification is unidentified without external data** — measurement bridges (race, cause coding) need linkage/validation; the duty is explicit, propagated uncertainty and the best *externally-informed* estimate (§II.5) — never fabrication, never a no-op.

These three are the honest edge of the system. Everything else is designed or defined labor.

---

## Appendices

### Appendix A — Glossary

- **Flow / stock / field** — occurrence (count+exposure) / conserved level (process/reconstruction) / standing property (typed covariate).
- **Extensive / intensive** — scales-with-size (needs denominator) / already-a-ratio (comparable).
- **Axis** — a coordinate system on an entity type; its structure (metric-hier / tree-hier / flat-cat) sets its role and prior.
- **Lattice** — Place × Time, the support domain; multi-resolution on every hierarchical axis.
- **LDO** — the structured precision (inverse covariance) over all variables across the lattice, plus the machinery to estimate it and read off links. The inference engine.
- **Precision matrix** — inverse covariance; nonzeros = direct (conditional) links.
- **Sparse / low-rank / separable / lagged / prior-regularized** — the five structural assumptions making the precision tractable and interpretable (§III.4).
- **Latent factor / epidemic wave** — a shared driver many variables move with; separated out (low-rank) so co-movement isn't mistaken for direct links.
- **Copula margin** — per-variable transform to a common Gaussian scale (extensive → count+exposure).
- **HSIC** — residual nonlinear-dependence detector (audit, not engine).
- **Stability selection** — multiplicity control by edge recurrence across subsamples.
- **Measurement model / bridge** — observation operator reconciling two sources measuring one latent attribute (RaceBridge is the first instance).
- **Foundational asset** — scope-invariant, national, full-history, versioned; sliced by queries, never rebuilt at reduced scope.
- **Adaptive Precision Controller** — uncertainty-driven compute scheduler; escalates approximation-limited decision-relevant results to exact.
- **Causal ladder** — Rungs 0–3, each adding typed assumptions from association to intervention.

### Appendix B — Compliance finding cross-reference

Status summary in §X.1. The compliance report remains the canonical detail; Phase-0 contract tests (`T0-*`) pin each remediation before dependent work. `XCUT-02` is the one genuinely open finding (→ `REFACTOR-01`).

### Appendix C — Document lineage map

- **Retained parents:** `MSD.md` (substrate contracts, by reference + §11.5 amendments), `PEGASUS_COMPLIANCE_AND_REMEDIATION.md` (correctness baseline).
- **Absorbed & superseded by MSD-III:** MSD-II (§III LDO, §XI TDD, §X compliance review) · gap analysis (§III diagnosis) · vision assessment (§III LDO design) · disease-semantic-axis (§II.6) · conceptual-foundations (Part I ontology) · architecture-explorations (§III.2–3, §IV, §V.5, §VII, §VIII) · closing-the-open-items (§II.5 RaceBridge, §VI lifecycle, §IX validation, Part XII limits) · mathematics companion (Part I + §III grounding; retained only as optional pedagogy).

### Appendix D — The program acceptance test

Specified in §XI.4. It is the one test whose passing means PegaSUS discovers a real, lagged, spatially-heterogeneous epidemiological relationship **without being told it exists** — the operational definition of "automated epidemiology" for this system.

### Appendix E — §11.5 MSD-I amendments (apply before dependent builds)

Amend `MSD.md`: **§1.6** two-lifecycles + CommonPanel + per-cell provenance + month/year resolution; **§2.13** denominator-asset family + two-layer reconstruction + the measurement-model layer; **§2.14/§6.10** SpatialWeightGraph + DiseaseGraph + circularity aborts; **§3.16** concept grammar + variable generation across the disease axis; **§6′ (rewrite of §6)** the LDO (copula margins → structured precision → residual audit → certification); **§8.3 (replace)** link record; **§3.17** entity-coordinate axis typing; **§1.5/§9.3** DataScope × ExecutionStage; **§10** the new aborts; **§12 (scope)** widen to "field compiler + joint spatiotemporal dependency discovery + causal escalation over a living foundational skeleton."

---

*End of MSD-III. This is the definitive plan: the ontology that types every quantity, the data plane that compiles the clean field, the LDO that discovers the dependency structure, the causal ladder that escalates it, the computation that fits it on commodity hardware with quantified validity, the foundational-asset lifecycle that maintains it, the interaction model that steers it, and the validation battery that makes it trustworthy — with the three permanent limits named as principles. Build from Phase 0; keep every contract test green; the acceptance test (§XI.4) is done-ness.*
