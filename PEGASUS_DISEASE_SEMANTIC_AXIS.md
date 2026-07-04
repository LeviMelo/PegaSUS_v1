# PegaSUS — The Disease Semantic Axis (MSD-II companion, §II.13)

**Lineage.** Extends `PEGASUS_MSD_II.md` (the LDO, the SpatialWeightGraph §II.4, the concept grammar §II.5, the CTR kernel §II.3, multi-resolution §II.7). Inherits the MSD and the compliance report as constitutional parents. This document is normative where it says MUST; it introduces one new spec section (§II.13) and a work-item family (`MII-DIS-*`).

**Provenance of inputs.** Prompted by a handoff (an external-AI discussion of ICD libraries: `simple-icd-10`, `icd-mappings`, AHRQ CCSR/CCIR/CCS, pediatric CCC, phecodes, `icdcodex`, ICD embeddings, Qwen3). That handoff is treated as raw material, not design. Several of its framings are corrected below; its concrete resources are accepted where they earn their place.

**One-sentence thesis.** The disease dimension should stop being a *per-field restriction parameter* (today's σ_C) and become the **primary organizing axis of the LDO's variable set** — carrying its own hierarchy, multi-label concept structure, and (optionally) an embedding geometry — so that disease structure enters the estimator as a **prior on the variable-dependency operator**, exactly parallel to how geography enters as the SpatialWeightGraph prior, but acting on the *variable* dimension rather than the *cell* dimension.

---

## 1. Critical assessment of the handoff — what is right, wrong, and missing

**Right, and worth adopting:**
- The three-layer decomposition — *mapping tables* (code→concept), *hierarchy/topology* (code→position), *NLP/embeddings* (code→vector) — is a sound taxonomy and I keep it (§4).
- Multi-label concept membership (CCSR lets a code sit in several categories) is correct and important: disease semantics are not a partition. This is a real upgrade over PegaSUS's current partition-only `icd_curated_groups.yaml`.
- Provenance/projection-lossiness typing (ICD-10-CM ≠ Brazilian CID-10) is exactly right and aligns with PegaSUS's anti-silent-fiction ethos.
- `simple-icd-10`'s deterministic hierarchy ops and `icd-mappings`' importable groupers are genuinely useful substrate.

**Wrong, or dangerously imprecise — corrected here:**
1. **"ICD as an axis just like geo/time/demographic" is a type error.** Geo and time are *support* axes (the domain of every field); age/sex/race are *stratification* axes (subdivisions within a cell). ICD is neither — it is a **variable-identity axis**: it determines *which quantity a variable measures*. Conflating it with support axes leads to nonsense like "adding an ICD dimension to the panel cells." The correct elevation is different and stronger (§2, §5).
2. **Embeddings do not "generate hypotheses."** The handoff repeatedly implies embeddings produce discovery. They do not. Embeddings can supply a *prior* and a *search order*; the **data** defines every edge. Letting an embedding assert a link is a confounding leak (§5.5).
3. **Empirical co-occurrence embeddings as a prior are circular.** The handoff proposes "Brazilian empirical embeddings from SIM/SIH/…" and then wants to use disease similarity to guide discovery on the same data. Using a co-occurrence-derived similarity as a prior in a test of that co-occurrence smuggles the hypothesis into the null — the exact circularity §II.4.1 forbids for spatial weights. Only *structural* similarity (hierarchy, label text, external mappings) is safe as a prior; empirical similarity is an **output** or must be strictly partition-separated (§5.5).
4. **Multi-label overlap is a correctness trap, not just a feature.** If two concept-variables share underlying codes (CCSR multi-label), they are **mechanically** correlated because they count overlapping events. Reporting that as a discovered epidemiological link would be a serious error. The overlap is *known structure* and must be modeled as such (§5.3).

**Missing entirely from the handoff — the load-bearing parts:**
- **How disease structure becomes a term in the LDO's mathematics.** The handoff never connects ICD to the precision operator. That connection (a disease-space Laplacian prior `L_D` on `Ω_var`) is the whole point (§5.2).
- **The hard-constraint / soft-prior duality.** Hierarchy gives *additivity constraints* (children sum to parent) for the CTR kernel; embedding gives *smoothness priors* for the LDO. Same structure, two different mathematical roles (§5.2, §6).
- **PegaSUS already has richer disease semantics than "code→concept":** the SIM cause **chain** (LINHAA–D ordered, LINHAII associated) distinguishes *underlying cause* from *mention*. The disease axis must respect that; a "link to a cause" is not the same as a "link to a mention" (§5.1).
- **The autonomous-discovery mechanism.** The disease axis is what turns the combinatorial link search into a *guided* search — the missing piece for "autonomous investigation" on a laptop (§5.4).

---

## 2. The type-theoretic correction: three kinds of axis

| Axis kind | Examples | Role | Enters the LDO as |
|---|---|---|---|
| **Support** | geography, time | the domain of every field (indexes cells `(s,t)`) | the lattice; spatial prior `L_W`, temporal dynamics `Σ_time` |
| **Stratification** | age, sex, race | subdivides *within* a cell | resolution of a variable; demographic tensor axes |
| **Variable-identity (semantic)** | **ICD/cause**, procedure, carrier-concept | determines *what a variable is* | the **index of the variable set** `X_1..X_p`; disease prior `L_D` on `Ω_var` |

The LDO estimates a dependency operator over `X ∈ ℝ^{p×S×T}`. Geography structures the `S` dimension (via `L_W`); time structures `T` (via `Σ_time`); **the disease axis structures the `p` dimension.** Elevating ICD "to a central axis, especially in PIRS" means, precisely: *make the disease axis the structured generator of, and prior over, the `p` variable dimension of the LDO.* This is a stronger and cleaner statement than the handoff's "add an ICD axis," and it is what the rest of this document specifies.

**A useful duality.** A single carrier stratified by disease — "admissions by ICD-block × municipality × month" — is a tensor with a disease *stratification* axis. The LDO **unfolds** that disease axis into separate variables so cross-disease dependence is estimable. So `disease-as-stratification-axis (tensor)` and `disease-as-variable-index (unfolded)` are two views of the same object; today's σ_C produces the stratified view one slice at a time, the LDO consumes the unfolded view wholesale.

---

## 3. What PegaSUS already does with ICD (extend, don't reinvent)

Verified in the current registries:
- **`diagnostic_topology.yaml`** already types ICD-bearing fields as `topology_role` (e.g. `underlying_cause`), `icd_system: ICD-10`, `unit: ICD10`, `aggregation: non_aggregable`, with operators `icd_chapter_projection`/`icd_block_projection`. **ICD is already a non-aggregable, topology-bearing axis — not a generic numeric.** Good foundation.
- **`icd_catalog.yaml`** holds a minimal chapter/block catalog (ranges + labels) sufficient for projection.
- **`icd_curated_groups.yaml`** (MSD §3.11 `G_curated`) defines cause groups as unions of 3-char intervals, **partition-based** with an explicit `OTHER` residual so σ_C partitions the death population exactly. Manually curated.
- **`icd_quality_groups.yaml`** flags well-formed / missing / ill-defined (R00–R99) codes.
- The executor applies σ_C via `_add_icd_stratum` (chapter/block restriction); parse machinery is `parse_icd` / `parse_icd_ordered_chain` (LINHAA–D) / `parse_icd_unordered_set` (LINHAII).

**The gaps the handoff's resources fill:** (a) a *proper deterministic hierarchy* (closure, nearest-common-ancestor, dot normalization) instead of ad-hoc range checks; (b) *external, multi-label* concept groupers (CCSR/CCIR/CCC/phecodes) beside the manual partition list; (c) *code-system provenance and projection typing* (CID-10 vs ICD-10-CM); (d) a *disease geometry* (graph + optional embeddings) usable as an estimator prior and a search policy. None of this discards the current design; it upgrades the catalog, generalizes the grouper, and adds a geometry layer.

---

## 4. The Disease Semantic Axis — normative architecture (§II.13)

Four components, each mirroring an existing MSD-II pillar so the axis is not a bolt-on.

### §II.13.1 Disease Concept Registry (extends the Registry Platform §II.1 and Concept Grammar §II.5)

A registry of **code→concept assertions** imported from multiple sources, every assertion provenance- and projection-typed. Normative record:
```
DiseaseConceptAssertion:
  code            # e.g. CID-10 I21.9 (dot-normalized)
  code_system     # CID-10 | ICD-10 | ICD-10-CM | ICD-9-CM | ICD-11-MMS
  concept_id      # e.g. ccsr_CIR007 | cci_chronic | phecode_411 | brazilian_icsap_cardio
  concept_family  # chronicity | body_system | clinical_category | pediatric_ccc | avoidable | custom
  source          # icd_mappings@0.6.2 | ahrq_ccsr@v2026 | cha_ccc@v3 | phecode_x | brazilian_list
  multi_label: true|false          # a code MAY hold several concepts (CCSR); MUST NOT be silently forced to one
  projection_status # exact | parent_projection | approximate | unmappable | source_system_specific
  chronic: true|false|mixed|unknown
  warnings[]
```
Contracts:
1. **No silent cross-system truth.** A concept imported from an ICD-10-CM grouper MUST be stored with `code_system: ICD-10-CM` and, when applied to Brazilian CID-10 data, MUST carry an explicit `projection_status`. `approximate`/`parent_projection` concepts propagate reduced certification (they enter fields at `state ≤ fragile`).
2. **Multi-label preserved.** The registry MUST allow a code to hold multiple concepts. Forcing a partition is permitted ONLY when the analysis explicitly requests a partition (e.g. σ_C mortality that must sum exactly), and then only via a declared tie-break rule with an `OTHER` residual (the existing `icd_curated_groups.yaml` mechanism becomes *one* partition-view over the multi-label registry).
3. **Round-trip validity.** Every `concept_id` MUST resolve to a definition (ranges or code-set) so σ_C restrictions remain materializable.

### §II.13.2 ICD/CID Adapter (extends §II.1 named-callable registry)

Deterministic hierarchy behind the callable registry, **not** hardwired into ingestion:
```
callables: {
  "icd_validate": ..., "icd_add_dot": ..., "icd_remove_dot": ...,
  "icd_ancestors": ..., "icd_descendants": ..., "icd_nearest_common_ancestor": ...,
  "icd_children": ..., "icd_is_leaf": ...
}
```
`simple-icd-10` MAY back the WHO ICD-10 hierarchy; `simple-icd-10-cm` MAY back ICD-10-CM for grouper import; both are wrapped so the **Brazilian CID-10** remains the authoritative code system for DATASUS data, with the WHO/CM hierarchies used for closure/labels/projection only. The adapter upgrades `icd_catalog.yaml` from range-checks to true closure and supplies nearest-common-ancestor (needed for disease-graph distance, §II.13.3). **Guard:** the adapter MUST NOT silently coerce a CID-10 code to an ICD-10-CM subcode; codes that exist in one system and not the other are typed `source_system_specific`.

### §II.13.3 DiseaseGraph (mirrors the SpatialWeightGraph §II.4)

One base disease-structure object with derived views and a legality class — the disease-axis analogue of `spatial_graphs.yaml`:
```
disease_graphs.yaml:
  cid10_hierarchy:          {kind: hierarchy, provenance: [who_icd10, cid10_catalog], legality_class: structural}
  ccsr_membership:          {kind: concept_bipartite, provenance: [ahrq_ccsr], legality_class: structural}
  label_embedding_knn_k8:   {kind: knn, provenance: [qwen3_label_text], legality_class: structural}
  empirical_cooccurrence:   {kind: knn, provenance: [datasus_events], legality_class: context_derived}   # GUARDED
```
API (parallel to `load_spatial_graph`):
```
load_disease_graph(id).view("adjacency"|"laplacian"|"membership"|"distance")
DiseaseGraph.laplacian() -> L_D        # the disease-space smoothness operator for the LDO prior
DiseaseGraph.groups()    -> overlapping concept groups   # for group penalties / overlap handling
```
**Legality class (critical, reuses §II.4.1):** `structural` graphs (hierarchy, external membership, label-text embedding) are data-independent and safe as priors. `context_derived` graphs (empirical co-occurrence embeddings) MUST be rejected as a prior for any edge whose variables share the graph's provenance — i.e. they cannot inform a test run on the same data. New §10 abort: *"context-derived disease structure shares provenance with the tested variables."*

### §II.13.4 Disease embeddings (bounded, build-time, provenance-typed)

Text embeddings of `{code, PT-BR label, EN label, synonyms, hierarchy path, mapping labels}` per code, via a local Qwen3-Embedding-family model. Contracts:
1. **Build-time asset, not a runtime dependency.** Vectors are precomputed once and cached (≈14k CID-10 codes + ~500 concepts × 1024 dims ≈ 60 MB). The embedding model MUST NOT be required at LDO runtime. This keeps the runtime within the §II.10 envelope; **Qwen3-0.6B** is the GPU-feasible default (≈1.2 GB fp16 at build time), 4B/8B are offline/CPU-only options.
2. **Provenance-typed.** Label/synonym/hierarchy/mapping-text embeddings are `structural` (data-independent) → usable as LDO priors and for search. Any embedding trained on DATASUS event co-occurrence is `context_derived` → **output-only / partition-separated** (§5.5).
3. **Never an edge.** Embeddings supply priors and search order only (§5.5).

---

## 5. LDO integration — the heart (extends §II.6, §II.7)

### 5.1 Disease axis as the variable generator (multi-resolution `p`)

The LDO variable set is no longer hand-seeded; the **concept grammar (§II.5) instantiates it by crossing carriers with disease concepts at a chosen disease resolution**:
```
variable = (carrier ∈ {Deaths, HospitalAdmissions, LiveBirths, ...})
         × (disease_selector at resolution r ∈ {chapter, block, category, curated_concept, ccsr_concept, leaf})
         × (stratification)
```
Resolution `r` is a first-class parameter (this is §II.7 multi-resolution applied to the **disease** dimension). The grammar MUST respect the SIM chain semantics: a disease selector carries a `topology_role ∈ {underlying_cause, mention, associated}` (from `diagnostic_topology.yaml`), and a variable built on `underlying_cause` is a distinct variable from the same disease as a `mention`. Each generated variable materializes or emits a typed `FailedBranch` (§II.5) — never silent absence.

### 5.2 Disease-space Laplacian `L_D` as a prior on `Ω_var` (the core mathematical move)

The LDO's variable-dependency operator `Ω_var` (§II.6.1) is estimated with a **disease-structure prior**, the exact analogue of the geographic GMRF prior `L_W` but acting on the `p` dimension:

- **Hierarchy → fused/group penalty.** Sibling codes and parent/child pairs are a priori related. Add a fused penalty `λ_H · Σ_{(i,j)∈hierarchy} ‖Ω_{i·} − Ω_{j·}‖²` (rows of related diseases should have similar dependency profiles) and/or overlapping-group sparsity from concept membership (`groups()` in §II.13.3). This lets a link discovered at the block level *inform* (not dictate) leaf-level estimation.
- **Embedding similarity → smoothness prior.** With the `structural` label-embedding kNN graph, `L_D = D − W_D` enters as `λ_D · tr(Ω_var L_D Ω_varᵀ)` — dependency profiles vary smoothly across semantically similar diseases. This is the cold-start prior when empirical co-occurrence is sparse (rare diseases, small municipalities).
- **Hierarchy → hard additivity constraint in the CTR (§II.3).** When a disease-stratified tensor is *reconstructed* (e.g. disaggregating a coarse cause to finer codes), children MUST sum to parent — a linear constraint in `CTRProblem`, identical in form to age-bin disaggregation (§II.3). **Duality:** hierarchy is a *hard constraint* for reconstruction, a *soft prior* for dependency estimation.

Net: `Prec(Z) ≈ Ω_var ⊗ Σ_space^{-1} ⊗ Σ_time^{-1}`, with `Ω_var` now regularized by `L_D` (disease) exactly as `Σ_space^{-1}` is built from `L_W` (geography). **Disease becomes an axis of the estimator in the same mathematical sense as geography.**

### 5.3 The shared-code overlap correction (mandatory)

When concept-variables share underlying codes (multi-label), they are **mechanically** correlated. The membership bipartite graph (`ccsr_membership`) makes the overlap *exactly known*. The LDO MUST:
1. Compute an **overlap operator** `O_{ij} = |codes(i) ∩ codes(j)| / |codes(i) ∪ codes(j)|` (Jaccard on code sets) from the registry.
2. Treat high-overlap pairs as **known structure**: either (a) fit on a partition-view (disjoint code sets) for the core dependency estimate, or (b) explicitly regress out the shared-count component, or (c) tag any surviving edge `mechanical_overlap` in the link record so it is never reported as a discovered epidemiological link.
New §10 abort: *"disease-concept edge reported without overlap accounting when Jaccard(codes) exceeds threshold."* This is the disease-axis analogue of the shared-denominator trap.

### 5.4 Coarse→fine disease resolution + semantic search-space expansion (autonomous discovery)

This is the mechanism that makes autonomous investigation tractable. The disease structure is the **search policy** over the otherwise combinatorial disease × disease × lag × geo space:
1. **Coarse pass:** run the LDO with disease resolution = {chapter, curated/CCSR concept}. Few variables → cheap → a global disease–disease–context link graph.
2. **Expand where edges fire:** for each certified coarse edge, expand the candidate set along the DiseaseGraph — *down the hierarchy* (block → category → leaf) and *across structural neighbors* (label-embedding kNN, sibling concepts) — and refit **locally** on just those variables (and, per §II.7, on the strong-signal spatial/temporal subset).
3. **Stability-select and certify** (§II.6.4) the refined edges.
4. **Stop policy:** expansion halts when refinement no longer changes the certified edge set (path agreement) or the compute budget is hit — an explicit, logged budget, never silent truncation.

The embedding/hierarchy thus answers *"where should the engine look next?"* — turning brute-force `p²×K×S` search into guided search. This is the concrete content of "autonomous epidemiological discovery": the disease axis provides the priors (where links are plausible), the generator (what variables to instantiate), and the search order (what to refine), while the **data alone decides which edges survive**.

### 5.5 The disease circularity guard (mandatory; mirrors §II.4.1)

- **Structural disease geometry** (hierarchy, external membership, label-text embedding) is data-independent → **safe** as an LDO prior and as a search policy.
- **Empirical disease geometry** (co-occurrence embeddings learned from DATASUS events) is `context_derived` → it MAY be produced as an **output** (a learned Brazilian disease map — scientifically valuable) but MUST NOT be used as a prior in, or a search policy for, a test run on the same data partition. If used at all as a prior, it MUST be learned on a disjoint partition (e.g. different years/regions) with the separation recorded in the link record.
- **No embedding ever asserts an edge.** The link record's `edge_type` values remain `{contemporaneous, lagged_directed, latent_shared, nonlinear_residual, mechanical_overlap}`; there is no `embedding_similarity` edge type. Similarity is a prior weight, not a finding.

---

## 6. EFG legality & aggregation-law extensions

The disease axis becomes a first-class legality term (extends MSD §3.8 / §3.5):
1. **Disease aggregation law.** Hierarchy aggregation is a **partition** (additive up the tree). Concept projection is **multi-label** (non-additive: summing overlapping concepts double-counts). The aggregation registry MUST distinguish `disease_hierarchy_additive` from `disease_concept_multilabel_nonadditive`; an operator that sums across multi-label concepts is illegal (`Δ = 0`).
2. **Code-system provenance term.** An operation combining fields of different `code_system` without a declared projection is illegal (the disease analogue of the race-bridge declaration gate). `approximate`/`parent_projection` projections force `state ≤ fragile`.
3. **Closure legality for σ_C.** A cause-specific restriction MUST either partition (with `OTHER` residual) or be explicitly typed multi-label; a restriction that neither partitions nor declares multi-label is illegal.

---

## 7. Compute envelope (fits the RTX 4050 trivially)

- Embedding vectors are **precomputed and cached** (~60 MB); the embedding model is not loaded at LDO runtime. Qwen3-0.6B is the GPU build-time default.
- `L_D` is sparse (hierarchy: ~1 parent + few children per node; kNN: k≈8) → the disease prior adds negligible VRAM, exactly like the sparse spatial `L_W`.
- Multi-resolution disease passes are the compute-management strategy on the `p` axis, as §II.7 is on the `S×T` axis: coarse global (tens of variables), fine local (a handful). National-scale disease × geo × time remains a tiling/streaming problem, never a dense one.

---

## 8. Guardrails summary (enforceable rules distilled from the corrections)

1. Disease geometry that is `context_derived` (empirical co-occurrence) is output-only / partition-separated — never a same-data prior (§5.5).
2. Embeddings supply priors and search order only; **data decides edges** (§5.5).
3. Multi-label concept overlap is known structure; high-Jaccard edges are `mechanical_overlap`, not discoveries (§5.3).
4. Cross-code-system application carries explicit `projection_status`; `approximate` ⇒ `state ≤ fragile` (§II.13.1, §6).
5. Multi-label concepts are never silently forced to a partition (§II.13.1).
6. `underlying_cause` vs `mention` vs `associated` are distinct variables (§5.1).
7. The embedding model is a build-time asset; runtime stays within §II.10 (§7).

---

## 9. TDD work items (`MII-DIS-*`) — insert after MSD-II Phase 2, interleave with Phase 3

> Sequencing: `DIS-01/02` (registry + adapter) and `DIS-03` (DiseaseGraph) close before the LDO consumes them at `MII-LDO-00/04`; `DIS-04` (L_D prior) pairs with `MII-LDO-02`; `DIS-05` (overlap) gates `MII-LDO-05` edge readout; `DIS-06` (semantic expansion) extends `MII-RES-01`; `DIS-07` (embeddings) is optional and build-time.

- **`MII-DIS-01` — Disease Concept Registry + multi-source import.**
  *Red:* on a fixture, a code with two CCSR concepts yields two assertions (multi-label preserved); an ICD-10-CM-sourced concept applied to CID-10 carries a non-`exact` `projection_status`; forcing a partition without an `OTHER` residual fails.
  *Green:* `config/registries/disease_concepts.yaml` + importer wrapping `icd-mappings` (CCSR/CCIR/CCS/CCC) and Brazilian lists; provenance/projection typing.
  *Files:* `registries/disease_concepts.yaml`, `she/disease/import.py`.

- **`MII-DIS-02` — ICD/CID adapter behind the callable registry.**
  *Red:* `icd_add_dot("I219")=="I21.9"`; `icd_nearest_common_ancestor("I21.0","I22")` is in `I20-I25`; a CID-10-only code is `source_system_specific`, not coerced.
  *Green:* wrap `simple-icd-10`(-cm) via `registries/callables.py`; upgrade `icd_catalog.yaml` closure.
  *Files:* `datasus/icd_adapter.py`, `registries/callables.py`.

- **`MII-DIS-03` — DiseaseGraph + legality class.**
  *Red:* `load_disease_graph("cid10_hierarchy").laplacian()` returns sparse `L_D`; a `context_derived` graph sharing provenance with tested variables is rejected with the §II.13.3 abort.
  *Green:* `config/registries/disease_graphs.yaml`, `pegasus/disease/graph.py` (mirror `geo/spatial_graph.py`).
  *Files:* `disease/graph.py`, `registries/disease_graphs.yaml`.

- **`MII-DIS-04` — `L_D` prior in the LDO.**
  *Red:* on synthetic data where sibling diseases share a dependency profile, adding the `L_D` smoothness/fused penalty improves edge recovery for rare (sparse) disease variables vs no prior.
  *Green:* extend `pirs/ldo/precision.py` with the `L_D` term (fused hierarchy + kNN smoothness); pairs with `MII-LDO-02`.
  *Files:* `pirs/ldo/precision.py`.

- **`MII-DIS-05` — Shared-code overlap accounting.**
  *Red:* two concept-variables with 80% code overlap do NOT produce a `contemporaneous`/`lagged_directed` edge; they are tagged `mechanical_overlap`; the §5.3 abort fires when overlap is unaccounted.
  *Green:* overlap operator from `membership`; overlap handling in `pirs/ldo/edges.py`; new `edge_type`.
  *Files:* `pirs/ldo/edges.py`, `pirs/ldo/records.py`.

- **`MII-DIS-06` — Disease-resolution multi-pass + semantic expansion.**
  *Red:* a coarse concept-level edge triggers a fine leaf-level refit only on hierarchy/kNN neighbors of the firing concepts; the expansion halts on path agreement; budget is logged.
  *Green:* extend `pirs/ldo/resolution.py` (§II.7) with the disease axis and the DiseaseGraph search policy.
  *Files:* `pirs/ldo/resolution.py`.

- **`MII-DIS-07` — (optional, build-time) Label embeddings.**
  *Red:* embeddings are precomputed/cached and NOT loaded at LDO runtime; a label-embedding kNN graph is `structural`; an empirical-co-occurrence embedding is `context_derived` and blocked as a same-data prior.
  *Green:* offline `disease/embed_build.py` (Qwen3-0.6B) → cached vectors → `label_embedding_knn` in `disease_graphs.yaml`.
  *Files:* `disease/embed_build.py`, cache artifact.

**Acceptance augmentation to the program test (Appendix D of MSD-II).** The Zika→microcephaly acceptance run SHOULD now be reachable *without any forced disease selectors*: the arbovirus and microcephaly variables are **instantiated by the disease-axis generator** (arbovirus admissions as an infectious-disease block concept; microcephaly as a congenital-anomaly concept via the SINASC anomaly ICD), and the lagged edge is discovered under the disease + spatial + temporal priors. Full autonomy over the disease axis is the point of this section.

---

## 10. Amendments this document requires

- **MSD-II:** add **§II.13** (this document); extend **§II.5** (concept grammar instantiates variables across the disease axis), **§II.6.1** (`Ω_var` carries the `L_D` prior; `mechanical_overlap` edge type), **§II.7** (resolution axis includes disease), **§II.8** (link record fields: `code_system`, `projection_status`, `topology_role`, `overlap_jaccard`), **§II.3** (hierarchy additivity as a CTR constraint instance).
- **MSD:** extend **§3.5** (disease aggregation law: hierarchy-additive vs multi-label-nonadditive), **§3.8** (disease-axis legality term: code-system provenance, closure legality), **§3.11** (`G_curated` becomes one partition-view over the multi-label Disease Concept Registry); add the three new §10 aborts (context-derived disease prior; unaccounted overlap; illegal multi-label sum).
- **Compute (`compute.yaml`):** register the build-time embedding step; assert the model is absent at LDO runtime.

*End. Use `MII-DIS-*` IDs in commits; cite the parent MSD-II section. The disease axis is accepted; the handoff's embeddings are accepted only as build-time, provenance-typed priors and search policy — never as a source of edges.*
