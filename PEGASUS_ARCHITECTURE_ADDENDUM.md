# PegaSUS — Architecture & Extensibility Addendum

**Companion to:** `PEGASUS_COMPLIANCE_AND_REMEDIATION.md`. That report fixes *correctness* (making the production tier honor the contracts the spec tier already encodes). This addendum addresses *architecture, extensibility, and platform shape* — the north star the correctness work should be steered toward.
**Audience:** PegaSUS developers and coding agents (including smaller LLMs). New work items use the `ARCH-` prefix; they reference compliance findings by their original IDs (`SHE-NORM-01`, etc.).
**Provenance of inputs:** This addendum evaluates a set of architectural proposals raised in discussion, then develops the accepted ones into concrete, grounded designs. The proposals are treated as raw direction, not settled requirements; several are accepted with substantial refinement, one is accepted only as a deferred extension point, and critical guardrails absent from the original proposals are added throughout.

---

## 0. Triage — what to accept, refine, defer, and why

A coding agent should not implement everything proposed. Here is the critical judgment, with the engineering rationale. "Accept" means it is directionally right *and* I have developed it below; "Refine" flags where the naive version is wrong; "Defer" means correct-but-not-now.

| # | Proposal | Verdict | Core rationale / refinement added |
|---|---|---|---|
| 1 | General `SpatialWeightGraph` (`W_ij`), not binary adjacency | **Accept + refine** | Right abstraction; already has a waiting consumer (`loss.py` `spatial_laplacian`). **Refinement:** one *base* graph with multiple *derived normalized views* (different consumers need different normalizations), plus a **legality class** to prevent circular use of context-derived weights. |
| 2 | ICAR *consumes* a spatial kernel, not *defines* space | **Accept** | Folds into #1. Corrects `PIRS-SPAT-01`: ICAR/CAR/GMRF consume the declared `W` (symmetric view); contiguity is one case. |
| 3 | OLAP cubes are first-class, not secondary to DATASUS events | **Accept** | Architectural framing of existing MSD intent (§2.9, §2.12). Formalized below as two explicit parallel lifecycles. |
| 4 | Synthetic SIDRA-like cubes are central SHE outputs | **Accept + bound** | Faithful to `B_reconstructed`/`B_latent`. **Refinement:** every synthetic cube MUST pass a certification gate and is quarantined/dashboard-unsafe by default — else this becomes fabrication. |
| 5 | Age-bin harmonization is latent inference, not crosswalk | **Accept** | Strong. It is literally a small instance of the §2.8 constrained-tensor problem. Unified under the CTR kernel below. |
| 6 | Source onboarding should be declarative/registry-driven | **Accept (core)** | The platform thesis. **Refinement:** "mostly declarative" = declarative routing + a registry of *named code callables*; not everything can be pure YAML. |
| 7 | Registered concepts are grammar/seeds, not fixed outputs | **Accept** | Faithful to the EFG-as-compiler vision; sharpens `EFG-REG-01`. Concept family → instantiation planner → concrete legal fields. |
| 8 | EFG higher-order context-conditioned functionals | **Accept as extension point, DEFER impl** | Genuinely new scope beyond the locked MSD operator set. Define the contract + guardrails now; implement only after Phases 0–3 and an MSD §3 amendment. Hardest legality risk. |
| 9 | PIRS manifests must be pedagogical/auditable | **Accept (cheap win)** | Aligns with the anti-silence ethos. A decision-record schema + renderer. |
| 10 | Run profiles must not make PIRS feel peripheral | **Accept as refinement** | Valid. **Refinement:** decouple two orthogonal axes — *DataScope* (which substrate exists) and *ExecutionStage* (validate/compile/investigate). PIRS is central to `investigate`. |
| 11 | The common geo×time panel is earned, not assumed | **Accept** | Correct. Formalize *CommonPanel construction* as an explicit compilation phase between SHE and EFG. |
| 12 | Onboarding must be centralized, plugin-like (capstone) | **Accept (capstone)** | The synthesis of #6/#7. The "Registry Platform" below is the concrete realization. |

**Nothing is rejected outright.** The only deferral is #8. The most important refinements the original proposals lacked are: the **spatial-weight legality/circularity guard** (#1), the **synthetic-cube certification gate** (#4), the **declarative-plus-named-callable middle path** (#6), and the **DataScope/ExecutionStage decoupling** (#10).

**Sequencing discipline (read this before building anything):** Correctness precedes platformization. Do not build the full Registry Platform before the normalizers emit real data. The leverage move is to **implement `SHE-NORM-01` Path A *as* the first slice of the Registry Platform** — i.e. the correctness fix and the architecture fix are the same work if done in the right order. The roadmap in §10 interleaves them accordingly.

---

## 1. The unifying thesis: PegaSUS as a registry-driven compiler platform with two lifecycles and one reconstruction kernel

Most of the accepted proposals are facets of a single coherent architecture. Stating it once prevents piecemeal implementation:

> **PegaSUS is a compiler platform.** It onboards heterogeneous public-health data through *declarative registries backed by named code plugins*; it processes that data through **two parallel lifecycles** — an **event lifecycle** (record tables: SIM/SIH/SINASC/CNES) and a **cube lifecycle** (OLAP tables: SIDRA/census/GDP/sanitation) — that converge on a **compiled common support panel**; it reconstructs missing structure (population, fine age bins, latent context) through **one shared constrained-tensor-reconstruction kernel** with a certification gate; it instantiates epidemiological **concept families** into concrete legal fields via the EFG; and it investigates them via PIRS. Every transformation is registry-declared, provenance-tagged, legality-checked, and — when reconstructed — certified and quarantined by default.

The four structural pillars:

1. **Registry Platform** (§2) — executable-authority registries + plugin adapters + a validator/generator backbone. Realizes proposals 6, 7, 12.
2. **Two Lifecycles + CommonPanel** (§3) — explicit event and cube pipelines converging on a compiled panel. Realizes proposals 3, 11.
3. **Constrained Tensor Reconstruction (CTR) kernel** (§4) — one solver family generalizing the population tensor, age-bin disaggregation, synthetic cubes, and ST-DFM, with a shared certification gate. Realizes proposals 4, 5.
4. **SpatialWeightGraph** (§5) — one spatial structure consumed by every spatial component, with derived views and a legality class. Realizes proposals 1, 2.

Plus three cross-cutting refinements: **Concept-family grammar** (§6, proposal 7), **PIRS decision records** (§7, proposal 9), **DataScope/ExecutionStage** (§8, proposal 10), and the **deferred higher-order operator extension point** (§9, proposal 8).

---

## 2. Pillar 1 — The Registry Platform (executable authority)

**Realizes proposals 6, 7, 12. Extends `SHE-NORM-01`, `PIRS-REG-01`, `REG-*`.**

### 2.1 The principle and the realistic target

The principle: **registries are executable authority, not post-hoc metadata.** Today, `config/registries/source_fields.yaml` is keyed by *canonical* names and carries only post-decode semantics; the actual raw→canonical decoding is hardcoded in the `normalize_*_events` functions (see compliance `SHE-NORM-01`). That is the gap.

The realistic target is **"declarative routing + named code callables."** Pure-YAML decoding is a fantasy — the SIM cause-chain parser and ICD topology need real code. So the contract is: registries declare *what* happens (route, decoder name, output field, axes); a small registry of *named callables* provides the *how* (the actual decode/parse/adapter functions). YAML references code by name; code is written once and reused everywhere.

### 2.2 The registry set (target state)

| Registry | Authority over | Status today | Action |
|---|---|---|---|
| `source_adapters.yaml` | source families → adapter plugin + config schema | **absent** | `ARCH-REG-01` create |
| `source_routing.yaml` (raw side of `source_fields.yaml`) | per `(source_system, raw_field)`: route ∈ {Decode,Parse,PreserveMark,Exclude}, decoder/parser callable, output canonical field | **absent** (registry is canonical-keyed) | `ARCH-REG-02` create — *this is `SHE-NORM-01` Path A* |
| `canonical_fields.yaml` (current `source_fields.yaml`) | post-decode semantics (carrier/unit/aggregation/role/axes/admissibility) | present, good | keep; extend coverage (SHE-NORM-01) |
| `composite_decoders.yaml` | named decoder params (unit maps, bounds, sentinels) | scaffold (909 B) | `ARCH-REG-03` populate; back the named-callable registry |
| `concepts.yaml` | concept-family grammar (§6) | partial (`core_seed_registry.yaml`) | `ARCH-REG-04` formalize |
| `operators.yaml` | legal operators + legality requirements | implicit in code | `ARCH-REG-05` externalize |
| `spatial_graphs.yaml` | spatial weight graphs (§5) | **absent** | `ARCH-REG-06` create |
| `cube_compendium` (`sidra_compendium.json`) | cube catalog/metadata | present | keep; extend (cube lifecycle, §3) |
| `model_registry.yaml` | PIRS families | stale (2 vs 8) | fix per `PIRS-REG-01`; regenerate from `glm.py` |

### 2.3 The named-callable registry

Create `pegasus/registries/callables.py` — the single resolution point mapping string names → functions, so every YAML `decoder:`/`parser:`/`adapter:` reference resolves to real code:

```python
# pegasus/registries/callables.py
from pegasus.datasus import decoders
from pegasus.datasus import icd_parsers

CALLABLE_REGISTRY: dict[str, Callable] = {
    "decode_sim_idade":        decoders.decode_sim_idade,
    "decode_sih_age":          decoders.decode_sih_age,
    "decode_physical_scalar":  decoders.decode_physical_scalar,
    "decode_count2":           decoders.decode_count2,
    "clamp_bool":              decoders.clamp_bool,
    "filter_cnpj":             decoders.filter_cnpj,
    "parse_icd_underlying":    icd_parsers.parse_icd,
    "parse_icd_ordered_chain": icd_parsers.parse_icd_ordered_chain,
    "parse_icd_unordered_set": icd_parsers.parse_icd_unordered_set,
    # ... every decoder/parser the routing registry references
}

def resolve_callable(name: str) -> Callable:
    if name not in CALLABLE_REGISTRY:
        raise RegistryError(f"unregistered callable {name!r}; add it to CALLABLE_REGISTRY")
    return CALLABLE_REGISTRY[name]
```

`declarative_normalize.py` already has `_resolve_decoder_callable`; this generalizes and centralizes it so adapters, decoders, and parsers share one resolver and one validation surface.

### 2.4 The SourceAdapter plugin interface (`ARCH-ADAPT-01`)

A new source *family* (not just a new table of a known family) is onboarded by adding one adapter:

```python
# pegasus/sources/base.py
class SourceAdapter(Protocol):
    source_family: str          # "datasus_dbc", "sidra_cube", "ftp_csv", "shapefile", ...
    realm: Literal["event", "cube"]   # routes into the event or cube lifecycle (§3)

    def discover(self, manifest_entry: dict) -> list[RawArtifactRef]: ...
    def read_raw(self, ref: RawArtifactRef) -> RawTable | RawCube: ...
    def raw_schema(self, ref: RawArtifactRef) -> RawSchema: ...   # columns/dims as-is
```

`source_adapters.yaml` maps `source_family → adapter class name + config schema`. The DATASUS DBC reader and the SIDRA cube reader become the first two registered adapters. The adapter's `realm` field is what routes data into the correct lifecycle.

### 2.5 Validator + generator backbone (`ARCH-REG-07`)

Replace the broken `validate_registry_tree` (compliance `REG-VALIDATOR-01`) with a real cross-registry validator that enforces the executable-authority invariants:

- Every `(source_system, raw_field)` in `source_routing.yaml` has a route; no raw field is silently dropped (Exclude must be explicit).
- Every `decoder`/`parser`/`adapter` name resolves via `resolve_callable`.
- Every canonical field referenced by routing exists in `canonical_fields.yaml` with consistent carrier/unit/aggregation.
- Every concept's `legal_operators`/`required_axes` reference registered operators/axes.
- Every spatial graph's `provenance` and `legality_class` are present.
- Round-trip: every admissible canonical field is reachable from some raw field OR is a declared reconstruction output (so nothing is admissible-but-unproducible).

Add a generator/scaffolder CLI:

```text
pegasus registry new-source --family datasus_dbc --system SIM-DO   # scaffolds routing stubs
pegasus registry new-field  --system SIH-RD --raw DIAGSEC1 --route Parse --parser parse_icd_unordered_set
pegasus registry check                                              # runs the full validator
```

### 2.6 The onboarding ergonomics target (acceptance criterion for the platform)

After this pillar, the "add new data" cost must match this table — and a test should assert that onboarding a fixture source touches **only** declarative files + at most one new callable:

| Scenario | Required change | Must NOT touch |
|---|---|---|
| Known protocol, new table | one manifest entry | code |
| New field encoding | one decoder callable + one `composite_decoders.yaml`/`source_routing.yaml` entry | CLI, workflow, normalizer, validator |
| New source family | one `SourceAdapter` plugin + one `source_adapters.yaml` entry | CLI, workflow, EFG, PIRS |
| New concept | one `concepts.yaml` entry | EFG operator code |

---

## 3. Pillar 2 — Two lifecycles converging on a compiled CommonPanel

**Realizes proposals 3, 11. Reframes `SIDRA-CTX-01`, `SIDRA-CTX-02`.**

### 3.1 The two lifecycles, stated explicitly

The MSD already implies both, but the code is event-biased and the cube path is inert (compliance `SIDRA-CTX-01`). Make them explicit, parallel, and named.

**Event lifecycle** (record tables → counts/functionals):
```text
raw record  --adapter-->  typed records
            --source_routing(decode/parse)-->  canonical event substrate (per-record)
            --EFG count/functional operators over declared support-->  field cells on panel
```

**Cube lifecycle** (OLAP tables → observed/reconstructed context & denominators):
```text
raw cube  --adapter-->  typed cube + metadata
          --regime classify (§2.9)-->  regime R(q)
          --classification projection (§2.12.2)-->  concept/category-aligned cube
          --[disaggregation / reconstruction if regime demands]  (CTR kernel, §4)
          --bounded pushforward (§2.12.3)-->  bounded cube
          --substrate tagging-->  B_official | B_harmonized | B_reconstructed | B_latent | B_cross-sectional
          --align to panel-->  context/denominator field cells on panel
```

Implementation: each `SourceAdapter.realm` (§2.4) dispatches into `she/lifecycle/event.py` or `she/lifecycle/cube.py`. The cube lifecycle is mostly *already-built modules that aren't wired* (`sidra/regime.py`, `sidra/projection.py`, `sidra/pushforward.py`, `she/stdfm/*`); this pillar is the *orchestration* that connects them (compliance `SIDRA-CTX-01`).

### 3.2 CommonPanel as an explicit compilation phase (`ARCH-PANEL-01`)

The shared `municipality × year [× age × sex × race]` panel is a **compiled product**, not a raw assumption. Make it a first-class phase with its own artifact:

```text
SHE substrate (events + cubes, each with native support)
        │
        ▼   CommonPanel planner  (new: she/panel.py)
   target panel P  ←  derived from UserIntent axes + legality
        │
        ├── events  → aggregated/counted to P cells           (provenance: observed)
        ├── official cubes → projected to P cells              (provenance: observed)
        └── reconstructed cubes → CTR/pushforward to P cells   (provenance: reconstructed/bounded)
        │
        ▼
   Panel manifest:  per (field, cell) provenance + state  →  EFG/PIRS
```

The panel manifest records, **per cell**, whether a value is observed, projected, reconstructed, or bounded. This is the §3.8.1 support intersection promoted to an auditable phase and is the structural fix for the "silent absence" anti-pattern (compliance `XCUT-05`): a panel cell is never blank — it is observed, reconstructed-with-uncertainty, or explicitly `unavailable_on_panel` with a reason.

### 3.3 Why this reframing matters

`SIDRA-CTX-01` in the compliance report says "wire SIDRA into EFG." This pillar makes that precise and general: implement the **cube lifecycle** and the **CommonPanel phase**, of which SIDRA is the first instance. Any future cube source (census, GDP, sanitation) then onboards through the *same* lifecycle with no new pipeline code — satisfying proposal 12's extensibility demand.


---

## 4. Pillar 3 — The Constrained Tensor Reconstruction (CTR) kernel

**Realizes proposals 4, 5. Generalizes `SHE-POP-01/02` and unifies with ST-DFM.**

### 4.1 The unification insight

Four things the MSD/proposals treat separately are the **same mathematical object** — a constrained optimization over a non-negative latent tensor:

1. **Population tensor** (§2.8) — `N[s,t,age,sex,race]` from census anchors + births + deaths + smoothness + cohort dynamics.
2. **Age-bin disaggregation** (proposal 5) — `observed_broad_bins = A · latent_fine_age + ε`, recovered under totals + smoothness + cohort + marginals.
3. **Synthetic context cubes** (proposal 4) — reconstructed sanitation/GDP/etc. surfaces over the panel.
4. **ST-DFM latent factors** (§2.10) — already implemented with a spatial-Laplacian penalty.

This is not a loose analogy. The existing `she/population/loss.py` **already** contains the general structure: anchors (equality/soft constraints), a linear aging operator, birth/death terms, ILR composition, second-difference smoothness over age and time, **and a `spatial_laplacian` penalty** — and the ST-DFM factor loss reuses the same `spatial_laplacian` einsum. The population solver is already 90% of a general CTR kernel; it is just not factored as one.

### 4.2 The CTR contract (`ARCH-CTR-01`)

Refactor `she/population/` into `she/reconstruction/` with the population tensor as the canonical *instance*, not the only implementation:

```python
# pegasus/she/reconstruction/problem.py
@dataclass(frozen=True)
class CTRProblem:
    latent_shape: tuple[int, ...]                 # e.g. (S, T, A, X, R)
    observations: list[Observation]               # each: (A_operator, observed_values, weight)
    anchors: list[Anchor]                         # hard/soft equality (known totals)
    nonnegativity: bool = True
    penalties: list[Penalty]                      # 2nd-diff age, 2nd-diff time, spatial Laplacian(W), ...
    dynamics: list[Dynamics]                      # cohort aging, birth entry, death prior
    marginals: list[MarginalConstraint]           # sex/race/geo totals must match
    composition: CompositionSpec | None = None    # ILR simplex (race), if applicable

# Each Observation carries a LINEAR operator mapping latent -> observed space:
#   - population anchor:        identity / selection
#   - age-bin disaggregation:   A = bin-aggregation matrix  (broad = A · fine)
#   - synthetic cube:           the cube's marginalization operator
```

`Penalty` for space takes the **SpatialWeightGraph Laplacian** (§5) — closing the loop: the spatial smoothness that today receives a raw `spatial_laplacian` tuple instead receives `Laplacian(W)` from the shared graph registry.

### 4.3 Age-bin disaggregation as a CTR instance (`ARCH-CTR-02`, proposal 5)

This replaces "crosswalk lookup" with inference. For SIDRA/census broad bins → canonical fine age grid:

```text
minimize   Σ_obs ‖ A · n_fine − n_broad ‖²_W
subject to n_fine ≥ 0
           Σ_fine within each broad bin = n_broad           (totals)
           Σ_a n_fine[s,t,a,·,·] = sex/race/geo marginals    (marginals)
penalties  λ_age · ‖Δ²_age n_fine‖²                          (smoothness over age)
           λ_cohort · ‖cohort_continuity(n_fine)‖²           (diagonal continuity over t)
```

Output is `B_reconstructed`, **dashboard-unsafe by default**, with per-cell uncertainty. It must never be treated as observed. This is the correct home for age harmonization — *in SHE, before EFG receives canonical age axes* (proposal 5's exact claim), and it reuses the population solver machinery rather than introducing a parallel system.

### 4.4 The certification gate — the guardrail the proposals omitted (`ARCH-CTR-03`)

Synthetic cubes are powerful and dangerous: without discipline they fabricate data that looks observed. **Every CTR output passes a certification gate** generalizing the ST-DFM §2.10.4 thresholds (which already exist and are faithful — `she/stdfm/certification.py`):

```text
CTRCertification:
    holdout_mape       ≤ verified: 0.15   fragile: 0.35
    reconstruction_var ≤ verified: 0.25   fragile: 0.40
    constraint_residual≤ tol               (totals/marginals actually satisfied)
    stability          ≥ 0.85              (multi-start / perturbation agreement)
  ⇒ status ∈ {verified, fragile, illegal_excluded}
```

Hard rules:
- A CTR output may enter substrate only as `B_reconstructed`/`B_latent`, never `B_official`.
- It is **dashboard-unsafe by default**; promotion to dashboard-safe requires `status=verified` AND propagated uncertainty present.
- New §10-style abort (`ARCH-CTR-04`): *"reconstructed tensor promoted to verified without holdout certification or propagated uncertainty."*
- The panel manifest (§3.2) tags every reconstructed cell with its certification status, so downstream consumers can filter.

### 4.5 Effect on existing roadmap items

`SHE-POP-02` (ADMM, state-space, primal-dual backends) becomes "implement CTR **solver backends**" — the same scale-tiered selection (§2.8.12), now serving all CTR instances. This is more reuse for the same effort.

---

## 5. Pillar 4 — The SpatialWeightGraph contract

**Realizes proposals 1, 2. Reframes `PIRS-SPAT-01`; feeds `EFG-Q-01`, CTR §4, HSIC nulls.**

### 5.1 Why one shared graph, and why "one base, many views"

Today `geo/adjacency.py` provides only **binary symmetric adjacency** (`load_adjacency(require_symmetric=True)` → node → neighbor tuples). Yet at least five components need spatial structure: ICAR/CAR (§6.3), Moran's I (§3.12, currently often `None` — compliance `EFG-Q-01`), spatial HSIC nulls (§6.8), spatial cross-fit folds (§6.6.1), and CTR/ST-DFM spatial smoothness (§4, the `spatial_laplacian`). They should consume **one declared graph**, not five ad-hoc notions.

But the naive "one `W` for everything" is wrong: different consumers need different *normalizations*, and some need symmetry that a gravity/flow matrix lacks. So the contract is **one base graph + derived normalized views**:

```text
base graph:  weighted, possibly directed edge list
   ├── binary view            (W_ij ∈ {0,1})            → contiguity baseline
   ├── row_standardized view  (Σ_j W_ij = 1)            → Moran's I, smoothing
   ├── symmetric view         (½(W+Wᵀ) or min)          → ICAR/CAR precision (needs symmetry)
   └── laplacian view         (L = D − W_sym)           → CTR/ST-DFM spatial penalty
```

### 5.2 The registry contract (`ARCH-SPATIAL-01`)

```yaml
# config/registries/spatial_graphs.yaml
graphs:
  contiguity_queen:
    kind: contiguity
    construction: queen            # rook | queen
    directed: false
    base_units: municipality_cod7
    provenance: [ibge_mesh]
    legality_class: structural      # geometry only — always safe (see §5.4)
    default_view: binary
  knn_centroid_k8:
    kind: knn
    construction: "k_nearest_centroids(k=8)"
    directed: true
    provenance: [ibge_centroids]
    legality_class: structural
  gravity_pop_distance:
    kind: gravity
    construction: "mass_i * mass_j / distance^alpha"
    params: {mass: population, alpha: 2.0}
    directed: true
    provenance: [ibge_population, ibge_centroids]
    legality_class: context_derived  # depends on population — NOT safe everywhere (§5.4)
  commuting_flow:
    kind: flow
    construction: observed_od_matrix
    directed: true
    provenance: [ibge_commuting_survey]
    legality_class: context_derived
```

Per-edge artifact columns: `left_id, right_id, weight, graph_kind, construction_method, directed, normalized, provenance, legality_class`.

API (`pegasus/geo/spatial_graph.py`, generalizing `adjacency.py`):
```python
load_spatial_graph(graph_id) -> SpatialGraph
SpatialGraph.view("symmetric"|"row_standardized"|"binary"|"laplacian") -> matrix
SpatialGraph.blocks(n) -> spatial blocks        # for HSIC nulls / cross-fit folds
```

### 5.3 Consumer wiring (`ARCH-SPATIAL-02`)

| Consumer | View | Replaces |
|---|---|---|
| ICAR/CAR precision (§6.3 `PIRS-SPAT-01`) | `symmetric` → `L` | (absent today) |
| Moran's I (`EFG-Q-01`) | `row_standardized` | `moran_i = None` |
| CTR / ST-DFM spatial penalty (§4) | `laplacian` | raw `spatial_laplacian` tuple |
| HSIC spatial null (§6.8) | `blocks()` | ad-hoc `spatial_blocks` labels |
| Spatial cross-fit folds (§6.6.1) | `blocks()` partition | ad-hoc |

This makes `PIRS-SPAT-01`'s ICAR concrete: ICAR consumes the `symmetric`/`laplacian` view of the declared graph (proposal 2 exactly).

### 5.4 The legality guardrail the proposals missed (`ARCH-SPATIAL-03`) — critical

A spatial weight can silently bias or fabricate association. If you weight space by GDP and then test whether mortality is associated with GDP, the weight has smuggled the hypothesis into the null structure. **This is a circularity the original proposal did not flag, and it must be enforced.**

Rule: every graph has a `legality_class`:
- **`structural`** — geometry only (contiguity, distance, k-NN on centroids). Always safe.
- **`context_derived`** — weight depends on a substantive variable (population, GDP, flows).

Enforcement in the PIRS selector and legality engine:
```text
if graph.legality_class == "context_derived"
   and graph.provenance ∩ {test.outcome_source, test.covariate_source} ≠ ∅:
       REJECT graph for this test   (or downgrade result to descriptive_association_only)
```
Default spatial structure for any inferential test is `structural` (contiguity). A `context_derived` graph may be used only when (a) intent explicitly declares it, (b) it shares no provenance with the outcome/covariate under test, and (c) the choice is recorded in the PIRS decision record (§7). Add a §10-style abort: *"context-derived spatial weight shares provenance with the tested variable."*

---

## 6. Concept-family grammar & EFG instantiation

**Realizes proposal 7. Sharpens `EFG-REG-01`.**

The clarification is correct and faithful to the EFG-as-compiler vision: a registered concept like `mortality_rate` is **grammar**, not a fixed output column and not a user formula. The EFG *instantiates* concrete legal fields from concept families against intent + availability + legality.

### 6.1 Concept registry (`ARCH-CONCEPT-01`)

```yaml
# config/registries/concepts.yaml
concepts:
  mortality_rate:
    family: rate
    numerator: {carrier: Deaths, event: SIM-DO}
    denominator_family: population_at_risk
    legal_operators: [RN]
    required_axes: [geography, time]
    optional_stratifiers: [age, sex, race]
    restriction_templates:
      cause_specific: {axis: icd_topology_role, operator: sigma_C, source: underlying_icd_norm}
    default_state: fragile_until_legal
  hospitalization_rate: { ... carrier: HospitalAdmissions ... }
  low_birth_weight_share: { family: proportion, numerator: {carrier: LiveBirths, restriction: weight<2500}, denominator_family: live_births ... }
```

### 6.2 The instantiation planner (`ARCH-CONCEPT-02`)

`efg/concept_planner.py` expands a requested concept family into concrete V-fields:

```text
intent: "cardiovascular mortality by municipality-year"
   → concept: mortality_rate
   → restriction: cause_specific(I20–I25) via σ_C over underlying_icd_norm
   → numerator: count(Deaths | icd ∈ I20–I25) over municipality×year
   → denominator: population_at_risk(municipality×year)   [check availability/profile]
   → RN(numerator, denominator)  →  legality check (Δ + declaration)
   → V-field: "I20–I25 mortality rate by municipality-year"  (state per Q-tensor)
```

The division of labor is the compiler analogy made literal:
- **Concept registry = grammar** (what fields are expressible).
- **Instantiation planner = parser/codegen** (expand grammar against intent + substrate).
- **Legality engine (`efg/legality.py`) = type checker** (reject illegal instantiations).
- **Executor (`efg/executor.py`) = backend** (materialize the legal field).

This reinforces `EFG-REG-01`: after the SHE fixes populate real substrate, the planner guarantees each requested concept either instantiates a legal field or emits a typed `FailedBranch` (never a silent absence). It also prevents the two failure modes proposal 7 warned about: the EFG does **not** discover rates by chance (only registered concepts instantiate), and users do **not** write arbitrary formulas (intent selects concept *families*, not expressions).

---

## 7. PIRS decision records — pedagogical, auditable manifests

**Realizes proposal 9. Low cost, high value.**

PIRS is currently a black box. The fix is both documentation and a manifest-schema enrichment, aligned with the anti-silence ethos.

### 7.1 The decision-record schema (`ARCH-PIRS-EXPLAIN-01`)

For every modeled outcome, emit a structured record with a **rationale string per decision**:

```json
{
  "outcome_field_id": "...",
  "decisions": {
    "model_family":  {"chosen": "negative_binomial",
                      "why": "count outcome with overdispersion (var/mean=4.2 > 1.5); Poisson rejected"},
    "exposure_offset": {"chosen": "log_population_at_risk", "why": "rate interpretation per §6.2"},
    "spatial_mode":  {"chosen": "municipality_FE",
                      "why": "|T|=12 ≥ 10 and ζ_Y=0.07 ≤ 0.10 (§6.3 precedence)"},
    "spatial_graph": {"chosen": "contiguity_queen",
                      "why": "structural graph; context-derived graphs share provenance with covariate (§5.4)"},
    "residual_mode": {"chosen": "cross_fitted", "why": "budget=standard (§6.6)"},
    "null_regime":   {"chosen": "season_preserving_moving_block_circular_shift",
                      "why": "monthly seasonal panel (§6.9)"},
    "fdr_method":    {"chosen": "BY", "why": "dependent tests across space (§6.8)"},
    "hsic_mode":     {"chosen": "nystrom", "why": "n_eff=8.2k > 5000, budget standard (§6.7)"}
  }
}
```

### 7.2 Renderer (`ARCH-PIRS-EXPLAIN-02`)

`pirs/explain.py` renders the records into a human-readable per-run markdown/HTML summary ("why the system modeled this count with NB, this proportion with beta-binomial, this cost with Gamma…"). This is what turns PIRS from a black box into a teachable, auditable instrument, and it composes with the §6.2/§6.3 family/spatial work (`PIRS-FAM-01`, `PIRS-SPAT-01`) — those decisions become the content of the records.

---

## 8. Run-profile refinement — decouple DataScope from ExecutionStage

**Realizes proposal 10. Refines compliance §3.2 (Run Profile).**

The concern is fair: framing PIRS as "skippable" risks making the system's investigative core feel peripheral. The fix is to recognize that the compliance report's single "Run Profile" axis was conflating two orthogonal things.

### 8.1 Two orthogonal axes (`ARCH-PROFILE-01`)

- **DataScope** (which substrate classes are populated): `core_vital` / `contextual` / `full` — unchanged from compliance §3.2.
- **ExecutionStage** (how far the pipeline runs): `validate` (SHE + EFG legality only, no materialization) / `compile` (materialize `V_fields` + `Q_tensor`, no PIRS) / `investigate` (full SHE + EFG + PIRS + HSIC).

PIPELINE identity: **`SHE + EFG` = compiler validity / field construction; `SHE + EFG + PIRS` = full investigative PegaSUS.** `investigate` is the *default* for a real run; `validate`/`compile` are explicit debugging/staging choices. PIRS is never "optional" — it is the stage that is skipped *only* when the operator deliberately asks for a sub-run.

### 8.2 Bundle validation depends on both axes

The §3.2 `PROFILE_NONEMPTY` rule becomes a function of `(DataScope, ExecutionStage)`:
- `ModelAssociations`, `ResidualAssociations`, `Hypotheses` are required-non-empty **only** at `ExecutionStage = investigate`.
- Context fields in `V_fields` are required-non-empty only at `DataScope ∈ {contextual, full}`.
- All 17 keys still always exist; empty keys still carry an explicit `empty_by_stage`/`empty_by_profile` reason (the anti-silence guarantee).

This keeps honest staging (you *can* run `core_vital`+`compile` for debugging) without ever implying PIRS is secondary to the mission.

---

## 9. Deferred extension point — higher-order context-conditioned functionals

**Realizes proposal 8, but DEFERRED with hard guardrails.**

Proposal 8 (e.g. `Ψ_var(ρ | GDP-quintile)` = variance of mortality rate within GDP quintiles) is intellectually sound and genuinely useful, but it is **new scope beyond the MSD's locked operator set (§12)** and carries the highest legality risk of anything in this addendum. Verdict: **define the contract now, implement after Phases 0–3, and only after an MSD §3 amendment.**

### 9.1 Why defer (be honest about the risk)

A higher-order functional composes three objects, each with its own provenance/state: (a) a compiled field `ρ` (its own legality/state), (b) a context axis `q` (possibly a reconstructed/`B_cross-sectional` cube — inherits quarantine), (c) a dispersion functional. Getting the legality wrong silently produces authoritative-looking heterodox statistics. Specifically: the context axis is frequently *derived from the same source* as the rate's denominator or a covariate — the same circularity as §5.4. This must not ship before the core is compliant and the guard is in place.

### 9.2 The contract (define now, `ARCH-HOF-01`)

```yaml
# config/registries/higher_order_operators.yaml  (extension point; impl deferred)
higher_order_operators:
  contextual_dispersion:
    input_field_family: [rate, proportion]
    context_axis: required                 # e.g. gdp_quintile (must be a legal axis)
    functional: [variance, cv, gini, iqr, theil]
    legality:
      input_must_be_materialized: true
      input_state_max: fragile
      context_axis_must_be_legal: true
      context_provenance_disjoint_from_input: true   # the §5.4 anti-circularity rule, reused
      output_state: quarantined_descriptive          # dashboard-unsafe by default
      output_never_denominator: true
```

### 9.3 The EFG/PIRS boundary (resolve before building)

A higher-order functional produces a **materialized descriptive field** (a number per context bin) — it is NOT an inferential claim. PIRS separately *tests* such dispersion. Keep them distinct: EFG materializes `Ψ_var(ρ|q)` as a quarantined descriptive field; PIRS may then test its association with other context. Building this prematurely blurs that boundary, which is why it is gated behind core compliance.


---

## 10. Integrated roadmap — how this interleaves with the compliance plan

The golden rule: **correctness precedes platformization, but the highest-leverage correctness fix *is* the first slice of the platform.** Do not build pillars 1–4 as a greenfield rewrite; grow them out of the compliance fixes.

### Phase 0 — Honesty & hygiene (unchanged from compliance §12, plus)
- Compliance: registry hygiene (`REG-DEAD-01`, `REG-VALIDATOR-01`, `REG-STALE-LIST-01`), Run Profile contract, `PIRS-REG-01`.
- **`ARCH-PROFILE-01`** — split DataScope vs ExecutionStage now (it is a small `UserIntent` change and clarifies everything downstream).
- **`ARCH-REG-07`** — replace `validate_registry_tree` with the real cross-registry validator (you are already touching it for `REG-VALIDATOR-01`).

### Phase 1 — Restore source reality *as* the Registry Platform's first slice
- Implement compliance `SHE-NORM-01` **via Path A = `ARCH-REG-02` + `ARCH-REG-03` + `ARCH-ADAPT-01` + §2.3 callable registry.** One effort delivers both the correctness fix (real substrate) and pillar 1's foundation.
- Compliance `SHE-SINASC-01`, `SHE-CNES-01/02/03`, `SHE-DEC-02/03` — these become *registry entries + callables*, exercising the new platform.
- `EFG-Q-01` (state tensor) + **`ARCH-SPATIAL-01/02`** — build the SpatialWeightGraph now because Moran's I (a Q-tensor component) needs it; the `structural` contiguity view is a thin wrapper over existing `adjacency.py`.

### Phase 2 — Correct inference, consuming the new spatial graph
- Compliance `PIRS-SPAT-01` ← **`ARCH-SPATIAL-02/03`**: ICAR/FE consume the declared graph; enforce the §5.4 legality guard.
- Compliance `PIRS-FAM-01` (families) + **`ARCH-PIRS-EXPLAIN-01/02`** (decision records — the new families/spatial modes become the record content).
- Compliance `RACE-01` steps 1–2.

### Phase 3 — Light up `contextual` via the cube lifecycle + CTR kernel
- Compliance `SIDRA-CTX-01/02` ← **`ARCH-PANEL-01`** (CommonPanel phase) + the cube lifecycle (§3.1).
- **`ARCH-CTR-01/02/03/04`** — refactor `she/population/` into the CTR kernel; deliver age-bin disaggregation (`ARCH-CTR-02`) and the certification gate. ST-DFM becomes a CTR instance.
- **`ARCH-CONCEPT-01/02`** — formalize the concept grammar + instantiation planner (the EFG now has real substrate + context to instantiate against).
- Compliance `RACE-01` steps 3–4.

### Phase 4 — `full` scale + reconstruction backends
- Compliance `SHE-POP-02` ← **CTR solver backends** (ADMM, state-space, primal-dual) serving all CTR instances at Brazil scale.
- Synthetic context cubes beyond population (sanitation/GDP surfaces) through `ARCH-CTR` + the certification gate.

### Phase 5 (research, gated) — higher-order functionals
- **`ARCH-HOF-01`** only after Phases 0–3 compliance and an MSD §3 operator-set amendment.

---

## 11. New work-item index (`ARCH-*`)

| ID | Title | Pillar | Severity-equiv | Depends on / pairs with |
|---|---|---|---|---|
| `ARCH-PROFILE-01` | Split DataScope vs ExecutionStage | refine | S2 | compliance §3.2 |
| `ARCH-REG-01` | `source_adapters.yaml` registry | 1 | S1 | ARCH-ADAPT-01 |
| `ARCH-REG-02` | `source_routing.yaml` (raw→canonical) | 1 | S0 | **= SHE-NORM-01 Path A** |
| `ARCH-REG-03` | Populate `composite_decoders.yaml` | 1 | S1 | ARCH-REG-02 |
| `ARCH-REG-04` | Formalize `concepts.yaml` | 1/6 | S1 | ARCH-CONCEPT-01 |
| `ARCH-REG-05` | Externalize `operators.yaml` | 1 | S2 | EFG-LEG-01 |
| `ARCH-REG-06` | `spatial_graphs.yaml` | 1/4 | S1 | ARCH-SPATIAL-01 |
| `ARCH-REG-07` | Real cross-registry validator + generator CLI | 1 | S1 | REG-VALIDATOR-01 |
| `ARCH-ADAPT-01` | `SourceAdapter` plugin interface | 1 | S1 | ARCH-REG-01 |
| `ARCH-PANEL-01` | CommonPanel compilation phase + manifest | 2 | S1 | SIDRA-CTX-01 |
| (lifecycle) | `she/lifecycle/{event,cube}.py` split | 2 | S1 | SIDRA-CTX-01 |
| `ARCH-CTR-01` | CTR kernel (refactor population → reconstruction) | 3 | S1 | SHE-POP-01 |
| `ARCH-CTR-02` | Age-bin latent disaggregation | 3 | S1 | ARCH-CTR-01 |
| `ARCH-CTR-03` | CTR certification gate | 3 | S0 | reuses §2.10.4 |
| `ARCH-CTR-04` | Abort: reconstructed promoted w/o certification | 3 | S0 | ARCH-CTR-03 |
| `ARCH-SPATIAL-01` | SpatialWeightGraph registry + API | 4 | S1 | geo/adjacency.py |
| `ARCH-SPATIAL-02` | Wire ICAR/Moran/HSIC/cross-fit/CTR to graph | 4 | S1 | PIRS-SPAT-01, EFG-Q-01 |
| `ARCH-SPATIAL-03` | Spatial legality/circularity guard | 4 | S0 | ARCH-SPATIAL-01 |
| `ARCH-CONCEPT-01` | Concept-family registry | 6 | S1 | EFG-REG-01 |
| `ARCH-CONCEPT-02` | EFG concept instantiation planner | 6 | S1 | ARCH-CONCEPT-01 |
| `ARCH-PIRS-EXPLAIN-01` | PIRS decision-record schema | 7 | S2 | PIRS-FAM/SPAT |
| `ARCH-PIRS-EXPLAIN-02` | Human-readable PIRS renderer | 7 | S3 | ARCH-PIRS-EXPLAIN-01 |
| `ARCH-HOF-01` | Higher-order functional extension point | 9 | deferred | Phases 0–3 + MSD amend |

Severity-equiv uses the compliance scale (S0 contract-critical … S3 hygiene) applied to the *target architecture*: e.g. `ARCH-SPATIAL-03` is S0 because shipping context-derived spatial weights without the circularity guard produces silently-wrong inference.

---

## 12. MSD amendments this addendum requires

Beyond the compliance report's §13 amendments, accepting these architectural directions requires the MSD to add:

1. **§1.6 Two Lifecycles + CommonPanel** — define the event vs cube lifecycle and name CommonPanel construction as a compilation phase with per-cell provenance.
2. **§2.13 Constrained Tensor Reconstruction** — generalize §2.8 (population) and §2.10 (ST-DFM) into one CTR family with a shared certification gate; declare age-bin disaggregation a CTR instance; add the `ARCH-CTR-04` abort.
3. **§2.14 / §6.10 SpatialWeightGraph** — define the graph contract, the derived views, the `legality_class`, and the circularity abort (`ARCH-SPATIAL-03`). Restate §6.3 ICAR as *consuming* the graph.
4. **§3.16 Concept grammar** — define concept families as grammar and the instantiation planner as the expansion mechanism (the EFG-as-compiler model made explicit).
5. **§8.7 / §9.3 ExecutionStage** — add the validate/compile/investigate axis orthogonal to DataScope; restate PIRS as central to the `investigate` stage rather than optional.
6. **§3.17 (deferred) Higher-order operators** — reserve the extension point with the legality guard; mark implementation as post-core.

Keeping spec and code in lockstep here is the whole point of the platform: every one of these amendments corresponds to a registry that becomes the executable authority, so the MSD describes *contracts the registries enforce*, not prose the code may quietly diverge from.

---

## 13. One-paragraph summary for a hurried implementer

Most of the proposed ideas are directionally right and converge on a single architecture: **PegaSUS as a registry-driven compiler platform.** Build it by *growing it out of the correctness fixes*, not by rewriting. Do `SHE-NORM-01` as the first slice of a real raw→canonical routing registry backed by named decoder callables (pillar 1). Add a `SpatialWeightGraph` registry with derived views and — critically — a legality class that blocks context-derived weights from contaminating tests of their own variables (pillar 4); wire ICAR/Moran/HSIC/CTR to it. Make the SIDRA cube path real by implementing the cube lifecycle and an explicit CommonPanel phase (pillar 2). Refactor the population solver into a general Constrained Tensor Reconstruction kernel that also does latent age-bin disaggregation and synthetic cubes — each behind a certification gate that keeps reconstructed data quarantined and never "observed" (pillar 3). Treat registered concepts as grammar the EFG instantiates (not fixed columns), make PIRS explain its choices in human-readable decision records, and split run profiles into orthogonal DataScope and ExecutionStage axes so PIRS stays central. Defer higher-order context-conditioned functionals until the core is compliant and the MSD is amended. The guardrails the original proposals lacked — spatial circularity, synthetic-cube certification, declarative-plus-named-callable, scope/stage decoupling — are not optional; they are what separate a rigorous platform from a faster way to produce confident, wrong numbers.

*End of addendum. Use `ARCH-*` IDs in commits; pair them with the compliance `*` IDs they extend.*
