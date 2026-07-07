# PegaSUS Architectural Advancements & Feature Proposals

Annexed from an ongoing project discussion (handed off 2026-07-07). These were framed against a *fairly early LDO build*; below each I record the CURRENT status (what's since been built) and how it connects to the math/modularization/GPU critique findings. Items 1–4 are buildable scope; item 5 is deferred pending its own spec pass.

---

## P1 — SpatialWeightGraph: richer, legality-typed adjacency
**Proposal:** multiple weighting schemes as selectable `view`s of one registered graph — distance-decay `exp(-d/ρ)`, k-NN, gravity `pop_i·pop_j/d²`, flow-based (SIH transfer volumes). **Hard constraint:** gravity/flow are `context_derived` (depend on population/flow variables) and MUST be rejected as the spatial prior for any tested variable sharing that provenance (circularity guard). Only contiguity/distance (`structural`) are safe as a default prior — the typing ships *with* the graph.

**Status / assessment (mostly a real gap):**
- `SPG-01` (task #7, done) built a `SpatialWeightGraph` with **views/blocks/laplacian + a circularity guard** — so the *typing scaffold* and the structural-vs-context-derived rejection exist. But the actual code the LDO uses (`geo/adjacency.py` → `structural_cod6_adjacency`) is **binary contiguity only**; the richer weight schemes (distance-decay/kNN/gravity/flow) are **not** wired into the GMRF whitening.
- **Strong connection to the critique:** `M1`/`A20` (Moran computed 6 ways; the non-geographic proxy) and Theme-6 (spatial mishandling) both point to the SAME consolidation — ONE `geo.spatial` module owning the graph + all weight views + Moran + effective-n. The circularity typing is *essential* and aligns with the critique's demand that context-derived graphs never inform an edge on the same data.
- **Verdict:** BUILD as a refinement of SPG-01, folded into the spatial-consolidation redesign (M1). Distance-decay/kNN are low-risk; gravity/flow require the circularity guard enforced at the whitening call site (refuse a gravity graph as the prior for a population-derived variable).

## P2 — Disease-tree "borrow strength": adaptive, pattern-only shrinkage
**Proposal (3 hard requirements):** (1) shrink **structure only** (spatial/temporal/reporting smoothness), never the disease's own **level/rate**; (2) estimate `τ²` **adaptively per block from the data** (genuinely-different diseases get little/no shrinkage — data can override the prior); (3) **flag** heavily-shrunk estimates in provenance/uncertainty.

**Status / assessment (partially built, key gap = adaptive τ²):**
- `O6` (this session) built `sum_of_scales_disease_operator` — per-scale disease Laplacians {category/block/chapter} — but with **FIXED per-scale precisions** (`disease_scale_precisions` a caller constant). It shrinks precision *rows* (structure), which satisfies requirement (1). It does NOT satisfy (2) — the precisions are not estimated adaptively from data — nor (3) — no flagging of shrunk estimates.
- **Connection to critique:** `LDO-DIS-IDENT-03` (per-scale precisions confounded/collinear) and Theme-4 (uncalibrated magic numbers, incl. `LDO-DIS-MAGIC-06`: γ_disease=0.1 unjustified) independently demand exactly requirement (2) — estimate the shrinkage strength, don't hard-code it. `LDO-DIS-SPARSE-07` (rare-disease copula invalid) motivates requirement (3) — a heavily-shrunk rare-disease estimate must not read as measured.
- **Verdict:** BUILD requirements (2) adaptive τ² (empirical-Bayes / marginal-likelihood per block) and (3) shrinkage-flag; (1) is largely satisfied but verify the prior never touches the level. This is the DiseaseGraph-shrinkage refinement + it discharges two critique findings.

## P3 — Export / Materialization layer (NEW subsystem)
**Proposal:** a first-class layer for user-controllable exportable datasets: user selection of fields/strata/resolutions to materialize; tidy typed Parquet/CSV carrying per-cell **uncertainty + provenance columns** alongside values; stable documented schemas for external tools; reproducibility manifests (foundation-asset versions + query → export). Flagged as a genuine architectural gap, likely its own MSD-III section/appendix.

**Status / assessment (genuine new gap, high strategic value):**
- No such layer exists; the `interrogate`/`lens` verbs + 17-key bundle are for inline reading, not controllable export.
- **Deep connection to the critique's #1 theme:** the export MUST carry per-cell **uncertainty + provenance** — which is exactly what `A1`/`A16`/`M6`/Theme-2/Theme-14 say the system computes-but-discards. The export layer is the natural FORCING FUNCTION for the `ObservationReliability` contract: if every exported cell must ship its uncertainty + provenance, the uncertainty pipeline can no longer be decorative. So P3 and the reliability-contract redesign should be co-designed.
- **Verdict:** BUILD as a new work-item family AFTER the reliability contract (M6) exists, so exports carry real (not fabricated) uncertainty. Highest strategic value of the four — it is how results actually leave the system.

## P4 — Default exposure/denominator declaration (per-query overridable)
**Proposal:** exposure choice a first-class, per-query-overridable declaration (not a fixed registry default) — support instantiating the SAME count under MULTIPLE denominators (e.g. proportional-morbidity `total_admissions` vs population-based) as distinct, both-first-class variables on demand.

**Status / assessment (real refinement of O7 / REG-07):**
- `O7` (this session) added the extensive/intensive routing gate (count-with-exposure only for extensive-count numerators). The registry requires an `exposure_ref` per extensive measure but has no per-query override / multi-denominator mechanism.
- **Connection to critique:** Theme-2 (`LDO-MARG-08`, `MQ-EXPOSURE-VARIANCE-01`, `RN-N-EVENTS-RATE-01`) — the denominator is treated as an exact known constant and the count-vs-rate wiring is buggy. P4's "same count under multiple denominators" is epidemiologically important (proportional mortality vs rate) AND forces the denominator to be an explicit, uncertainty-carrying declaration rather than a silent default.
- **Verdict:** BUILD as a REG-07/EFG-OUT-01 refinement, co-designed with the denominator-uncertainty fix (Theme-2): each denominator instantiation carries its own reconstruction uncertainty into the margin.

## P5 — Causal-ladder implementation detail (DEFERRED)
**Proposal:** concrete build specs for Rung-1 orientation (collider/CI-pattern + LiNGAM) and Rung-2 quasi-experimental estimators (ITS, DiD). **Explicitly deferred to its own installment — do NOT build until spec'd.**

**Status / assessment:** Rung-1 (LiNGAM + collider) and Rung-2 (ITS + negative-control veto) were built this session (CAUSAL-01, O8) — BUT the critique's Theme-15 found them fragile (LiNGAM on Gaussianized data destroys the non-Gaussianity it needs; collider without faithfulness / Meek propagation; ITS autocorrelation/single-break). So P5's "own spec pass" should INCORPORATE the Theme-15 corrections before any further causal build. Hold, per the proposal, and fold the Theme-15 findings into that spec.

---

## How the proposals map to the redesign roadmap
- P1 → folds into the **spatial-consolidation** redesign (`M1` + Theme-6): one `geo.spatial` (graph + weight views + Moran + effective-n + circularity typing).
- P2 → **disease-shrinkage adaptive-τ²** (Theme-4 `LDO-DIS-*` + `O6` refinement).
- P3 → **new Export/Materialization layer**, co-designed with the reliability contract.
- P4 → **denominator declaration** refinement (Theme-2 + `O7`/REG-07).
- P5 → held for its own spec, seeded with Theme-15.
- The unifying prerequisite for P3 and P4 is the **`ObservationReliability` contract** (`M6`/Theme-14) — the single highest-leverage redesign.
