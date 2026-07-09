# PegaSUS Completion Roadmap (2026-07-09)

The settled assessment + architected solutions + a sequenced roadmap to finish the project.
Consolidates the repo-health assessment + the two review waves + three user reframings:
(1) **full national × full-temporal (2000–2024) is the DEFAULT operating regime**, not an edge case;
(2) the `data/` layout is accreted and needs a persistence contract + cleanup;
(3) RaceBridge's problem is **ecological, not individual** — with real implications for its design.

Companion docs: `PEGASUS_REPO_HEALTH_ASSESSMENT.md` (findings), `PEGASUS_OUTPUT_QUERY_LAYER.md`.

---

## 0. Operating reality (the frame that changes priorities)

The routine job is **all systems, all UFs, 2000–2024**. Local scopes are the exception. Therefore:
- The **data/fetch/normalize layer is on the critical path for every run** and must be optimized for
  the full-scale case — out-of-core, bounded-memory streaming, and parallelism are **first-class
  requirements, not "later at full scale."** My earlier "not now" framing was wrong; corrected here.
- Full-temporal is non-negotiable (you can't pretend time doesn't exist): the LDO's lag machinery,
  the population tensor's 2000–2025 coverage, and the normalizers' 25-year streaming all must hold
  at that span.

---

## 1. RaceBridge — problem reformalization + redesign (centerpiece)

### 1.1 The problem, objectively defined

We want race-stratified rates (e.g., mortality by **self-declared** race — the census/IBGE concept).
The **numerator** (DATASUS deaths/events) carries an **administrative** race label (often recorded by
a third party, not the person); the **denominator** (census population) carries **self-declared** race.
The two concepts diverge systematically in Brazil (administrative over-reports *branca*, under-reports
*parda/preta*). Dividing admin-race events by self-declared population yields biased race-specific rates.

### 1.2 Your key insight, formalized: the distortion is ECOLOGICAL, not individual

- **Individual level — unidentifiable.** For a single death labeled *branca*, nothing tells us whether
  that person would have self-declared *branca*, *parda*, or *preta*. There is no individual ground
  truth (no record linkage). Individual reclassification is fundamentally not identifiable.
- **Aggregate level — identifiable.** Across a cell (municipality×year), distortion **is** detectable:
  if the census population is 10% *branca* but 60% of deaths are labeled *branca*, and we don't believe
  *branca* mortality is ~13× higher, admin over-labeling is implicated. **The signal lives in the
  discrepancy between the admin-race event distribution and what the self-declared population + plausible
  rates predict.** This is a classical **ecological-inference** structure.

This reframing was under-acknowledged in the original formalization, and it matters (§1.5).

### 1.3 The model: a hierarchical Poisson ecological-deconvolution / latent-class measurement model

Indices: `j` = self-declared (true, census) race; `k` = administrative (recorded) race; `s` = cell.
Observed: `N_{s,j}` = census population of race `j`; `Y_{s,k}` = admin events labeled race `k`.
Latent target: `λ_{s,j}` = true self-declared-race-`j` event rate in cell `s`.
The single small carrier of the distortion: `C_{k|j} = P(recorded=k | self-declared=j)` — the confusion matrix.

Generative model (forward, not a plug-in reversal):
```
Y_{s,k} ~ Poisson( Σ_j  C_{k|j} · λ_{s,j} · N_{s,j} )
```
i.e. true-race-`j` people generate events at rate `λ_{s,j}`, and each event is admin-labeled `k` with
prob `C_{k|j}`. We infer the **posterior over `λ_{s,j}`** (the actual quantity of interest), with `C`
the shared misclassification structure.

### 1.4 Identifiability — and why your "collapse to a small bounded entity" trick is the key

Per-cell, `{λ_{s,·}, C}` is hopelessly under-identified (K observations, K rates + K² confusion params).
**The identification strategy IS the small-C trick, formalized:**
- Make `C` **shared** across cells (one national `C`, or region-level `C_r`) and **small/structured**
  (a 5×5 stochastic matrix — or fewer free params: mostly-diagonal with a few dominant off-diagonal
  "flows" like *parda→branca*).
- Then the **many cells with varying census composition `N_{s,j}`** identify the few parameters of `C`
  (via cross-cell contextual variation — the same mechanism that identifies King/Goodman ecological
  models) **separately** from the cell-specific `λ_{s,j}` (pinned by hierarchical shrinkage).
- **Honest scrutiny (you invited it):** pure ecological estimation of `C` is *fragile* — it leans on the
  shared-`C` homogeneity assumption, sufficient contextual variation, and the small parameterization to
  break the rate-vs-distortion confound. So the **robust design is HYBRID**: a literature/expert-informed
  **prior** on `C` (your "empirical, educated guess" — the bounded entity) + the ecological likelihood to
  **refine** `C` where the data identify it + **wide posterior uncertainty** where they don't. The
  small-`C` parameterization is exactly what makes even a rough prior useful and keeps the model *not
  "overly hairy"* — your architectural instinct is the identifiability mechanism, not just an aesthetic.

### 1.5 Why the current design is mis-levelled, and the wiring implications

Current bridge: per-cell **count reallocation** with a **fixed** `C`:
`posterior_self_{s,j} = Σ_k Y_{s,k}·W_{s,k,j}`, `W ∝ C_{k|j}·π_{s,j}` (a Bayes *reversal* with local
composition `π`). This:
1. Frames it as **individual reallocation** (redistribute K admin counts into J bins per cell) — the
   very thing that isn't individually identifiable; the reallocation is a plug-in, not an inference.
2. Uses a **fixed** `C` — never estimates it from the ecological signal, which is the *only* identifiable
   thing.
3. Produces reallocated **counts**, then rates from those points — not a **rate posterior** with the
   deconvolution's (often large) uncertainty.
4. `local-π` is a heuristic anchor, not a principled hierarchical prior.

**Wiring implication:** race-rate estimation is a **joint, panel-wide** problem (all cells share `C`), so
it belongs as a **measurement-model layer over the whole municipality×year×race panel** that emits
`λ_{s,j}` posteriors into the rate/LDO — not a per-cell plug-in buried in the denominator build. This is
tractable *because* `C` is small (the only global parameters are `C`/`C_r` — a handful of numbers — plus
shrunk cell rates: a standard hierarchical Poisson model).

### 1.6 Redesign — phased

- **W-RACE-0 (reframe + honesty, days):** relabel the current fixed-`C` output as what it is — an
  *ecological plug-in under a fixed prior `C`* (the degenerate, no-refinement case of the model above),
  and mark its rates *descriptive* where `C` is uncalibrated. No math change; stops overclaiming.
- **W-RACE-1 (code-only, ~1 wk):** propagate the bridge's already-computed uncertainty into the rate +
  LDO measurement-error term (currently dropped); finish region-conditioned `C_r` selection; replace
  `local-π` with a census-anchored shrinkage prior. Makes the pipeline *ready* for an inferred `C`.
- **W-RACE-2 (the real redesign, ~3–4 wk):** implement the hierarchical Poisson ecological model
  (§1.3) — literature prior on `C_r` + ecological refinement via the panel, EM+Laplace first (fast,
  integrates with numpy/polars), MCMC later if multimodality bites. Emit `λ_{s,j}` posteriors → rates.
  Data: literature/PNS/expert priors now; record-linked `C` if ever acquired (upgrades the prior).
- **W-RACE-3 (optional):** full MCMC + region hierarchy if sensitivity analysis demands it.

**Verdict:** a genuine, warranted redesign — but *evolutionary* (the small-`C` core survives and becomes
a proper inferential object), not a rewrite. Highest-value first step is W-RACE-1 (honest uncertainty).

---

## 2. Data layer for the full-scale default (reprioritized to first-class)

Because full national×temporal is the norm (§0), the data-plane deep-dive's Tier-C (out-of-core) is
**promoted** alongside its quick wins:
- **DP-1 (byte-safe, now):** R-subprocess poll→backoff; drop redundant warm-cache SHA256; SIDRA
  metadata content-hash cache. ~15–30 min/run, zero behavior change.
- **DP-2 (~1 wk):** decouple normalize parallelism from the 2-way cap → memory-budgeted 3–4-way
  (+25–40% wall-clock), byte-safe.
- **DP-3 (first-class, ~1–2 wk):** end-to-end streaming (`scan_parquet`→`sink_parquet`) with an explicit
  **memory ceiling** so a 25-year national normalize is bounded, not best-effort; DuckDB out-of-core for
  the passes that still materialize. This is the difference between "runs on this machine" and "OOMs" at
  the default scale, so it's not deferrable.
- **DP-4:** per-chunk→quarterly combine batching; combine-output caching by `(system,uf,years)`.

---

## 3. Storage / cache / persistence — a contract + a cleanup

**Problem (confirmed by inspection):** `data/` mixes the pipeline layers (`raw/cache/processed/
normalized`, 47G) with accreted dev-run detritus (`actual_state_panels/…/run_{full,win,epifix2,lag,…}`
+ stale `__efg/__pirs_stage_workspace` siblings) and empty stubs (`intermediate/`, `assets/`,
`_peryear_probe/`). No documented persistence contract; redesigns accreted.

**STORE-ORG (design):**
1. **A documented storage contract** (`data/` layout spec): a clean two-tier split —
   `data/lake/{raw,cache,processed,normalized,sidra,metadata}` (the content-addressed data plane) vs
   `data/runs/<intent>/<run_id>/` (immutable run bundles) vs `data/assets/<name>/<version>/` (versioned
   foundational assets: population tensor, spatial/disease graphs). One convention, enforced by a path
   registry, so persistence stops re-accreting.
2. **A GC/cleanup pass:** delete the dev-run detritus (`actual_smokes`, ad-hoc `run_*` variants, all
   `__pirs_stage_workspace`), fold `actual_state_panels` into the `runs/` convention, remove empty stubs.
   Reclaims disk + removes the confusion. Gate: keep the newest canonical run per intent; archive-list
   what's dropped (never silent).
3. **Stage-workspace lifecycle:** `__efg_stage_workspace` dirs are transient — they should be created
   under a temp/scratch path and cleaned on success (like the DATASUS ancillary cleanup already does),
   not left beside run bundles.

---

## 4. The rest (from the reviews — folded in, unchanged in priority logic)

- **Registries:** W-REG-1 operationalize vectorized transforms (→ "add data ≈ registry edit", ~2 wk) →
  health-registry typing/de-orphan → W-REG-2 codebooks/SIDRA policies → REG-07 consolidation (deferred).
- **Per-module robustness:** silent-exception logging, ICD/concept cache mtime-invalidation, spatial-graph
  view caching, bundle consistency checks, multi-reason exclusions. (Rejected: query silent-fallback.)
- **Integrity tail:** maternal_child_linkage truthfulness flag; delete `pirs/` after test migration.
- **Output Query Layer:** P3d materialized_field read + P3e CLI.
- **Refactor debt (gated):** god-module decomposition (compile/dag/executor/kernels); REG-07.

---

## 5. Sequenced roadmap to completion

**Phase 1 — Full-scale readiness + integrity (pre-live-test).**
Data plane DP-1/DP-2/DP-3 (bounded-memory national normalize) · storage contract + GC (STORE-ORG) ·
Tier-1 integrity (maternal flag; empirical_compression deleted ✓) · W-RACE-0 + W-RACE-1 (honest race
uncertainty) · per-module safe robustness fixes. **Exit:** a full national×temporal `validate`-stage run
completes within memory + emits honest provenance.

**Phase 2 — Live test + measurement upgrades.**
Run the reduced-statewide then national `validate`→`investigate` live test · W-RACE-2 (ecological `C`
model) · W-REG-1 (registry-driven transforms) · health-registry typing. **Exit:** a national investigate
run produces certified, race-honest, self-describing hypotheses.

**Phase 3 — Polish + hardening.**
God-module decomposition · REG-07 · Output Query P3d/P3e · DP-4 · W-RACE-3 (if warranted) · `pirs/` removal.

**Phase 4 — The studies.**
National C25 + disease-comorbidity at full 2000–2024 scale, on the now-first-class data plane.

**The single highest-leverage next move:** Phase-1 data-plane bounded-memory streaming (DP-3) —
without it, the *default* run risks OOM; with it, everything downstream is unblocked at real scale.
