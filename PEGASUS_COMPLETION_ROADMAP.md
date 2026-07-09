# PegaSUS Completion Roadmap (2026-07-09)

The settled assessment + architected solutions + a sequenced roadmap to finish the project.
Consolidates the repo-health assessment + the two review waves + three user reframings:
(1) **full national × full-temporal (2000–2024) is the DEFAULT operating regime**, not an edge case;
(2) the `data/` layout is accreted and needs a persistence contract + cleanup;
(3) RaceBridge's problem is **ecological, not individual** — with real implications for its design.

Companion docs: `PEGASUS_REPO_HEALTH_ASSESSMENT.md` (findings), `PEGASUS_OUTPUT_QUERY_LAYER.md`.

---

## STATUS — autonomous burn (last updated 2026-07-09)

Each roadmap item was executed OR verified against live code (code over docs — many audit flags were
stale). Outcome triage:

**DELIVERED (validated + committed):**
- **W-RACE-2 ecological estimator** — `measurement/race_ecological.py` (commit 75f1afe). Generative
  Poisson deconvolution per §1.3, per-stratum emission C = covariate-dependent confusion (§1.7).
  Proof-of-capability `tests/unit/test_race_ecological_deconvolution.py`: gradcheck 1e-7, recovers a
  planted 1.5 rate-ratio the naive crosswalk erases to 0.91, strong-identity-prior re-erases guard,
  two-stratum recovery, identifiability diagnostic. This is the §1.8 planted-signal validation the
  user required *before* implementation, now promoted to the shipped test.
- **STORE-ORG workspace GC** — `datasus/storage_gc.py::gc_stale_stage_workspaces` (commit 0585171).
  Dry-run-default, safe-by-construction; 366 MiB / 6740 files reclaimable on live data/. Removed the
  dead `data/intermediate/pirs` lake entry. (Execution left as a user one-liner — irreversible.)
- Earlier this session: W-RACE-1 (race_bridge_cv → LDO reliability weight), DP-1a (subprocess poll
  backoff), EFG-QT (classify_q_state wiring), T1.2 coverage honesty, Output Query P3a-c, AMC honesty,
  maternal-child truthfulness, bundle first-class consistency.

**VERIFIED → REDIRECTED (not built — measuring the prescription changed the answer):**
- **DP-2** (parallel normalize 3-4-way): `pipeline.py:292-296` documents a MEASURED rationale for the
  cap-of-2 (per-system decode already saturates BLAS/polars; >2 only stacks RAM). Needs a real
  national-normalize benchmark, NOT a speculative build against the measured rationale (CLAUDE.md §I).
- **DP-1b/DP-1c**: already-adequate (sidecar-cached / mtime-lru stable within a run).
- **W-REG-1** (declarative transform engine): correct-but-marginal — per-system transforms are
  irreducibly custom domain logic; the codebook is already registry-centralized. Real lever is REG-07.
- **Health-registry orphans** (icd_curated_groups etc.): stale claim — actually wired (3 refs).

**STAGED — two real, invasive builds for focused fresh-context execution (not rushed at depth):**
- **W-RACE-2-wire**: reconcile the registry's row-stochastic reclassification prior P(self|admin) with
  the estimator's column-stochastic emission P(admin|self), then wire posterior λ_{s,j} into the
  denominator path (replacing the fixed-C crosswalk). Moves point estimates → validate ground-truth
  recovery survives before adopting (CLAUDE.md §V).
- **REG-07**: unify the 3 divergent registry loader contracts (`validators.py:25-72`) → one typed
  loader. Decode-path rewrite needing byte-parity validation.

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

- **W-RACE-0 (reframe + honesty) — VERIFIED largely-done.** The guard already exists: `compile.py`
  detects a fixture/uncalibrated prior (`epistemic_status`), **blocks it from a dashboard-safe `full`
  run**, and stamps `race_bridge_prior_uncalibrated_assessment_only`; `source_reality_guard` whitelists
  that as legitimate epistemic metadata (not fake source data). The §1.8 validation **confirms this
  guard is essential** (uncalibrated-`C` rates erase signals) — so it is a justified safe default, not a
  lazy one. No new code needed; the substantive race work is W-RACE-1/2.
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

### 1.7 The sophisticated form: a covariate-dependent confusion `C(x)` (the general answer)

`C` is not one matrix — misclassification demonstrably varies by **source system** (physician-on-death-
certificate vs hospital-birth-clerk vs admission-clerk; already half-acknowledged via separate
`C_SIM/SIH/SINASC`), **time** (a 25-year drift as the self-declared-race push took hold — non-negotiable
for a full-temporal study), **age/cohort** (proxy-reporting + older-cohort *embranquecimento*), **region**
(the `C_r` of §1.4), and **reporting mode** (proxy vs self — the root of the discrepancy). The general
model makes `C` a **function of covariates** `x = (system, year, age, region, …)`.

**Parameterization that preserves the small-entity discipline** (the crux — this must NOT become a
K×K×covariate tensor). A **structured multinomial-logit emission anchored at identity**:
```
log[ P(admin=k | self=j, x) / P(admin=j | self=j, x) ] = α_{kj} + x·β_{kj}
```
with `α` the identity-dominant baseline and `β` **heavily regularized toward 0** (default = the
parsimonious shared-`C`; deviate only where the ecological signal demands). Keep the free off-diagonal
structure on the **dominant distortion axis** — the *ordinal* branca–parda–preta "whitening" gradient —
so covariate-dependence collapses to a few interpretable coefficients (a scalar whitening propensity
`ρ(x) = logit⁻¹(β₀ + β_sys + β_year·t + β_age·a + β_region)` driving *parda/preta→branca* mass, plus one
for the *parda↔preta* boundary), not a full tensor. This subsumes the current per-source fixed matrices
as the special case `β_year=β_age=β_region=0`, `β_sys` = fixed offsets.

**A new identifiability asset — cross-system consistency.** The systems observe (partly) the same
underlying self-declared population under *different* reporting. Holding the census composition fixed,
the **difference in admin-race distributions across systems identifies the system-specific `C`** — a
clean, quasi-instrumental restriction (the covariate that makes `C` vary is the covariate that identifies
the variation). Time drift is identified by temporal change against roughly-fixed population; age/region
lean harder on priors (age also drives rates, so the rate model must absorb age×race rate variation,
leaving the residual age effect on `C`). Where separation fails → honest wide posteriors.

This is the **general, robust answer**: `C` becomes a small, regularized, identity-anchored, ordinal-
structured regression object rather than a fixed matrix — sophisticated and covariate-responsive while
staying bounded. It folds into **W-RACE-2** (start with system+time effects, where identification is
cleanest; add age/region as the data support them).

### 1.8 Empirical validation of the formalization (planted-signal probe, 2026-07-09)

A disposable probe (CLAUDE.md §II) generated synthetic ecological data with a **known** whitening `C`
(true *preta* 32% mislabelled lighter, *parda* 25%→*branca*) **and** a **planted genuine race
differential** (*preta* mortality truly 1.5× *branca*), then tested whether the ecological-deconvolution
estimator (§1.3) recovers both without erasing the legitimate signal. Truth-init and neutral-init
converged to the same optimum (→ identified, not a lucky local min). Results:

| scenario | naive preta ratio | model | `C` recovery | verdict |
|---|---|---|---|---|
| A. high contextual variation, single system | **0.91** (erases/inverts) | **1.49** | exact (err 0.01) | recovers signal + `C` |
| B. low contextual variation | 0.90 | 1.30 (partial) | good | degrades — variation is the identifier |
| C. strong prior anchored at **identity-`C`** | 0.91 | **1.07 (erases)** | suppressed | over-strong/mis-anchored prior is harmful |
| D. two systems (shared rates, different `C`) | 0.91 | **1.49** | best (err 0.007) | cross-system is the strongest identifier |

- **Validated:** given adequate contextual variation (A) or cross-system data (D), the model recovers
  both `C` and the planted differential (1.49 vs 1.50), while the **naive admin/census estimate erases
  and inverts the signal** (0.91 — reads *preta* mortality as *lower* when it is genuinely 50% higher).
- **The user's central worry is real, quantified, and resolved** by the model given the identifying
  resources. Failure mode to avoid (scenario C): an over-strong prior anchored at identity-`C` actively
  erases the signal by mis-attributing whitening to the rate.
- **The current production bridge (fixed identity/synthetic `C`) is scenario C at infinite prior strength
  → not merely inert but actively harmful** where real misclassification exists.
- **W-RACE-2 design constraints (measured, not assumed):** (i) exploit cross-municipality contextual
  variation; (ii) prefer the multi-system joint model (cross-system `C` is the strongest identifier —
  validating §1.7); (iii) anchor the prior on a real literature `C`, **not identity**, with calibrated
  strength; (iv) report wide posteriors where contextual variation is low (scenario B).

Probe: `scratchpad/racebridge_validation_probe.py` — becomes W-RACE-2's proof-of-capability test at build.

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
