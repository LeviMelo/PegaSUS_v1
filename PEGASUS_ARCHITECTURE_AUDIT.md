# PegaSUS — Architectural / Mathematical / Computational-Compliance Audit

Whole-codebase audit vs PEGASUS_MSD_III.md (esp. Part V) + the operational plan. Multi-agent, adversarially verified (8 dimensions, 41 agents). Generated 2026-07-06.

**Findings:** 21 total, 20 survived verification, 1 refuted.

## Remediation progress (Workstream A — LDO core math integrity)

Center of gravity is the LDO math core, not cleanup. Landed so far, each test-gated (full suite 352 green):

- ✅ **CRITICAL — CPW S/L collapse** (06acb2f): incoherence gate (a latent factor must span ≥3 variables, else it is a direct edge) + S/L mutual exclusion. Clean planted-recovery across seeds; acceptance test `test_ldo_sparse_lowrank_recovery.py`. Also retired the lag-0 double-emit.
- ✅ **HIGH — residual-scan precision misalignment** (95cbfa4): `S[:p,:p]` raw slice → aligned `lag0_precision` built through the kept-feature map; no more misattribution / silent scan-disable when low-coverage vars drop.
- ✅ **HIGH — anticonservative iid HSIC null** (1847294): within-UF restricted-permutation structured null + descriptive-only gate when <5 spatial blocks; test proves it suppresses a spatially-confounded edge the iid null flags.
- ✅ **MEDIUM — unconverged ADMM certified edges** (0f0c324): §V.6 convergence gate downgrades to descriptive.
- ✅ **VAL-03 identifiability** (4cd7388): planted factor made identifiable (3-var); scorer no longer counts `latent_shared` as false-positive direct edges → perfect recovery.
- ✅ **Chokepoints** (separate): SIDRA `facts_to_frame` schema crash (e8d13b3); SIM normalizer O(1) age-maps + lazy projection (e4550d8).

**Workstream B/C (compute + inert milestones) — landed:**
- ✅ **HIGH — spatial GMRF whitening wired** (fit_lagged_links now whitens by Σ_space^{-1/2}=(κI+L_W)^{1/2}; κ/L_W no longer inert). Test: GMRF Moran's I → ~0; fit changes under whitening.
- ✅ **HIGH — causal LiNGAM orientation wired** into run_ldo (CAUSAL-01 was built+tested but never called); orients contemporaneous edges on the RAW non-Gaussian values. Test: non-Gaussian A→B oriented source=A; Gaussian left undirected.
- ✅ **MEDIUM — §V.3 randomized SVD** for the low-rank factors (auto at scale, exact certifies approximate) + a noise-floor readout threshold that also removed spurious latent_shared factor pairs.
- ✅ **#2 Kronecker (§V.2) reframed**: the LDO forms only the bounded (p·(K+1))² variable precision from cell *samples* — it never builds the (p·S·T)² object §V.2 targets, so that "biggest lever" does not apply to the LDO's sample-based design (space→GMRF whitening, time→lags). The R-step logdet prox stays exact (full-spectrum).
- ✅ **#11/#13 disease-axis truth gaps** reconciled (adaptive-ℓ1 scope stated precisely; DIS-04b Laplacian-quadratic scoped; §5.1 canonical path documented).

**Workstream D (dead code) — CORRECTION: the audit's dead-code list was unreliable.** Verification before deletion found **3 of its "clean deletes" are governed PANEL-01 KEEPs** with explicit `# do not reap (§1c)` banners — `she/sih_costs.py`, `she/cnes_capacity.py`, `sources/sidra/projection.py` — and its importer detection **missed the `from pkg import module` form** (`output/maternal_child_compile_attach.py` has a live contract test). Only `dashboard/hsic_readonly.py` (zero importers, no banner) + the empty `src/pegasus/studies/` dir were safely removed. The remaining candidates (maternal-child cluster, `output/sidra_denominator_anchor`, `workflows/construct/build_efg`, `workflows/acquire/ingest_sidra` [claimed-live], `pegasus.pirs` [pins the core inference-baseline guardrail]) are **tested** and require per-module domain judgment — **not** autonomous bulk deletion. Deferred deliberately.

**Genuinely remaining:** STORE-02 lazy `scan_parquet` views (a data-plane refactor, orthogonal to the LDO); the tested dead-code candidates above (need confirmation); DIS-04b Laplacian-quadratic disease prior (scoped enhancement).

## Reframe (what verification established)

- The annexed Pylance dump is **stale/noise**: `pirs/ldo`, `she/reconstruction`, `sidra/population_cube` "zombie trees" are already deleted; the B-section type/None items are annotation noise, not runtime defects; `ldo/precision.py:101` is correct.

- The real damage is in the **LDO mathematical core**: the estimator emits `Hypotheses.parquet` with a systematically miscalibrated edge classifier, unwired spatial-confounder control, and a residual detector that is both misaligned and mis-nulled. MSD-III Part V is ~30-40% wired — the building blocks (spatial GMRF whitening, ST-DFM GPU solver, Nystrom/RFF, separable backbone, causal LiNGAM) exist as modules but are not connected to the live estimator.

## Findings (ranked)

| # | Sev | Category | Verdict | Title |
|---|-----|----------|---------|-------|
| 1 | critical | math_correctness | CONFIRMED | CPW sparse+low-rank decomposition has no stable λ2 operating point — S/L separation collapses either |
| 2 | high | arch_divergence | CONFIRMED | pegasus.causal (CAUSAL-01 LiNGAM orientation + collider detection) is built and test-covered but nev |
| 3 | high | math_correctness | CONFIRMED | LDO residual scan feeds a misaligned/undersized precision block (S[:p,:p] over the kept-feature subs |
| 4 | high | arch_divergence | CONFIRMED | Spatial GMRF prior Σ_space⁻¹=κI+L_W is absent from the live estimator (whitening code is orphaned; k |
| 5 | high | math_correctness | CONFIRMED | Residual HSIC scan uses an iid permutation null on spatiotemporally autocorrelated residuals (antico |
| 6 | high | compute_noncompliance | CONFIRMED | Kronecker/spatial separability (MSD-III §V.2) absent from the live LDO path; separable backbone orph |
| 7 | medium | dead_code | CONFIRMED | Dead maternal-child / SINASC output cluster — three unimported output modules pulling one she module |
| 8 | medium | arch_divergence | CONFIRMED | disease/variable_grammar.py (§5.1 disease variable generator) is bypassed by investigate.py's own me |
| 9 | medium | dead_code | CONFIRMED | Three off-dispatch workflow entrypoints (2 dead, 1 orphaned-but-claimed-live) plus a build_efg modul |
| 10 | medium | math_correctness | PLAUSIBLE | A lag-0 pair can emit both a `contemporaneous` (from S) and a `latent_shared` (from L) LinkRecord —  |
| 11 | medium | numerical_stability | CONFIRMED | Unconverged ADMM low-rank solves still emit certified 'selected' edges with no convergence gate (vio |
| 12 | medium | perf_chokepoint | CONFIRMED | §V.3 randomized NLA (randomized SVD for low-rank factors, stochastic Lanczos for logdet) is entirely |
| 13 | low | dead_code | CONFIRMED | Stale pylance dump: the three named 'zombie trees' are already deleted — do not act on the A-section |
| 14 | low | dead_code | PLAUSIBLE | pegasus.pirs package (354 LOC) is test-pinned dead code; its HSIC functionality lives in ldo/hsic.py |
| 15 | low | dead_code | PLAUSIBLE | projection.py is a deletable superseded duplicate; the other 3 orphans are intentionally-parked PANE |
| 16 | low | dead_code | PLAUSIBLE | Test-only/unwired helper modules — reachability flag (with two evidence errors); age_standardization |
| 17 | low | dead_code | CONFIRMED | Empty package directory src/pegasus/studies with no __init__.py |
| 18 | low | type_nullsafety | CONFIRMED | Pylance B-section 'type/None-safety' items in live code are annotation noise, not runtime defects |
| 19 | low | arch_divergence | CONFIRMED | Disease L_D 'Laplacian-regularized' prior is implemented as an adaptive ℓ1 penalty, not the §III.4(5 |
| 20 | low | dead_code | CONFIRMED | precision.py:101 '4-tuple/2-tuple' pylance finding is STALE — current graphical_lasso unpack is corr |

## Detailed evidence + fix (critical & high)

### [CRITICAL] CPW sparse+low-rank decomposition has no stable λ2 operating point — S/L separation collapses either way
- **Files:** src/pegasus/ldo/lowrank.py:59; src/pegasus/ldo/lowrank.py:93; src/pegasus/workflows/investigate.py:151
- **Evidence:** The core estimator fit_sparse_plus_lowrank is the Chandrasekaran–Parrilo–Willsky decomposition Ω=S−L (§III.4.2). Numerical test with a ground-truth mix of one shared factor (vars 0,1,2 co-moving) plus one genuine direct edge (3–4): at the run_ldo default lambda2=0.1, `S` comes back PURELY DIAGONAL (S off-diag max = 0.000), `direct_edges` is EMPTY, and the true direct 3–4 edge is misclassified as `latent_shared` (0.71) — L absorbed all off-diagonal structure. Sweeping λ2: at λ2≤0.05 L absorbs everything (0 direct edges); at λ2≥0.2 L collapses to rank 0 (zero factors), so the shared epidemic-wave driver — the entire reason the low-rank layer exists per §III.4.2 ('prevents everything-riding-a-wave from being mistaken for a web of direct links') — is never separated and vars 0,1,2 become a dense direct-edge clique. NO single λ2 recovers both structures. The investigate.py docstring at :151 documents the same discovery from the other side ('the raw run_ldo default of 0.1 lets the low-rank layer over-absorb... collapsing every relationship into latent_shared') and 'fixes' it with lambda2=1.0 — which is the rank-0 degenerate regime.
- **Fix:** Do not ship a single fixed λ2. The CPW estimator requires λ1/λ2 chosen on the identifiability curve (Chandrasekaran incoherence condition). Add a stability-path / cross-validated selection of (λ1, λ2) that certifies a non-degenerate rank(L)>0 AND non-diagonal S on held-out subsamples, or replace the hand-tuned scalar-ADMM with a validated LVGLASSO (e.g. regpath with the Ma-Xue-Zou solver) and gate the run with a synthetic sparse+low-rank recovery acceptance test that asserts the direct edge lands in S and the shared factor in L simultaneously.
- **Leverage:** This is the analytical heart of the whole system — every LinkRecord's edge_type (contemporaneous/lagged_directed vs latent_shared) is decided by this split. At the current defaults the classification is systematically wrong: genuine direct disease links are reported as latent confounding, or genuine shared epidemic drivers are reported as dense webs of direct links. Every downstream causal-escalation and hypothesis output inherits the error.  **Effort:** L

### [HIGH] pegasus.causal (CAUSAL-01 LiNGAM orientation + collider detection) is built and test-covered but never wired into the LDO/investigate output path — a 'completed' milestone that is inert at runtime
- **Files:** src/pegasus/causal/orient.py:1; src/pegasus/causal/quasi.py:1; src/pegasus/workflows/investigate.py:1
- **Evidence:** grep for external importers of pegasus.causal returns ZERO runtime importers; only tests/synthetic/test_causal_orient.py and test_causal_quasi.py import it. workflows/investigate.py (the LDO driver) contains no reference to 'causal', 'orient', 'collider', or 'lingam'. Task #10 is marked completed ('CAUSAL-01: rung-1 non-Gaussian orientation (LiNGAM) + collider detection') but the produced module is orphaned — the LDO emits undirected link_records and the orientation stage is never invoked. 262 LOC (orient 165 + quasi 97).
- **Fix:** Either wire causal.orient/quasi into the investigate/LDO output stage (orient the hypotheses before write_hypotheses) so CAUSAL-01 is actually live, or, if orientation is deliberately deferred, mark it as such and keep the modules — but recognize that as-is the 'completed' status is misleading and the code is untested against real runs.
- **Leverage:** Either delivers the CAUSAL-01 capability the task claims (edge orientation on real hypotheses) or reclaims 262 LOC; resolves a plan-vs-code tension where a 'done' milestone is inert.  **Effort:** M

### [HIGH] LDO residual scan feeds a misaligned/undersized precision block (S[:p,:p] over the kept-feature subset)
- **Files:** src/pegasus/ldo/orchestrator.py:151; src/pegasus/ldo/residual_scan.py:62; src/pegasus/ldo/lags.py:90; src/pegasus/ldo/covariance.py:65
- **Evidence:** orchestrator.py:151 `lag0 = lagged.fit.S[:p, :p]` then :154-155 passes it to `scan_residual_nonlinear_edges(gf, lag0, ...)`. But `lagged.fit.S` is produced by `fit_sparse_plus_lowrank(pw.correlation, ...)` (lags.py:90-93) where `pw.correlation` is (q×q) with q=len(kept), and `kept = [i for i in range(F) if coverage[i] >= min_coverage and stds[i] > 1e-9]` (covariance.py:65) DROPS low-coverage features. lags.py itself maps positions correctly via `pos = {f: a for a, f in enumerate(kept)}` (lags.py:85, entry() at :99-101), but run_ldo bypasses that and slices S[:p,:p] directly, assuming the first p kept rows are the p lag-0 base variables in order. residual_scan.py:62 then does `precision @ Zc` with Zc shape (p, n) and attributes results to `field.variables[i]` (residual_scan.py:87-88). Reproduced with p=4, one lag-0 var dropped: `pw.kept = (0,2,3,4,6,7,8,10,11)`, so `kept[:4]=(0,2,3,4)` -> slot 3 holds feature 4 = (lag1,var0), which residual_scan labels field.variables[3]. When enough features drop that q<p, `S[:p,:p]` is q×q and `precision @ Zc` = (q,q)@(p,n) raises ValueError, caught by the additive try/except at orchestrator.py:157 -> residual scan silently disabled (residual_scan_error recorded but run proceeds).
- **Fix:** Do not slice S[:p,:p]. Build a p×p lag-0 precision aligned to gf.variables by mapping through `lagged.fit`'s kept index: allocate an identity-diagonal p×p, then for each pair (i,j) of base variables look up their lag-0 feature positions via the same `pos`/`kept` map lags.py uses (feature index = i for lag 0), and fill from S only where both survived (default 0/identity where a lag-0 variable was dropped). Alternatively expose a `lag0_precision` on LaggedFit computed inside lags.py where the kept-map is in scope. Add a synthetic test that runs run_ldo with run_residual_scan=True on a field where a lag-0 variable is dropped for low coverage and asserts the residual edges carry the correct variable names (and that q<p does not silently disable the scan).
- **Leverage:** run_investigate (workflows/investigate.py:179) calls run_ldo without run_residual_scan, so it defaults True (orchestrator.py:95) — the buggy path runs in every real investigate on multi-source panels, precisely where sparse coverage drops lag-0 features. Every existing LDO test passes run_residual_scan=False, so the entire residual-HSIC layer (MSD-III §III.6, the 'named-terms + residual detector' architecture) is untested and currently either silently inert or emitting misattributed nonlinear_residual links into Hypotheses.parquet.  **Effort:** M

### [HIGH] Spatial GMRF prior Σ_space⁻¹=κI+L_W is absent from the live estimator (whitening code is orphaned; kappa is dead)
- **Files:** src/pegasus/ldo/precision.py:44; src/pegasus/ldo/precision.py:65; src/pegasus/ldo/lags.py:63; src/pegasus/ldo/orchestrator.py:129
- **Evidence:** §III.4 makes the spatial GMRF one of the five structural assumptions: (3) separable Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹ and (5) prior-regularized with L_W; §V.2 lists the spatial GMRF (~6 neighbors/row) as a primary structural lever. precision.py implements build_spatial_precision (κI+L_W from structural_cod6_adjacency) and _whiten_spatial to remove spatial autocorrelation before the graphical lasso — but grep shows fit_contemporaneous_precision / _whiten_spatial / build_spatial_precision have ZERO runtime callers (only the __init__ re-export). The live path is run_ldo → fit_lagged_links → fit_sparse_plus_lowrank, none of which whiten by space. `kappa` is threaded run_ldo(:87)→fit_kwargs(:129)→fit_lagged_links(:63) but its body never references kappa. So Ω_var is estimated from raw pairwise correlations that still contain spatial autocorrelation — exactly what precision.py's own docstring says whitening must remove ('a link in Ω_var is dependence net of space'). Every apparent variable link is inflated by shared spatial structure.
- **Fix:** Either wire the GMRF whitening into fit_lagged_links (whiten each lag slice by Σ_space^{-1/2} before assembling the correlation) so kappa/L_W actually act, or, if the intent is to defer, remove the dead kappa parameter and the orphaned precision.py functions and record the spatial-prior gap explicitly in the plan rather than presenting a threaded-but-inert knob.
- **Leverage:** Without spatial whitening the LDO cannot distinguish 'two diseases co-vary' from 'two diseases both cluster in the Northeast'; at national municipal scale spatial autocorrelation is the dominant confounder, so essentially every contemporaneous edge is suspect.  **Effort:** L

### [HIGH] Residual HSIC scan uses an iid permutation null on spatiotemporally autocorrelated residuals (anticonservative)
- **Files:** src/pegasus/ldo/residual_scan.py:68; src/pegasus/ldo/residual_scan.py:91
- **Evidence:** scan_residual_nonlinear_edges calls numpy_kernel_hsic_permutation_test WITHOUT permutation_indices, so the null is an iid within-sample permutation (hsic.py:299 default), and it stamps null_strategy='iid_permutation' (residual_scan.py:91). The residuals are the joint-model conditional residuals over (space×time) cells, which retain spatial and temporal autocorrelation. §III.6 mandates HSIC with panel-appropriate handling and the entire nulls.py registry exists precisely for this (spatial_block_cyclic_time_shift, season_preserving_moving_block_circular_shift) — an iid null breaks exchangeability and inflates HSIC false positives on autocorrelated fields. run_hsic_scan (the structured-null path via generate_null_indices) exists and is unused by the residual layer.
- **Fix:** Feed structured permutation_indices (from generate_null_indices with the panel's spatial_blocks/panel_shape) into the residual-scan HSIC, selecting the regime from the panel type (nulls.select_null_regime). Do not emit nonlinear_residual edges under an iid null on a spatiotemporal panel.
- **Leverage:** The residual scan is the §III.6 nonlinear detector; with an anticonservative null its BH-FDR-'significant' nonlinear_residual edges (certified 'selected') are unreliable, and they are promoted as discoveries.  **Effort:** M

### [HIGH] Kronecker/spatial separability (MSD-III §V.2) absent from the live LDO path; separable backbone orphaned
- **Files:** src/pegasus/ldo/lowrank.py:120
- **Evidence:** dense eigh on (p(K+1))^2; grep kron zero hits; spatial factor orphaned.
- **Fix:** factor and Kronecker matvec/solve/logdet.
- **Leverage:** national-scale lever.  **Effort:** XL

## Remediation roadmap (synthesis)

I have a complete, evidence-backed picture from the 20 verified findings and dimension summaries. Let me synthesize the final assessment. No further code reads are needed — the verification passes already anchored every claim to file:line and adjudicated severity/verdict. Writing the synthesis directly.

---

# PegaSUS — Final Compliance Assessment & Remediation Roadmap

## Framing: what the verification actually established

Two things reframe the whole audit. First, **the "zombie tree" premise in the brief is stale** — `pirs/ldo`, `she/reconstruction`, `sidra/population_cube` are already deleted from disk; the live homes (`ldo/`, `denominators/{population,reconstruction}/`) import cleanly. So the roadmap's center of gravity is *not* cleanup — it's the **mathematical heart of the LDO**, which is where the real damage is.

Second, the Pylance dump is **suspect evidence, not authority**: every group-B "runtime hazard" was adjudicated as annotation/stub noise or already-guarded (`precision.py:101`, `spine.py:303`, `orchestrator.py:399`, `graph.py:127`). The one real reachable defect the sweep *missed* (LDO residual-scan index misalignment) is the pattern to internalize — **green suites and linters both mask the live-path math bugs**.

---

## Part 1 — Ranked Remediation Roadmap (by severity × leverage)

### WORKSTREAM A — LDO core math integrity `[CRITICAL, highest leverage]`

This is the analytical heart (MSD-III §III.4). Every `LinkRecord.edge_type` the system exists to produce is decided here, and it is **systematically wrong at the production defaults**.

| Finding | Sev | What's broken |
|---|---|---|
| *"CPW sparse+low-rank decomposition has no stable λ2 operating point"* | **critical** | No single λ2 recovers both a direct edge (in S) and a shared factor (in L). Default λ2=0.1 → S diagonal, genuine direct links mislabeled `latent_shared`; production λ2=1.0 → rank(L)=0, shared epidemic drivers become dense direct-edge cliques. Reproduced. |
| *"Spatial GMRF prior κI+L_W absent from the live estimator"* | high | The §III.4(3/5) spatial whitening exists in `precision.py` but has **zero runtime callers**; `kappa` is threaded through `run_ldo→fit_lagged_links` and never referenced. Ω_var is estimated from raw correlations still carrying spatial autocorrelation — the dominant confounder at municipal scale. |
| *"Residual HSIC scan uses an iid permutation null on autocorrelated residuals"* | high | `scan_residual_nonlinear_edges` calls the HSIC test without structured `permutation_indices`, stamping `null_strategy='iid_permutation'`. Anticonservative; the `nulls.py` registry (`spatial_block_cyclic_time_shift`, etc.) and `run_hsic_scan` structured path exist and are bypassed. Certified `selected` nonlinear edges are unreliable. |
| *"LDO residual scan feeds a misaligned/undersized precision block"* | high | `orchestrator.py:151 lag0 = fit.S[:p,:p]` slices the *kept-feature* q×q matrix as if the first p rows were the p lag-0 base variables in order. When any lag-0 var is dropped for low coverage (the common DATASUS case), edges are misattributed; when q<p it crashes into a swallowed try/except and silently disables the scan. Runs by default in every `run_investigate`; exercised by **zero** tests. |
| *"A lag-0 pair can emit both `contemporaneous` and `latent_shared`"* | medium | No mutual exclusion between the S-loop (`edges.py:169-182`) and L-loop (`edges.py:183-194`); one pair emits two contradictory records. (Note: the finding's cited `lowrank.py` `direct_edges` path is dead — fix at `edges.py`.) |
| *"Unconverged ADMM solves still emit certified 'selected' edges"* | medium | `converged=False` is recorded in diagnostics but never gates promotion. Violates §V.6(2) "never present an approximated number as exact" and the §III.8 abort. |
| *"Disease L_D 'Laplacian-regularized' is an adaptive ℓ1, not the §III.4(5) quadratic"* | low | Doc-vs-code truth gap; honest docstring, but DIS-04/task naming overstates. Cannot borrow strength across the hierarchy. |

**Fix approach:** (1) Replace the fixed scalar λ2 with **identifiability-curve selection** of (λ1, λ2) — a stability/CV path that certifies rank(L)>0 *and* non-diagonal S on held-out subsamples, gated by a synthetic sparse+low-rank recovery acceptance test that asserts the planted direct edge lands in S while the planted factor lands in L. This is the keystone; the other five are cheaper once the split is trustworthy. (2) Wire `_whiten_spatial`/`build_spatial_precision` into `fit_lagged_links` (whiten each lag slice by Σ_space^{-1/2}) so κ/L_W actually act — or delete the dead knob and record the gap honestly. (3) Feed structured `permutation_indices` from `generate_null_indices` into the residual HSIC and refuse to emit nonlinear edges under an iid null on a spatiotemporal panel. (4) Build the p×p lag-0 precision aligned to `gf.variables` via the same `pos`/`kept` map `lags.py` already uses (expose `lag0_precision` on `LaggedFit`). (5) Partition each pair to one record type. (6) Gate promotion on `fit.converged` (widen uncertainty → `descriptive`, mirror the `low_n_eff` gate).

**Effort:** L (keystone λ2) + M×3 + S×1. **Risk:** medium — touches the promotion path every hypothesis flows through; strongly mitigated by the acceptance tests these fixes force into existence. **Unblocks:** trustworthy `Hypotheses.parquet`; every downstream causal escalation; the entire residual-HSIC layer (currently untested and either inert or misattributing).

---

### WORKSTREAM B — §V "Computation & Scale" adaptations `[HIGH leverage, structural]`

The MSD-III Part V contract is the "deep-optimization" promise. On the live path it is **largely aspirational-over-orphaned** — the §V.1(c) anti-pattern the plan itself warns against.

| Finding | Sev | Gap |
|---|---|---|
| *"Kronecker/spatial separability (§V.2) absent; separable backbone orphaned"* | high | Live LDO densifies to (p·(K+1))² and runs dense `eigh`; `grep kron` = zero. The separable backbone (`precision.py`) is unwired. Concrete cost *today*: `_adaptive_lag_order` caps K=1 when p>200 because the dense cube is unaffordable at context scale — the missing lever already sacrifices lag fidelity. Plan §132/499 claims "never densified"; code densifies. |
| *"§V.3 randomized NLA entirely unimplemented"* | medium | Every ADMM eigen-op is full dense `eigh` at O(p³)/iter × up to 500 iters × n_subsamples refits. `randomized_svd`/`svd_lowrank`/stochastic Lanczos: zero hits. §V.3 names this exact hotspot and promises bounded-error randomized NLA. |
| *(cross-cut) ST-DFM GPU solver disconnected from the LDO low-rank layer* | — | `she/stdfm/torch_solver.py::solve_stdfm` is called only by the standalone ST-DFM pipeline, **never by `ldo/lowrank.py`** — the GPU-capable solver the plan designated as the low-rank layer is disconnected. (POP-02 GPU "justified" but the designated kernel is unused — the storage/compute plan-tension.) |

**Fix approach:** Sequenced, not big-bang. **First cheap win:** wire the *existing* spatial whitening (Workstream A#2) before any Kronecker rewrite — it reclaims correctness without new NLA. **Then** replace the L-step PSD projection and factor readout with randomized/truncated SVD at expected factor rank r (HMT power-iteration bound), keeping the R-step logdet prox on the full spectrum via stochastic Lanczos (the auditor's correction: randomized SVD is inapplicable to the logdet prox). **Finally** the full separable operator (Kronecker matvec/solve/logdet) and connect `solve_stdfm` as the low-rank layer, gated by the Adaptive Precision Controller (§V.5) escalating to exact only when an edge is decision-relevant.

**Effort:** XL (full separability) → L (randNLA) → M (whitening wire). **Risk:** high for the full operator, low for the whitening wire. **Unblocks:** national municipal scale without the K=1 lag cap; the §V.6 bounded-error guarantee; honest §V.2/V.3 compliance instead of orphaned backbone.

---

### WORKSTREAM C — Plan-vs-code truth reconciliation (inert "done" milestones) `[MEDIUM, high truth-value]`

Milestones marked *completed* whose code never runs. A green suite masks this because every unit is reachable **only by tests**.

| Finding | Sev | Reality |
|---|---|---|
| *"pegasus.causal (CAUSAL-01 LiNGAM) built, test-covered, never wired into investigate"* | high | Zero runtime importers; `investigate.py` never references orient/collider/lingam. The plan (§IV, CAUSAL-01) mandates it as a live post-LDO stage. 262 LOC inert. |
| *"disease/variable_grammar.py (§5.1 generator) bypassed by investigate's own path"* | medium | `investigate.py:88` docstring: "no regeneration through the disease variable grammar." Two divergent §5.1 impls that can silently disagree. |

**Fix approach:** For CAUSAL-01, **wire `causal.orient/quasi` into the output stage** (orient hypotheses before `write_hypotheses`) — delivering claimed science — or explicitly mark deferred. For §5.1, pick one source of truth: delete `variable_grammar.py` (accept the V_fields-reading path as canonical) or make `investigate.py` delegate to it; reconcile the DIS task note. **Effort:** M each. **Risk:** low-medium (CAUSAL wiring adds an untested-on-real-runs stage — gate with one proof test). **Unblocks:** honest milestone status; either real edge orientation or reclaimed LOC.

---

### WORKSTREAM D — Dead-code de-engorgement (§X.2) `[LOW severity, real hygiene]`

~2,970 LOC of genuine dead code *inside live directories*, masked by test-pinning. **Nuance the verification forced:** not all "orphans" are deletable — several are governed KEEPs.

**Clean deletes (zero refs at all, or superseded duplicates):**
- `she/sih_costs.py`, `dashboard/hsic_readonly.py`, the `she/cnes_capacity.py` *summarizer* (NOT `registries/cnes_capacity.py`, which is heavily live — the finding conflated them)
- `sources/sidra/projection.py` (superseded by `sidra_projection.py`; kill the name collision) + `stitching.py`
- `workflows/construct/build_efg.py` (module-vs-function collision with `efg.dag.build_efg`) + `workflows/acquire/ingest_sidra.py`
- The dead maternal-child cluster: `output/maternal_child_compile_attach`, `output/sinasc_efg_bundle`, `she/maternal_child`, `output/sidra_denominator_anchor` (~583 LOC; also *contradict* compile.py's stated "manual attachers forbidden" invariant)
- Empty untracked `src/pegasus/studies/` dir
- `pegasus.pirs` package (354 LOC) — **but** its §226 Nyström/RFF is only diagnostic shells; the real approximation lives in `ldo/hsic.py`, so this is pure cleanup, **not** the compliance gap the original finding claimed (that half was REFUTED in verification).

**Do NOT delete (governed KEEPs / conditional):**
- `she/high_dimensional.py`, `sources/sidra/stitching.py` carry "PANEL-01 pre-build, unwired — do not reap" banners per REFACTOR_MASTER_PLAN §1c
- `efg/empirical_compression.py` — un-wired EFG-CMP-01 spec feature, not dead
- `measurement/age_standardization.py` — **runtime-wired** into the C25 study (win-condition); the finding's "test-only" evidence was wrong
- `curated_cause_report.py` — the operational plan's current C25 describe-only reporter; re-wire, don't delete

**Fix approach:** two commits — clean deletes + pin-tests first; then triage the conditionals against intended-but-unwired capability. **Effort:** S-M. **Risk:** low (verified zero runtime reach). **Unblocks:** accurate importer-greps and package scans; stops the strangler-fig half-migrations from misleading future audits.

---

### WORKSTREAM E — Lint debt clearance `[TRIVIAL, do last]`

The Pylance group-B items on live paths: single annotation-cleanup pass (`scipy.sparse` typing, stale `sim_deaths` line numbers, `graphical_lasso` stub). **Not remediation** — do not spend audit effort treating these as defects, and do not touch the copies inside deleted trees. **Effort:** S. **Risk:** none.

---

## Part 2 — The 5 highest-leverage items

Ranked by impact on **mathematical integrity / scale-compliance / correctness**:

1. **CPW λ2 has no stable operating point** *(critical)* — the keystone. Fixing the S/L split is prerequisite to every edge classification being meaningful. Nothing downstream is trustworthy until this is on the identifiability curve with an acceptance test.
2. **Spatial GMRF (κI+L_W) absent from the live estimator** *(high)* — at national municipal scale, spatial autocorrelation is *the* dominant confounder; without whitening the LDO cannot distinguish "two diseases co-vary" from "both cluster in the Northeast." Also the cheapest §V correctness win (code exists, just unwired).
3. **Kronecker/spatial separability (§V.2) absent** *(high)* — the national-scale lever; its absence *already* forces the K=1 lag cap at p>200. Unlocks scale the plan promises.
4. **LDO residual-scan precision misalignment** *(high)* — runs by default in every real investigate, on exactly the sparse panels where it breaks, with zero test coverage. Either misattributes nonlinear edges or silently disables the entire §III.6 detector.
5. **Residual HSIC iid-null anticonservatism** *(high)* — the structured-null machinery exists one function over; certified `selected` nonlinear discoveries are currently statistically unreliable on autocorrelated fields.

(#4 and #5 compound: the residual layer is simultaneously fed a misaligned precision *and* tested against the wrong null — it is the least-trustworthy output the system emits, yet promoted as discoveries.)

---

## Part 3 — Honest bottom line

**Distance from MSD-III Part V compliance: substantial and structural.** Part V is the "deep-optimization contract," and on the *live* path it is largely **aspirational-over-orphaned**:

- **§V.2 separable Kronecker backbone** — the code exists (`precision.py`) but has zero runtime callers; the live LDO densifies to (p·(K+1))² and runs dense `eigh`. The plan's claim (§132/499) that precision is "never densified" is **false against the code** — the code is right about what it does, the plan is aspirational. This already costs lag fidelity (forced K=1 at scale).
- **§V.3 randomized NLA** — entirely unimplemented; every eigen-op is full dense at O(p³)/iter.
- **§V.5 Adaptive Precision Controller / §V.6 bounded-error propagation** — the convergence gate is absent (unconverged solves emit certified edges), directly violating "never present an approximated number as exact."
- **The designated GPU low-rank solver (`solve_stdfm`) is disconnected** from `ldo/lowrank.py` — POP-02's "GPU justified" holds, but the kernel it justifies is unused by the LDO.

The pattern is consistent: **the structural building blocks (spatial precision, ST-DFM GPU solver, Nyström/RFF, stability paths) exist as modules but are not wired into the estimator that Part V governs.** So the codebase is closer to Part V *in inventory* than *in execution* — perhaps 30-40% wired.

**The deepest unimplemented computational adaptations, in order:**
1. **A trustworthy sparse+low-rank split** — not a scale adaptation per se, but the correctness foundation Part V's whole precision machinery presupposes. Currently broken at defaults.
2. **Kronecker-separable operators** (matvec/solve/logdet) so space/time/variable never densify — the single largest scale lever.
3. **Randomized NLA** (HMT randomized SVD for factors, stochastic Lanczos for logdet) — the bounded-error inner-loop the ADMM hotspot demands.
4. **Structured-null statistical validity** in the residual layer, and **convergence/error propagation gating** so approximated results are never certified as exact.

**Framing for the maintainer:** the highest-value work is **not** the ~2,970 LOC of dead code (that is low-severity hygiene, and the verification showed several "orphans" are governed KEEPs or the win-condition C25 path). The value is concentrated in **Workstream A** — the LDO produces its central artifact (`Hypotheses.parquet`) with a systematically miscalibrated edge classifier, an unwired spatial confounder control, and a residual detector that is both misaligned and mis-nulled. Fix the S/L split and wire the spatial whitening first; those two alone move the system from "emits confident but unreliable edges" to "emits defensible ones," and the spatial wire doubles as the cheapest Part-V correctness down payment before the XL Kronecker build.