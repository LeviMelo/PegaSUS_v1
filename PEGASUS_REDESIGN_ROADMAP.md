# PegaSUS Redesign Roadmap — corrections + enhancements over the MSDs

Synthesis of the research phase (2026-07-07). Sources, all preserved at FULL granularity (this doc is a navigation + sequencing layer, not a replacement — every workstream cites finding IDs):
- `PEGASUS_MATH_CRITIQUE.md` — 165 workflow findings (16 themes) + `PEGASUS_MATH_CRITIQUE_STEELMAN.md` (20 adversarial verdicts, mostly UPHELD) + the lead's 23 direct LDO/population/q-tensor findings (`A1`–`A23`).
- `PEGASUS_MODULARIZATION_CRITIQUE.md` — `M1`–`M6` + workflow findings.
- `PEGASUS_GPU_OPTIMIZATION_CRITIQUE.md` — `G1`–`G6` + 50 workflow findings.
- `PEGASUS_FEATURE_PROPOSALS.md` — `P1`–`P5`.

**Governing principle (per the lead's directive):** the MSDs are FALLIBLE theory. Several top findings are cases where the *spec itself* is the trap (Poisson vs its own NB mandate; flat-GMRF vs SPDE; separability; uncalibrated constants). Where the roadmap diverges from an MSD, it says so and proposes the enhancement. **No live study runs until Tier-0 is closed** — the current pipeline would produce scientifically invalid results with false precision.

Priorities: **P0** = invalidates a scientific claim, blocks any valid study. **P1** = materially biases results. **P2** = performance/scale. **X** = cross-cutting enabler.

---

## TIER 0 — Foundational validity (BLOCKS the study)

### W1 · The `ObservationReliability` contract — the single highest-leverage redesign
*(discharges `A1`, `A13`, Theme-14: `LDO-MARG-05/11`, `LDO-LAG-NEFF-13`, `NEFF-KISH-WEIGHT-01`, `KISH-ABS-NEGATIVE-01`, `DENOMFRAG-SPEC-DIVERGE-01`, `CV-COUNT-VS-SAMPLE-01`, `TEMPROUGH-SPEC-01`, `LDO-NEFF-15`; `M6`; enables `P3`/`P4`)*
- **Problem:** the EFG/§3.12 layer computes a rich per-cell/per-field reliability state (`W`, effective-n, fragility, provenance risk) that the LDO **consumes almost none of** — the covariance is unweighted, the edge SE uses the raw cell count, and the n_eff/Kish/CV/fragility statistics measure the *wrong* quantity vs the spec names. No data contract binds producer to consumer.
- **Redesign:** a typed `ObservationReliability` object (per-cell weight tensor + per-field effective-n + fragility) that `assemble_ldo_tensor` PRODUCES and the covariance/whitening/SE estimators are REQUIRED to consume:
  1. weighted pairwise/whitened covariance (`Σ w·xᵀx / Σ w`); weighted ECDF for the rank margin.
  2. Fisher-z SE from the effective-n, not the raw cell count.
  3. fix each spec-divergent statistic: Kish over the *denominator* (not the value), fragility = §3.12.8 below-μ_min share, CV = the *sampling* CV of the estimate, roughness = second-difference curvature.
  4. contract test: a low-reliability cell must MEASURABLY change the fit (guards against re-decoupling).
- **Why first:** it is both the #1 validity fix and the archetypal module-boundary redesign, and it is the forcing function for the export layer (`P3`).
- **Progress (2026-07-07):**
  - ✅ **point 1 (covariance)** — `pairwise_correlation` and `whitened_lagged_correlation` now take an optional per-cell `weights` and compute reliability-weighted moments; `fit_lagged_links` / `fit_contemporaneous_precision` thread `field.W` by default (`use_reliability_weights=True`). `weights=None` ⇒ binary mask ⇒ **byte-identical** to the old unweighted fit (verified), so an all-observed run never drifts; only genuinely sub-unity cells change. Commit `59e8226`. Directly confirmed by inspection (`A1`) + independent agent (`LDO-B01`) before the fix.
  - ✅ **point 4 (contract test)** — `tests/synthetic/test_ldo_state_weight.py`: a correlation carried *entirely* by low-reliability cells is now suppressed once down-weighted (imputed demography can't be certified); byte-identity when absent; `field.W` provably reaches the fit. Closes the pre-existing "asserts plumbing not effect" gap.
  - ⏳ **remaining:** point 1's *weighted ECDF* for the rank margin (`randomized_pit_gaussianize` still builds the empirical CDF unweighted — a reconstructed cell still distorts ranks); **point 2** Fisher-z edge SE from effective-n not raw cell count (`A13`); **point 3** the spec-divergent statistics (Kish over the *denominator*, fragility = §3.12.8 below-μ_min share, sampling-CV, second-difference roughness).
  - **Ceiling logged:** the whitened path uses the *feature-scaling* form of reliability (attenuate a low-reliability cell's signal), not an exact heteroscedastic-GLS whitening (reliability folded into the metric `Q` + per-feature effective-n centering). Documented in the `whitened_lagged_correlation` docstring.

### W2 · Noise-model redesign — NB/ZINB margins + structured expected-count offset
*(discharges Theme-1: `LDO-MARG-01/06/10`, `LDO-LAG-GAUSSCOUNT-09`, `LDO-HSIC-12`, `LDO-CAUSAL-COUNT-DIST-13`, `LDO-DIS-SPARSE-07`, `POP-GAUSS-07`, `ASTD-POISSON-03`, `LDO-CERT-ENH-NB-15`; Theme-2: `LDO-MARG-08/02`, `MQ-EXPOSURE-VARIANCE-01`; Theme-11; `A2`/`A3`/`A5`)*
- **Problem (the MSD contradicts itself):** MSD-I §6.2 prescribes NB/quasi-Poisson/hurdle for overdispersed/zero-inflated counts; the code hardcodes an **equidispersed Poisson** margin with a **single global rate λ**, so (a) overdispersion is re-manufactured as spurious latent structure, and (b) the between-municipality rate variation — the actual epidemiological object — is pushed into the "surprise" scale as an artefactual national development axis.
- **Redesign:**
  1. per-variable **negative-binomial (or ZINB/hurdle) exposure-offset margin** with a data-estimated dispersion; family selected from the carrier registry, recorded in provenance.
  2. the offset is the **structured expected count** (small-area/BYM `E_i = exp(offset + spatial RE)`), not a scalar λ — so `Z` carries local anomaly (a variance-stabilized SMR residual), aligning with MSD-III §III.3 BYM.
  3. carry the **exposure's own reconstruction uncertainty** into the margin (errors-in-variables on the offset) — ties to W1 and `A16`.
  4. integrate out the randomized-PIT jitter (multiple realizations, MI-combine) — `LDO-MARG-03`, Theme-12.

### W3 · Spatial + temporal dependence — one graph, whiten both axes, exchangeable nulls
*(discharges Theme-6: `MORAN-DOUBLECORRECT-01`, `LDO-WHITEN-04`, `LDO-LAG-KAPPA-11`, `LDO-HSIC-02/11`, `MORAN-ORDERING-PROXY-01`, `ECOFALLACY-GUARD-01`; Theme-7: `LDO-TIME-03`, `LDO-AR1-14`, `LDO-LAG-STAT-02`, `LDO-LAG-OVERLAP-05`; `A6`/`A14`/`A20`; `M1`; `P1`)*
- **Problem:** spatial dependence is variously ignored (raw missing→0 impute, unnormalized Laplacian), double-corrected (Moran deflation on top of the GMRF), or measured with a **non-geographic 1-D Moran proxy**; the permutation null is non-exchangeable; **temporal autocorrelation is never whitened** (space is, time isn't, yet γ_time is applied); a single pooled AR(1)/lag structure is imposed across the epidemiological transition.
- **Redesign:**
  1. **one `geo.spatial` module** (P1 + M1): the registered `SpatialWeightGraph` + weight *views* (contiguity/distance-decay/kNN/gravity/flow) with **legality typing** (context-derived gravity/flow refused as a prior for variables sharing that provenance) + `moran_i(values, graph)` + `effective_n(values, graph)` — deletes the 6 Moran copies, fixes A20/A21.
  2. **row-standardized** Laplacian + principled missing-cell handling in the whitening (not impute-0); decide ONCE where spatial dependence lives (the GMRF) and remove the redundant Moran deflation.
  3. **whiten temporal dependence before the fit** (estimate Σ_time⁻¹ and apply it, not as telemetry); allow the lag/φ structure to vary across regime boundaries (change-point / time-varying), and correct effective-n for overlapping-window serial correlation.
  4. an **exchangeable** spatial null (conditional/toroidal or model-based) with a hard guard against the iid fallback.

### W4 · Calibrate the decision boundaries + dependence-aware multiplicity
*(discharges Theme-4: `LDO-LAMBDA-09`, `LDO-KAPPA-05`, `LDO-EXH-01/06`, `LDO-CAUSAL-NC-BINOM-04`, `LDO-CAUSAL-COLLIDER-THRESH-08`, `LDO-CERT-STABTHRESH-06`, `STATE-GATE-THRESHOLDS-01`, `LDO-DIS-MAGIC-06`, `POP-MIGBOUND-06`, `DIVERGENCE-EPSILON-01`, `LDO-MARG-13`, `LDO-RANDSVD-13`; Theme-5: `LDO-LAG-MULTTEST-12`, `LDO-EXH-07`, `LDO-HSIC-05/06`, `LDO-CAUSAL-MHT-14`, `LDO-CERT-MULTITEST-07`, `LDO-DIS-STAB-14`; `A8`/`A23`; `P2`)*
- **Problem:** ~15 hand-set constants (λ, κ, ρ, screen 0.15/0.30, stability 0.6, veto 0.5, q-state cliffs…) silently determine which findings survive, with no calibration; the p²·(K+1) hypothesis space has FDR fields *plumbed but unused*, and where FDR runs its PRDS assumption is violated by the spatial panel.
- **Redesign:** select λ by a stability-selection error-control criterion (Meinshausen-Bühlmann / eBIC, scaled `√(log p / n_eff)`); estimate κ, disease τ² (**P2 adaptive shrinkage**), ρ from data; replace hard cliffs with continuous scores + documented sensitivity sweeps; adopt a **dependence-aware FDR** (block/knockoff permutation respecting the spatial structure), actually wire the FDR fields, and re-derive the certifiability budget so the permutation floor doesn't make national edges structurally uncertifiable (`LDO-HSIC-06`).

### W5 · Identifiability honesty — stop reporting priors as inference
*(discharges Theme-3: `POP-IDENT-01`, `RACE-IDENTIFY-07`, `POP-GRAV-05`, `LDO-INCOH-07`, `LDO-LAG-CONF-01`, `LDO-DIS-IDENT-03`, `ALGEBRA-CLOSURE-DIV-01`; Theme-13: `LDO-MARG-04`, `LDO-HSIC-01`, `LDO-CERT-SLICE-SCOPE-04`, `POP-DEATHRATE-09`, `LDO-CAUSAL-NC-NEFF-12`; `A11`/`A16`/`A19`)*
- **Problem:** intercensal demography, the race bridge, and gross migration are **prior pushforwards reported as inference**; several statistics double-dip (in-sample HSIC residual scoring, full-sample ECDF then subsampled, circular δ = deaths/anchor).
- **Redesign:** for each unidentified estimand, either add an external identifying anchor OR demote it to explicitly **prior-dominated / descriptive with a sensitivity envelope** (quantify data-vs-prior contribution, `A16`); enforce genuine **cross-fitting** (fit margins/joint model on disjoint folds from the HSIC/certification data); break the circular denominator estimation; test the CPW incoherence condition empirically instead of the `min_factor_support=3` proxy.

---

## TIER 1 — Estimator + inference integrity

### W6 · Optimizer + covariance-matrix integrity
*(Theme-8: `LDO-PSD-01`, `LDO-NCORR-02`, `LDO-LAG-NEARCORR-07`, `LDO-GLASSO-11`; Theme-10: `LDO-ADMM-06`, `LDO-SMOOTH-12`, `POP-ITER-02`, `POP-CLOSURE-03`; `A7`/`A9`)*
- Higham nearest-correlation (or a shrinkage estimator that never produces an indefinite input) replacing eigen-clip+rescale; make the glasso→pinv fallback loud. ADMM stops on **primal AND dual** residuals with adaptive ρ + exact (or error-bounded) smoothness prox; remove the pop-solver `max_iter=12` cap (tolerance-based); enforce compositional closure **multiplicatively** (ILR/softmax), not Euclidean clip.

### W7 · Causal-claim discipline (fold into P5's spec pass)
*(Theme-15: `LDO-CAUSAL-RUNG-SEMANTICS-11`, `LDO-CAUSAL-FAITHFUL-07`, `LDO-CAUSAL-COLLIDER-CONFLICT-09`, `LDO-CAUSAL-LINGAM-POOL-05/LINEAR-06`, `LDO-CAUSAL-DID-ORPHAN-15`, `LDO-HSIC-15`; Theme-16: `LDO-CERT-FISHERZ-01`, `LDO-CERT-UNITS-02/LATENTBAND-03/SLICE-BIAS-05/HOLDOUT-NULL-10/CONJ-CONFIRM-11/BAND-ZERO-12`; `A11`)*
- Downgrade auto-applied Rung-2 labels lacking an adjustment set + exchangeability; **Meek propagation** for order-independent orientation; make LiNGAM residual/scoring consistent (both nonlinear) and autocorrelation-aware; correct ITS for autocorrelation/seasonality/multi-break; fix the Fisher-z SE for whitened partials + the unit-mismatched uncertainty quadrature + the latent-loading-as-partial-corr band. **Per P5, this becomes the causal installment's own spec** — seed it with these findings before building further.

### W9 · Separability escape hatch
*(Theme-9: `LDO-SEP-08`, `LDO-LAG-KRONSEP-10`, `POP-SEP-10`)*
- Add at least a low-rank **non-separable** component (or a spatially-varying-coefficient / spatiotemporal-interaction term) so lagged spatial epidemic spread and migration coupling are representable; validate separability empirically before assuming it. (The Kronecker operator stays as the scalable *backbone*; the interaction is the correction.)

---

## TIER 2 — Performance + scale (validity-neutral)

### W8 · GPU acceleration — the two high-value, validity-neutral wins
*(`G1`/`G2`; `LDO-GPU-01/02/03`, `LDO-SPEED-04`)*
- **Batched ADMM eigh on GPU (float64):** the ~19 fit_lagged_links/run × ~1000 eigh each is THE wall-clock hotspot and the ideal `torch.linalg.eigh` batch (`b×q×q`); keep S/L/U/R on-device; **eigh stays float64** (per §V.1 — f32 corrupts the logdet). ~5–15× on the LDO fit. Batch the 12 stability subsamples as one `(12,q,q)` call.
- **GPU HSIC residual scan:** the 48 GB dense-kernel cache → on-device RFF/Nyström feature maps (~5 GB f32) with the permutation null as batched matmul — moves the scan from "refuses for lack of 48 GB RAM" to "fits in 6 GB VRAM", ~10–30×, AND unifies the two HSIC implementations (discharges `O9`). *(Reframe: the earlier "minor gains" verdict judged single ops; this workload is repeated-small-dense + one huge scan — GPU-favorable.)*
- Prerequisite: **VRAM admission serialization** (`G6`) before turning on more CUDA paths.
- Deferred/measured: Kronecker einsum (`G3`), whitening sparse matvec (`G4`, precompute-once first per `A15`), population blocked-GPU (`G5`).

### W10 · Numerical-method + streaming redesigns (algorithmic, not hardware)
*(GPU-workflow `numerical-methods`/`population-solve` clusters — **re-run pending after the session-limit reset**; plus §V.4 streaming sufficient-statistics, the full float32 solve working-set)*

---

## CROSS-CUTTING — modularization + new subsystems

### W11 · Consolidation (enables everything above)
*(`M1`–`M6`; workflow `WF-01..06`: orphaned `run_attach_race_bridge`, bypassed `build_autonomous_efg` wrappers, fixture-EFG scaffolding in production, duplicated mode mappers, `_race_bridge_prior_artifact` name-collision, `ingest_sidra` duplication)*
- `geo.spatial` (W3/P1), `core.text`/`core.io` (M2), `measurement.composition` (M4), collapse the two reconstruction packages (M5), fix the geo→denominators layering (M3), delete the confirmed orphans, resolve the compile/pipeline seam duplications. **Depends-on note:** the reliability contract (W1) and the geo.spatial consolidation (W3) are the two structural anchors the rest hang off.

### W12 · New subsystems (post-W1)
- **P3 Export/Materialization layer** — co-designed with W1 so every exported cell carries real uncertainty + provenance (the forcing function).
- **P4 per-query denominator declaration** — same count under multiple denominators as first-class variables, each carrying its reconstruction uncertainty (W2).

---

## Sequencing summary
1. **W11 (consolidation anchors: reliability contract scaffold + geo.spatial)** → unblocks W1/W3 cleanly.
2. **W1, W2, W3, W5** (Tier-0 validity) — in parallel where independent; these gate any study.
3. **W4, W6, W7/P5-spec, W9** (Tier-1 integrity).
4. **W8** (GPU) — can proceed in parallel with Tier-1 (validity-neutral); do W1/W2/W3 first so we accelerate the *correct* computation, not a flawed one.
5. **W12** (export + denominator subsystems) after W1/W2.
6. **W10** — re-run the pending GPU/numerical clusters after the session-limit reset, then fold in.

**The one-line thesis:** the LDO is a sophisticated engine that currently (a) uses the wrong noise model, (b) discards the uncertainty it computes, (c) mishandles the spatial/temporal dependence it exists to study, and (d) gates findings on uncalibrated constants — so its outputs are not yet scientifically trustworthy. Tier-0 fixes those four; everything else is integrity, speed, and structure on top. **Do not run the flagship study until Tier-0 lands.**
