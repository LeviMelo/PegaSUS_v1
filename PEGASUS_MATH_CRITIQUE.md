# PegaSUS Mathematical / Theoretical Critique Ledger

Research phase (2026-07-07): a deep critique of the MSD theory AND the implementation, favoring **direct code+math inspection** over tests, aimed at (1) architectural/intent completeness + *enhancement over* the MSDs (which are treated as fallible theory), (2) modularization, (3) optimization/GPU. **Full granularity is preserved — no finding is compacted away.** Findings from the 12-agent `math-theory-critique` workflow (`wf_99a2944f-2d7`) are appended verbatim below the lead's own direct-inspection findings.

Severity: CRITICAL (invalidates a scientific claim) · HIGH (materially biases results) · MEDIUM · LOW.

---

## Part A — Lead's own direct-inspection findings (LDO core)

These came from reading the actual math in `margins.py`, `covariance.py`, `lowrank.py`, `lags.py`, `assemble.py` — not from tests.

### A1 · `DIRECT-W-01` — CRITICAL — the reliability-weight tensor `W` never enters the precision estimator
- **kind:** implementation_gap / statistical_validity_risk
- **detail:** `assemble_ldo_tensor` builds a per-cell reliability tensor `W` from the §2.3 provenance ladder (`observed`=1.0, `geo/time_invariant_broadcast`=0.5, `reconstructed`=0.5, `bounded`=0.4, …), the §3.12 Kish/Moran `n_eff`, and (as of O12) the Q-tensor `denom_fragility`/`provenance_risk`. The MSD premise is that fragile/reconstructed cells are DOWN-WEIGHTED in the dependency estimate. But the core estimator ignores `W` entirely: `covariance.pairwise_correlation` and `covariance.whitened_lagged_correlation` compute the (whitened) correlation from `Z` with **no weighting**; `lags.fit_lagged_links` and `precision.fit_contemporaneous_precision` never read `field.W`. Grep for `field.W`/`weights` in `covariance.py`/`lags.py`/`precision.py` = **zero hits**. `W` is only carried (margins/edges/resolution) and used in ONE peripheral place — the ≤24-edge `spatial_field` BYM readout. So every cell — a directly-observed SIM death count and a gravity-reconstructed intercensal population denominator — contributes EQUALLY to the precision.
- **evidence:** `covariance.py` (no W); `lags.py:108-137` passes `field.Z` only; `precision.py:118-119`; W consumers = `margins.py:142`, `edges.py:46`, `resolution.py:46/72`, `spatial_field.py:84`.
- **recommendation:** make the covariance a genuinely WEIGHTED estimator: `pairwise_correlation` and `whitened_lagged_correlation` must weight each cell's contribution by `W` (weighted means/cross-moments: `Σ w·x / Σ w`, weighted `n_eff` for the Fisher-z SE). This also finally makes the O12 state→W wiring functional. Without it, the entire §3.12 uncertainty architecture is decorative and the precision is biased toward reconstructed structure.
- **risk_if_ignored:** the LDO discovers dependence driven by the population-reconstruction prior (a shared latent driver across all rate variables via the common denominator), reported as epidemiological structure. This is a first-order validity failure.

### A2 · `DIRECT-MARG-01` — CRITICAL — count-exposure margin uses a single POOLED rate λ for all cells & all years
- **kind:** msd_theoretical_fragility / statistical_validity_risk
- **detail:** `count_exposure_gaussianize` sets `λ = Σx / Σexposure` — ONE scalar over the whole variable (all municipalities, all 2000–2024 years) — then `μ_i = λ·E_i` and PITs each count under `Poisson(μ_i)`. So the latent `Z_i` encodes deviation from the **national-average rate**, not from a local/temporal expectation. A high-mortality municipality has systematically high `Z` in every year (a pure spatial artefact); if national rates trend over 25 years, the whole panel's `Z` trends. The margin therefore INJECTS spatial-mean and temporal-trend structure that the LDO subsequently "discovers" as dependence. The docstring claims dependence is modelled "net of exposure" — true for exposure, but NOT net of the rate's spatial/temporal variation, which is the dominant structure.
- **evidence:** `margins.py:111` `lam = x.sum() / total_e`; `margins.py:112` `mu = lam * e`.
- **recommendation:** use a spatially/temporally-varying `μ` — fit a baseline offset GLM (log μ = log E + s(space) + f(time)) or at least an empirical-Bayes shrunk per-(space) and per-(time) rate — so `Z` is deviation from the LOCAL expected count. Equivalent to modelling the count with a proper mean structure before extracting residual dependence.
- **risk_if_ignored:** spurious spatial and temporal "edges" that are just the marginal rate surface; the spatial GMRF whitening cannot fully remove this because the copula transform is nonlinear.

### A3 · `DIRECT-MARG-02` — HIGH — Poisson margin ignores overdispersion → invalid PIT → non-Gaussian latent
- **kind:** statistical_validity_risk
- **detail:** Municipal disease/mortality counts are almost always overdispersed (Var ≫ Mean) from unobserved heterogeneity and clustering. Under overdispersion the Poisson PIT `u = F(x-1;μ)+U·p(x;μ)` is NOT Uniform(0,1) (it is over-dispersed → u concentrates near 0 and 1), so `Z = Φ⁻¹(u)` is NOT standard Gaussian. The Gaussian-copula precision then estimates a correlation on a non-Gaussian latent — biased, and the Fisher-z SE is wrong.
- **evidence:** `margins.py:113` Poisson CDF/PMF; no dispersion parameter anywhere.
- **recommendation:** negative-binomial (or quasi-Poisson) margin with a per-variable dispersion estimate; or a moment-based dispersion correction. Test PIT uniformity (a KS/χ² diagnostic on `u`) and warn when it fails.
- **risk_if_ignored:** systematically miscalibrated edge uncertainties and biased partial correlations for the extensive-count variables (i.e. the disease outcomes themselves).

### A4 · `DIRECT-MARG-03` — HIGH — rank-PIT over the joint (S×T) pool bakes the marginal rate surface into `Z`; nonlinear-vs-linear tension with whitening
- **kind:** statistical_validity_risk
- **detail:** `randomized_pit_gaussianize` ranks each variable across ALL cells jointly (space×time flattened). A variable with a strong spatial gradient becomes monotone-in-gradient in `Z` → strong spatial autocorrelation in `Z`. The downstream GMRF whitening is a LINEAR operator; it cannot undo the NONLINEAR rank transform's induced autocorrelation. So marginal transform and spatial de-correlation fight each other. Same pooled-reference problem as A2 but for the rank margin.
- **evidence:** `margins.py:42-63`, `gaussianize_field` flattens `field.X[j].reshape(-1)`.
- **recommendation:** rank within a stratification that removes the nuisance surface (e.g. rank within-time, or residualize a spatial spline before ranking), consistent with whichever de-correlation the whitening performs.
- **risk_if_ignored:** residual spatial autocorrelation in `Z` inflates apparent dependence between any two spatially-structured variables (Clifford-Richardson effective-df problem).

### A5 · `DIRECT-MARG-04` — HIGH — extreme zero-inflation makes the randomized-PIT latent mostly noise for rare diseases
- **kind:** epidemiological_risk / statistical_validity_risk
- **detail:** For a rare cause (e.g. C25 pancreatic deaths) most of Brazil's 5,570 municipalities have 0 deaths/year. The randomized PIT spreads that huge tie mass uniformly over `[0, F(0)]`, so all zero-cells get i.i.d. uniform-noise `Z`. The LDO then estimates dependence largely from noise for those cells, diluting real signal and potentially manufacturing weak spurious edges. This is intrinsic to the discrete copula under mass-at-zero.
- **evidence:** `margins.py:57-62` ties handled by `less/leq/eq` uniform spread.
- **recommendation:** a hurdle/zero-inflated margin (model P(zero) separately), OR aggregate rare outcomes to a spatial grain where counts are non-degenerate (the multiresolution machinery should choose the grain by count adequacy, not just memory).
- **risk_if_ignored:** the flagship rare-cancer study estimates municipal dependence from tie-breaking noise.

### A6 · `DIRECT-WHIT-01` — HIGH — GMRF whitening imputes missing cells to the mean (0), attenuating spatial correlation for sparse panels
- **kind:** statistical_validity_risk
- **detail:** `whitened_lagged_correlation` sets `Zc = where(finite, Z, 0.0)` before the sparse `Q`-matvecs. Imputing missing cells to 0 (the standardized mean) and treating them as observed injects a mass of "average" cells the whitening smooths across → attenuation of the estimated spatial correlation toward 0, and a biased whitened cross-moment `Gᵀ Q G`. The docstring concedes "a dense op cannot honour per-cell missingness," but for sparse national panels this is a material bias, not a footnote.
- **evidence:** `covariance.py:167` `Zc = np.where(np.isfinite(Z), Z, 0.0)`.
- **recommendation:** either restrict the whitened Gram to observed cells per pair (pairwise-complete whitening), or carry a per-cell mask into the `Q`-quadratic and normalize by observed mass — this is where `W` (A1) should also enter.
- **risk_if_ignored:** spatial edges attenuated where data is sparse (exactly the small/rural municipalities), a coverage-correlated bias.

### A7 · `DIRECT-WHIT-02` — MEDIUM — `_nearest_correlation` is a single eigen-clip, not a true nearest-correlation projection
- **kind:** computational_trap
- **detail:** `_nearest_correlation` clips eigenvalues at `floor=1e-3` then divides by `√diag`. That is one step, not Higham's alternating projection onto the intersection of the PSD cone and the unit-diagonal set; after the renormalization the matrix can again have eigenvalues < floor or |off-diag| slightly > 1, and it is not the Frobenius-nearest correlation matrix. It biases the input to CPW.
- **evidence:** `covariance.py:29-36`.
- **recommendation:** Higham's `nearcorr` (a few alternating projections) or `scipy`/`statsmodels` `corr_nearest`; cheap at `p·(K+1)` scale.
- **risk_if_ignored:** a subtly non-PSD or mis-scaled correlation fed to the ADMM, degrading the S/L split.

### A8 · `DIRECT-CPW-01` — HIGH — λ1, λ2 are fixed magic numbers; the entire edge set depends on an unselected penalty
- **kind:** msd_theoretical_fragility / statistical_validity_risk
- **detail:** CPW/LVGLASSO edge recovery depends critically on `(λ1, λ2)`; the CPW theory ties the admissible range to sample size and the incoherence parameter. The code fixes `λ1=0.1`, `λ2=1.0` (production) with NO data-driven selection — no CV, no theoretical `√(log p / n)` scaling, no stability-path selection of λ itself (the O4 path-agreement conjunct only checks robustness at a fixed operating point, it does not choose λ). The number of discovered edges is therefore an artefact of an arbitrary threshold.
- **evidence:** `orchestrator.py` defaults; `lowrank.py:fit_sparse_plus_lowrank(lambda1=0.1, lambda2=0.1)`; investigate `lambda2=1.0`.
- **recommendation:** select `λ1` by a stability-selection criterion (Meinshausen-Bühlmann, control per-family error at a target) or by an eBIC/CV over a path; scale with `√(log p / n_eff)`. Report the selected λ + the selection's error guarantee.
- **risk_if_ignored:** "we found K edges" is not reproducible or defensible; a reviewer changes λ and the finding set changes.

### A9 · `DIRECT-CPW-02` — MEDIUM/HIGH — fixed ADMM ρ, no adaptive penalty / residual balancing → convergence risk misread as "descriptive"
- **kind:** computational_trap
- **detail:** The CPW ADMM uses a fixed `ρ=1.0` with no Boyd-style primal/dual residual balancing. For an ill-conditioned `C` the split can converge very slowly or stall; the `max_iter=500, tol=1e-5` gate then yields `converged=False`, which the certifier downgrades to "descriptive." But an unconverged S/L split is not merely low-confidence — it can be QUALITATIVELY wrong (mass that belongs in `L` sits in `S` as spurious dense edges, or vice versa). Downgrading to descriptive ships a wrong graph, not a cautious one.
- **evidence:** `lowrank.py:fit_sparse_plus_lowrank(rho=1.0, max_iter=500, tol=1e-5)`; no ρ update in the loop.
- **recommendation:** adaptive ρ (residual-balancing), over-relaxation, and a convergence certificate (primal+dual residual, duality gap) rather than an iteration cap; refuse (not "descriptive") if the gap is large.
- **risk_if_ignored:** on the hardest (most interesting) national fits, a wrong graph is emitted with a "descriptive" label that reads as merely uncertain.

### A10 · `DIRECT-CPW-03` — MEDIUM — the incoherence gate (`min_factor_support=3`) is a crude proxy for CPW identifiability
- **kind:** msd_theoretical_fragility
- **detail:** CPW identifiability rests on an incoherence parameter μ(L) (the low-rank part must be "spread"/incoherent) and a transversality/degrees-of-freedom condition. The code approximates this with `min_factor_support=3` (a factor supported on <3 strong loadings is reclassified as a direct edge). A genuine 2-variable confounder (a shared driver of exactly two variables) is then misassigned to `S` as a direct edge — the exact ambiguity CPW is supposed to resolve. The number 3 has no theoretical grounding.
- **evidence:** `lowrank.py:110/135/226` `min_factor_support`.
- **recommendation:** compute the actual incoherence (max leverage of `L`'s eigenvectors) and gate on it; or report both hypotheses (direct vs confounded) with the identifiability caveat when support is small.
- **risk_if_ignored:** two-variable confounding is systematically reported as a direct causal-candidate edge.

### A11 · `DIRECT-LAG-01` — HIGH — distributed-lag "directed" edges are Granger-predictive, not causal, and assume 25-year stationarity
- **kind:** epidemiological_risk / msd_theoretical_fragility
- **detail:** The lag-extended precision reads a nonzero `X_i(t−k) → X_j(t)` block as a "directed lag-k link." This is Granger-style temporal precedence, which is confounded by any common trend or a slow third driver; the MSD's framing risks over-reading it as mechanism. Worse, the single lag structure is fit over all of 2000–2024 — a period of major epidemiological transition (mortality decline, coverage expansion) — implicitly assuming a stationary lag response. At annual aggregation, lag-0 vs lag-1 is also barely separable.
- **evidence:** `lags.py:154-179` directed peak-lag readout; no stationarity test/segmentation.
- **recommendation:** state the Granger (not causal) semantics explicitly on lagged edges (the causal ladder already helps — enforce it); test stationarity / allow a regime split; consider sub-annual grain where lag identification is meaningful.
- **risk_if_ignored:** temporal-precedence associations over-interpreted as lagged mechanisms across a non-stationary period.

### A12 · `DIRECT-TEMP-01` — MEDIUM — temporal-smoothness prior can manufacture a smooth response curve that isn't there
- **kind:** statistical_validity_risk
- **detail:** The §III.4(5) temporal quadratic `(γ/2)tr(SᵀL_lag S)` shrinks adjacent-lag precision blocks toward each other. With `γ=0.1` default this biases the discovered distributed-lag response_curve toward smoothness. If the true response is a sharp single-lag spike, the prior smears it across neighbours (a spurious-smoothing trap); the reported "response curve" then reflects the prior, not the data.
- **evidence:** `lowrank.py:build_smoothness_operator`, `orchestrator.py gamma_temporal=0.1`.
- **recommendation:** make `γ` data-driven (marginal-likelihood/CV) rather than a fixed default; report the curve's prior-sensitivity.
- **risk_if_ignored:** response-curve shapes are partly artefacts of an unjustified smoothing weight.

### A13 · `DIRECT-SE-01` — HIGH — Fisher-z edge SE uses the RAW cell count, not the Moran-corrected effective-n → over-certification
- **kind:** statistical_validity_risk
- **detail:** Edge uncertainty is `stat_se = 1/√(n_eff−3)` with `n_eff = _n_eff(field) =` the raw count of observed cells (`isfinite.any(axis=0).sum()`). Under strong spatial (and temporal) autocorrelation the EFFECTIVE sample size is far smaller than the raw cell count (Clifford-Richardson / Dutilleul); using the raw n makes the SE too small → edge confidence intervals too narrow → the §III.8 conjunction certifies edges that a correct SE would leave descriptive. The Moran-corrected effective-n IS computed in the Q-tensor (`_moran_corrected_n_eff`), but — exactly like `W` (A1) — it never reaches the SE. Same architectural gap: the effective-df correction is computed and discarded.
- **evidence:** `edges.py:151` `_n_eff`, `edges.py:173` `stat_se`; `compile_attach.py:253` computes the corrected n_eff but the LDO reads the raw one.
- **recommendation:** feed the Moran/Kish-corrected effective-n (per the panel's actual autocorrelation) into the Fisher-z SE; this is the same fix vector as A1 (make the estimator consume the computed uncertainty state).
- **risk_if_ignored:** systematic over-certification of spatially-autocorrelated edges — anticonservative exactly where epi data is most dependent.

### A14 · `DIRECT-STAB-01` — HIGH — spatial stability-selection randomly subsamples municipalities, fragmenting the GMRF adjacency
- **kind:** statistical_validity_risk / msd_theoretical_fragility
- **detail:** `stability_select` draws a random 70% subset of municipalities; each refit calls `build_spatial_precision_sparse(subsampled space_ids)`, which keeps only adjacency edges BETWEEN surviving municipalities. Randomly deleting 30% of munis shatters the contiguity graph (every deleted muni removes its edges), so each subsample whitens with a much sparser, inconsistent GMRF than the true one → the whitening under-corrects differently every subsample → stability frequencies are measured under a degraded and varying spatial model. You cannot i.i.d.-subsample spatial units and preserve the spatial dependence structure.
- **evidence:** `edges.py:34-41` `space_ids=tuple(field.space_ids[i] for i in idx)`; `lags.py:113` rebuilds `Q_space` on the subset.
- **recommendation:** use spatial BLOCK subsampling (drop whole contiguous regions / UF blocks, keeping the within-block graph intact) or a spatial bootstrap that respects contiguity; or subsample TIME (already added in MIN-3) and keep space whole for the whitening.
- **risk_if_ignored:** the stability guarantee — the LDO's headline "edges survive subsampling" claim — is measured under a broken spatial model, so it neither controls false selection nor reflects the fitted whitening.

### A15 · `DIRECT-STAB-02` — MEDIUM — per-subsample GMRF rebuild + whitening is redundant national compute
- **kind:** computational_trap
- **detail:** Beyond A14's validity issue, rebuilding `build_spatial_precision_sparse` + the matrix-free whitened Gram on every one of `n_subsamples` refits re-walks the national adjacency and re-accumulates the `O(F·S)` Gram each time — a large repeated cost at national `S≈5570`. The full-graph whitening is (or could be) computed once; only the subsample selection changes.
- **evidence:** `edges.py:_subsample_edges → lags.fit_lagged_links → whitened_lagged_correlation` per subsample.
- **recommendation:** precompute the full-graph whitened features once; subsample in the whitened space (respecting A14's block structure). Candidate for GPU batching (many small refits) — see the GPU-rematch ledger.
- **risk_if_ignored:** stability selection dominates LDO wall-clock at national scale for no benefit.

### A16 · `DIRECT-POP-01` — CRITICAL — intercensal (age×sex×race) denominators are prior-dominated ("fabricated demography"), and that uncertainty is never propagated
- **kind:** identifiability_concern / epidemiological_risk
- **detail:** The population tensor `P` over (S,T,A,X,R) is pinned by data only at census years (anchors: 2000/2010/2022) and by the per-(locality,time) closure total (a marginal). The full age×sex×race JOINT structure in every INTERCENSAL year is determined by the PRIORS — the cohort-aging recursion from the last census, migration/age second-difference smoothness, and the race-ILR prior — not by observation. So intercensal municipal age-sex-race denominators are substantially MODEL OUTPUT. The flagship C25 study age-standardizes on exactly these denominators, so its rates in ~22 of 25 years rest on a prior-dominated interpolation whose uncertainty is neither quantified nor propagated (compounding A1: even if it were quantified, `W` is ignored by the LDO).
- **evidence:** `loss.py` — anchors only where observed; closure by projection; aging/race/smoothness are priors; `orchestrator.py` two-layer denominator (closed-form interp + refine).
- **recommendation:** quantify the data-vs-prior contribution per intercensal cell (e.g. the posterior variance / effective-prior-weight), emit it as a per-cell denominator uncertainty, and propagate it into the rate margins and the LDO `W`. Consider a proper cohort-component demographic model with credible intervals rather than a penalized LS point estimate.
- **risk_if_ignored:** age-standardized municipal trends reported with false precision; the denominator's model structure leaks into "epidemiological" findings.

### A17 · `DIRECT-POP-02` — HIGH — reconstruction objective weights are hand-set magic numbers governing the data-vs-prior bias-variance tradeoff
- **kind:** msd_theoretical_fragility / statistical_validity_risk
- **detail:** `PopulationObjectiveWeights` fixes anchor=10, aging=1, migration=0.1, migration_total=0, race=0, age_smooth=0.05. These are the relative influence of DATA (anchor=10) vs PRIORS (age_smooth=0.05) — a 200:1 hand-chosen ratio — with no calibration to the actual variances of each term. The entire reconstruction's bias-variance tradeoff, and hence the intercensal demography, is set by unjustified constants.
- **evidence:** `loss.py:33-46` `PopulationObjectiveWeights` defaults.
- **recommendation:** set weights as inverse noise variances (a proper GLS/hierarchical-Bayes weighting), or profile them by cross-validating held-out census years; report sensitivity.
- **risk_if_ignored:** reconstructed demography is an artefact of arbitrary regularization strength.

### A18 · `DIRECT-POP-03` — MEDIUM — cohort survival uses `1 − nan_to_num(death_rate, 0)` → missing-mortality cells are treated as immortal
- **kind:** computational_trap
- **detail:** The aging recursion uses `survival = 1 − nan_to_num(death_rates, nan=0.0)`. A cell with a MISSING death rate gets `survival = 1.0` (nobody dies), so its aged population is overestimated. Missingness is silently mapped to "no mortality," a directional bias in the cohort projection precisely where mortality data is absent (small/rural municipalities).
- **evidence:** `loss.py:143-144`.
- **recommendation:** impute a regional/age-typical survival for missing cells (or widen its uncertainty), not survival=1.
- **risk_if_ignored:** population over-projected in data-poor cells, deflating their apparent rates.

### A19 · `DIRECT-POP-04` — MEDIUM — the migration tensor is an unidentified smoothness-prior residual reported as a specific flow field
- **kind:** identifiability_concern
- **detail:** Municipal age-sex-race net migration is unobserved; the solve produces a specific migration tensor `η` driven almost entirely by the 2nd-difference smoothness prior (w=0.1) plus the per-locality net-total anchor (w=0 by default). The result is a smooth but essentially fabricated flow field that then feeds the population aging recursion. The MSD notes migration unidentifiability, but the pipeline still emits and consumes a point migration field with no uncertainty flag.
- **evidence:** `loss.py:288-312`; `migration.py` gravity prior; `migration_total` weight default 0.
- **recommendation:** treat `η` as a nuisance with wide uncertainty (or marginalize it), and do not let its point estimate silently shape the denominator without an uncertainty band.
- **risk_if_ignored:** the denominator inherits a fabricated migration structure as if observed.

*(Lead's direct reading continues; the modularization + GPU thrusts follow in their own ledgers. Workflow findings append to Part B.)*

---

## Part B — `math-theory-critique` workflow findings (full, verbatim)

*(Appended when `wf_99a2944f-2d7` returns — every finding, no compaction.)*
