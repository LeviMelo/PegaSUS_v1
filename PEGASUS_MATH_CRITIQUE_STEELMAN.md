# Steelman verdicts (adversarial defense of the CRITICAL/HIGH critiques)

## LDO-MARG-01 [CRITICAL] — Poisson-offset margin hardcodes equidispersion, contradicting MSD-I's own NB/ZINB mandate

This fully confirms the picture. There is even a `select_count_family` router (MSD §6.2 data-aware routing) that the LDO margin bypasses entirely. I have everything needed.

---

**VERDICT: UPHELD.** The critique is mathematically correct, its spec citations are accurate, and — the sharpest point — the corrective machinery it recommends *already exists in the repo* but is not wired into the LDO margin.

**1. The spec citations are exactly right.**
- MSD-I §6.2 (lines 6270-6272, read verbatim): "zero-inflated support routes to hurdle or zero-inflated models… negative binomial or quasi-Poisson is used" for counts.
- MSD-III §III.5 (line 222): "The MSD-I GLM families are the per-variable transforms (Poisson/NB/binomial/beta-binomial/lognormal/…)".

The spec mandates a *family set* and a *data-aware selection*. `count_exposure_gaussianize` (margins.py:88-115) hardcodes exactly one member (Poisson) with a single pooled rate `lam = x.sum()/total_e`, and never estimates dispersion or a zero-inflation gate. The implementation is narrower than the contract it cites.

**2. The PIT math is correct.** The randomized PIT `u_i = F(x_i-1; μ_i) + U·p(x_i; μ_i)` is Uniform(0,1) **iff** `F` is the true generating CDF (this is the defining property of the Dunn–Smyth randomized-quantile transform for discrete data). Under an overdispersed truth (Var ≫ μ) fed through a Poisson `F`, the tails of `F` are too thin, so realized counts land disproportionately in the extreme CDF bands → `u_i` is over-dispersed relative to uniform → `ndtri(u_i)` yields a heavy-tailed `Z` with inflated second moment concentrated in high-exposure/high-count cells. The mechanism the critique describes is real.

**3. The consequence for the precision graph is a genuine failure mode, not hand-waving.** The whole point of the copula layer (§III.4) is that a *correctly Gaussianized* `Z` lets Gauss-Markov structure (sparse + low-rank) be read off the second moment. A margin that systematically fat-tails the large municipalities makes those cells jointly "surprising" in a common direction. The CPW decomposition (`sparse − low-rank`) is precisely the machinery that will absorb a shared inflation as a `latent_shared` factor or, failing that, spurious dense edges — i.e. overdispersion re-manufactured as a discovered driver. This directly threatens the determinant readout.

**4. The decisive point — the fix is already built and merely unrouted.** `src/pegasus/compute/glm.py` contains the complete apparatus the recommendation asks for:
- `_estimate_nb_theta` (line 207) — method-of-moments MoM θ from mean/variance, with a clean Poisson-limit fallback (`var ≤ mean → θ=1e6`).
- `_negative_binomial`, `_nb_log_pmf`, NB CDF pairs (`_untruncated_cdf_pair`, line 267).
- `_hurdle_cdf_bounds` (line 280) + `randomized_quantile_residuals` (line 302) — the exact Dunn–Smyth randomized-PIT-to-Gaussian map, hurdle/ZI-aware.
- `select_count_family` (line 313) — "zero-inflated → hurdle; overdispersed → NB", the §6.2 router.

And a `Grep` confirms **the LDO package never imports `compute.glm`** (no matches for `compute.glm` under `src/pegasus/ldo`). So the situation is worse than "Poisson is a simplification": the codebase has the NB/hurdle randomized-PIT machinery, the family-selection router, *and* a `dispersion_screen` (var/mean index of dispersion) in `ldo/exhaustiveness.py:39` — and the count-exposure margin ignores all three, re-deriving a Poisson-only path from scratch.

**Corrections/sharpenings to the critique:**
- The critique credits the spec's family set; note the repo *implements* it — so the recommendation is a wiring/routing task, not a build. `count_exposure_gaussianize` should compute an offset-adjusted MoM θ (Pearson-residual dispersion of `x` against `μ = λ·E`, not the marginal var/mean of `x`, since exposure heterogeneity itself inflates the raw variance) and swap `poisson.cdf/pmf` for `nbinom`, routing zero-inflated support to the existing hurdle CDF bounds. The θ estimate must condition on the exposure structure or it will over-attribute exposure-driven variance to dispersion.
- One point of nuance the critique slightly overstates: the *other* branch, `randomized_pit_gaussianize` (the empirical rank margin), is dispersion-agnostic and would NOT suffer this — it is asymptotically uniform by construction regardless of the truth. So the defect is scoped precisely to variables that get routed onto the count-exposure Poisson branch (`n_exposure_margins` in diagnostics, orchestrator.py:464) — the extensive-count RN fields. That is a real and central subset (mortality/incidence counts with population denominators — i.e. the pancreatic-C25 win-condition outcome itself), so the severity rating stands, but it is not "every variable."

**Net:** CRITICAL, upheld. The margin's own distributional assumption is falsified by Brazilian municipal count data, the cited spec disallows it, the PIT/precision-graph consequence is sound, and the fix is a routing job into already-present, spec-compliant NB/hurdle machinery.

Key files: `src/pegasus/ldo/margins.py:88-115` (the defect), `src/pegasus/compute/glm.py:207,280,302,313` (the unwired fix), `src/pegasus/ldo/exhaustiveness.py:39` (an unused dispersion diagnostic), `src/pegasus/ldo/orchestrator.py:464` (the routing point).

---

## LDO-MARG-02 [CRITICAL] — Single global rate lambda pushes real spatial/temporal rate variation into the latent 'surprise' scale

I now have complete evidence for a rigorous verdict.

---

## VERDICT: PARTIAL (the mechanism is real; the severity is overstated because two downstream layers absorb it — but a residual concern is genuine and worth sharpening)

### What the critique gets exactly right

1. **The code fact is correct.** `margins.py:111` is a single scalar `lam = x.sum() / total_e`. There is no stratum, no spatial random effect — the offset is the flat national rate. `μ_i = λ·E_i` residualizes every cell against one national mean.

2. **The statistical consequence is correctly derived.** For a variance-stabilized Poisson-offset PIT, `Z_i ≈ (x_i − λE_i)/√(λE_i)`, which is (up to the VST) the **standardized morbidity ratio residual against a global rate** — the textbook naive SMR, *not* a BYM/shrinkage expected count. A persistently high-rate municipality does get a positive Z on every high-rate outcome, so a "municipal development/access" gradient is injected into the latent scale coherently across outcomes. This is real and the critique states it precisely. The recommendation (per-stratum rate at minimum, BYM `E_i = exp(offset + spatial RE)` ideally, Freeman–Tukey/Anscombe VST, EM-style copula-margin refinement) is the correct small-area-estimation fix.

### Why the severity is nonetheless overstated (PARTIAL, not CRITICAL)

The critique's own escape hatch is load-bearing and the codebase **actually implements it**. The claim "the dominant discovered structure is a mechanical artefact masquerading as a determinant edge, and genuine edges are shrunk beneath it" requires the artefact to **contaminate the direct-edge set**. Two wired layers block that:

- **Spatial GMRF whitening (Layer 1, `precision.py:87-98`, `covariance.py:151-198`, task #34/#41 = DONE, live).** The estimator does **not** feed raw Z to the graphical lasso. It whitens by `Σ_space^{-1/2}` with `Σ_space^{-1} = κI + L_W` *before* estimating `Ω_var`. A spatially-structured common offset — which is exactly what the North/Northeast gradient and urban/rural access axis are, being smooth over the adjacency graph — lies **in the low-frequency eigenspace of `L_W`** and is precisely what the whitener suppresses. The docstring is explicit: "a link in `Ω_var` is dependence *net of* space." So the mechanically-injected spatial gradient is largely removed before edges are read, not promoted to edges.

- **CPW sparse+low-rank split (Layer 2, `lowrank.py`, DONE).** Whatever coherent cross-outcome common mode survives whitening loads on the low-rank `L` and is emitted as `latent_shared`, with `S` (the direct edges) recovered *net of* it (`lowrank.py:12-13, 289-292`: a pair in a direct edge is dropped from `latent_shared` — one role per pair). The critique concedes this is "correct MSD-III behavior" but calls the wave "an artefact." Here the framing overreaches: the LDO's contract (§III.8) is to **produce hypotheses with typed provenance, not certify causation**. An emitted `latent_shared` factor labelled as a shared driver is not silently mis-sold as a determinant edge — it is exactly the "confounded pair" bucket. The failure the critique fears (genuine determinant edges "shrunk beneath this artefactual factor") is the specific pathology the pairwise-complete covariance and the S-vs-L role split were built to prevent — see `covariance.py:5`, which names *this exact collapse* ("a spurious shared factor that collapses every relationship into `latent_shared`") as the thing the design fights.

### The residual concern that IS genuine (this is what to sharpen)

The two layers don't make the flat-λ harmless — they relocate the damage rather than eliminate it, and the critique correctly identifies that the margin **cannot see the structured model** (Layer 0 precedes Layer 1). Two real leakage paths remain:

1. **Whitening is spatial-only and by a *fixed prior* `κI+L_W`, not the fitted rate field.** It removes smooth spatial structure but (a) `κ` is a fixed hyperparameter (default 1.0), so the whitener is a mis-specified prior, not the shrunk posterior expected-count — under- or over-whitening is possible; and (b) any part of the municipal-development gradient that is **not spatially smooth over the cod6 adjacency** (e.g., isolated high-access urban islands surrounded by low-access rural neighbours — a high-frequency component of `L_W`) survives whitening and then loads on `L`. So the flat-λ artefact still inflates the `latent_shared` factor's magnitude/rank even if it rarely reaches `S`.

2. **Variance mis-scaling from the wrong μ is not undone by whitening.** The GMRF whitens the *mean* structure across space; it does not correct the fact that a low-exposure high-rate cell was PIT'd against `λE_i` with the wrong dispersion. Heteroscedastic residual variance from flat-λ standardization biases the correlation entries themselves before whitening — whitening a wrongly-scaled Z does not restore the right scale.

So: **the artefact is prevented from masquerading as a *direct determinant edge* (the CRITICAL claim fails), but it does mechanically inflate the shared-latent factor and perturb correlation scale (a real, PARTIAL-severity defect).** The recommendation is correct in direction and would strictly improve fidelity; its most defensible minimal form here is the **per-stratum (health-region/state) rate offset in the margin** — that alone converts the flat-λ into a coarse structured expectation and removes the bulk of the coherent gradient at Layer 0, before it ever depends on the whitener's fixed `κ` being well-chosen. The full EM-style copula-margin refinement (fit precision → re-derive `E_i` from the fitted spatial field → re-Gaussianize) is the principled version and is consistent with the architecture, but it breaches the clean Layer-0/Layer-1 separation the design currently prizes, so it is an enhancement, not a bug-fix.

### Bottom line

- **REFUTED** as stated at CRITICAL severity ("dominant *discovered structure* / genuine determinant *edges* shrunk beneath an artefact"): the spatial GMRF whitening + CPW S/L role-split, both wired and live, are exactly the machinery that keeps a coherent spatial common-mode out of the direct-edge set and into a typed `latent_shared` bucket. The design anticipated this collapse explicitly (`covariance.py:5`).
- **UPHELD** as a genuine PARTIAL-severity margin defect: the flat scalar λ is a real small-area-estimation shortcut that injects an avoidable coherent gradient at Layer 0, inflating the low-rank factor and mis-scaling correlations; whitening by a *fixed-`κ`* spatial prior mitigates but does not eliminate it (high-frequency spatial and variance-scale leakage remain). The recommended per-stratum-or-BYM expected count is the correct fix and is architecturally admissible.

**Relevant files:** `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (line 111, the defect), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\precision.py` (lines 87-129, the spatial whitening that mitigates), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py` (lines 269-302, the S/L role split), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (lines 1-8 and 151-198, the anticipated-collapse note + whitened correlation), `C:\Users\Galaxy\LEVI\PegaSUS\PEGASUS_MSD_III.md` (§III.3 lines 201-208 BYM/small-area, §III.4 lines 214-218, §III.5 line 222).

---

## LDO-MARG-03 [HIGH] — Randomized-PIT jitter drawn once and frozen; stability selection treats one stochastic realization as data

The MSD spec is clear: it specifies the rank/PIT transform as Layer 0 but says nothing about integrating out the auxiliary randomization U. The spec describes it as a deterministic map `Z = Φ⁻¹(F_j(X))`, glossing over that for discrete F_j the transform is only distributionally-Gaussian under U-averaging. So the spec does not mandate the fix, but neither does it forbid it — the critique targets a genuine gap between the theory (valid in expectation over U) and the implementation (one frozen draw).

Let me confirm one final point: whether `stability_select` re-randomizes the PIT per subsample. From the code, it does NOT — it slices `field.Z` directly (edges.py:45). And `count_exposure_gaussianize` has the identical single-draw structure (line 113). I have everything I need.

## Verdict: PARTIAL (mechanism real, severity overstated)

The critique is **factually correct on the code** and **identifies a real statistical gap**, but its **HIGH severity and its "majority of latent values are pure jitter" characterization are overstated**. The core recommendation (multiple randomizations pooled MI-style, or a deterministic point estimate + randomization band) is sound and worth adopting.

### What the critique gets right (upheld)

1. **Single frozen draw — confirmed.** `gaussianize_field` is called exactly once (`orchestrator.py:71`), `rng.uniform(...)` fires once per variable (`margins.py:60`; same for the count-exposure margin `margins.py:113`), and `stability_select` slices the *frozen* `field.Z` (`edges.py:45`). The auxiliary randomization U is **never integrated out** — not across subsamples, not across holdout/certification/λ-path refits, all of which reuse the same `gf`. This is exactly as described.

2. **The distributional-validity point is correct.** The Dunn–Smyth randomized PIT `z = Φ⁻¹(F(x⁻) + U·p(x))` is standard normal only *marginally over U ~ Uniform*. For a single draw, each tie-band member gets one arbitrary latent value, and two epidemiologically identical zero cells get different Z. Freezing U turns the auxiliary randomization into a **fixed nuisance common to every subsample**, so stability selection cannot see it as variability — correct. Stability selection controls sampling variability of the *lattice*, conditional on the margin; it says nothing about margin-randomization variability. The critique's structural claim is sound.

3. **Reproducibility-in-the-scientific-sense is genuinely weakened for low-count edges.** A different margin seed *will* move the jittered ranks of the zero-mass cells, and for edges whose signal lives in the low-count tail this can flip selection. This is a real, unquantified Monte-Carlo-over-U error component that the current uncertainty budget omits. The propagated `edge_uncertainty` (edges.py:173–176) combines only Fisher-z statistical SE and randomized-SVD numerical error — **the auxiliary-randomization SE is not in it.** That omission is a legitimate finding.

### Where the critique overreaches (why PARTIAL, not UPHELD)

1. **"Majority of every variable's latent values are pure jitter frozen into the correlation structure" — this overstates the impact on the *estimand*.** The zero cells do all get randomized ranks, but they get them from a **common tie-band** `[#{<x}/n, #{≤x}/n]` — for the zero mass, `[0, q₀]` where q₀ is the zero-fraction. Their Z values are i.i.d. Uniform-mapped-to-Gaussian *within that band*, so across cells they are **exchangeable noise, not a coherent spurious signal**. Two variables that are both >90% zeros get independent U-draws, so their jitter is uncorrelated by construction — it does **not** manufacture a spurious partial correlation between them in expectation; it **inflates the effective noise floor** (attenuating true partial correlations, i.e. biasing toward *null*, not toward false discovery). The critique's framing "it looks like signal to all subsamples" is imprecise: the frozen jitter is a fixed *nuisance realization*, but it is high-entropy noise, not structured signal — its main effect is variance/reproducibility, not systematic false-positive edges. This materially lowers the risk relative to the "biases every refit identically" reading.

2. **Rare-event routing partly mitigates.** The variables with >90% zero mass are exactly the extensive counts that, when an exposure/denominator is declared, are routed to `count_exposure_gaussianize` (the Poisson-offset margin, `margins.py:88`, `orchestrator.py` count-exposure wiring). Under a Poisson(μ=λ·E) CDF the tie-bands `[F(x−1;μ), F(x;μ)]` are **cell-specific** (they depend on E), so the "all zeros collapse to one arbitrary band" picture is weaker there — though the single-draw problem still applies within each band. The pure rank-PIT worst case the critique describes is the *no-denominator* path.

3. **Low-power edges in the tail are already gated to descriptive.** `_MIN_N_EFF = 100` (edges.py:23) and the `n_eff < 100 → descriptive_only` gate (edges.py:168, 202) mean the sparsest configurations cannot be *promoted* on one jitter draw — they ship as descriptive. This doesn't eliminate the reproducibility concern for the moderately-sparse-but-powered regime, but it caps the "false confidence" failure mode the critique invokes.

### The correction (what should actually change)

The critique's recommendation is the right shape but should be sharpened:

- **Cheapest correct fix:** wrap the existing single-draw path in an **R≥20 outer loop over independent margin seeds**, refit the precision each time, and combine MI-style — average partial correlations, and inflate `edge_uncertainty` by the between-realization (Rubin between-imputation) variance component `(1+1/R)·B` added in quadrature to the current within-fit SE. The stability frequency then becomes a *pooled* selection frequency across (subsample × realization), which is exactly the quantity that averages the jitter away. This is the mathematically defensible version.
- **Cheaper approximation:** use a **deterministic mid-quantile PIT** (`u = (#{<x} + 0.5·#{=x})/n`, i.e. `U ≡ ½`) for the *point estimate / selection*, and use a handful of random draws only to produce a **diagnostic band** on each edge. Mid-P is not perfectly Gaussian for heavy ties, but it removes the RNG dependence of the reported result and is reproducible.
- **Minimum viable (do regardless):** the margin seed is already plumbed (`seed=seed` at orchestrator.py:71 / `gaussianize_field(..., seed=)`), so **run the whole LDO under a few seeds and report the per-edge Monte-Carlo SE attributable to U.** Even without pooling, this converts an invisible nuisance into a reported, honest uncertainty — consistent with the project's own honesty-layer doctrine (every promoted edge carries propagated uncertainty; this is a missing term in that budget).

### Bottom line

**PARTIAL — upheld as a real, unaddressed statistical-validity gap (the auxiliary randomization is frozen and never integrated out, and its variance is absent from the edge uncertainty budget), but downgrade from HIGH to MEDIUM:** the frozen jitter is exchangeable noise that attenuates toward the null and inflates variance/irreproducibility rather than manufacturing structured false-positive edges "identically across refits," the sparsest edges are already gated to descriptive by `n_eff<100`, and the extensive-count worst case is partly mitigated by the exposure-offset margin. The recommended fix (R independent PIT realizations pooled by Rubin's rules, or a deterministic mid-P point estimate with a randomization band) is correct and should be adopted; at minimum, exercise the already-exposed margin seed across a few runs and add the between-realization component to `edge_uncertainty`.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (lines 42–64 rank PIT, 88–115 count-exposure, 118–144 single-seed driver), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py` (line 71 single call; 173–176 uncertainty budget missing the U-term), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (lines 41–48 subsample slices frozen Z; 23/168/202 n_eff descriptive gate). Governing spec: `PEGASUS_MSD_III.md` §III.5 and `PEGASUS_MSD_II.md` §MII-LDO-01 — both specify the rank/PIT margin but neither addresses integrating out the randomization, so the gap is a spec silence, not a spec violation.

---

## LDO-MARG-04 [HIGH] — Empirical CDF computed on the full national sample then subsampled — information leak across stability folds

Confirmed: neither `lags.py` nor `resolution.py` re-gaussianizes. The margin is computed exactly once, globally, and every downstream refit (stability subsamples, temporal holdout train/test, exact-certify slice, path-agreement) consumes the same pre-computed `gf.Z`. I have all the evidence needed for the verdict.

---

## VERDICT: PARTIAL — one prong REFUTED (minor), one prong UPHELD (and should be sharpened to a different subsystem)

The critique bundles two claims of very different severity under one banner. The stability-selection claim is largely overstated; the temporal-holdout claim is a genuine train/test leak. The recommendation should be split accordingly.

### Factual premises: both confirmed
- `margins.py:56-60` ranks each value against `np.sort(obs)` over the **full** finite sample, `u=(#{<x}+U·#{=x})/n`, `n`=full-sample size. Confirmed.
- `orchestrator.py:139-142` calls `gaussianize_field` **once**; `stability_select(gf,…)` (line 250), `temporal_holdout(gf,…)` (line 246), `certify_approximation_on_slice` (line 233) and `regularization_path_agreement` (line 314) all consume that single pre-computed `gf.Z`. No downstream stage re-gaussianizes (verified in `lags.py`, `resolution.py`, `edges.py`, `holdout.py`). So the marginal transform is a **global statistic shared across every fold**. Confirmed.

### Prong A — stability selection (spatial subsampling): REFUTED as a material threat

The critique's own framing gives away why this is weak: *"the folds are positively correlated through the common CDF and the recurrence frequency overstates stability."* Two mathematical reasons this is close to negligible in this pipeline:

1. **Stability selection was never premised on independent folds.** Meinshausen–Bühlmann stability selection resamples *without* replacement at fraction ≤ ½ and its error control (the `E[V]≤q²/((2π−1)p)` bound) is derived for **overlapping** subsamples of a *shared* dataset — folds are correlated by construction (subsample_frac=0.7 here means any two folds share ~0.4·S localities *directly*, a far larger dependency than the CDF coupling). The MSD-III §III.7 text says "refit on lattice subsamples; keep edges recurring" — it does not claim fold independence. The critique invents a "quasi-independence premise" that neither the method nor the doc asserts.

2. **The shared global CDF is a monotone, near-deterministic map, so it contributes almost no *extra* fold-coupling.** For a spatial subsample of size `n_keep≈0.7S`, the fold-internal rank of a retained cell equals its global rank restricted to that fold — and the global empirical CDF `F_n` converges to the true `F` at rate `O_p(n^{-1/2})` (DKW). With national `S` in the thousands×`T`, `n` is large, so `F_full` and any `F_fold` differ by `O_p(n_fold^{-1/2})` — a vanishing perturbation of the latent Z. The partial-correlation sign/support that stability selection keys on is invariant to a common monotone transform to first order; the leakage moves selection frequencies by a lower-order amount than the 0.3-fraction *direct* sample sharing already present. It does **not** manufacture recurrence for a null edge: a spurious pair has near-zero partial correlation in every fold regardless of whose CDF ranked it.

So for the spatial-stability prong the critique is real in *direction* but negligible in *magnitude*, and it misattributes to the CDF a "validity premise" the method doesn't have. Refuted as a HIGH risk.

### Prong B — temporal holdout (§IX.3): UPHELD — this is a genuine leak, and it is the real bug

Here the critique is correct and sharper than it states. `temporal_holdout` (`holdout.py:98-144`) splits the **already-Gaussianized** `gf.Z` by time (`_time_slice`, `_edges(_time_slice(field,0,train_T))` vs `(…,train_T,T)`). But `gf.Z[:,:,t]` for a training year `t` was produced by `randomized_pit_gaussianize` ranking each cell against **all cells including the held-out tail years**. Concretely: `u = (#{obs<x}+U·#{obs=x})/n` with `obs` and `n` spanning the whole time axis. A training cell's latent value therefore encodes *how many future-year cells fall below it*. This is textbook target leakage — the "out-of-sample" window is not out-of-sample at the margin layer.

Why this matters more than the stability prong: temporal holdout's entire epistemic purpose (per §III.8 / §IX.3 "real links persist/predict; flukes evaporate") is to be an **honest hold-out**. A transform that bakes future ranks into the training Z defeats that purpose *by construction*, not to lower order. If the disease rate trends over the period (C25 mortality does), the future-informed CDF systematically shifts training-year Z toward the full-period distribution, aligning train and test edge signs and **inflating the persistence rate** — the exact quantity the certifier reads. This is not a small perturbation; it is a directional contamination of the one prong designed to be leakage-free.

The same objection applies, more weakly, to `certify_approximation_on_slice`'s **spatial** slice (it re-uses global-CDF Z on a locality subset) — but there the comparison is approx-vs-exact on the *same* Z, so the shared CDF cancels and it's harmless. The temporal case does **not** cancel because train and test are meant to be distributional strangers.

### Correction (what to actually fix)

Do **not** rewrite spatial stability selection to re-gaussianize per fold — that would be expensive, would *reduce* stability-selection validity (fold-internal ranks on `0.7S·T` points are noisier), and fixes a non-problem. The fix is scoped to the temporal prong:

- In `temporal_holdout`, re-estimate the margin on the **training window only** and apply that same training CDF to map the test window (`F̂_train` extrapolated to test values, ties/PIT as usual). This makes the held-out years genuine strangers and is the standard way to run a leakage-free temporal CV of a copula/rank model. It requires threading raw (pre-Gaussian) values into `temporal_holdout` (currently it only receives `gf`), or re-running `randomized_pit_gaussianize` on `raw_field.X[:,:,0:train_T]` before slicing.
- Document in `margins.py`/§III.5 that the rank margin is a **global statistic**, hence any *temporal* holdout must re-fit it on the training window; spatial subsampling may share it.

### Net
- `risk_if_ignored` ("reported stability frequencies are anticonservative; edges certified that wouldn't survive an honest fold-internal transform") is **overstated for spatial stability** (Prong A) and **understated/misdirected for temporal holdout** (Prong B, where it's a true leak inflating `persistence_rate`).
- Severity should be **MEDIUM**, retargeted from "stability PIT" to "temporal-holdout margin leak (§IX.3)." The anti-spurious gate (§III.8) is weakened specifically at its *temporal-persistence* conjunct, not at its stability-frequency conjunct.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (global CDF, lines 56-62), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py` (gaussianize-once at 139-142; holdout call at 246), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\validation\holdout.py` (`temporal_holdout`/`_time_slice`, lines 81-144 — the leak site), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (`stability_select`, lines 61-148 — Prong A, no fix needed).

---

## LDO-MARG-05 [HIGH] — Per-cell reliability weights W are computed and propagated but NEVER used in the margin or the precision estimator

I now have everything needed for a definitive verdict. §III.2 is the smoking gun: *"estimation is by weighted likelihood... Sparsity is encoded as zero observation weight on unobserved cells."* The MSD mandates a **weighted-likelihood** estimator where the observation weight is per-cell. The code implements only the binary {0 = unobserved, 1 = observed} corner of that weight — the continuous reliability gradation (0.25 broadcast, 0.4 bounded, 0.5 reconstructed) is built, propagated, and then dropped at the one place it would change the answer.

Here is my verdict.

---

## VERDICT: UPHELD (with one evidence correction that, if anything, sharpens it)

The critique's central claim is **correct and material**: the per-cell reliability tensor `W` is assembled with a rich provenance-graded scale (`_STATE_WEIGHT`, assemble.py:51-60), propagated verbatim through `GaussianField` (margins.py:34), sliced through every subsample/restrict/coarsen op — and then **the core precision estimator that actually produces the edges never enters it into a moment**. A national-domain-scalar broadcast cell (W=0.25) contributes to the correlation that drives every link record with *exactly the same unit weight* as a directly-observed high-`n_eff` cell.

### The mathematical confirmation

The two estimators in `fit_lagged_links` (lags.py:109-117) are:

- **`whitened_lagged_correlation`** (covariance.py:151-220, the default `spatial_whiten=True` path): the Gram is `G += Ft @ (Q_space @ Ft.T)` where `Ft = Zc[:, :, t-lag]` is the NaN-zeroed data. The only cell-level weighting present is the *implicit* 0/1 from `Zc = where(isfinite, Z, 0)`. **W appears nowhere.**
- **`pairwise_correlation`** (covariance.py:48-109, the `S==1` / no-whiten fallback): every moment (`Sx = X0 @ M.T`, `Cxy = X0 @ X0.T`, `n_ab = M @ M.T`) is built from `M = isfinite(X)` — a **binary finiteness mask**, not W. `mean_a = Sx/n_ab`, `cov = Cxy/n_ab − mean_a·mean_b` are ordinary unweighted pairwise-complete moments.

So both estimator paths implement only the **{0,1} corner** of the observation weight (`w=0` unobserved, `w=1` observed). The continuous reliability gradation `W ∈ {0.25, 0.4, 0.5, ...}` is dead on arrival at the estimator.

### This is a genuine MSD violation, not just an unfulfilled docstring

§III.2 is normative and explicit: *"each dataset provides observations… through a **weighted observation operator**. Sparsity is encoded as zero observation weight on unobserved cells; **estimation is by weighted likelihood**."* The spec's weight is not binary — "zero on unobserved" is one endpoint of a per-cell weight whose graded interior is precisely what `_STATE_WEIGHT` encodes. The prime directive (§0.4 / MSD-I §12) forbids "missingness with silence." A broadcast cell is a form of soft-missingness (a fabricated-fill with ~no independent information); giving it unit weight in the moment is the "missingness with silence" failure applied to the low-reliability tail. **Confirmed.**

### The one evidence error (in the critique's favor to correct, against it to note)

The critique's evidence bullet asserts `field.W` is *"only ever sliced/carried, never in a moment."* **That specific claim is factually false.** Two estimators *do* enter W into a moment:

- `spatial_field.py:42-61` (`_within_locality_stats`): the BYM varying-coefficient fit computes `A_s = Σ w·dX²`, `D_s = Σ w·dX·dY`, `c_s = Σ w` — genuinely reliability-weighted normal equations.
- `resolution.py:72-81` (`coarsen_field_spatial`): coarse cells are the reliability-weighted mean `Σ(w·Z)/Σw`.

This does **not** rescue the design, because both are *downstream/ancillary* readouts (effect-modification surface; coarse-pass aggregation), not the primary precision estimate. The edges themselves — the actual `link_records` — come from the unweighted `covariance.py` path. But it means the correct framing is sharper: **the weighting apparatus is applied inconsistently — honored in the spatial-field and coarsening secondary paths, ignored in the primary edge estimator** — which is arguably worse than "uniformly dead," because it means a coarse-pass edge and its fine-pass confirmation are weighted under two different measures.

### The only W-derived quantity reaching the primary fit — and why it doesn't count

`field_weights` (a *per-variable scalar*, e.g. an `n_eff`-derived reliability from the §3.12 state tensor) does thread through `assemble → run_ldo` (orchestrator.py:56-67, 428) and multiplies a whole variable's W column. But (a) it is a per-variable scalar, not per-cell, so it cannot distinguish an observed municipality from a broadcast one *within* the same variable; and (b) it only scales W, which the estimator then ignores anyway. So even this coarse hook is inert at the moment level. `n_state_weighted` in the diagnostics counts how many variables were scaled — reporting an apparatus that has no effect on the numbers.

### Severity assessment: HIGH is justified, with a scope caveat

The `risk_if_ignored` is real and quantitatively bounded by the broadcast fraction. In a context-heavy national panel, SIDRA socioeconomic covariates and any domain-scalar/geo-invariant field can be a **large** share of cells at W=0.25–0.5. Under equal weighting, the correlation is dominated by the *repeated fabricated-fill value* (a broadcast variable is literally constant across the broadcast axis), which manufactures spurious low-variance structure and inflates apparent associations — a textbook "fabricated-fill drives the edge" failure. The whitening path partly masks but does not fix this (whitening addresses spatial autocorrelation, orthogonal to reliability). 

**One honest mitigation the critique understates:** stability selection (edges.py) *does* provide a partial backstop — a broadcast-driven edge that is an artifact of one fabricated-fill pattern will often fail to recur across Place×Time subsamples and be demoted to `descriptive`. But this is downstream damage-control, not the honest-edge principle honored at source; a broadcast value that is *stably* constant will recur stably and survive. So stability does not close the gap.

### Correction / recommendation (endorsed, with a refinement)

The critique's fix is correct: enter `w_ab = √(w_a·w_b)` (or `min`) into the moments and derive a Kish effective sample size `n_eff = (Σw)²/Σw²` for the `min_coverage`/`min_overlap` gates and the Fisher-z SE. Two refinements:

1. **Do it in `whitened_lagged_correlation` too, not just `pairwise_correlation`** — the whitened path is the *default* (`spatial_whiten=True`), so a fix that only touches `pairwise_correlation` leaves the primary national path unweighted. In the whitened Gram, weight per cell as `Ft ← √W ⊙ Ft` before `Ft @ (Q @ Ft.T)` (a reliability-weighted whitened second moment), and accumulate `Σw`, `Σw²` per feature for the Kish `n_eff`.
2. **Feed the Kish `n_eff` into `edges.py:_n_eff`**, which currently counts `isfinite(Z).any(axis=0).sum()` — a raw cell count that treats a broadcast cell as a full observation. This is a *second, independent* place the same binary-vs-graded error surfaces: the power gate (`_MIN_N_EFF = 100`) and the promoted-edge Fisher-z SE (`1/√(n_eff−3)`) both over-count reliability. Fixing only the moment but not `_n_eff` would still report over-confident uncertainty on every promoted edge.

**Bottom line:** UPHELD. The reliability-weighting is genuinely absent from the primary edge estimator and from the power/SE gate, in violation of the §III.2 weighted-likelihood mandate and the §0.4 honest-edge principle. The critique's one factual overreach (W "never in a moment") should be corrected to "never in the *primary precision* moment, though inconsistently honored in the spatial-field and coarsening secondary paths" — which strengthens rather than weakens the finding.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (lines 48-109, 151-220 — the unweighted estimators), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\assemble.py` (51-60 — `_STATE_WEIGHT`), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (34 — W carried, unused), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (151-152 — `_n_eff` binary count), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` (109-117 — estimator dispatch); MSD mandate at `C:\Users\Galaxy\LEVI\PegaSUS\PEGASUS_MSD_III.md` §III.2 (line 197).

---

## LDO-MARG-06 [HIGH] — Gaussianization discards count-scale variance the denominator principle exists to preserve

This is now fully confirmed. `fit_lagged_links` reads only `field.Z` and the mask; `field.W` is carried on the `GaussianField` and sliced along, but it is **never** passed into the correlation estimator or the ADMM fit. The estimator sees a 0/1 observed-mask, not the count-based precision weight.

I have everything I need. Let me verify one loose end the critique raises about heteroscedasticity: whether the count-exposure PIT genuinely equalizes marginal variance regardless of count magnitude (the "2 deaths vs 2000 deaths → same Z" claim), which is the mathematical heart.

The randomized PIT gives `u_i` marginally Uniform(0,1) *exactly* for any margin (that is the defining property of the PIT, and randomization makes it hold for discrete distributions too). Therefore `z_i = Φ⁻¹(u_i)` is *exactly* marginally N(0,1) for every cell, regardless of `μ_i`. This is provably true and is the intended behavior. So the marginal-variance-equalization the critique describes is real and by construction.

Now I can render the verdict.

---

**VERDICT: PARTIAL (leaning UPHELD on the diagnosis, but the framing over-claims and the "false precision" mechanism is misattributed).**

The critique correctly identifies a real, load-bearing gap, but it mislabels *which* variance is destroyed and *where* the resulting error actually bites. Let me separate the true core from the overreach.

## What the critique gets RIGHT (upheld)

1. **The PIT provably equalizes marginal variance across cells.** By the randomized-PIT construction in `count_exposure_gaussianize` (margins.py:113–114), `u_i = F(x_i-1;μ_i) + U·p(x_i;μ_i)` is *exactly* Uniform(0,1) for every cell, so `z_i = Φ⁻¹(u_i)` is *exactly* marginally N(0,1) regardless of `μ_i = λ·E_i`. A cell with `μ=2` and one with `μ=2000` both produce unit-variance latent Gaussians. The critique's "2 deaths and 2000 deaths map to nearly the same Z at the same standardized-residual sign" is mathematically correct. This is intrinsic to *any* copula/Gaussianization margin, not a bug in this implementation.

2. **W (the count/reliability precision weight) never reaches the estimator.** Verified directly: `pairwise_correlation` (covariance.py:48–109) and `whitened_lagged_correlation` (covariance.py:151–220) consume only `Z` and a **0/1 finite-mask** `M`. `fit_lagged_links` (lags.py:114–117) passes `field.Z` only; `field.W` is carried on `GaussianField` and sliced through `_subfit` (edges.py:46) but is **dead weight** — it enters no moment product and no ADMM penalty. So a noisy 2-count cell and a precise 2000-count cell contribute to `M@Mᵀ`, `X0@X0ᵀ` with **identical unit weight**. The LDO-MARG-05 premise ("W is the intended vehicle and it is unused") is factually correct in the current code.

3. **The consequence — rare-event municipal cells given false precision — is real.** Low-count cells have genuinely noisier rate estimates (CV = 1/√Y per §3.12.9), and nothing in the Z-scale or the estimator down-weights them. Correlation entries are effectively count-unweighted. This *does* contradict the §II.3 promise as literally worded ("loses the count-variance").

## Where the critique OVERREACHES (correction)

**The "offset preserves the mean but not the variance" framing is imprecise — the count-exposure margin *does* inject count-magnitude information into the mean/location of Z, and that is not nothing.** The critique concedes this ("honors it for the MEAN") but then treats it as cosmetic. It is not. Under the Poisson-with-offset PIT, the *quantization granularity* of `u` carries count information: for a small `μ`, the PMF steps `p(x;μ)` are coarse, so the randomization band is *wide* and `z_i` is dominated by the injected Uniform noise `U` — i.e. a 2-count cell's Z is *deliberately* made noisy/diffuse by construction. For large `μ`, the band is narrow and `z_i` tracks the standardized deviation tightly. So the heteroscedasticity is **not fully annihilated**: it survives as *reduced information content per low-count cell*, in the form of PIT-randomization noise, not as marginal variance. The critique's claim that count-precision is "ANNIHILATED at Layer 0" is too strong — it is *attenuated and converted into added latent noise*, which is a milder (but still real) defect.

The distinction matters for the fix. The critique's own recommendation is right in spirit but the cleanest correction is not "keep an Anscombe representation alongside" (which reintroduces the materialized-transform the denominator principle forbids). It is:

- **Carry a per-cell Fisher weight `w_i ∝ μ_i` (≈ `1/CV²` ≈ `Y`) into the moment accumulation** so `M` in covariance.py:80–86 becomes a *weighted* mask (`M_w = w ⊙ finite`), making `n_ab`, `Sx`, `Cxy` count-information-weighted. This is a ~5-line change localized to `pairwise_correlation` / `whitened_lagged_correlation`, and it is exactly the W that is already plumbed to the estimator boundary but dropped. That closes LDO-MARG-05 and LDO-MARG-06 with one weld.

## Net

- The **mechanism** (PIT forces unit marginal variance; W is unused; low-count cells get unwarranted influence) is **confirmed in code** — margins.py:113, covariance.py:80–96, lags.py:114–117.
- The **normative contradiction** with §II.3/I.2 (lines 63, 132) is **real**: the denominator principle's stated purpose is to "supply the correct count-variance," and the estimator currently discards it.
- The critique **overstates** by saying the count information is fully annihilated — the randomized offset-PIT retains it as location + randomization-noise granularity, so the failure is "false precision for low-count cells," not "total loss." Severity HIGH is defensible but should be reworded from "annihilated" to "not carried into cell influence."

Sharpened recommendation: **weight the covariance moments by a count-derived Fisher weight `w_i ≈ μ_i` (the already-plumbed but unused W), applied inside `pairwise_correlation`/`whitened_lagged_correlation`, rather than adding a parallel variance-stabilized representation** — the latter would reintroduce a materialized transform the denominator principle exists to avoid.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (lines 88–115, 118–144), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (lines 48–109, 151–220), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` (lines 108–139), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (line 46); MSD refs `PEGASUS_MSD_III.md` lines 63, 132.

---

## LDO-MARG-07 [HIGH] — Gaussian copula imposes tail-independence that epidemic threshold/clustering dependence violates

I have everything needed. The spatial varying-coefficient field is a *linear* within-locality slope `Y ≈ α_s + β_s·X`, smoothed spatially — it models effect-modification-by-place, not a threshold or upper-tail-dependence structure. My assessment is complete.

## Verdict: PARTIAL (leaning UPHELD on the core mechanism; the "systematically under-estimated / attenuated toward the bulk" harm claim is overstated but the theoretical gap is real)

### What the critique gets exactly right

**1. The mechanism claim is factually correct.** `margins.py` maps every variable to a latent normal via rank/PIT (`randomized_pit_gaussianize`, or a Poisson-offset PIT for exposure counts), and `precision.py` then fits `sklearn.covariance.graphical_lasso` on the resulting Gaussian-scale correlation. That is definitionally a **Gaussian copula**: a single second-moment object (Σ, or its sparse inverse Ω) carries *all* the dependence on the latent scale. The Gaussian copula's coefficient of upper-tail dependence is
$$\lambda_U=\lim_{q\to1}P(U_1>q\mid U_2>q)=0 \quad\text{for all } \rho<1,$$
and its dependence is radially symmetric and monotone. So the critique's core mathematical statement — the LDO's primary graph cannot represent asymptotic tail dependence or co-presence/co-absence asymmetry — is **true of the design, not just the code**. §III.5 line 222 ("dependence is modeled there") is the exact Gaussian-copula commitment.

**2. The "HSIC is detector-not-engine" backstop is correctly characterized as insufficient for the primary graph.** I confirmed the orchestration order in `run_ldo`: fit Gaussian precision → `to_link_records` → `scan_residual_nonlinear_edges` **appends** `nonlinear_residual` LinkRecords → certify. HSIC-flagged pairs are emitted as a *separate typed edge downstream*; they are **never promoted into a named threshold/interaction term that re-enters `Ω_var` estimation**. The primary precision graph — which drives stability selection, certification, the causal ladder, and the Kronecker joint operator — remains pure Gaussian-copula. So recommendation (ii) ("promote flagged pairs to explicit terms BEFORE the final graph, not merely as annotations") targets a real seam. §III.6 line 226 ("flags candidates for promotion... it never *is* the model") states this by design.

**3. The interaction/varying-coefficient term does not close the gap.** §III.4(5)/§III.3 name "interaction (effect-modification) terms... as varying-coefficient structure," and `spatial_field.py` implements it — but as a **linear** within-locality slope `Y ≈ α_s + β_s·X` with a spatial-smoothness prior. That models *where* an edge is stronger, not a *temperature/rainfall threshold cutoff* on transmission, and it is fit only *post hoc for already-selected edges* as a sidecar. It is not a threshold basis feeding the graph. So the spec's "named nonlinear terms" inventory (copula-marginal + varying-coefficient + lags/seasonality) genuinely lacks a threshold/joint-tail primitive.

### Where the critique overreaches (why PARTIAL, not UPHELD)

**A. "Attenuated toward the bulk / systematically under-estimated" is only half-right, and conflates two different objects.** The LDO does not fit a Pearson correlation on raw counts — it fits graphical-lasso on **rank-PIT-normal-scored** data. Because the PIT is a monotone transform, the Gaussian-copula ρ it recovers is (a rank-based estimate of) the **normal-scores / van-der-Waerden correlation**, which equals `2·sin(πρ_S/6)`-type functions of Spearman's ρ_S. So the edge weight tracks the *whole-support monotone concordance*, not a linear Pearson slope that a threshold would flatten. A Zika→microcephaly relationship that is strong across the co-outbreak support is captured with roughly full strength; it is **not** "attenuated toward a weak linear partial correlation." The specific harm claim is precise only for dependence that is *concentrated in the joint tail and near-absent in the bulk* (λ_U > 0 with weak central concordance) — genuine threshold-triggered co-explosions. For those, yes, a single ρ under-weights the conditional-on-both-extreme co-occurrence. So: the attenuation is **real but confined to tail-concentrated dependence**, not the blanket under-estimation the "risk_if_ignored" asserts.

**B. The lag layer partially rescues the *epidemiologically load-bearing* case.** The win-condition edge (Zika→microcephaly, arbovirus co-circulation) is **lagged and directed** — it lives in `lags.py`'s cross-lag precision, where the distributed-lag response profile over k is read as the edge. A joint outbreak that co-explodes and then co-decays is a strong *lagged concordance*, which the rank-based lagged precision detects with high stability (the residual-scan `nonlinear_residual` type and the temporal-holdout persistence gate further protect it). The critique's headline example is therefore among the *better*-served cases, not the worst. The genuinely under-served case is **contemporaneous same-season joint upper-tail co-occurrence of two rare diseases with weak off-peak concordance** — narrower than "the signature epidemiological signals."

**C. Co-absence asymmetry is already handled, for a different reason.** The critique says "co-absence of two rare diseases is uninformative, co-presence is informative," implying the Gaussian copula mistreats it. But the **randomized PIT on zero-inflated counts** (documented in `margins.py`) spreads the tied zero-mass *uniformly across its CDF band* rather than collapsing all co-zeros onto one latent point. This deliberately prevents the mass of co-absent cells from manufacturing spurious latent concordance — it neutralizes the co-absence problem at the margin, before the copula sees it. The copula still can't *reward* co-presence asymmetrically (that needs an asymmetric/vine copula), but the "co-absence pollutes the estimate" half of the concern is already mitigated.

### Sharpened, corrected recommendation

The defensible, minimal fix is **not** a t-copula globally (recommendation (i) as stated is weak: a t-copula adds *symmetric* tail dependence with one d.o.f., which does not capture the asymmetry the critique itself emphasizes, and a single global ν is a poor fit to a mixed national panel). The tighter version:

- **Report a tail-dependence diagnostic per certified edge** (recommendation iii) — cheap, honest, and consistent with the LDO's existing "coverage manifest / descriptive-vs-selected" honesty layer. A nonparametric upper-tail-dependence estimate (e.g. `λ̂_U` from the empirical copula at high quantiles) computed on the *residual-scan candidate pairs* would flag exactly the pairs where the Gaussian weight is a lower bound. This is the change that fits the architecture with least violence.
- **Escalate HSIC-flagged pairs into an explicit threshold/interaction basis that re-enters the precision fit** (recommendation ii) — this is the real structural gap and the correct long-term fix, but it is a genuine new modeling layer, not a one-line augmentation.
- **At minimum, document in §III.5 that contemporaneous Gaussian-copula edge weights are *lower bounds* for tail-concentrated pairs** — the "at minimum" clause is correct and currently absent from the spec.

### Bottom line

The critique correctly identifies a **real theoretical limitation of the MSD design**: the primary discovery graph is Gaussian-copula (λ_U = 0, symmetric), HSIC only annotates rather than promotes, and no threshold/tail primitive feeds the graph. That much is **UPHELD**. But its damage model is overstated — the LDO fits *rank/PIT-normal* dependence (not attenuation-prone linear Pearson), the flagship Zika/arbovirus edges are *lagged* and among the best-served, and zero-inflation co-absence is already neutralized at the margin. The residual risk is specifically **contemporaneous, tail-concentrated joint-extreme co-occurrence with weak bulk concordance** — a narrower but genuine blind spot. Severity is better rated **MEDIUM** than HIGH, and the correct remediation is a per-edge tail-dependence diagnostic + a documented lower-bound caveat, with HSIC-to-threshold promotion as the real (larger) fix — not a global t-copula.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\precision.py`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\residual_scan.py`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\spatial_field.py`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py` (order at lines 195–332), `C:\Users\Galaxy\LEVI\PegaSUS\PEGASUS_MSD_III.md` §III.5 (line 222) / §III.6 (line 226).

---

## LDO-MARG-08 [HIGH] — Exposure denominator treated as a known constant despite being a reconstructed uncertain tensor

The audit only ever treats exposure as a known offset — the denominator-uncertainty term is never mentioned. This is a genuine gap, not a deferred-and-documented one. I have everything I need.

Let me verify one more claim: that the margin uses a single global `lam` (pooled MLE), which matters for the spatial-bias argument.

The code confirms `lam = x.sum() / total_e` (a single pooled rate across all cells), and `mu = lam * e`. So the standardized residual `z_i` measures each cell's deviation from the pooled-rate expectation, and `Var(E)` enters nowhere.

## VERDICT: UPHELD (with one refinement that sharpens, not weakens, it)

**The critique is mathematically correct on its central claim.** `count_exposure_gaussianize` (margins.py:88-115) computes `mu = lam * e` and forms the Poisson-offset PIT `u_i = F(x_i-1; mu_i) + U·p(x_i; mu_i)` treating `e` (the exposure/denominator) as an error-free constant. The population tensor that supplies `e` is, per MSD-III §II.4 (lines 149, 151), an explicitly *reconstructed, uncertainty-typed* asset whose per-cell error grows monotonically with anchor distance. That uncertainty is dropped at the margin. I confirmed end-to-end that:

1. **Exposure enters as point values only.** `assemble.py` builds `exposure` as a bare `(p,S,T)` float tensor (lines 182-200); there is no companion `sigma_E` / `cell_uncertainty` channel. The §II.4 per-cell uncertainty asset exists upstream but is never transported to the margin.
2. **The margin variance is understated.** The PIT calibrates `z_i` to `Var(count | E) = mu_i` (Poisson). The correct predictive variance of the count given an *uncertain* log-offset is a compound Poisson-lognormal, `Var ≈ mu_i + mu_i^2 · sigma_{logE,i}^2` (equivalently `mu + lambda^2 Var(E)` to first order). Feeding `z_i` through a CDF built on the smaller variance makes `|z_i|` systematically too large on projected/intercensal cells — false precision, exactly as claimed. (Minor correction to the critique's algebra: the deflation term is `mu^2 σ²_logE`, i.e. it scales with `mu²`, not the critique's `lambda² Var(E)` written as an additive count term — same idea, but the compound-Poisson-lognormal form is the precise one, and it means the deflation is worst for *high-count* cells on uncertain denominators, not uniform.)
3. **The spatial-bias mechanism is real and is the serious half.** Because `lam` is a single **pooled** MLE (`x.sum()/total_e`), `z_i` is literally "this cell's deviation from the national-average rate implied by its denominator." A mis-vintaged IBGE projection that biases a whole region's `E` in the same direction (the precise §II.4(147) anti-discontinuity failure — the 2022 census enumerating ~10M fewer than the 2021 projection) shifts every `mu_i` in that region coherently, producing a spatially-coherent shift in `Z`. The spatial-GMRF whitening step (`κI + L_W`, task #34, now live) is designed to read residual spatial covariance as epidemiological structure — it has no way to distinguish denominator-vintage bias from genuine spatial dependence. So the error is not merely inflated confidence; it *manufactures* the spurious spatial trend §II.4 was written to prevent, and routes it into the causal/spatial layer. This is the correct and important escalation, and it is why HIGH severity is justified.

**Not addressed by any existing mechanism.** The `W` reliability weight *does* encode provenance state (`_STATE_WEIGHT`: `projected`→0.5, `reconstructed`→0.5, `bounded`→0.4 in assemble.py:50-60), which is a coarse proxy for denominator uncertainty — but `W` is propagated only as a field attribute and is **not consumed inside the margin variance** (the margin takes only `counts`, `exposure`, `rng`). The completeness audit (WP3 / BLO-9/10) wired count-with-exposure specifically as "the fix is wiring, not new math," and its requirement text (audit lines 154, 263) frames exposure purely as a known offset. So the denominator-*uncertainty* term was never in scope — this is a true residual gap, not a documented-and-deferred one.

**Where the critique slightly over-reaches:** it is only an issue for variables actually routed through `count_exposure_gaussianize` (extensive counts with a positive exposure tensor); intensive/rank-PIT variables are untouched. And when the denominator is a *census-year* cell, `sigma_E ≈ 0` and the current margin is essentially correct — the defect is concentrated exactly on intercensal/extrapolated cells, which is consistent with (and narrows) the critique's own framing.

**Recommended fix — the critique's option (a) is the right default, sharpened:** deflate `z_i` by the compound variance. Concretely, transport a per-cell `sigma_logE` (derivable from the §II.4 anchor-distance envelope already computed by `FAL-POP-PROJ`, task #18) alongside `exposure`, and replace the Poisson PIT with a Poisson-lognormal (negative-binomial-like) PIT whose variance is `mu(1 + mu·(e^{σ²_logE}-1))`. Option (b) — resampling `E` from its FAL-POP-PROJ posterior inside a multiple-randomization loop — is more faithful (it also captures spatial correlation structure of `E`, which a per-cell scalar `sigma_E` cannot) but costs a posterior draw per LDO replicate. For the spatial-bias half specifically, only option (b) or an explicit spatially-correlated `Var(E)` term fully neutralizes the GMRF misread; the scalar-`sigma_E` deflation fixes the false-precision half but still leaves coherent regional denominator bias in the *mean* of `Z`. That distinction is worth recording: **variance deflation alone does not remove the spatial bias — it only stops the whitening from being over-confident about it.**

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (lines 88-115, 133-134), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\assemble.py` (lines 50-60, 174-207), `C:\Users\Galaxy\LEVI\PegaSUS\PEGASUS_MSD_III.md` (§II.4 lines 147-151, §III.5 line 222).

---

## LDO-PSD-01 [HIGH] — Pairwise-complete correlation is provably indefinite; the MSD treats it as a valid covariance

I now have the complete data flow. Let me verify my mathematical claim about `_nearest_correlation` producing a genuinely PSD matrix, and note the one subtle gap: the diagnostic reporting the critique asks for.

The data flow is confirmed:
1. `pairwise_correlation` / `whitened_lagged_correlation` build the raw pairwise matrix `C` (indefinite, as the critique says).
2. Both return `_nearest_correlation(C)` — an eigen-clip to `floor=1e-3`, rescaled to unit diagonal.
3. `lags.py:137` passes `pw.correlation` (the projected matrix) into `fit_sparse_plus_lowrank`.
4. `lowrank.py:168` adds another `+1e-4·I` ridge before the ADMM logdet.

So the objective never sees the raw indefinite matrix. Let me write the verdict.

---

## VERDICT: PARTIAL (leaning REFUTED on the core claim; UPHELD on one narrow diagnostic gap)

### The critique's mathematical premise is correct...
The pairwise-complete correlation matrix assembled at `covariance.py:83–100` **is** generically indefinite. This is not in dispute and the code's own docstring says so. Each entry `corr[a,b]` is built from `n_ab = M @ M.T` — the overlap of `a` and `b`'s observed cells — with `mean_a`/`mean_b` computed on *different* subsamples (line 90 `mean_a` is a's mean over overlap-with-b; line 91 `mean_b` is b's mean over overlap-with-a). Wothke (1993) / Lounici (2014) apply: this matrix is not the Gram matrix of any single dataset and can carry negative eigenvalues that are structural (varying SIM/SIH/SIA overlap patterns), not roundoff. That part of the critique is right.

### ...but its central factual claim — "the MSD treats it as a valid covariance" and "feeds an indefinite C into the log-likelihood" — is FALSE at the level of the actual code.

The critique's own evidence pointer (`covariance.py:83–100`) stops **five lines short of the operative code**. The raw matrix `C` never leaves the function. Both estimator entry points return the matrix **after** projection:

- `covariance.py:105` — `correlation=_nearest_correlation(C)`
- `covariance.py:216` — `correlation=_nearest_correlation(corr)` (whitened path)

`_nearest_correlation` (lines 38–45) does exactly the eigen-repair the critique says is missing:
```python
vals, vecs = np.linalg.eigh(C)
vals = np.clip(vals, floor, None)   # floor = 1e-3, strictly positive
psd = (vecs * vals) @ vecs.T
return psd / np.outer(d, d)          # rescale to unit diagonal
```
This clips **to a strictly positive floor `1e-3`, not to zero** — so the returned matrix is positive *definite* with `λ_min ≥ 1e-3/max_scale`, not merely PSD. The consumer (`lags.py:137`) passes `pw.correlation` — the projected matrix — into `fit_sparse_plus_lowrank`, which at `lowrank.py:168` adds a **further** ridge: `C = 0.5*(emp_cov+emp_cov.T) + 1e-4*I`.

**Consequence for the objective the critique invokes.** The critique's collapse argument is: an indefinite `C` makes `tr((S−L)C)` unbounded below along `C`'s negative eigenspace, so `−logdet(S−L)+tr((S−L)C)` is non-convex/unbounded. But the `C` that reaches the objective has **`λ_min > 0` by construction**. For a strictly-PD `C`, the Gaussian term `−logdet(Θ)+tr(ΘC)` (with `Θ=S−L≻0` enforced by the logdet barrier and the PSD projection in the L-step) is the *standard* graphical-lasso likelihood: jointly convex in `Θ` and **bounded below**, minimized at `Θ=C⁻¹`. There is no negative-eigenspace direction to run the objective to `−∞`, because there is no negative eigenvalue. The unboundedness the critique describes is a real pathology of the *raw* pairwise matrix — which is precisely why the code projects it before the objective ever sees it. The design does not "silently assume" PSD; it *enforces* PD.

So recommendation (a)/(b) — "estimate a jointly-PSD C" — is aimed at a failure mode the code has already closed. The nearest-correlation projection is one legitimate member of the family the critique itself lists ("do not paper over with a one-shot eigen-clip" — but a Higham-style nearest-correlation projection *is* the principled repair, not a naive clamp; the only quibble is that this is a single-pass eigen-clip rather than the full alternating-projection Higham 2002 algorithm, which matters only when a unit-diagonal constraint conflicts with PSD — a second-order concern here).

### Where the critique lands a real hit (the UPHELD sliver)

The critique's **last sentence of the recommendation is not implemented**:
> "Report the pre-projection minimum eigenvalue and the Frobenius distance moved by projection as a data-quality diagnostic — a large move should abort the fit."

This is correct and currently missing. `_nearest_correlation` throws away `vals` before clipping and returns nothing about how far it moved. A grep for `min_eig`/`Frobenius`/`distance moved` in `ldo/` finds nothing. The genuine residual risk is **not** the one the critique headlines (a fabricated objective) but a quieter one: if the raw pairwise matrix is *badly* indefinite (large-magnitude negative eigenvalues from near-disjoint SIM/SIH/SIA overlap), the projection can move `C` a large Frobenius distance to a well-conditioned surrogate that differs materially from any real covariance — and the current code has **no gate that notices**. The projected matrix always looks clean to the downstream ADMM; the data-quality signal (how much repair was needed) is silently discarded. That is a legitimate honesty-layer gap consistent with the MSD's own anti-silence invariant (§II.11) and its abort discipline (§II.6.4).

### Correction / recommended fix (narrowed)
Do **not** re-architect the estimator (EM/IPF/matrix-completion) — the PSD-feeding claim is refuted and that work would solve a non-problem. Instead, make `_nearest_correlation` return `(projected, min_eig_pre, frob_shift_rel)` where `frob_shift_rel = ‖proj−C‖_F / ‖C‖_F`, surface both on `PairwiseCovariance`, and add a §II.6.4-style abort `pairwise_covariance_indefiniteness_exceeds_threshold` (e.g. trip when `min_eig_pre < −τ` or `frob_shift_rel > 0.1`) rather than silently fitting. This converts the critique's valid concern from "the math is broken" (it isn't) to "the repair is unmonitored" (it is), which is the accurate and actionable framing.

**Files:** `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (`_nearest_correlation` L38–45; return sites L105, L216), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py` L168 (second ridge), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` L137 (consumer), `C:\Users\Galaxy\LEVI\PegaSUS\PEGASUS_MSD_II.md` §II.6.2 L247.

---

## LDO-NCORR-02 [HIGH] — '_nearest_correlation' is eigen-clip+rescale, NOT the metric-nearest correlation matrix (Higham)

I now have everything needed to adjudicate both defects. Let me verify the two mathematical claims precisely.

**Defect (1): does the final `psd / outer(d,d)` rescale reintroduce indefiniteness?**

The rescale computes `R = D⁻¹ · psd · D⁻¹` where `D = diag(sqrt(diag(psd)))`. Since `d_i = sqrt(psd_ii) > 0` (clipped to ≥1e-6), D is a real, invertible diagonal matrix. This is a **symmetric congruence transform** `R = Dᶜ psd Dᶜ` with `Dᶜ = D⁻¹`. By **Sylvester's law of inertia**, a congruence by a nonsingular matrix preserves the inertia (# of positive/negative/zero eigenvalues) exactly. Since `psd` after eigen-clip to floor=1e-3 has all eigenvalues ≥ 1e-3 > 0 (strictly PD), R is strictly PD. The critique's claim that "rescaling by outer(d,d)... can reintroduce indefiniteness because it is a congruence that changes the spectrum" is **mathematically false**: congruence changes the spectrum but *not the inertia*, and PD-ness is an inertia property. So defect (1)'s PSD-safety claim is refuted.

The "single step of alternating projection doesn't converge to the nearest correlation matrix" part is **literally true** — this is not Higham's projection. But that's a naming/optimality issue, not a validity issue, because the output IS a valid PD correlation matrix (unit diagonal by construction, PD by the inertia argument). It just isn't the *metric-nearest* one.

**Defect (2): the absolute floor=1e-3 biases small eigenvalues → distorts precision/edges.**

This is the substantive claim. Let me check whether downstream actually inverts this correlation and whether the input is already ~unit-scaled.

The input `C` is a **correlation** matrix already (built at lines 96–101: each entry is `cov/sqrt(var_a·var_b)`, diagonal forced to 1.0). So its eigenvalues sum to trace = q (number of variables), with λ_max ≤ q and typically O(1). A floor of 1e-3 on a matrix whose eigenvalues live on an O(1) scale is a *relative* floor of ~1e-3/λ_max, not an arbitrary absolute one. The critique's framing ("applied to RAW eigenvalues, not a relative scale... for a near-degenerate block it inflates small eigenvalues arbitrarily") **misreads the input as a general covariance**. For a correlation matrix it is effectively already relative. So the "inflates arbitrarily" concern is much weaker than stated.

**But** — the downstream consumers genuinely do invert it:
- `precision.py:121-123`: adds `1e-4·I` then runs `graphical_lasso` (or `pinv`) → precision.
- `lags.py:137`: feeds to `fit_sparse_plus_lowrank` (ADMM) → precision.

So the smallest eigenvalues *do* dominate the precision. The floor at 1e-3 sets a condition-number ceiling of ~q/1e-3 ≈ 1e3·q, which is a real (if mild) regularization. Combined with the *additional* `+1e-4·I` at precision.py:121 and the graphical-lasso `alpha`, the near-null directions are further regularized before inversion. The clip is therefore not the sole determiner of the precision's ill-conditioned directions. The distortion is real but bounded and partially redundant with existing regularizers — not the "ad-hoc spectral surgery concentrating distortion exactly on the edges" catastrophe the critique paints.

Here is my verdict.

---

## Verdict: **PARTIAL** (leaning REFUTED on the dangerous claims, UPHELD on the naming/hygiene)

The critique correctly identifies that `_nearest_correlation` is **not** Higham (2002) — it is one eigen-clip + diagonal rescale, and the name/docstring overclaim. That much is UPHELD and worth fixing as truth-in-labeling. But the two *substantive* defects it uses to justify HIGH severity are mathematically wrong or overstated.

### Defect (1) — "rescale can reintroduce indefiniteness" — **REFUTED**
The final step is `R = D⁻¹ · psd · D⁻¹` with `D = diag(√diag psd)`, a real invertible diagonal. This is a **symmetric congruence**, and by **Sylvester's law of inertia** congruence preserves inertia exactly. After the eigen-clip every eigenvalue of `psd` is ≥ floor = 1e-3 > 0, so `psd` is strictly PD, hence `R` is strictly PD. The critique's own words — "congruence... changes the spectrum" — are true but irrelevant: PD-ness is an *inertia* property, not a spectral-magnitude one. The result is a valid unit-diagonal PD correlation matrix. There is no indefiniteness risk. (The only way to break this is if a diagonal of `psd` were ≤0, impossible for a PD matrix, and it's floored at 1e-6 regardless.)

The "one projection step ≠ intersection of the two cones" observation is *true* but harmless: the output still satisfies **both** constraints (unit diagonal by the rescale, PSD by inertia). It simply isn't the *metric-nearest* such matrix. For a downstream that only needs "a valid PD correlation to invert," minimal Frobenius distance is not required — so this is a cosmetic gap, not a validity gap.

### Defect (2) — "absolute floor biases the precision/edges" — **PARTIAL, mostly overstated**
The critique's core error is treating the input as a general covariance. It is not: `_nearest_correlation` is only ever called on an **already-constructed correlation matrix** (covariance.py:96–101 divides by `√(var_a·var_b)` and forces unit diagonal; whitened path does the same at 212–214). Its eigenvalues sum to `trace = q` and sit on an O(1) scale, so `floor = 1e-3` acts as an *effectively relative* floor of ~1e-3/λ_max — a condition-number cap of ~10³·q, not "arbitrary inflation." The premise "applied to RAW eigenvalues, not a relative scale" is factually incorrect for the only inputs this function receives.

What *is* fair: the smallest eigenvalues do dominate the inverted precision, and a hard floor is a form of regularization that shapes the near-null directions. But (a) it's a bounded, well-behaved cap, and (b) it is **not the operative regularizer** — `precision.py:121` adds `+1e-4·I` and runs `graphical_lasso(alpha)`; `lags.py:137` runs sparse+low-rank ADMM with `lambda1/lambda2`. The edges are governed by those explicit penalties, not by whether the correlation floor was 1e-3 vs 1e-4·λ_max. So "distorts the scientific output" / "biases exactly the edges" is not demonstrated; the effect is second-order and largely absorbed downstream.

### The one correct, worth-doing recommendation
Make the floor relative and record the achieved minimum eigenvalue / projection move — cheap, honest, and removes the last bit of scale-dependence:

```python
def _nearest_correlation(C, *, rel_floor=1e-6):
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    floor = rel_floor * max(vals.max(), 1.0)
    clipped = np.clip(vals, floor, None)
    psd = (vecs * clipped) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    return psd / np.outer(d, d)   # PD preserved by Sylvester inertia
```

Swapping in a true Higham iteration (statsmodels `corr_nearest`) is optional polish, not a correctness fix — the current output is already a valid PD correlation matrix, and full national LDO runs invert it thousands of times per stability resample, where an iterative Newton/Dykstra loop is a real cost for no validity gain.

### Net
- **Naming/docstring overclaim ("nearest ... Higham"):** UPHELD — fix the docstring, it's misleading.
- **Indefiniteness-after-rescale (defect 1):** REFUTED — Sylvester's law of inertia guarantees PD is preserved.
- **Absolute-floor biases edges (defect 2):** PARTIAL/overstated — the input is already a correlation matrix (effectively relative scale), and the actual edge regularization lives in graphical-lasso/ADMM penalties + the `+1e-4·I`, not this clip.
- **Severity:** should be **LOW** (truth-in-labeling + a one-line relative-floor hardening), not HIGH. There is no demonstrated scientific-output distortion.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (lines 38-45 the function; 96-101 and 212-214 confirm the input is a correlation matrix), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\precision.py:121-125` and `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py:137` (the inverting consumers and their independent regularizers).

---

## LDO-TIME-03 [CRITICAL] — Temporal autocorrelation is never whitened before the fit; Σ_time⁻¹ is post-hoc telemetry only

I now have everything needed. My assessment:

---

**VERDICT: UPHELD (with two sharpenings and one important qualification the critique understates).**

**The factual claims are all correct — verified in code:**

1. **Only space is whitened.** `fit_lagged_links` (lags.py:109-114) builds `Q_space = κI+L_W` and calls `whitened_lagged_correlation(field.Z, Q_space, K, …)`. That routine (covariance.py:151-220) computes the Gram matrix `G += Ft @ (Q_sparse @ Ft.T)` — a quadratic form in the **spatial** metric only. There is no temporal metric anywhere in the estimator. The lag-extended features are stacked (`_build_lagged_feature_matrix`, lags.py:48-67) and the resulting `n_cells = S·T_eff` columns are correlated as if exchangeable.

2. **Σ_time⁻¹ is telemetry-only, in a swallowing try/except.** kron.py's `KroneckerPrecision` is instantiated exactly once, at orchestrator.py:400 inside `try/… except: kronecker_report = None` (lines 385-407). Its `joint_logdet` feeds a report dict; `_op.matvec/solve/logdet` never touch the CPW correlation, ADMM, stability, or edge SEs. Even the data-estimated `_phi` (orch:391-399) is used *only* to build the AR(1) factor for that log-det — it never deflates any variance. kron.py's own module docstring (lines 26-31) explicitly states the operator is "deliberately NOT swapped into the CPW ADMM R-step." So the identity `Prec(Z) ≈ Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹` is claimed but only two of the three factors act on the estimate.

3. **The iid effective-n is uncorrected.** edges.py:151-176: `n_eff` is the raw count of observed cells (`isfinite(Z).any(axis=0).sum()` ≈ S·T_eff), and `stat_se = 1/√(n_eff−3)` is the textbook Fisher-z SE **for iid samples**. There is no `√(1−φ²)` (or block-bootstrap, or AR-adjusted DOF) deflation. So autocorrelation inflates the reported precision of every partial correlation, and this feeds the certification conjunction (stability AND uncertainty) directly.

**The mathematics is right.** For an AR(1) series with lag-1 autocorrelation φ, the variance of the sample mean/correlation scales by the integrated autocorrelation time ≈ (1+φ)/(1−φ), so the honest effective sample size is n·(1−φ)/(1+φ), not n. Treating n serially-dependent draws as iid understates SE by roughly √((1+φ)/(1−φ)). For Brazilian secular-trend series (epidemiological transition, SUS coverage expansion) φ→1 makes this severe, and Yule's nonsense-correlation says two independent trending series show large *marginal* correlation. This is a real identity gap.

**Two sharpenings (the critique is if anything too kind to the code):**
- The stacked-window design makes the dependence worse than plain AR: adjacent sample columns `(s,t)` and `(s,t+1)` **share K of their K+1 lag entries** by construction, so consecutive "samples" are mechanically near-duplicates, not merely correlated. The overlap is deterministic, not just stochastic.
- The stability layer does **not** rescue this. `stability_select` (edges.py:90-97) perturbs *contiguous* time windows (`np.arange(start, start+L)`), so every subsample is itself a block of serially-correlated points fit under the same iid estimator. Subsampling contiguous blocks resamples the *trend*, so a spurious trend-driven edge is stable across subsamples — stability selection will *confirm* the nonsense correlation, not filter it. The critique's "manufactures spurious distributed-lag curves" is correct and the stability gate is not a mitigant.

**The one qualification the critique understates (why PARTIAL is tempting but UPHELD is right):** the sparse-plus-low-rank split gives *partial* protection the critique dismisses too fast. The shared low-rank `L` component (lags.py, `fit_sparse_plus_lowrank`) is designed to absorb a *common* driver — and a shared secular trend across variables is exactly a rank-1 common factor. To the extent the epidemiological-transition trend is *common* across the paired series, `L` can soak it into `latent_shared` rather than a sparse lagged edge. This blunts the *cross-series* Yule problem for co-trending variables. But it does **not** fix the effective-n inflation (the SE is still computed as iid on whatever survives into `S`), and it does nothing for the *within-series* autocorrelation that inflates a genuine variable's own lagged autocovariance — precisely the "discover Zika's 6-9 month lag" self-driven distributed-lag curve. Since `L` addresses only one of the two failure modes and the SE/certification math is unconditionally wrong, the critique holds.

**Correction / recommended fix (endorsing the critique's, made precise):**
- The cheap, correct fix is not necessarily to build the full temporal whitener into the Gram accumulation (though that is the principled route: replace `Ft @ (Q_space @ Ft.T)` with a form carrying `Q_time^{1/2}` along the T axis so the correlation is of `Σ_space^{−1/2} Z Σ_time^{−1/2}`). The *minimum* honest fix is to stop reporting iid `n_eff`: compute an autocorrelation-corrected effective sample size `n_eff_ac = n_eff·(1−φ̂)/(1+φ̂)` using the φ̂ **already estimated** at orchestrator.py:391-399 (it is computed and then thrown into telemetry — wire it into `stat_se` instead), and propagate that into `edge_uncertainty` and the `_MIN_N_EFF` low-power gate. That alone converts the manufactured significance into honest wide intervals.
- Separately, the stability layer should use a **block/circular-block** time resample rather than a single contiguous window if it is to have any power against trend-driven spurious edges.
- And the report should not present the Kronecker log-det as characterizing the fit's operator, since the fit did not use Σ_time⁻¹.

Net: the critique correctly identifies a genuine estimation-vs-claimed-model mismatch (severity CRITICAL is defensible for the flagship lagged-discovery capability); the low-rank `L` provides only partial, one-sided mitigation; and the recommended fix is directly actionable because the needed φ̂ is already computed and merely discarded into telemetry.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` (109-141), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (151-220), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\kron.py`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py` (380-407), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (61-176).

---

## LDO-WHITEN-04 [HIGH] — (κI+L_W)^{1/2} whitening imputes missing cells to 0 and uses the unnormalized degree Laplacian — over/under-corrects heterogeneously

Confirmed: `kappa` is a fixed default of 1.0, never fit. All three factual claims of the critique hold in the code. Now I can render my verdict.

Let me quickly verify the mathematical structure of the degree argument by considering what `featᵀ(κI+L_W)feat` actually does to the correlation — since the whitener is applied identically to every variable's spatial slice, the degree-scaling question is subtle.

The critique's mathematical claims are all factually grounded. My verdict:

---

## Verdict: PARTIAL (leaning UPHELD) — both mechanisms are real and present in code, but one is somewhat overstated and the fixes are partly already available in-repo but unwired.

### The code confirms every factual claim
- **Mean-imputation into a difference operator:** `covariance.py:176` (`Zc = np.where(np.isfinite(Z), Z, 0.0)`), `precision.py:93`, and the explicit admission in `lags.py:104-106` are all exactly as cited. The whitener is applied as the metric `featᵀ Q feat` (`covariance.py:194`, `G += Ft @ (Q_sparse @ Ft.T)`), with `Q = κI+L_W`.
- **Unnormalized combinatorial Laplacian:** `precision.py:73` writes `κ + degree` on the diagonal and `-1` off-diagonal (`precision.py:77`) — this is `L_W = D − A`, unnormalized, exactly as claimed.
- **κ fixed, not fit:** `orchestrator.py:101` `kappa: float = 1.0`, never estimated. The critique's "fit κ" recommendation is unaddressed.

### Problem (1) — mean-imputation gradients: UPHELD, and it is the more serious of the two

This is the sharper half of the critique and it is correct. Because `L_W` is a difference operator, the whitened residual at municipality *s* is `κ·z_s + Σ_{n∈N(s)}(z_s − z_n)`. When a neighbour *n* is missing and set to 0 (the margin mean), the term `(z_s − 0) = z_s` is injected as if *n* truly equalled the global mean. For an observed municipality *s* sitting next to a cluster of missing (typically poor/remote) municipalities, this manufactures a spurious spatial gradient `z_s − 0` at every observed/missing boundary. Since DATASUS missingness is **spatially clustered and correlated with the outcome** (under-reporting is worse exactly where burden is high/access is low), the induced bias is systematic and outcome-correlated — the worst kind. The recommendation (restrict `L_W` to the observed sub-graph and recompute degrees per time-slice) is exact and honours sparsity; it is not currently done. This is a genuine specification defect.

One important caveat the critique understates: the `whitened_lagged_correlation` path centers in whitened space (`G − HHᵀ/n`, `covariance.py:203`) and the pairwise-complete correlation elsewhere is genuinely missing-aware. So the impute-0 damage is confined to *the spatial-whitening step specifically* — it does not contaminate the non-whitened correlation estimator. The blast radius is the spatial term only, but within that term the critique is right.

### Problem (2) — degree-confounded whitening: PARTIAL / overstated

The factual premise (unnormalized `L_W`, eigenvalues scale with degree) is true, but the stated *consequence* — "distorts partial correlations non-uniformly across space" — is weaker than it sounds, for a structural reason the critique misses.

The whitener multiplies **every variable's spatial slice by the same operator `Q^{1/2}`**. The estimated object is the `p×p` (well, `p(K+1)²`) **variable** precision `Ω_var`, obtained from the metric `featᵢᵀ Q featⱼ` reduced *over all municipalities*. A high-degree municipality does get up-weighted in that sum relative to a low-degree one — but it is up-weighted **identically for variable *i* and variable *j***. A correlation is a ratio: `⟨fᵢ,fⱼ⟩_Q / √(⟨fᵢ,fᵢ⟩_Q⟨fⱼ,fⱼ⟩_Q)`. A common per-node reweighting that hits numerator and both denominator terms does **not** produce a per-*edge* bias the way a variable-specific reweighting would; it reweights *which municipalities dominate the national correlation*, not *which edges appear*. So "whitens municipality A more than B" is true, but "distorts the partial correlation between variables non-uniformly" does not follow directly — the spatial index is marginalized out before any partial correlation is read.

The real residual concern from (2) is subtler and correct: the unnormalized `L_W` gives **heterogeneous marginal variances** to the whitened field (interior nodes are shrunk harder), so the effective sample is geographically reweighted toward dense-interior municipalities — under-weighting exactly the sparse coastal/border/remote nodes that are the epidemiological blind spots. That is a real *representativeness* distortion and it compounds problem (1) (same remote municipalities), but it is not the "partial correlations distorted per-edge" claim as written. The BYM2/scaled-Laplacian fix (homogenizing marginal variances so `κ` and geography are separated) is the right remedy for this, and the repo already contains the machinery (`spatial_field.py` implements a BYM/ICAR field with a scaled structured precision) — it is simply **not wired into the whitener**, which still uses the raw `κI+L_W`.

### Net
- Claim (1): **UPHELD** — real, systematic, outcome-correlated bias; fix is exact and not implemented.
- Claim (2): **PARTIAL** — premise true, "per-edge partial-correlation distortion" overstated (spatial index is marginalized, common reweighting cancels in the ratio); the defensible residual is a geographic **representativeness/marginal-variance** distortion, correctly remedied by BYM2/scaled-Laplacian.
- κ-fixed-not-fit: **UPHELD** as a minor, honestly a Mutable-Solver knob rather than a spec violation.

The recommendation is sound and the two fixes are cheap relative to their payoff: (a) restrict `L_W` to the observed sub-graph per time-slice instead of impute-0 (removes the boundary-gradient bias exactly), and (b) scale the structured precision to unit generalized marginal variance (Riebler/Sørbye) so whitening strength stops tracking degree. The repo's `spatial_field.py` BYM code is the natural donor for (b).

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py` (whitening path, lines 176/194/203), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\precision.py` (unnormalized `κI+L_W`, lines 73/77/93), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` (impute-0 admission, lines 104-106), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\spatial_field.py` (existing BYM/ICAR machinery, unwired to the whitener).

---

## LDO-ADMM-06 [HIGH] — ADMM stopping test uses only the primal residual — can certify convergence at a non-stationary iterate

## VERDICT: PARTIAL (leaning UPHELD on the mechanism, downgraded on the stated consequence)

The critique's **core mathematical claim is correct**: the stopping test at `lowrank.py:229` checks only the primal residual, and standard ADMM (Boyd 2011 §3.3) requires both primal and dual residuals to be small. But two of the critique's framing points are wrong, and its severity is inflated by a downstream safeguard it acknowledges but misjudges.

### What the critique gets right

**1. The stopping test is primal-only, and a real dual residual exists and is uncomputed.**
The loop is a 3-operator ADMM with consensus constraint `R = S − L` (R is the log-det block's copy of the consistency variable `M := S − L`). The dual update is `U += R − (S−L)` (line 224–225), so:
- primal residual `r = R − (S−L)` — this **is** what's checked (`pn`, line 227)
- dual residual `s = −ρ·Δ(S−L) = −ρ(M^{k+1} − M^k)` — this is **never formed anywhere in the loop**.

The critique's algebra (`s = ρ(M^{k+1}−M^k)`) is exactly right for this splitting. A tiny `r` with a non-negligible `s` is the textbook false-convergence mode: `R` tracks `S−L` closely while `M` is still moving. So the mechanism is real, and the recommendation (dual residual + `ε = √n·ε_abs + ε_rel·max(...)` scale-normalized tolerances) is the correct fix.

**2. The unnormalized `tol=1e-5` is scale-dependent.** `rn = max(1, ‖R‖)` normalizes only by the magnitude of R, not by `√n` (the dimension) or the covariance scale, so a fixed `1e-5` does mean different things across `p` and C-scale. This part is unambiguously correct and worth fixing regardless.

### Where the critique overreaches

**3. "No adaptive ρ" — true but not a convergence *bug*.** ADMM converges for any fixed ρ > 0 (for this convex problem). Adaptive ρ (Boyd §3.4.1) only accelerates and rebalances; its absence is a performance/robustness gap, not a correctness defect. Bundling it into a HIGH-severity "certifies a non-stationary iterate" claim conflates "slower" with "wrong."

**4. The stated *consequence* is substantially mitigated by a safeguard the critique itself cites.** The critique's risk is: "the convergence gate downstream trusts a flag that can be true at a non-stationary point → wrong direct-vs-latent split gets locked in." But the flag is asymmetric in exactly the protective direction:

- The §V.6 gate (`certify.py:38–40`, `orchestrator.py:289–294`) only ever uses `converged` to **downgrade** edges to `descriptive` when `converged=False`. A *false positive* on `converged` (the failure mode the critique describes) causes the gate to **not fire** — i.e., it fails to add protection it otherwise would have. It does **not** manufacture a `selected` certification on its own.
- Promotion to `selected` still requires the **independent conjuncts** of §III.8 (`test_ldo_convergence_gate.py`): holdout **stability** across resamples *and* propagated **uncertainty**. A premature stop that produced an unstable S/L split would generically fail the stability conjunct, because the split would differ across the stability resamples. The convergence flag is a *belt*; stability is the *suspenders*.

So the honest statement of the risk is narrower than "the core scientific claim may reflect an unconverged saddle." It is: **a premature stop can weaken one of three independent guards, and only in cases where the stability resampling happens to also be fooled would a wrong split survive.** That is a real hole, but it is a defense-in-depth erosion, not a single-point failure of the scientific claim.

### One technical sharpening the critique misses

The dual residual here is even more load-bearing than in generic ADMM because of the **incoherence gate** (`min_factor_support`, lines 270–282) and the **mutual-exclusion** rule (lines 288–295). Both operate on `S` and `L` **separately** at readout, not on `M = S−L`. The primal residual can be small (M is settled) while S and L individually are still trading mass between the sparse and low-rank blocks — which is precisely the direct-vs-latent split the critique worries about. A dual residual on `Δ(S−L)` alone would *not* fully catch that; a genuinely rigorous certificate should also monitor the change in `S` (or `L`) per iteration, since the split can drift within a fixed `M`. This actually *strengthens* the critique's underlying concern while showing its proposed fix (dual residual on M only) is necessary but not sufficient for the CPW split specifically.

### Bottom line

- The **stopping-test defect is UPHELD**: primal-only, unnormalized, real dual residual uncomputed. Fix is cheap and correct as recommended, and should additionally track `Δ‖S‖`/`Δ‖L‖` given the split-at-readout.
- The **severity is PARTIAL**: the downstream §V.6 gate + §III.8 stability/uncertainty conjuncts mean the flaw degrades defense-in-depth rather than directly certifying a false edge; the "core scientific claim reflects an unconverged saddle" framing overstates it. Adaptive-ρ absence is not a correctness bug.

**Recommended disposition:** accept the fix (add dual residual + scale-normalized abs/rel tolerances at `lowrank.py:224–231`; consider adding split-drift monitoring), but reclassify from HIGH to MEDIUM given the intact stability/uncertainty conjuncts that gate `selected` independently of the `converged` flag.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py` (loop 206–231; readout split 258–295), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\certify.py:38–40`, `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py:289–294`, `C:\Users\Galaxy\LEVI\PegaSUS\tests\unit\test_ldo_convergence_gate.py`.

---

## LDO-INCOH-07 [HIGH] — min_factor_support=3 and loading_threshold=0.3 are magic proxies for the CPW incoherence condition, not a guarantee it holds

This confirms the mathematical picture completely. I have everything I need for a rigorous verdict.

The key finding: the `loading_threshold=0.3` gate combined with the L2-normalized eigenvector geometry creates a hard cutoff at **k ≈ 11 variables** — any latent driver spread across more than ~11 variables has per-variable loadings below 0.3 and is silently dropped, regardless of how strong the confounder is. But the critique's mechanism (shattering into spurious S edges) is *wrong* — CPW keeps S clean; the driven-block of S is exactly zero.

Let me write up the verdict.

---

## Verdict: **PARTIAL** — the concern is real and HIGH, but the critique misdiagnoses the failure *mechanism*. The correction makes it sharper, not weaker.

### What the critique gets right

1. **The readout gate is not the CPW identifiability condition.** Correct. `lowrank.py:270-282` operates on the post-ADMM output `L`, not on the problem's incoherence. The docstring at `lowrank.py:147-156` explicitly *calls* the `min_factor_support` gate "the Chandrasekaran–Parrilo–Willsky **incoherence identifiability condition** at readout" — that naming is an overclaim. CPW incoherence is a property of the *true* (S*, L*) pair (the tangent-space transversality condition μ(L*)·deg(S*) < c of Chandrasekaran et al. 2012, Thm 4.1); it governs whether ADMM can recover the truth at all. A support-count on the recovered `L` cannot certify it. If the truth is non-identifiable, ADMM lands on the wrong (S,L) and no readout gate repairs it. This half of the critique is **UPHELD**.

2. **The thresholds are absolute and p-blind.** Correct and, it turns out, worse than the critique argues. `min_factor_support=3` and `loading_threshold=0.3` are fixed constants; neither scales with p. The recommendation to make support scale with p and threshold relative to the loading distribution (participation ratio / energy concentration) is sound.

3. **The dense-weak confounder is the genuine hazard.** Correct that this is exactly the case the low-rank layer exists for and exactly where the design fails.

### Where the critique is mathematically wrong (and must be corrected)

The critique's stated failure — *"a weak-dense driver is shattered into dozens of spurious direct disease-disease edges in S"* — **does not occur**, and this matters because it names the wrong artifact. I ran it (25 vars, a factor loading 20 of them):

- The CPW ADMM **correctly absorbs the dense-weak confounder into L**. The driven-block of the recovered S is **exactly zero** (`max|S_offdiag| = 0.0000`) at confounder strengths from 0.28 to 0.45. There is no shattering. This is CPW working *as designed* — the trace penalty on L plus the ℓ1 penalty on S is precisely what prevents the everything-riding-a-wave clique.

The real failure is subtler and arguably worse: the confounder is **silently dropped from the reported output entirely.** The mechanism is the interaction the critique half-saw but mislabeled:

- An L2-normalized eigenvector of a factor spread uniformly over k variables has per-variable loading ≈ **1/√k**. The `0.3` gate therefore admits a factor only while **1/√k ≥ 0.3, i.e. k ≲ 11 variables.** Beyond ~11 variables the loadings fall under 0.3 *no matter how strong the confounder is* (I confirmed loadings stay ~0.24 as I raised strength 0.28→0.45; strength scales the eigenvalue, not the normalized loading). A 20-variable epidemiological wave gets **zero** `latent_shared` edges — it is neither in S (CPW cleaned it) nor in the report (the gate killed it). The scientific artifact is not "spurious edges" but a **false negative: the confounder vanishes**, and any downstream S estimated *conditional* on that dropped factor is still net-of-the-driver, so direct edges elsewhere remain trustworthy but the driver itself is unreported.

So the `risk_if_ignored` sentence is factually incorrect (no shattering), but the underlying HIGH severity is *validated by a different, cleaner failure*: the very "weak-dense driver (climate, poverty)" the critique names is **suppressed**, and the `0.3` constant hard-caps discoverable factor breadth at ~11 variables — a national wave touching 20+ diseases is structurally invisible.

### On point (3) of the critique — the incoherence assumption itself

Partially right but overstated. The claim that "a dense-but-weak confounder violates the sparse-S incoherence assumption so CPW cannot separate it regardless" is not what I observed — CPW *did* separate it cleanly into L (S stayed zero). CPW incoherence constrains L's *column space* (μ(L) small = spread, which a dense confounder *satisfies*) and S's *degree* (bounded). A dense-weak confounder is the **good** case for the low-rank side (high spread = low coherence). What breaks is not CPW separation but the *readout*. So this sub-point conflates the estimator's identifiability (fine here) with the reporting gate (broken here).

### Corrected recommendation

The critique's recommendations are directionally right; sharpen them:

1. **Drop the "incoherence" naming** from `lowrank.py:147-156` — the gate is a *coherence/concentration heuristic on recovered L*, not the CPW identifiability condition. Keeping the name hides the gap.
2. **Replace `loading_threshold=0.3` absolute with a participation-ratio / energy-concentration criterion** so a genuinely spread factor is *kept because it is spread*, not dropped because its per-variable loadings are individually small. The current gate has its logic inverted for large factors: high spread (the mark of a real shared driver) drives every loading down and *fails* the gate.
3. **Scale `min_factor_support` with p.**
4. **Add the dense-weak confounder to the planted-structure test** (`test_ldo_sparse_lowrank_recovery.py` currently only plants a 3-variable factor — `factor_vars=(0,1,2)`, k=3, loading 0.577, comfortably above 0.3). The test passes *because it never exercises k>11*. Assert the k=20 confounder lands in `latent_shared`, not that it vanishes.
5. **Optionally** compute a real post-hoc diagnostic (μ of recovered L's column space, max degree of S) and flag when μ·deg approaches the theoretical bound — but as a *fit-quality flag*, not as the classification gate.

### Bottom line

**PARTIAL.** The critique correctly identifies that the gate is a magic-number surrogate, not the CPW condition, and that it fails on the dense-weak confounder — both HIGH-severity, both real. It mis-states the failure as spurious S-edges; the true failure is silent suppression, driven by a 1/√k geometry that caps discoverable factor breadth at ~11 variables. That correction strengthens the case for the recommended fix (relative/participation-ratio thresholding) rather than weakening it. The planted-structure test's green status is not evidence of correctness here — it only ever plants a 3-variable factor.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py` (gate at lines 269-295; docstring overclaim at 147-156; `1/√k` interaction with `loading_threshold` default at line 121), `C:\Users\Galaxy\LEVI\PegaSUS\tests\synthetic\test_ldo_sparse_lowrank_recovery.py` (test only exercises k=3, never the failing k>11 regime).

---

## LDO-SEP-08 [HIGH] — Separability Ω_var⊗Σ_space⁻¹⊗Σ_time⁻¹ forbids space-varying and lag-varying dependence — the epidemiologically interesting case

I have everything needed. Here is my verdict.

---

**VERDICT: PARTIAL — the critique's headline mechanism is real but it misreads the estimator architecture, so its severity and its central risk claim are wrong.**

## Where the critique is factually wrong

**1. The separable Kronecker operator is NOT the estimator backbone.** The critique's load-bearing premise — "the fit uses one global Ω_var and one global whitener" combined into `Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹` — conflates two distinct objects.

- `kron.py`'s `KroneckerPrecision` is used in exactly one place: `orchestrator.py:387-403`, feeding `joint_logdet` into the §V.6 validity *telemetry* report. Its own docstring (kron.py:26-31) states it is "deliberately NOT swapped into the CPW ADMM R-step." It is a joint-likelihood *evaluator*, not the fitter.
- The actual estimator (`lags.fit_lagged_links` → `covariance.whitened_lagged_correlation` → `fit_sparse_plus_lowrank`) never forms a Kronecker product. It uses the spatial GMRF `Q = κI+L_W` as a **whitening metric** (lags.py:109-114), computing `featᵢᵀ Q featⱼ` to strip spatial autocorrelation *before* the graphical-lasso, then estimates Ω_var on the whitened residuals. The temporal axis is handled by explicit lag-stacking, not an AR(1) Kronecker factor.

So the estimator's structural assumption is **not** "Σ_space is identical for every variable." Whitening by a shared spatial metric assumes the *nuisance* spatial autocorrelation (the municipal-clustering confounder) is common — a far weaker and more defensible claim than "the spatial range of every disease's *signal* is identical." The critique's example (airborne vs vector-borne spatial range) is a property of the signal covariance, which the whitened graphical-lasso does not constrain to be shared. The critique attacks the telemetry operator's assumption as if it were the estimator's.

**2. The "cancellation → fictitious homogeneous Brazil" risk is directly defended against, and not post-hoc.** The critique asserts region-opposite couplings "cancel to no edge" and are "averaged away." But:

- `exhaustiveness.heterogeneity_screen` (exhaustiveness.py:25-29) explicitly scores on the *std* of subgroup effects, and its docstring names the exact failure the critique raises: "a case whose pooled mean is ~0 because subgroup effects have opposite signs still scores high here." `should_drill_down` fires the fine pass on heterogeneity OR max-subgroup signal, so a cancelling edge is *not* pruned — it is escalated. This is a **recall-tuned sensitivity filter**, wired live into `run_multiresolution_ldo` (resolution.py:206), plus a random deep audit (`_random_deep_audit`) that empirically *measures* the false-negative rate. The "averaged away silently" outcome the critique predicts is precisely what the §VIII cancellation machinery exists to prevent, and it is not post-hoc — it gates which pairs get the fine fit.

## Where the critique is correct (and should be sharpened)

**3. The estimator still fits ONE global Ω_var.** This is true and it is the real residue. `fit_lagged_links` produces a single national partial-correlation matrix. Region-varying *variable-dependency structure* (dengue-rainfall strong in NE, weak in South) is not estimated as a stratified Ω_var; it is only recovered *per-edge, post-hoc* by `fit_spatial_varying_coefficient` (spatial_field.py). The critique is right that this is a "post-hoc per-edge readout, not a relaxation of the estimator." The BYM field is fit only for edges the global estimator already surfaced. An edge that is strong-in-NE / absent-elsewhere and whose *national whitened partial correlation* falls below `edge_threshold` will never trigger a spatial-field readout — the heterogeneity screen operates on subgroup effects in the multiresolution scan, but the base national LDO fit itself has no per-region Ω_var re-estimation. So there is a genuine gap: **strong-somewhere-below-national-threshold couplings can be missed by the base estimator**, caught only if the coarse subgroup screen happens to fire on them first.

**4. The sign-flip warning is genuinely absent as a first-class output.** `_heterogeneity_summary` (spatial_field.py:157-174) emits `beta_min`, `beta_max`, `beta_std`, and per-scale variance shares — enough to *infer* a sign flip (`beta_min<0<beta_max`) but there is no explicit "sign-flips-across-space" flag surfaced to the reader. Recommendation (c) is not yet met. This is a low-cost fix: derive a `sign_heterogeneous` boolean in `_heterogeneity_summary` from `beta_min·beta_max < 0` weighted by mass on each side.

**5. Recommendations (a) and (b) are unbuilt.** There is no region-stratified Ω_var homogeneity test (Kronecker-vs-non-Kronecker likelihood ratio), and the low-rank non-separable correction is only *anticipated* (`kron_cg_solve` accepts a `correction=` argument, kron.py:210, but nothing constructs one). The §V.6 audit tests reproduction of held-out data and residual HSIC, which — as the critique correctly notes — can pass even when a coupling's *strength* varies by region, because a spatial average can still predict held-out cells adequately. The homogeneity of Ω_var across regions is not itself a tested hypothesis.

## Corrected severity

Down from HIGH to **MEDIUM**. The catastrophic outcome the critique names (regionally-opposite couplings silently cancelling to a fictitious homogeneous Brazil) is *specifically* defended by the §VIII heterogeneity/cancellation screen and the random-audit false-negative measurement, and the estimator does not carry the "identical Σ_space per variable" assumption the critique ascribes to it (whitening ≠ Kronecker signal model). What remains is real but narrower: (i) the base national estimator fits a single Ω_var, so region-varying *dependency strength* is recovered only per-edge and post-hoc, with a threshold-gated recall gap; (ii) Ω_var homogeneity is not a tested hypothesis; (iii) sign-flip heterogeneity is inferable but not surfaced as an explicit warning.

**Actionable residue:** implement recommendation (c) (cheap: sign-heterogeneity flag in `_heterogeneity_summary`), then (a) (a region-stratified Ω_var homogeneity LRT as a §V.6 conjunct). Recommendation (b) (low-rank non-separable correction) is architecturally staged in `kron_cg_solve` but would only matter if the Kronecker operator were promoted to the estimator, which it currently is not — so (b) is lower priority than the critique implies.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\kron.py` (evaluator, not fitter), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py:108-141` (real estimator, single Ω_var), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\covariance.py:151-198` (shared-metric whitening), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\exhaustiveness.py:25-62` (cancellation screen — refutes the core risk), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\spatial_field.py:64-174` (post-hoc per-edge BYM readout, no sign-flip flag).

---

## LDO-MARGIN-10 [HIGH] — Count-with-exposure margin is Poisson-offset (equidispersed); Brazilian municipal counts are strongly overdispersed → mis-calibrated latent Z → biased correlations

Confirmed: there is no dispersion parameter, no NB margin, and no PIT-uniformity (KS) gate anywhere in the LDO. The `dispersion_screen` in `exhaustiveness.py` is a coarse-scan heterogeneity filter, not a margin calibration gate — it does not feed back into the Poisson margin. I have enough to render a verdict.

## VERDICT: UPHELD (with one technical sharpening and one scope correction)

The critique is mathematically correct and confirms code that is exactly as described. It is also independently corroborated by the project's own math audit (`PEGASUS_MATH_CRITIQUE.md`), which already logs this as **two** separate findings — meaning LDO-MARGIN-10 is real, not a misread.

### What the code actually does (confirmed)
`margins.py:88-115`, `count_exposure_gaussianize`:
- `lam = x.sum() / total_e` — a single pooled scalar rate over **all municipalities and all years**.
- `mu = lam * e`; `u = poisson.cdf(x-1, mu) + U·poisson.pmf(x, mu)`; `z = ndtri(u)`.
- No dispersion parameter exists anywhere in `src/pegasus/ldo/`. No NB margin. No PIT-uniformity/KS gate before the covariance step.

The path is genuinely reached in production: `orchestrator.py:70-73` passes `raw.exposure` into `gaussianize_field`, which routes any variable with positive exposure to the Poisson-offset margin (`margins.py:133-134`). So this is not a dead branch.

### The mathematics is sound
The randomized PIT `u = F(x−1) + U·p(x)` is exactly Uniform(0,1) **iff `F` is the true CDF** (Brockwell/Dunn–Smyth). Poisson forces Var = Mean. Under municipal overdispersion (Var ≫ Mean — the empirical norm for SIM/SIH counts), the Poisson CDF is too light in the tails, so the PIT `u` piles up near 0 and 1, and `Z = Φ⁻¹(u)` is heavy-tailed, **not** standard Gaussian. Since the LDO estimates a Gaussian-copula precision on the `Z` scale, the correlation estimates and their Fisher-z SEs are biased. That chain is correct.

The pooled-λ point is also correct and is the more damaging half: `Z_i` measures deviation from the **national-average rate**, not a local expectation. A structurally high-mortality municipality gets systematically negative-then-positive `Z` (per the sign, high count vs. low national μ → high `u` → positive `Z`) in *every* year — a manufactured spatial rate surface baked into the latent field. The nonlinear rank/PIT transform means the downstream **linear** GMRF whitener (`κI + L_W`) cannot fully remove it, which is precisely the coupling to LDO-WHITEN-04 the critique invokes.

### Two corrections / sharpenings

1. **Scope correction — the pooled-λ defect is more severe than "compounds it."** The critique frames overdispersion as the headline and pooled-λ as an aggravator. The project's own audit ranks them oppositely: pooled-λ is logged **CRITICAL** (DIRECT-MARG-01) and overdispersion **HIGH** (DIRECT-MARG-02). This is the right ordering. Even a *perfectly dispersion-calibrated* NB margin with a single pooled mean rate would still inject the spatial-mean and temporal-trend surface as spurious dependence. So the NB fix alone (the critique's primary recommendation) is **necessary but not sufficient** — a spatially/temporally-varying baseline (log μ = log E + s(space) + f(time), or an empirical-Bayes shrunk per-cell rate) is the load-bearing fix. The critique does mention the varying-rate GLM as a secondary recommendation; it should be promoted to primary.

2. **Technical nit in the critique's phrasing.** "The whitener then tries (and fails) to remove [the gradient]" is right about the *rate-surface* gradient, but note the whitener operates on `Z`, not on `u`; its failure mode is the linear-vs-nonlinear mismatch, not a calibration failure per se. Minor — does not weaken the finding.

### Spec check — the critique is right that Poisson is hardcoded against a broader license
MSD-III §III.5 explicitly lists the admissible family set as "Poisson/NB/binomial/beta-binomial/lognormal/…" and MSD-I §6.2 states NB/quasi-Poisson is the intended family "when overdispersed." The Operational Plan's own red-test spec (`PEGASUS_OPERATIONAL_IMPLEMENTATION_PLAN.md:475`) named a "Poisson/**NB**/beta-binomial/Gamma column" transform. So the count-with-exposure path **hardcoding Poisson is a spec-under-implementation**, not a spec-honoring choice — the family the MSD licenses (NB) simply was never wired for the exposure margin. The critique's read of the spec is accurate.

### Recommendation (as it should land)
- **Primary:** replace the pooled scalar `λ` with a varying baseline (offset GLM `log μ = log E + s(space) + f(time)`, or EB-shrunk per-(space)/per-(time) rate) so `Z` is deviation from a *local* expectation. This kills the manufactured gradient.
- **Secondary:** make the margin Negative-Binomial-with-offset with a method-of-moments or MLE dispersion `α` (the existing `dispersion_screen` in `exhaustiveness.py:39` already computes var/mean and can seed it) so the PIT is dispersion-calibrated.
- **Gate:** add a KS/χ² uniformity test on `u` per variable as a hard gate before the covariance step, downgrading edges for variables whose margin fails calibration. None currently exists.

**Net:** UPHELD. Correct math, correctly identified code, corroborated by internal audit. Sharpen by promoting the varying-rate (pooled-λ) fix above the NB fix in priority.

---

## LDO-LAG-CONF-01 [CRITICAL] — Lag-extended precision identifies conditional-Granger association, not directed effects; low-rank L cannot absorb LAGGED common trends

I now have everything needed to render a rigorous verdict. Let me confirm the critical mathematical point about whether `L`'s low-rank structure over the lag-extended space can represent a lag-shifted common factor. The correlation is a Pearson correlation of PIT-ranks — it captures cross-lag co-movement in the off-diagonal `Ω^(k,0)` blocks. The question: is a lag-shifted common factor a *low-rank* structure that `L` absorbs, or does it leak into `S`?

I have enough to assess. Let me write the verdict.

---

## VERDICT: PARTIAL — leaning UPHELD

The critique's central mathematical claim is **correct**, and the design does not currently defend against it. But two of its supporting sub-claims are imprecise, and the fix is smaller than the critique implies because much of the ladder machinery it recommends already exists (just not wired as a gate). Here is the fair, sharpened assessment.

### What the critique gets RIGHT (the load-bearing claim)

**1. The margin is fit on levels, not innovations — confirmed.** `margins.py` (`randomized_pit_gaussianize`) applies the PIT/rank transform to the raw observed values per variable. There is *no* detrending, differencing, or deterministic time-trend covariate anywhere upstream of the covariance. MSD-III §III.5 (line 222) describes the margin as "rank/PIT … dependence is modeled there" with no trend-residualization. So `Z_{j,s,t}` carries the full secular level, including the shared 2000–2024 monotone trend. The correlation in `covariance.py` is therefore a Pearson correlation of PIT-ranks **of levels**. A pair of monotone-trending series with a phase offset will produce exactly the peaked cross-lag partial-correlation curve `{S^(k,0)_ij}` that `lags.py:158-183` reads off as a "discovered distributed-lag response curve." This is real. The headline capability is exposed to trend-phase artifacts.

**2. The `edge_type='lagged_directed'` / docstring "directed lag-k link … time licenses the direction" (lags.py:5-13, §II.6.1 line 254) is conditional Granger, not a directed effect — correct and already conceded by the design elsewhere.** §IV line 251 explicitly says "Classical bivariate Granger is *subsumed* … the LDO's conditional, latent-controlled lagged edges," and Rung 0 (line 253) is labeled "**Directed only by time**." So the *taxonomy* already knows these are Granger-predictive, not causal. But the **naming** the critique targets (`lagged_directed`, "the engine finds the lag") is over-strong for the Rung-0 object and does invite causal misreading. The recommendation to rename toward `granger_predictive` and gate directional-causal language behind Rung 2 is well-founded and cheap.

### Where the critique is IMPRECISE (why PARTIAL, not clean UPHELD)

**3. "Low-rank L cannot absorb LAGGED common trends" — overstated as a categorical.** This is the critique's key mechanistic claim and it is *half* wrong. Crucially, `fit_sparse_plus_lowrank` in `lags.py:136` is fed `pw.correlation`, which is the **full** `p(K+1) × p(K+1)` lag-extended correlation — **not** just the lag-0 block. So `L` is fit over the entire lag-extended feature space and *can* in principle place mass on off-diagonal cross-lag blocks. A lag-shifted common factor (driver `f` loading on variable `i` at lag `k_i` and `j` at lag `k_j`) is, in the lag-extended coordinates, still a **genuinely rank-1 outer product** of a loading vector that happens to have nonzero entries in different lag blocks. CPW's `L` is a full dense PSD matrix; nothing restricts its eigenvectors to be lag-aligned. So the claim "L does not naturally represent a lag-shifted common factor" is **not** true by construction — the representation exists.

**The real failure is subtler and the critique should be sharpened to it:** whether the lag-shifted factor lands in `L` vs. `S` is a **separation/identifiability** question, governed by (a) the CPW incoherence condition and (b) the `min_factor_support ≥ 3` gate in `lowrank.py:270-282`. That gate deliberately **routes any rank-1 component supported on fewer than 3 variables INTO `S` as a direct edge** — precisely to fix the S/L collapse noted in memory. So for a **pairwise** lagged trend co-movement (exactly the "any pair with a phase offset" case the critique names), the incoherence gate *forces* it into `S` as a `lagged_directed` edge and *forbids* `L` from claiming it. The critique's conclusion (spurious directed edge) is right, but the mechanism is the opposite of what it states: it is not that `L` *can't* represent the factor — it is that the identifiability gate **actively excludes** a 2-variable lagged co-trend from `L` and hands it to `S`. A shared national trend loading on *many* variables at various lags *would* be caught by `L`; a pairwise phase-offset trend would not. Sharpen the critique to: "the low-rank defense only fires for high-support common factors; pairwise or few-variable lagged co-trends bypass it by the very incoherence gate that fixes the collapse."

### What already exists (reduces the fix cost)

- **Rung 2 (ITS/DiD/negative-control)** is built (`causal/quasi.py`, task O8 "Rung-2 negative-control veto + DiD"). The critique's recommendation (3) — require Rung 2 before directional-causal language — is a *wiring/gating* change, not new math.
- **Temporal holdout** (`orchestrator.py:239-246`, §IX.3) fits on a training window and checks tail-year persistence. This is a partial defense: a pure trend-phase artifact that is *stable* across the whole 25-year monotone trend would **survive** holdout (it persists precisely because the trend persists), so holdout does **not** neutralize this critique — worth stating explicitly as a non-defense.
- **`gamma_temporal` smoothness** (`build_smoothness_operator`, lag-chain Laplacian) is the opposite of a fix: it smooths precision rows *across* adjacent lags, which if anything *broadens* a spurious peaked response curve rather than removing the trend.

### The correction (what the design must add)

The critique's recommendation (1) is the correct primary fix and none of it exists today:

- **Add a per-variable deterministic time-trend/spline covariate to the copula margin (§III.5), or fit the margin on innovations (first-difference / detrended series).** Right now `gaussianize_field` has an `offset`/`covariates` hook in the *formula* (§II.6.1 line 234 writes `F_j(X | offset, covariates)`) but `margins.py` never consumes a time covariate — so the doc promises trend-residualization the code doesn't do. This is a genuine spec-vs-code gap.
- **Add an explicit common-trend null** (recommendation 2): refit a shared smooth national trend factor and require each lagged edge to survive its removal. This is stronger than the existing rank-`r` `L`, because `L` is fit jointly and (per the incoherence gate) will *not* hold a pairwise lagged co-trend.
- **Rename `lagged_directed` → `granger_predictive` at Rung 0** and reserve directed-causal language for Rung 2+.

### Bottom line

The critique correctly identifies a real, uncontrolled confounding pathway (shared secular trend → spurious peaked lagged partial-correlation curve) that strikes the flagship capability, and correctly notes the code fits levels with no detrending. It is **downgraded from CRITICAL-as-stated to a confirmed HIGH/CRITICAL with one corrected mechanism**: the low-rank `L` is fit over the full lag-extended covariance and *can* represent a lag-shifted factor in principle — the actual leak is that the `min_factor_support` incoherence gate deliberately routes low-support lagged co-trends into `S` as directed edges, so the defense only covers many-variable common trends. Temporal holdout does not save it (a persistent trend artifact persists out-of-sample). The recommended fixes are sound; the margin-side trend covariate is also a spec-vs-code discrepancy (§II.6.1/§III.5 imply covariate-conditioned margins the code omits).

**Relevant files:** `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\margins.py` (levels, no detrend — the root gap), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py:5-13,136-183` (edge naming + full lag-extended fit), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py:270-282` (incoherence gate that routes low-support co-trends to S), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py:239-246` (temporal holdout — non-defense here), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\causal\quasi.py` (Rung-2 machinery already exists for the recommended gate).

---

## LDO-LAG-STAT-02 [CRITICAL] — Stationarity: a SINGLE response curve is pooled over 2000-2024, assuming the lag structure is time-invariant across a major epidemiological transition

Now I have the complete picture. The `temporal_holdout` splits time into train (all but last year) and a 1-year tail — but this is a *forecast-persistence* check with a tiny tail, not a regime-split stationarity diagnostic. Critically, the *main* national fit (the one that produces the response curves) still pools all 25 years into one `Ω_var`. The holdout re-fits on the pooled train window, so it does not surface a pre-2015 vs post-2015 curve divergence; it asks only "does the pooled edge recur in the final year."

I have enough to render a verdict.

---

**VERDICT: UPHELD** (with two sharpenings and one narrowing).

**The code claim is exactly correct.** Both response-curve estimators accumulate a *single* second-moment matrix over all time slices:
- `whitened_lagged_correlation` (covariance.py:191–203): the loop `for t in range(K, T)` accumulates one Gram matrix `G += Ft @ (Q @ Ft.T)` and one running sum `R`, then centers with a single global mean `H/n_cells`. There is exactly one `Ω_var` for the whole window.
- `pairwise_correlation` (covariance.py:83–86): the moments `M@M.T`, `X0@M.T`, `X0@X0.T` are summed over *all* cells with no time index. `_build_lagged_feature_matrix` (lags.py:62–66) flattens `(t-K)*S+s` into one undifferentiated sample axis.

So the estimator is mathematically a **time-pooled maximum-likelihood covariance**: `Σ̂ = (1/n)Σ_t f_t f_tᵀ` (mean-centered). This is the MLE of a *single* joint Gaussian only under the assumption that `f_t` is drawn i.i.d. (or at least identically distributed) across `t`. That is precisely the strict-stationarity/regime-invariance assumption the critique names.

**The MSD claim is also correct.** §II.6.1 (line 239) writes `Σ_time⁻¹` as "temporal smoothness/seasonality." That is a **within-series smoothing prior** on the residual temporal covariance — it regularizes how adjacent time points covary; it says nothing about whether the *cross-variable lag operator* `Ω_var` is constant across `t`. The MSD never states, tests, or bounds a regime-invariance assumption. No `stationar*`, `changepoint`, `CUSUM`, or regime-split logic exists anywhere in `pegasus/ldo` (the "regime" hits are SIDRA cube-classification and null-permutation regimes — unrelated). The math is genuinely silent here.

**Why pooling is not benign (the sharpened mechanism).** The pooled `Ω_var` is not "the average response curve." A precision/correlation estimate is dominated by the *high-variance* regime, because covariance is a second moment: the regime contributing the largest `Σ_t (f_t − f̄)(f_t − f̄)ᵀ` mass sets the off-diagonals. Under an epidemiological transition where late-window variance (rising counts, arbovirus outbreaks, higher SUS-captured incidence) exceeds early-window variance, the pooled curve is effectively the *late-regime* curve wearing a whole-window label — which "describes no actual period" is even understating it; it describes one period and mislabels it as all. This is a real, directional distortion, not just loss of resolution.

**The completeness-trend point is valid and compounds it.** I confirmed there is **no detrending anywhere** — `margins.py` Gaussianizes each variable by a single global rank-PIT over all `(s,t)` with no year offset, and grep for `detrend|time-varying|completeness|trend` in the LDO returns nothing. A monotone completeness ramp (SIM/SINASC ~80%→95%+) enters every affected variable as a shared upward trend; a shared trend inflates pairwise correlation among all trending variables → exactly the spurious-lagged-edge fabrication LDO-LAG-CONF-01 describes. The two critiques reinforce.

**The two sharpenings / narrowings for fairness:**

1. **A partial mitigation exists but does not do what's needed.** `temporal_holdout` (holdout.py:98) *does* split time — but into a train window (all but 1 year) and a 1-year tail, and it only checks *sign persistence* of pooled-train edges into the final year. This is a forecast-fluke filter, not a regime diagnostic: (a) the train window itself still pools ~24 years into one `Ω_var`, so a pre-2015/post-2015 curve divergence is invisible to it; (b) a 1-year tail at annual resolution is far too short to characterize a post-transition regime; (c) it compares sign, not curve shape, so a lag-1 vs lag-4 peak shift within a persisting edge passes. It is the *wrong axis of split* for this failure. So the design is not entirely defenseless, but the existing prong does not cover the critique — the critique should acknowledge `temporal_holdout` exists and explain why it is insufficient (which strengthens, not weakens, the finding).

2. **Severity nuance.** The "Zika undetectable" risk is real but slightly overstated as stated: the shared low-rank `L` (lags.py:11–13, designed to absorb co-epidemic waves) will tend to soak a 2015 arbovirus surge into `latent_shared` rather than emit it as a dated lagged edge. That is arguably *worse* for the critique's purpose, not better — the transition signal is not just smeared into baseline, it may be actively routed into the nuisance factor and discarded. Either way the phenomenon does not surface as a regime-specific response curve. Severity **CRITICAL** stands for any run that reports response curves as epidemiologically interpretable across 2000–2024.

**Recommended correction (endorsing the critique's, with priority order):**
- **Minimum viable, and directly analogous to an existing gate:** add a stationarity diagnostic that re-fits `Ω_var` on the pre/post temporal-midpoint (or a CUSUM changepoint) and downgrades any edge whose *response curve* (not just sign) diverges across halves — mirroring the `regularization_path_agreement` conjunct in certgates.py. This reuses machinery already present (`_time_slice`, `_edge_signset` generalized to curve distance) and slots into `certify_link`'s existing soft-downgrade path.
- **Report divergence as a first-class finding**, not noise: a link whose curve differs pre/post-2015 *is* the epidemiological-transition signal the win-condition study is meant to find.
- **Model completeness as a time-varying offset** in the count margin (or detrend `Z` per-variable by year before Gaussianizing) so the reporting ramp cannot masquerade as a lagged relationship — this closes the LDO-LAG-CONF-01 interaction at its source.

The critique reads the math correctly, identifies an assumption the MSD genuinely never makes explicit, and proposes a fix that is consistent with the design's own existing certification idiom. **UPHELD.**

---

## LDO-LAG-SMOOTH-03 [HIGH] — Temporal-smoothness prior biases the discovered response curve toward smoothness that may not exist (spurious-smoothing trap)

I have everything I need. The critique is a known one (it appears in PEGASUS_MATH_CRITIQUE.md as DIRECT-TEMP-01, MEDIUM, but this LDO-LAG-SMOOTH-03 version escalates to HIGH). Let me render the verdict.

---

**VERDICT: UPHELD (with two sharpenings and one over-reach corrected).**

## The mathematical mechanism is correct

The critique's core claim is verified against the code, not misread.

`build_smoothness_operator` (lowrank.py:336-337) adds `G = γ_t·(L_lag ⊗ I_p)` where `L_lag` is the chain (path-graph) Laplacian over lag positions 0..K (lowrank.py:306-319). In the S-step this enters as the prox-gradient term `∇q = G·S + ρ(S−M)` (lowrank.py:219). The penalty is `(1/2)tr(Sᵀ G S)`. Because `G = γ_t (L_lag ⊗ I_p)` and the feature index is `f = lag·p + var`, this quadratic equals

  `(γ_t/2) · Σ_var Σ_{adjacent lags a,a±1} ‖ S[:, a·p+var] − S[:, (a±1)·p+var] ‖²`

i.e. it penalizes the **squared L2 difference between the full precision column of a variable at adjacent lags**. The reported response curve is `curve_ji[j,i,lag] = partial_full[j, lag·p+i]` (lags.py:160-161) — exactly row `j` of those coupled columns. So the penalty directly shrinks `|curve(lag) − curve(lag±1)|` of the very quantity read out as the distributed-lag response. The chain of reasoning "penalizes precision entries → deforms the response curve the user reads" is exactly right. Under the separable model `Prec ≈ Ω_var ⊗ Σ_space⁻¹ ⊗ Σ_time⁻¹` (MSD-II §II.6.1), an L2 chain Laplacian on the lag axis is a Gaussian random-walk prior on the lag profile, whose posterior mean is a smoothing spline — it demonstrably broadens a delta-like true response, pulls interior argmax, and can borrow strength to raise neighbors. The statistical characterization is sound.

And MSD-II §II.6.3 line 254 is explicit that this profile **is** the scientific output ("the profile … is the discovered distributed-lag response curve — the engine finds the lag; it is not told it"). So a prior of unknown strength sits on the headline deliverable. The conceptual objection lands.

## The two things that make this genuinely HIGH, not MEDIUM — and are NOT handled

1. **γ_t is never selected and never perturbed in certification.** The default 0.1 (orchestrator.py:118) is a fixed magic number. I checked the certification path: `regularization_path_agreement` (certgates.py:35-57) perturbs only `lambda1` (`base_lambda1 * g`, line 50); stability selection (edges.py:70-90) perturbs the spatial and time *lattice* (subsampling), not the smoothing weight. **No gate refits at γ_t ∈ {0, ...} and checks whether peak_lag moves.** So the one discipline that would have caught "peak lag is an artifact of the smoother" — path-agreement across the regularizer that actually controls smoothness — is absent for this specific hyperparameter. The critique's recommendation (1) is therefore not already satisfied. This is the decisive fact.

2. **No non-smoothing discovery default.** `fit_lagged_links` defaults to `gamma_temporal=0.0` (lags.py:82) — a byte-identical no-smoothing fit exists — **but the orchestrator overrides it to 0.1** (orchestrator.py:118) and that is what the pipeline runs. So the honest-discovery path exists in the low-level API and is thrown away at the wiring layer. Recommendation (2) is a small, correct fix: flip the orchestrator default to 0 for discovery and report smoothing as a separately-labeled robustness variant.

## One correction to the critique (why PARTIAL-worthy, but I still say UPHELD)

The critique slightly overstates the *epistemic opacity* claim ("it is a prior on conditional dependence magnitudes, not on the response the user reads"). Under the separable Gaussian model the response curve **is** exactly a slice of the precision (partial correlation), so a smoothness prior on precision columns *is* a smoothness prior on the response curve — the meaning is not opaque, it is a random-walk prior on the lag profile. That is a cleaner and more defensible framing than "indirectly deforms," and it does not weaken the objection; it strengthens the case that recommendation (3) (fused-lasso / total-variation on the lag profile, allowing sharp jumps) is the mathematically apt regularizer for piecewise-smooth incubation kernels. So: the *mechanism* framing should be tightened (not "opaque indirect," but "a literal RW prior on the reported curve"), which if anything makes the bias more legible and the fix more obviously warranted.

## Sharpened statement of the finding

> The distributed-lag response curve — the LDO's headline output (MSD-II §II.6.3) — carries an L2 chain-Laplacian smoothness prior of fixed strength γ_t=0.1 that is (a) not data-selected, (b) not included in any certification perturbation axis (only λ₁ and the lattice are perturbed; certgates.py:50, edges.py:90), and (c) overridden ON from the API's honest γ_t=0 default at the orchestrator layer (orchestrator.py:118 vs lags.py:82). A true single-lag incubation spike is therefore systematically broadened and its argmax pulled interior, with no reported sensitivity of peak_lag to γ_t. Minimum remediation: (i) add γ_t to `regularization_path_agreement` and report peak_lag stability across {0, γ_t, 2γ_t}; (ii) default the discovery run to γ_t=0 and emit smoothing only as a labeled robustness variant; (iii) replace L2 chain-Laplacian with fused-lasso/TV on the lag profile if regularization is wanted, since epidemiological lag responses are piecewise-smooth, not globally smooth.

Note: this is already logged as DIRECT-TEMP-01 (MEDIUM) in PEGASUS_MATH_CRITIQUE.md:90-93. The HIGH escalation here is justified because the completed O5 certification work (task #48) added λ- and time-perturbation axes but **specifically did not add a γ_t axis**, leaving the discovered peak lags uncertified against the smoother that shapes them.

Relevant files: `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lowrank.py` (306-347, 188-220), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\lags.py` (82, 124-133, 158-183), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\orchestrator.py` (118), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\certgates.py` (35-57), `C:\Users\Galaxy\LEVI\PegaSUS\src\pegasus\ldo\edges.py` (70-90).

---

## LDO-LAG-ANNUAL-04 [HIGH] — Lag-0 contemporaneous vs lag-k directed is not cleanly separable under annual aggregation; default K=8 means 8 YEARS

I have a complete and confident picture. Here is my verdict.

---

**VERDICT: PARTIAL — the core epidemiological aliasing physics is CORRECT and unmitigated at the record level, but two of the three sub-claims misread the code (adaptive-K, and "structurally impossible").**

## What the critique gets RIGHT (upheld)

**1. The lag-0 aliasing physics is real and the code confirms it.** Under annual aggregation any delay < 12 months collapses into the same year. `lags.py:163-169` peaks the directed link over lags `1..K` and explicitly excludes lag 0; the lag-0×lag-0 block is emitted only as an *undirected* `contemporaneous` edge (`lags.py:186-194`, `edges.py:205-219`). So a genuine sub-annual cause→effect mechanism (incubation, acute care, arboviral dynamics) is mathematically forced into the undirected bucket — no directed curve is recoverable for it at annual grain. This is not a bug; it is a correct consequence of the annual sampling interval (you cannot orient a lag shorter than your sampling period). The critique's core physics is sound.

**2. The honesty gap at the record level is real and is the sharp, actionable finding.** This is where the critique lands hardest and the code confirms it:
- `LinkRecord` (records.py) carries `lag_k: int` and `response_curve_ref` **with no time-unit field**. A curve indexed in years is byte-indistinguishable from one indexed in months. Recommendation (3) is unmet.
- The `contemporaneous` `LinkRecord` (edges.py:205-219) carries **no aliasing warning**. Nothing states "delay unresolved, ≤ 1 resolution unit." A downstream consumer reading `lag_k=0` will legitimately read "no delay / simultaneous," which under annual aggregation is false — it means "delay somewhere in [0, 12) months." Recommendation (2) is unmet.
- The `resolution` string does propagate on the `EdgeReadout`/field container (edges.py:47, margins.py:35), but it never reaches the individual record, and no code binds K's semantics to it. The unit is *carried* but not *stamped per claim*.

This is a genuine HIGH-severity honesty defect: the emitted evidence object is unit-ambiguous and silent about aliasing. Sharpen it as **the** finding.

## What the critique gets WRONG (refuted)

**3. "adaptive-K ignores the time unit and spends the budget on 6-8-year delays" — misreads the risk.** `_adaptive_lag_order` (orchestrator.py:80-94) caps `K ≤ T-2` and shrinks with `p`. The critique is right that it doesn't reason about the *unit* — but its stated harm ("spends the entire estimand budget on 6-8-year delays that have near-zero prior plausibility") is backwards. The engine does **not** impose a prior that mass sits at high lags; it fits the whole distributed-lag profile and reads off the empirical peak (`peak_lag = 1 + argmax(mags[...,1:])`). If the true signal is at lag 0 (aliased sub-annual), the fit simply won't find a large lag-k partial correlation, and the `edge_threshold` + stability selection (edges.py) will drop the spurious high-lag edges. Fitting K annual lags does not *fabricate* multi-year delays; it makes the profile *available* and lets sparsity/stability zero the implausible ones. The real cost of a too-large K is compute (O((p(K+1))³)) and reduced T_eff — which is exactly what the adaptive cap addresses. So the "fabricates implausible multi-year delays" half of the risk statement is not supported by the math; the estimator is peak-reading, not prior-loading.

**4. "The flagship Zika use-case is structurally impossible at the default resolution" — overstated.** The default `resolution="year"` is a dataclass *default* (margins.py:35, assemble.py propagates `panel.resolution`), not a hard-wired constraint. The panel axis supports monthly grain: `assemble.py:67` documents `time_ids` as `year*12+month at month grain`, and the entire HSIC null machinery has a first-class `monthly_seasonal_panel` regime (`nulls.py:41-48`, `residual_scan.py:189-206`) that keys on `"month" in resolution`. The MSD's own acceptance spec (MSD-III:476, MSD-II:578) demands a **monthly** Alagoas run — so monthly is the *intended* configuration for any lagged-directed claim, and the plumbing exists. Zika is impossible at *annual* resolution (true), but not "structurally impossible" — it requires running at the resolution the spec already prescribes. (Caveat that strengthens the critique: no acceptance test exists yet — `grep zika tests/` is empty, per memory this is deferred build #29 — so the monthly path is plumbed but unproven.)

## The correction / sharpened finding

The critique should be re-scoped from "annual K=8 fabricates multi-year lags and Zika is impossible" to the tighter, fully-supported claim:

> **Under annual aggregation the LDO cannot orient sub-annual mechanisms and silently emits them as unit-ambiguous `lag_k=0` contemporaneous records with no aliasing warning and no time-unit stamp — so a fast dynamic is indistinguishable from a truly simultaneous one, and any lag-k curve is indistinguishable from months vs years.**

The right fixes are the critique's (2) and (3), which are cheap and unambiguously correct:
- Stamp `resolution`/time-unit onto every `LinkRecord` (thread `field.resolution` through `edges.py` into the record — the string is already in scope at edges.py:47).
- Add a warning like `("lag0_delay_unresolved_at_{resolution}",)` on contemporaneous records when `resolution != "cell"`/sub-daily, so `lag_k=0` is never read as "no delay."
- Optionally, a soft warning when `resolution=="year"` and a *selected* lagged edge has `peak_lag ≥ 3` (multi-year), flagging it for a chronic-exposure prior review.

Recommendation (1)'s "refuse annual K>3" is too blunt — it would wrongly block legitimate cancer-latency studies (the pancreatic C25 national study is exactly an annual, multi-year-plausible use case). A *warning*, not a refusal, is correct there.

**Files:** `src/pegasus/ldo/lags.py:73,163-169,186-194` (lag-0 exclusion, annual K=8 default), `src/pegasus/ldo/margins.py:35` (`resolution="year"` default), `src/pegasus/ldo/records.py:20-49` (no time-unit field, no lag-0 aliasing warning), `src/pegasus/ldo/edges.py:47,205-219` (resolution carried on container, absent on record; contemporaneous record unwarned), `src/pegasus/ldo/orchestrator.py:80-94` (adaptive-K is compute-bound, not prior-loaded — refutes the "fabrication" claim).

---



# Theme index (lossless)

I'll build the lossless thematic index. Let me organize all 165 findings into cross-cutting themes, ensuring every id appears.

---

# Thematic Index of LDO / Measurement-Layer Mathematical Critique Findings

**Total findings indexed: 165.** Every id below appears at least once. Themes are cross-cutting; a handful of findings are cross-referenced where they legitimately span two themes, but each is *anchored* (primary home) in exactly one theme so the count is auditable.

---

## THEME 1 — Wrong noise model: overdispersion, Poisson/Gaussian misspecification on counts

The pervasive defect: count data (deaths, cases, population) is forced through equidispersed-Poisson or Gaussian-OLS machinery when Brazilian municipal counts are strongly overdispersed / zero-inflated, and the NB/ZINB mandate of MSD-I is ignored.

- **LDO-MARG-01** — Poisson-offset margin hardcodes equidispersion, contradicting MSD-I's own NB/ZINB mandate.
- **LDO-MARG-06** — Gaussianization discards count-scale variance the denominator principle exists to preserve.
- **LDO-MARGIN-10** — Count-with-exposure margin is Poisson-offset; municipal counts overdispersed → mis-calibrated latent Z → biased correlations.
- **LDO-LAG-GAUSSCOUNT-09** — Gaussian-copula partial-corr on sparse over-dispersed counts; no NB dispersion, ties break PIT monotonicity.
- **LDO-HSIC-12** — Gaussian-graphical conditional residuals impose Gaussian/linear precision on sparse over-dispersed death counts; HSIC audits the wrong residual.
- **LDO-CAUSAL-COUNT-DIST-13** — Entire causal ladder uses Gaussian-OLS where Poisson/NB-with-exposure is mandatory.
- **LDO-CAUSAL-ITS-GAUSS-01** — ITS segmented regression runs on Gaussianized latent Z, destroying level/slope semantics.
- **LDO-DIS-SPARSE-07** — Cause-specific counts are sparse/overdispersed/zero-inflated/rank-degenerate; rank-PIT-to-Gaussian copula invalid for rare diseases.
- **POP-GAUSS-07** — All-quadratic (Gaussian) loss on population COUNTS is the wrong noise model; ignores heteroscedasticity.
- **ASTD-POISSON-03** — Fay-Feuer gamma CI assumes Poisson counts; real DATASUS counts overdispersed and person-time uncertain.
- **LDO-EXH-11** — dispersion_screen (index of dispersion) defined and spec-named but never wired into the drill decision.
- **LDO-CERT-ENH-NB-15** — Gaussian-copula partial corrs on rank-PIT counts should be replaced by NB / small-area-shrunk margins before certification.

**Cross-cutting recommendation:** Replace every count-path margin and every causal/ITS regression with a negative-binomial (or ZINB) exposure-offset model, propagate the dispersion parameter into the latent-Z calibration, and wire the already-defined dispersion_screen into the drill/certification gates. Nothing downstream is trustworthy until the marginal noise model matches the data.

---

## THEME 2 — Exposure / denominator treated as a known constant (offset assumption broken)

The population/person-time denominator is a *reconstructed, uncertain* tensor, but the math treats it as an exact offset — and several places compute the denominator itself wrongly.

- **LDO-MARG-08** — Exposure denominator treated as a known constant despite being a reconstructed uncertain tensor.
- **MQ-EXPOSURE-VARIANCE-01** — MeasuredQuantity carries count+exposure but discards exposure's own reconstruction uncertainty, breaking the offset assumption.
- **LDO-DIS-DIAG-DENOM-12** — Disease-stratified counts enter with no denominator, so the prior smooths raw counts dominated by population size, not risk.
- **LDO-EXH-08** — Reliability-weighted MEAN pooling ignores population denominators; coarse cells dominated by tiny municipalities.
- **MORAN-VALUE-NOT-RATE-STD-01** — Moran's I on raw counts/rates without population standardization; measures density autocorrelation, not risk clustering.
- **RN-N-EVENTS-RATE-01** — For an RN rate field, n_events sums the RATE vector, not the numerator count — nonsensical event count.
- **ASTD-CI-PROP-15** — Standardized-rate CIs computed but never propagated into the LDO; downstream treats point ASRs as exact.

**Cross-cutting recommendation:** Carry the denominator's reconstruction covariance forward as a genuine error term (measurement-error / errors-in-variables model on the offset), fix the rate-vs-count field wiring (n_events, exposure routing), and standardize any autocorrelation/pooling statistic by population before it enters an estimator.

---

## THEME 3 — Identifiability / under-determination: the prior IS the posterior

Estimands that are structurally unidentified from the available data, so reported "inference" is a deterministic pushforward of assumptions.

- **POP-IDENT-01** — Intercensal (age,sex,race) tensor structurally under-determined; prior IS posterior, demography fabricated.
- **RACE-IDENTIFY-07** — Race bridge with uniform local π + fixed emission matrix is a deterministic linear pushforward, not inference.
- **POP-GRAV-05** — Gross migration volume unidentified from net marginals; whole flow scale set by magic base_rate=0.01.
- **LDO-INCOH-07** — min_factor_support=3 / loading_threshold=0.3 are magic proxies for the CPW incoherence condition, not a guarantee it holds.
- **LDO-LAG-CONF-01** — Lag-extended precision identifies conditional-Granger association, not directed effects; low-rank L can't absorb lagged common trends.
- **LDO-DIS-IDENT-03** — Per-scale disease precisions confounded: nested scales collinear, band-splitting can't separate contributions.
- **ALGEBRA-CLOSURE-DIV-01** — Kind/aggregation algebra not closed: divergence and Ψ-functional outputs are non-aggregable terminal leaves; no legal ratio-of-ratios/interaction.

**Cross-cutting recommendation:** For each unidentified estimand, either (a) add an external identifying anchor (registry/survey data, informative structural prior with stated provenance) or (b) demote the output to explicitly "prior-dominated / descriptive" with a sensitivity envelope, and STOP reporting it as an inference. Test the CPW incoherence condition empirically rather than proxying it.

---

## THEME 4 — Uncalibrated magic-number thresholds

Hard-coded constants that set decision boundaries with no data-driven calibration, sensitivity analysis, or false-negative/positive guarantee.

- **LDO-MARG-13** — Magic epsilon clip 1e-6 caps |Z| at ~4.75 uniformly, censoring true extremes.
- **LDO-KAPPA-05** — κ=1 default unmotivated; sets spatial smoothing scale with no estimation/sensitivity.
- **LDO-LAMBDA-09** — λ1,λ2,α fixed at 0.1/0.1/0.05 with no data-driven selection.
- **LDO-RANDSVD-13** — 2%-of-top-eigenvalue noise floor is a magic threshold; can drop weak drivers or admit PSD noise.
- **LDO-EXH-01** — Screen thresholds 0.15/0.30 arbitrary, no false-negative guarantee.
- **LDO-EXH-06** — Pooled-signal drill reuses weight_floor=0.05 and "any stability>0" OR-gate.
- **LDO-CAUSAL-NC-BINOM-04** — Negative-control veto is ad-hoc fraction≥0.5, uncalibrated to null FP rate.
- **LDO-CAUSAL-COLLIDER-THRESH-08** — Collider decision uses hard 0.1/0.2 thresholds, no sample-size scaling or test.
- **LDO-CERT-STABTHRESH-06** — Stability threshold uncalibrated magic (0.6 policy / 0.5 readout), no Meinshausen–Bühlmann error control.
- **LDO-CERT-PATHGRID-09** — Reg-path agreement over 3-point λ grid with floor 0.66 is a coarse binary vote.
- **STATE-GATE-THRESHOLDS-01** — verified/fragile thresholds (n_eff≥100, frag<0.05, miss<0.10, risk<0.5) uncalibrated, applied across incommensurable estimands.
- **LDO-DIS-MAGIC-06** — β=0.7, min_frac=0.2, γ_disease=0.1, affinity tiers 1/0.5/0.25 unjustified, no calibration.
- **POP-MIGBOUND-06** — Symmetric migration box bound 0.25·E/n_strata unjustified magic fraction.
- **DIVERGENCE-EPSILON-01** — Fixed additive epsilon=1e-9 in log-ratio divergence makes structural zeros produce huge spurious divergences.

**Cross-cutting recommendation:** Every listed constant must be either estimated from data (empirical Bayes / cross-validation / marginal likelihood), replaced by a hypothesis test with sample-size scaling, or accompanied by a documented sensitivity sweep. Constants that gate inclusion/exclusion decisions (screens, vetoes, stability, collider) are the highest priority — they silently determine which findings survive.

---

## THEME 5 — Multiple-testing / FDR under strong dependence

The p²·(K+1) edge/lag/break hypothesis space is tested with no calibrated multiplicity control, and where FDR is applied its independence/PRDS assumptions are violated by spatial dependence.

- **LDO-LAG-MULTTEST-12** — Lag extension multiplies hypotheses by (K+1)² with fixed edge_threshold=0.05, no curve-level FDR.
- **LDO-EXH-07** — Per-pair screen with no multiple-testing control over the O(p²) grid.
- **LDO-HSIC-05** — BH/BY/Storey assume PRDS/independence; spatial panel p-values strongly non-PRDS, FDR unproven.
- **LDO-HSIC-06** — BY×harmonic(m) inflation collides with 1/(perms+1) floor → national-scale edges structurally uncertifiable at α=0.1.
- **LDO-CAUSAL-MHT-14** — No multiple-testing control across edges, break candidates, and control series simultaneously escalated.
- **LDO-CERT-MULTITEST-07** — No MHT over p²·(K+1) edge tests; FDR fields plumbed but unused.
- **LDO-DIS-STAB-14** — Disease prior lowers ℓ1 for "related" pairs, biasing stability frequencies upward — multiplicity control no longer calibrated.

**Cross-cutting recommendation:** Adopt a dependence-aware error-rate procedure (e.g. permutation/knockoff-based FDR that respects the spatial block structure, or hierarchical curve-level FDR for the lag axis), actually wire the plumbed-but-unused FDR fields, and re-derive the certifiability budget so the permutation-count floor doesn't make national edges structurally uncertifiable. Recalibrate stability-selection frequencies whenever a structured penalty is applied.

---

## THEME 6 — Spatial-autocorrelation mishandling and double-correction

Spatial dependence is variously ignored, double-counted, or corrected with an ad-hoc formula lacking sampling-theory basis; permutation nulls are non-exchangeable on a spatial panel.

- **MORAN-DOUBLECORRECT-01** — 1/(1+MoranI) n_eff deflation double-counts spatial dependence the GMRF precision already models; ad-hoc, no sampling basis.
- **LDO-WHITEN-04** — (κI+L_W)^{1/2} whitening imputes missing cells to 0 and uses unnormalized-degree Laplacian → heterogeneous over/under-correction.
- **LDO-LAG-KAPPA-11** — Spatial whitening degrades to no-op κ-scaling for municipalities absent from adjacency, mixing whitened/unwhitened variables.
- **LDO-HSIC-02** — Within-(block×bucket) restricted permutation is NOT exchangeable for a spatially autocorrelated panel; stays anticonservative.
- **LDO-HSIC-11** — With one spatial OR temporal block, code falls back to full iid permutation — the maximally anticonservative null.
- **MORAN-ORDERING-PROXY-01** — q_tensor Moran fallback uses 1-D tensor-ordering contiguity — essentially random adjacency, meaningless autocorrelation.
- **LDO-CAUSAL-ITS-AGG-02** — National spatial-mean ITS ignores ecological heterogeneity, treats one national series as unit.
- **ECOFALLACY-GUARD-01** — Ecological-fallacy guard only aggregates numerator to denominator geography; does NOT prevent inference across silently-broadcast demographic strata.

**Cross-cutting recommendation:** Decide *once* where spatial dependence is modeled (the GMRF precision) and remove redundant deflation; use a normalized/row-standardized Laplacian with principled missing-cell handling (not impute-to-0); and replace restricted permutation with a genuinely exchangeable spatial null (e.g. conditional/toroidal or model-based resampling) with a hard guard against the iid fallback.

---

## THEME 7 — Temporal autocorrelation not whitened; stationarity over an epidemiological transition

Serial dependence and seasonality inflate significance; a single time-invariant structure is imposed across 2000–2024.

- **LDO-TIME-03** — Temporal autocorrelation never whitened before the fit; Σ_time⁻¹ is post-hoc telemetry only.
- **LDO-AR1-14** — A single globally-pooled AR(1) φ stands in for all temporal dependence across every variable and municipality.
- **LDO-LAG-STAT-02** — Single response curve pooled 2000–2024, assuming time-invariant lag structure across a major transition.
- **LDO-LAG-OVERLAP-05** — Overlapping lag windows induce mechanical serial correlation, inflating effective sample size.
- **LDO-CAUSAL-ITS-AUTOCORR-03** — ITS t-stats assume iid Gaussian residuals; segmented health series autocorrelated/seasonal → inflated significance.
- **LDO-CAUSAL-ITS-SINGLEBREAK-10** — Single-break, no-anticipation, no-concurrent-intervention ITS indefensible for national policy shocks.
- **LDO-LAG-ANNUAL-04** — Lag-0 contemporaneous vs lag-k directed not separable under annual aggregation; K=8 means 8 YEARS.
- **POP-SMOOTH-08** — Flat RW2 smoothness on age/migration is a stationary global prior fighting the epidemiological transition.

**Cross-cutting recommendation:** Whiten temporal dependence *before* the fit (not as telemetry), allow φ/lag structure to vary across regime boundaries (time-varying or change-point model), correct ITS inference for autocorrelation+seasonality (Newey–West / GLS, multi-break, anticipation windows), and account for the mechanical serial correlation of overlapping windows in the effective-n.

---

## THEME 8 — PSD / covariance-matrix integrity

Pairwise-complete correlation is indefinite and the "nearest correlation" repair is not the metric-nearest matrix, silently distorting the reported partial correlations.

- **LDO-PSD-01** — Pairwise-complete correlation provably indefinite; MSD treats it as a valid covariance.
- **LDO-NCORR-02** — "_nearest_correlation" is eigen-clip+rescale, NOT metric-nearest (Higham).
- **LDO-LAG-NEARCORR-07** — Nearest-correlation PSD projection (eigen-clip floor 1e-3) silently distorts the off-diagonal partial correlations being reported.
- **LDO-GLASSO-11** — graphical_lasso failure silently falls back to pinv, abandoning sparsity/regularization with no signal.

**Cross-cutting recommendation:** Replace eigen-clip+rescale with a true Higham nearest-correlation solve (or a positive-definite estimator that never produces an indefinite input, e.g. shrinkage), and make the glasso→pinv fallback loud (flag/refuse) rather than silent.

---

## THEME 9 — Separability (Kronecker) assumptions forbidding the interesting physics

The Ω⊗Σ_space⁻¹⊗Σ_time⁻¹ factorization rules out exactly the space-varying, lag-varying, spatially-spreading epidemic dependence the study exists to find.

- **LDO-SEP-08** — Separability forbids space-varying and lag-varying dependence — the epidemiologically interesting case.
- **LDO-LAG-KRONSEP-10** — Kronecker separability assumed, but space/var interaction is exactly where lagged-spatial epidemic spread lives.
- **POP-SEP-10** — Locality-block separability exact only because net-migration is imposed as a per-locality total, discarding spatial coupling migration actually has.

**Cross-cutting recommendation:** Introduce at least a low-rank non-separable component (or a spatially-varying-coefficient / spatiotemporal interaction term) so lagged spatial spread and migration coupling are representable; validate separability empirically before assuming it.

---

## THEME 10 — Optimizer / convergence integrity

Solvers that certify convergence too early, cap iterations below convergence, use inexact prox steps, or return prior-dominated iterates.

- **LDO-ADMM-06** — ADMM stopping test uses only primal residual; can certify convergence at a non-stationary iterate.
- **LDO-SMOOTH-12** — Smoothness prior enters via a single linearized prox-gradient step per ADMM iteration, not the exact proximal operator; convergence claim overstated.
- **POP-ITER-02** — max_iterations=12 hardcoded guarantees the solver never converges, returns prior-dominated iterate.
- **POP-CLOSURE-03** — Closure via Euclidean simplex projection (subtract-and-clip) distorts composition non-multiplicatively, can zero small strata.

**Cross-cutting recommendation:** Use both primal AND dual residual stopping criteria, an exact proximal operator (or prove the linearization's error bound), remove hardcoded iteration caps in favor of tolerance-based stopping, and enforce compositional closure multiplicatively (e.g. in ILR/softmax space) rather than by Euclidean clipping.

---

## THEME 11 — Copula / rank-PIT margin defects

Beyond the count-noise issue (Theme 1), the copula transform itself has structural, tie-handling, tail-dependence, and leakage flaws.

- **LDO-MARG-07** — Gaussian copula imposes tail-independence that epidemic threshold/clustering dependence violates.
- **LDO-MARG-09** — Structural vs sampling zeros conflated; rank PIT spreads them across the same CDF band.
- **LDO-MARG-10** — inverse_gaussianize uses biased off-by-one nearest-rank empirical quantile, no interpolation.
- **LDO-MARG-12** — Count-vs-rank routing is a numeric heuristic (any positive exposure), not a typed extensive/intensive decision.
- **LDO-CERT-COPULA-08** — Uncertainty ignores the randomized-PIT copula transform's own sampling noise and its instability for rare-event counts.

**Cross-cutting recommendation:** Adopt a tail-dependent copula (or an explicit extremes model) for epidemic clustering, separate structural from sampling zeros before ranking, use interpolated (not nearest-rank) quantiles, make the extensive/intensive routing a typed metadata decision, and propagate the PIT transform's own sampling noise into certification.

---

## THEME 12 — Frozen randomness / seed-determined bias

Stochastic transforms whose single frozen realization is treated as data, making "randomized" procedures deterministically biased and un-integrated into nulls.

- **LDO-MARG-03** — Randomized-PIT jitter drawn once and frozen; stability selection treats one realization as data.
- **LDO-HSIC-09** — Feature-map HSIC uses fixed seed-determined landmark/frequency draw for statistic AND null → deterministic approximation bias not in the null.
- **RACE-BOOT-10** — Bootstrap/posterior use fixed RNG seeds, resample only source-category multinomial; ignore dominant small-count variance → frozen intervals.

**Cross-cutting recommendation:** Integrate the randomization over its distribution — average the statistic over multiple PIT jitters / landmark draws, and let the null inherit the same randomization — rather than freezing one seed and treating it as ground truth.

---

## THEME 13 — Data-leakage / double-dipping / in-sample scoring

Statistics computed on the same data (or full-sample quantities) they then validate against, violating cross-fitting.

- **LDO-MARG-04** — Empirical CDF computed on full national sample then subsampled — leak across stability folds.
- **LDO-HSIC-01** — Residual scan double-dips: HSIC on in-sample joint-model residuals, violating cross-fitting mandate.
- **LDO-CERT-SLICE-SCOPE-04** — Exact-certifies-approximate on a same-slice re-run certifies the METHOD, not the national data scope claimed.
- **POP-DEATHRATE-09** — δ estimated as SIM_deaths/census_anchor injects circular self-reference, active only at census years.
- **LDO-CAUSAL-NC-NEFF-12** — Break detection and negative-control reuse the same min_level_t=3.0 OLS test → perfectly correlated errors, veto not independent.

**Cross-cutting recommendation:** Enforce genuine cross-fitting / sample-splitting: fit margins and joint model on disjoint folds from the HSIC/certification data, certify approximation on the *stressed out-of-scope* population, break the circular denominator estimation, and ensure vetoes use statistically independent tests.

---

## THEME 14 — Weights defined-but-unused / spec-divergent reliability handling

Per-cell reliability/n_eff weights that MSD mandates are computed and then dropped, plus estimators that measure a different quantity than the spec names.

- **LDO-MARG-05** — Per-cell reliability weights W computed and propagated but NEVER used in margin or precision estimator.
- **LDO-MARG-11** — Rank margin ignores reliability/observation weights when forming the ECDF (weighted ECDF needed).
- **LDO-LAG-NEFF-13** — Per-cell reliability weights w_{j,s,t} defined but never used in the lag covariance.
- **NEFF-KISH-WEIGHT-01** — Live n_eff uses rate VALUES as Kish weights, not the denominator — dimensionless, meaningless effective sample size.
- **KISH-ABS-NEGATIVE-01** — Kish n_eff takes abs() of weights, accepting negative-valued fields (log-ratios, z-scores) as if counts.
- **DENOMFRAG-SPEC-DIVERGE-01** — denom_fragility measures MISSING-denominator share, not the §3.12.8 below-μ_min share; n=3 counts as non-fragile.
- **CV-COUNT-VS-SAMPLE-01** — Live CV is cross-cell sample CV (heterogeneity), not the §3.12.9 sampling CV of the estimate — opposite things.
- **TEMPROUGH-SPEC-01** — Temporal roughness uses first-difference RMS/mean, not the §3.12.11 second-difference curvature per spec.
- **LDO-NEFF-15** — n_samples reported as min pairwise coverage; ignores overlap heterogeneity and serial/spatial dependence → overstated edge precision.

**Cross-cutting recommendation:** Actually thread the mandated reliability weights through the ECDF, margin, precision, and lag-covariance estimators; and fix each spec-divergent statistic (Kish must use the denominator not the value, fragility must be the below-μ_min share, CV must be the sampling CV, roughness must be second-difference) so the reported quantity matches the spec name.

---

## THEME 15 — Causal-claim overreach / faithfulness & orientation fragility

The causal ladder labels associations as quasi-experimental and orients edges with fragile, order-dependent, faithfulness-dependent logic.

- **LDO-CAUSAL-RUNG-SEMANTICS-11** — Auto-applied Rung-2 "quasi-experimental" tag with no adjustment set / exchangeability check — association wearing a causal badge.
- **LDO-CAUSAL-FAITHFUL-07** — Collider orientation presumes faithfulness, routinely violated by cancellation / near-deterministic denominators.
- **LDO-CAUSAL-COLLIDER-CONFLICT-09** — Conflicting v-structures resolved "last write wins", no Meek propagation → order-dependent, globally inconsistent.
- **LDO-CAUSAL-LINGAM-POOL-05** — Pairwise LiNGAM on flattened S×T panel as iid; autocorrelation + municipality mixture fabricate non-Gaussianity, break ANM.
- **LDO-CAUSAL-LINGAM-LINEAR-06** — Orientation uses LINEAR OLS residual but scores NONLINEAR dependence — inconsistent ANM, mis-orients.
- **LDO-CAUSAL-DID-ORPHAN-15** — difference_in_differences implemented but never wired, no parallel-trends test — dead or dangerously naive.
- **LDO-HSIC-15** — HSIC on a pair of joint-model residuals can't distinguish a real nonlinear edge from a shared unmodeled latent driving both.

**Cross-cutting recommendation:** Downgrade every auto-applied Rung-2 label lacking an explicit adjustment set + exchangeability argument; add Meek propagation for order-independent orientation; use a faithfulness-robust and autocorrelation-aware causal-discovery method; make the LiNGAM residual/scoring consistent (both nonlinear or both linear); and either wire DiD with a parallel-trends test or remove it.

---

## THEME 16 — Certification-layer statistical validity

The exact-vs-approx certification combines wrong SEs, mismatched units, and confirmatory conjunctions with no aggregate error budget.

- **LDO-CERT-FISHERZ-01** — Fisher-z SE 1/√(n_eff−3) is the wrong SE for a whitened partial correlation on pairwise-complete data.
- **LDO-CERT-UNITS-02** — stat_se (Fisher-z) and numerical_error (eigenvalue ratio) combined in quadrature despite different units.
- **LDO-CERT-LATENTBAND-03** — Exact-vs-approx band compares latent_shared LOADINGS as if partial correlations.
- **LDO-CERT-SLICE-BIAS-05** — Certification slice is the max-coverage least-sparse 700 localities — opposite of where the approximation is stressed.
- **LDO-CERT-HOLDOUT-NULL-10** — Persistence rate has no null; high value can arise from trend/autocorrelation; sign-only matching weak.
- **LDO-CERT-CONJ-CONFIRM-11** — Conjunction of confirmatory filters, no aggregate error budget; correlated conjuncts treated as independent.
- **LDO-CERT-BAND-ZERO-12** — Zero-uncertainty edges collapse agreement band to 0 (any float diff disagrees); missing overlap silently passes.
- **LDO-CERT-SVDERR-13** — numerical_error uses largest DROPPED eigenvalue ratio, not the Halko–Martinsson–Tropp probabilistic bound the MSD invokes.
- **LDO-CERT-ENH-CONFORMAL-14** — Recommends replacing analytic-Fisher-z + heuristic-stability gate with subsampling-bootstrap conformal edge inference.
- **LDO-EXH-13** — Two divergent CoverageManifest classes, different schemas, both claim §VIII.2(3).

**Cross-cutting recommendation:** Derive the correct SE for whitened partial correlations (or drop analytic SE for a subsampling-bootstrap/conformal interval), stop combining incommensurable units in quadrature, compare like-with-like (loadings vs loadings), certify on the *stressed sparse* slice, give the conjunction a real aggregate error budget, and use the proper HMT probabilistic bound for randomized-SVD error. Unify the duplicate CoverageManifest.

---

## THEME 17 — Exhaustiveness / coverage-guarantee weaknesses

The "no false negatives" machinery asserts rather than certifies recall, with point estimates, wrong pooling, and a refit that reintroduces the blind spot it claims to remove.

- **LDO-EXH-02** — Per-unit correlation over T=26 points is high-variance; its std is sampling noise, not true heterogeneity.
- **LDO-EXH-03** — False-negative-rate estimator is a bare point estimate — no Wilson CI, no sample-size justification, mismatched denominator.
- **LDO-EXH-04** — Coarse→fine is a naive REFIT, not the multiresolution-posterior readout §III.3 mandates, reintroducing Simpson/cancellation blind spot.
- **LDO-EXH-05** — Sparsity-of-truth prior indefensible for a dense socioeconomic field; stated but never tested against actual edge density.
- **LDO-EXH-09** — cod6-prefix pooling assumes stable municipality codes, ignores boundary changes / spatial autocorrelation.
- **LDO-EXH-10** — Subgroup screen uses raw bivariate correlation, not the LDO's conditional/latent-controlled effect — screens a different quantity.
- **LDO-EXH-12** — Audit/screen swallow all exceptions to None: broken audit reports "no false negatives measured" indistinguishably from "FNR truly low".
- **LDO-EXH-14** — No conformal / distribution-free guarantee on the drill decision; recall could be certified, not asserted.

**Cross-cutting recommendation:** Replace point-estimate FNR with a Wilson/conformal lower bound on recall, do the true multiresolution-posterior readout (not a refit), test the sparsity prior against observed edge density, screen the *same conditional quantity* the engine estimates, and never swallow audit exceptions into a "clean" result.

---

## THEME 18 — HSIC residual-scan specifics (nonlinear-dependence audit)

Beyond the double-dip (T13) and non-exchangeable null (T6): permutation-regime mislabeling, signal-destroying coarsening, biased subsampling, and reproducibility divergences.

- **LDO-HSIC-03** — Names a §6.8 cyclic-time-shift regime but executes a different, weaker permutation.
- **LDO-HSIC-04** — Memory-triggered coarsening AVERAGES residuals per block×bucket, destroying the nonlinear/non-Gaussian signal HSIC exists to detect.
- **LDO-HSIC-07** — Complete-case + sparsest-variable trimming makes the residual sample non-random, biased toward dense urban high-count municipalities.
- **LDO-HSIC-08** — Spatial block = first 2 digits of space id assumes stable 7-digit IBGE code, ignores AMC boundary changes.
- **LDO-HSIC-10** — Median-heuristic RBF bandwidth on seeded 2048-pair subsample, ill-defined for heavy-tailed/near-degenerate count residuals.
- **LDO-HSIC-13** — Memory guard compares float64 estimate against 60% of INSTANTANEOUS RAM — unstable, non-reproducible, can OOM or spuriously coarsen.
- **LDO-HSIC-14** — Two HSIC paths use different tie-handling (>= vs >=−1e-15) — subtle reproducibility/validity divergence.

**Cross-cutting recommendation:** Execute the actual named permutation regime, replace signal-destroying averaging with sub-sampling that preserves the joint distribution, weight/stratify the residual sample to restore representativeness, harmonize boundary codes, make the bandwidth and memory gate deterministic and reproducible, and unify tie-handling across both code paths.

---

## THEME 19 — Boundary-change / geography-harmonization robustness

Municipality codes are treated as stable across the study window; splits/merges/AMC changes create spurious discontinuities.

- **LDO-EXH-09** (also T17) — cod6-prefix pooling assumes stable codes, ignores boundary changes.
- **LDO-HSIC-08** (also T18) — 2-digit spatial block assumes stable 7-digit IBGE code, ignores AMC changes.
- **LDO-DIS-BOUNDARY-13** — Disease counts keyed to municipality_cod6 with no boundary-change harmonization → discontinuities misread by prior/whitening.
- **POP-DISC-04** — Method discontinuity at informative/non-informative and census-year boundaries produces artefactual denominator jumps.
- **INTERP-NEGMIG-13** — Net-migration residual conflates registration undercount and boundary changes with true migration; structurally negative/unbounded.

**Cross-cutting recommendation:** Harmonize all geography onto stable minimum-comparable-areas (AMCs) across the full 2000–2025 window before any pooling, whitening, or interpolation, and separate boundary-change artifacts from true migration in the net residual.

---

## THEME 20 — Age-standardization / small-area measurement layer

Direct standardization and its CIs are statistically indefensible at municipal scale; age-handling and numeric-stability defects compound it.

- **ASTD-SAE-02** — Direct standardization used for small-area municipal rates where indefensible; MSD prescribes SAE/BYM, ignored.
- **ASTD-AGEMISS-01** — Age-unknown deaths silently binned into 0–4, inflating infant rates, deflating standardized total.
- **ASTD-STDPOP-04** — WHO World Standard default; tiny 85+ weight discards most signal for an elderly cancer, destabilizes municipal estimates.
- **ASTD-OPENEND-05** — 85+ open-ended person-years vs single-year weights; ages >100 collapse to one bin, no person-time consistency check.
- **ASTD-NUMSTAB-06** — Fay-Feuer df = 2·asr²/v can underflow to df≤0 for tiny ASRs; ASR-per-1 scale makes v minuscule.

**Cross-cutting recommendation:** Switch municipal rates to a small-area estimator (BYM/SAE) with shrinkage, choose a reference standard appropriate to an elderly-cancer signal, handle age-unknown deaths by proportional redistribution (not 0–4 dumping), reconcile open-ended person-time, and guard the Fay-Feuer df/scale against underflow. (Overdispersion of the CI itself is anchored in T1 via **ASTD-POISSON-03**.)

---

## THEME 21 — Race bridge / demographic reconstruction internals

Beyond the top-level identifiability (T3): fabricated uncertainty, uniform-fallback reallocation, and non-cohort interpolation.

- **RACE-PRIOR-08** — Emission-matrix uncertainty is fabricated Dirichlet(250) around an unvalidated C; race_bridge_cv/CIs understate epistemic uncertainty.
- **RACE-DECISION-09** — Missing-race reallocation uses same (possibly uniform) π_local → missing race spread evenly across five categories.
- **INTERP-COHORT-11** — Intercensal composition is LINEAR share interpolation, not cohort-component; ages nobody, demographically impossible age structures.
- **INTERP-GEOM-12** — Geometric total-closure interpolation assumes constant growth, clamps composition beyond last census, misses the transition.
- **POP-GEOM-13** — Geometric closure interpolation + extrapolation beyond census range compound errors, can diverge for high-growth frontier municipalities.
- **POP-AGGREG-11** — Per-locality net-migration distributed across strata by unweighted gradient, spreading migration uniformly over all (a,x,r) cells.
- **POP-BIRTH-15** — Birth loss ignores infant mortality and mid-year timing; the UF→Region→Brazil race fallback cascade is not implemented.
- **POP-VALTUNE-14** — Hyperparameter tuning against census cells cannot observe intercensal reconstruction error — the quantity that matters is unmeasured.

**Cross-cutting recommendation:** Move to a cohort-component reconstruction that actually ages the population, ground emission-matrix uncertainty in validated data (not an invented Dirichlet), replace uniform missing-race reallocation with an informative local prior, and design a validation target that actually observes intercensal error rather than only census-coincident cells.

---

## THEME 22 — Numerically explosive ILR / structural-zero handling

Compositional transforms with tiny epsilon floors blow up on structural-zero strata in small municipalities.

- **POP-RACEILR-12** — ILR race loss uses epsilon=1e-12 flooring and 1/P chain-rule division, numerically explosive for structural-zero strata.
- **RACE-ILR-14** — Race-composition prior enforced in ILR coordinates with epsilon=1e-12, explosive for structural-zero categories in small municipalities.

**Cross-cutting recommendation:** Handle structural zeros as genuine zeros (zero-aware compositional model / Bayesian multiplicative replacement) rather than an ad-hoc 1e-12 floor, and bound the chain-rule division to prevent blow-up in sparse strata.

---

## THEME 23 — Disease-axis prior construction (L_D)

The ICD-similarity prior is bureaucratic not epidemiological, partly dead code, and mis-projects US groupers onto CID-10.

- **LDO-DIS-TREE-01** — ICD tree-distance (1/0.5/0.25) is bureaucratic adjacency, not epidemiological similarity; couples unrelated, separates linked diseases.
- **LDO-DIS-SCALES-02** — Sum-of-scales per-scale GMRF is dead code: disease_scale_precisions never populated, only flat single-γ Laplacian runs.
- **LDO-DIS-MULTICAUSE-04** — Coercing the disease axis into a σ_C count variable discards multi-cause/comorbidity structure.
- **LDO-DIS-MAXWEIGHT-05** — Variable-to-variable affinity is MAX closeness over the code cross-product, inflating coupling between broad multi-code variables.
- **LDO-DIS-CHAPTER-08** — WHO-gap table collapses the arbovirus block so dengue/chikungunya/Zika are only same-CHAPTER — under-couples Brazil's highest-burden cluster.
- **LDO-DIS-RANGE-09** — Custom-concept range membership uses lexical string comparison, mis-handling cross-letter/zero-padding boundaries.
- **LDO-DIS-OVERLAP-10** — Mechanical-overlap guard only works for enumerated starts_with_any predicates; range/regex restrictions yield code_set=None, silently bypassing the check.
- **LDO-DIS-PROV-11** — CCSR/CCIR groupers are US ICD-10-CM applied to CID-10, returning only PRIMARY category → multi-label concepts inconsistent/incomplete, yet enter as "approximate".

**Cross-cutting recommendation:** Rebuild L_D from an epidemiological-similarity source (shared etiology/transmission/risk-factor structure) rather than ICD tree bureaucracy, actually populate the per-scale precisions or delete the dead path, retain multi-cause structure as a multivariate outcome, fix the arbovirus block coupling, replace lexical range membership with proper code comparison, and reject (not "approximate") US-grouper multi-label projections that can't be validated on CID-10.

---

## THEME 24 — GPU-acceleratable / dense-scaling heavy math

Operations that both threaten national-p scalability and are natural GPU/randomized-linear-algebra targets.

- **LDO-DIS-QUAD-DENSE-15** — Disease smoothness path densifies: variable_affinity/L_D dense p×p, smoothness operator dense Kron over p·(K+1), contradicting sparse-L_D claim, blocking national p.
- **LDO-RANDSVD-13** (also T4) — Randomized-SVD noise-floor / point-estimate error propagation — a randNLA primitive whose error bound (HMT) should be probabilistic and is GPU-parallelizable.
- **LDO-CERT-SVDERR-13** (also T16) — numerical_error should use the Halko–Martinsson–Tropp probabilistic bound — the same randomized-SVD kernel amenable to GPU batching.

**Cross-cutting recommendation:** Keep L_D and the smoothness operator sparse (or block-structured) so national-p is feasible, and route the randomized-SVD / precision-estimation kernels through the existing GPU path with a *proper* HMT probabilistic error bound rather than a point estimate. This theme is where scalability and error-quantification fixes coincide.

---

# Ranked list of CRITICAL findings

The input flags **20 CRITICAL** findings. Ranked by blast radius (how much downstream inference each one invalidates):

1. **NEFF-KISH-WEIGHT-01** — n_eff uses rate values not denominators → *every rate field's* effective sample size is dimensionless and meaningless, poisoning all precision/gating downstream. Foundational; fix first.
2. **POP-IDENT-01** — Intercensal demographic tensor under-determined, prior IS posterior → the denominator every rate divides by is fabricated. Corrupts the entire measurement layer.
3. **LDO-MARG-01** — Poisson-offset margin hardcodes equidispersion against MSD-I's own NB/ZINB mandate → mis-calibrated latent Z for every count field.
4. **LDO-MARG-02** — Single global λ pushes real spatial/temporal rate variation into the latent "surprise" scale → systematic contamination of every edge.
5. **LDO-TIME-03** — Temporal autocorrelation never whitened before the fit → serial dependence inflates every temporal edge; Σ_time⁻¹ is decorative.
6. **LDO-HSIC-01** — Residual scan double-dips on in-sample residuals → the nonlinear-edge audit is anticonservative by construction.
7. **LDO-HSIC-02** — Restricted permutation is non-exchangeable on a spatial panel → HSIC nulls stay anticonservative even when double-dipping is fixed.
8. **LDO-CAUSAL-ITS-GAUSS-01** — ITS runs on Gaussianized Z, destroying level/slope semantics → the quasi-experimental causal claims measure nothing interpretable.
9. **LDO-LAG-CONF-01** — Lag precision identifies conditional-Granger association, not directed effects → the directed/lagged findings overclaim causality.
10. **LDO-LAG-STAT-02** — Single response curve pooled across a major epidemiological transition → time-invariance assumption is false where it matters most.
11. **LDO-CERT-FISHERZ-01** — Wrong SE for whitened partial correlations → the certification uncertainty is miscalibrated at its core.
12. **LDO-CERT-SLICE-SCOPE-04** — Exact-certifies-approximate on same slice certifies the method, not the national scope claimed → certification doesn't cover what the MSD says it does.
13. **RACE-IDENTIFY-07** — Race bridge is a deterministic linear pushforward → self-declared race counts are not inferred.
14. **ASTD-SAE-02** — Direct standardization statistically indefensible at municipal scale (SAE/BYM prescribed, ignored) → municipal ASRs unreliable.
15. **LDO-EXH-01** — Screen thresholds 0.15/0.30 arbitrary with no false-negative guarantee → the exhaustiveness claim rests on magic numbers.
16. **LDO-EXH-04** — Coarse→fine naive refit reintroduces the Simpson/cancellation blind spot §VIII.2 claims to avoid → coverage guarantee is self-undermined.
17. **LDO-EXH-02** — Per-unit correlation over T=26 is dominated by sampling noise, misread as heterogeneity → the drill's heterogeneity signal is noise.
18. **LDO-EXH-03** — FNR is a bare point estimate with mismatched denominator → recall is asserted, not bounded.
19. **LDO-HSIC-01/02 aside**, **LDO-LAG-STAT-02 aside** — (already listed).

(For completeness, the two remaining CRITICALs, ranked lower because their damage is more localized:)

20. **LDO-MARGIN-10** — Count-with-exposure margin Poisson-offset on overdispersed municipal counts → biased correlations (a specific, high-severity instance of the T1 pattern already led by LDO-MARG-01).

**Bottom line for the lead:** the CRITICAL cluster collapses into three root causes — (a) the denominator/effective-sample-size foundation is fabricated or dimensionless (NEFF-KISH-WEIGHT-01, POP-IDENT-01, RACE-IDENTIFY-07, ASTD-SAE-02), (b) the marginal/temporal/spatial noise model is misspecified and un-whitened (LDO-MARG-01/02, LDO-MARGIN-10, LDO-TIME-03, LDO-LAG-STAT-02, LDO-HSIC-01/02, LDO-CAUSAL-ITS-GAUSS-01, LDO-LAG-CONF-01), and (c) the honesty layer that is supposed to catch all this — certification and exhaustiveness — is itself miscalibrated (LDO-CERT-FISHERZ-01, LDO-CERT-SLICE-SCOPE-04, LDO-EXH-01/02/03/04). Fix (a) before (b) before (c); certifying on top of a fabricated denominator certifies nothing.