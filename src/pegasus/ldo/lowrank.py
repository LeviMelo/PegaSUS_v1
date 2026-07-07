"""LDO Layer 2 — sparse + low-rank latent-variable decomposition (MSD-II §II.6.1, MII-LDO-03).

Decompose the variable-dependency operator ``Ω_var = S − L`` with ``S`` sparse
(the direct + lagged links) and ``L`` low-rank PSD (shared latent drivers — the
ST-DFM factors; e.g. an epidemic wave co-moving many variables). This is the
Chandrasekaran–Parrilo–Willsky latent-variable graphical decomposition, solved by
ADMM:

    min_{S,L}  -logdet(S − L) + tr((S − L) C) + λ₁‖S‖₁,off + λ₂ tr(L),   L ⪰ 0

Variables that co-move only through a common factor load on ``L`` and are reported
as ``latent_shared`` (a confounded pair), NOT as dense direct edges in ``S`` — so
``S`` recovers the direct structure *net of* the shared driver. ``L``'s eigenfactors
are the latent drivers (the ST-DFM factors as Layer 2).

§V.3 Johnson–Lindenstrauss neighborhood-regression sketch (scope boundary): that clause targets a
**Meinshausen–Bühlmann per-node neighborhood-regression** estimator, where each variable is
lasso-regressed on the others and a JL projection sketches the ``(1±ε)`` regression geometry. The
LDO does NOT use neighborhood regressions — it fits the JOINT precision by this CPW ADMM — so there
is no per-node regression design to JL-sketch. The scale reduction §V.3 seeks is instead provided
by the **randomized SVD** low-rank readout (:func:`_low_rank_factors`, ``O(p²r)`` vs ``O(p³)``) and
the §V.2 Kronecker factoring of the joint operator. The JL-sketch requirement is therefore
architecturally N/A to the joint estimator, not an unbuilt feature of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseLowRankFit:
    S: np.ndarray                       # sparse direct precision
    L: np.ndarray                       # low-rank PSD latent component
    precision: np.ndarray               # Ω_var = S - L
    factor_loadings: np.ndarray         # (p x r) loadings of the recovered latent factors
    factor_values: np.ndarray           # (r,) eigenvalues of L
    converged: bool
    iterations: int
    numerical_error: float = 0.0        # §V.6(2): relative randomized-SVD truncation error (0 = exact)
    work_dtype: str = "float64"         # §V.1: bulk-iterate precision actually used (after any escalation)
    cond_escalated: bool = False        # §V.4: float32 was requested but cond(C) forced float64
    direct_edges: list[tuple[int, int, float]] = field(default_factory=list)   # (i,j,partial_corr) from S
    latent_shared: list[tuple[int, int, float]] = field(default_factory=list)  # (i,j,shared_loading) from L
    incoherence: float | None = None    # CPW identifiability score in [0,1]; None when L is rank-0
    well_identified: bool = True        # incoherence above the CPW guard threshold (readout warns if False)


def _soft_threshold_offdiag(M: np.ndarray, tau: float) -> np.ndarray:
    out = np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)
    np.fill_diagonal(out, np.diag(M))  # do not shrink the diagonal
    return out


def _prox_neg_logdet(M: np.ndarray, rho: float) -> np.ndarray:
    """argmin_R -logdet R + (rho/2)||R - M||^2  (R symmetric PD) via eigenvalues.

    §V.1 mixed precision: the eigendecomposition (a *reduction / log-det*) is always done
    in float64 even when the ADMM iterates are stored in float32 — the spectrum-reshaping
    step is where float32 round-off would corrupt small eigenvalues (and hence the log-det).
    The result is returned as float64; the caller casts back to the working dtype.
    """
    M = np.asarray(0.5 * (M + M.T), dtype=np.float64)
    vals, vecs = np.linalg.eigh(M)
    d = (vals + np.sqrt(vals**2 + 4.0 / rho)) / 2.0
    return (vecs * d) @ vecs.T


def _psd_project_shifted(M: np.ndarray, shift: float) -> np.ndarray:
    """PSD projection of (M - shift*I): eigen-clip at 0.

    ``M - shift·I`` shares its eigenvectors with ``M`` and only shifts the eigenvalues
    by ``-shift``, so the shift is applied to the eigenvalues after a single ``eigh(M)``
    — identical result, without allocating/subtracting a ``shift·I`` matrix every ADMM
    iteration (this runs twice per iteration for up to ``max_iter`` iterations). Like
    :func:`_prox_neg_logdet`, the eigh runs in float64 (§V.1 reduction) and returns float64.
    """
    M = np.asarray(0.5 * (M + M.T), dtype=np.float64)
    vals, vecs = np.linalg.eigh(M)
    vals = np.clip(vals - shift, 0.0, None)
    return (vecs * vals) @ vecs.T


def _low_rank_factors(
    L: np.ndarray, *, rank_cap: int, randomized: bool, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenfactors of the PSD low-rank component ``L`` (values desc, loadings aligned).

    MSD-III §V.3: the shared-driver factors are few, so at scale the top-``rank_cap``
    eigenpairs are recovered by **randomized SVD** (Halko–Martinsson–Tropp) in O(p²·r)
    instead of a dense O(p³) eigh — bounded error since ``L`` is genuinely low-rank
    (the tail past its numerical rank is zero). This is a post-convergence readout; it
    does not touch the ADMM iteration (or the CPW S/L split). The dropped-tail fraction is
    propagated as ``numerical_error`` into each latent_shared edge's uncertainty. This
    intra-fit noise floor is NOT itself the §V.6 exact-certifies-approximate contract — that
    contract is enforced separately by :func:`pegasus.validation.holdout.certify_approximation_on_slice`,
    which runs the whole pipeline exact-vs-approximate on a real slice. Small ``p`` uses the exact eigh.
    """
    p = L.shape[0]
    if randomized and p > 2 * rank_cap and rank_cap >= 1:
        from sklearn.utils.extmath import randomized_svd

        # L is symmetric PSD → its SVD is its eigendecomposition (U singular vectors are
        # eigenvectors, singular values are the non-negative eigenvalues).
        U, s, _ = randomized_svd(L, n_components=min(rank_cap, p - 1), random_state=seed,
                                 n_oversamples=10, n_iter=4)
        return s, U
    vals, vecs = np.linalg.eigh(L)
    return vals[::-1], vecs[:, ::-1]


def _cpw_incoherence(
    S: np.ndarray, factor_loadings: np.ndarray, *, edge_threshold: float
) -> tuple[float | None, float, float]:
    """CPW-style identifiability score of a recovered (S, L) split in [0,1].

    Chandrasekaran–Parrilo–Willsky is identifiable when L is *spread* (its column space is
    diffuse, not aligned to a handful of coordinates) and S is *spiky* (few off-diagonal
    edges). L's spread is the effective support of the projector ``P = V Vᵀ`` onto its column
    space: ``diag(P)`` sums to the rank ``r``, so ``k_eff = (Σ diag P)² / Σ diag(P)²`` is the
    participation number — ``k_eff = p`` for a perfectly uniform driver, ``k_eff ≈ k`` for one
    concentrated on ``k`` coordinates (a coherent, direct-edge-like component). ``spread =
    k_eff/p ∈ (0,1]``. A concentrated split (sparse mass collapsed into L's span) drives spread
    down. Discounted by S's off-diagonal edge density ``deg`` (a dense S is not the CPW spiky
    part): score = spread·(1−deg)."""
    p = S.shape[0]
    r = factor_loadings.shape[1]
    if r == 0:
        return None, 0.0, 0.0  # rank-0 L → nothing to disambiguate
    # Orthonormalize the loadings (randomized-SVD vectors may drift from exact orthonormal).
    Q, _ = np.linalg.qr(factor_loadings)
    dP = np.einsum("ij,ij->i", Q, Q)          # diag of the column-space projector, Σ = r
    k_eff = float(dP.sum() ** 2 / max(np.sum(dP**2), 1e-12))
    spread = k_eff / p
    off = ~np.eye(p, dtype=bool)
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = np.abs(-S / np.outer(d, d))
    deg = float(((partial >= edge_threshold) & off).sum()) / max(1, p * (p - 1))
    return spread * (1.0 - deg), spread, deg


def fit_sparse_plus_lowrank(
    emp_cov: np.ndarray,
    *,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-5,
    edge_threshold: float = 0.05,
    loading_threshold: float = 0.3,
    min_factor_support: int = 3,
    penalty_matrix: np.ndarray | None = None,
    smoothness_operator: np.ndarray | None = None,
    factor_rank_cap: int = 24,
    randomized_factors: bool | None = None,
    work_dtype=np.float64,
    cond_escalate_threshold: float = 1e8,
    adaptive_rho: bool = False,
    incoherence_threshold: float = 0.35,
    seed: int = 0,
) -> SparseLowRankFit:
    """LVGLASSO ADMM: ``Ω = S - L`` from an empirical covariance.

    ``penalty_matrix`` (p×p, symmetric ≥0) overrides the scalar ``lambda1`` with a
    per-pair ℓ1 penalty — the disease-structure prior (§II.6/§5.2): structurally
    related disease-concept variables get a *lower* penalty so their (sparse) edges
    survive, i.e. dependency profiles vary smoothly across the disease hierarchy.
    Absent it, the estimator is the plain scalar-penalty LVGLASSO (a no-op prior).

    ``smoothness_operator`` (``p×p``, symmetric PSD — the combined graph Laplacian
    ``G = γ_t·L_time + γ_d·L_D`` in the fit's feature space) adds the §III.4(5) **quadratic
    prior-regularizers** ``+(1/2)·tr(Sᵀ G S)`` to the LVGLASSO objective: temporal-smoothness
    (adjacent lags) and disease-Laplacian (structurally-close diseases get similar precision
    rows) shrinkage. It enters the S-step as a linearized proximal-gradient term
    ``−(G·S)/ρ`` before the soft-threshold; at ``G=None`` the step is exactly the original
    CPW soft-threshold, so the S/L split is unperturbed and the fit is byte-identical.

    ``min_factor_support`` enforces the Chandrasekaran–Parrilo–Willsky **incoherence
    identifiability condition** at readout: a genuine low-rank latent driver must be
    *spread* across variables, so a recovered factor whose strong loadings concentrate
    on fewer than ``min_factor_support`` variables is not a shared driver but a direct
    edge (a rank-1 component on 2 variables is exactly a strong pairwise link). Such
    concentrated factors are dropped from ``latent_shared`` (they surface in ``S`` as
    ``direct_edges``). Without this, one direct edge is double-reported as both a
    ``contemporaneous`` and a ``latent_shared`` link, and there is no fixed ``lambda2``
    that separates the two roles — the collapse the LDO exhibited at every operating
    point.

    ``work_dtype`` (§V.1 mixed precision): the ADMM bulk iterates ``S,L,U,R`` are stored
    in this dtype — ``float32`` halves the working-set memory for the national envelope,
    while every *reduction* (the eigh in ``_prox_neg_logdet`` / ``_psd_project_shifted``,
    the residual-norm accumulation, and the final edge readout) is always computed in
    ``float64``. ``cond_escalate_threshold`` (§V.4) monitors ``cond(C)``: if ``float32`` is
    requested but the covariance is ill-conditioned (``cond`` above the threshold), the
    solve **escalates** to ``float64`` (``cond_escalated=True``). Default ``float64`` is
    byte-identical to the historical fit (all casts become no-ops).
    """
    p = emp_cov.shape[0]
    C = 0.5 * (emp_cov + emp_cov.T) + 1e-4 * np.eye(p)
    # §V.1/§V.4 precision policy: float32 bulk only when C is well-conditioned; else escalate.
    work_dtype = np.dtype(work_dtype)
    cond_escalated = False
    if work_dtype == np.float32:
        evC = np.linalg.eigvalsh(C)
        condC = float(evC[-1] / max(float(evC[0]), 1e-12))
        if not np.isfinite(condC) or condC > cond_escalate_threshold:
            work_dtype = np.dtype(np.float64)
            cond_escalated = True
    if penalty_matrix is not None:
        penalty_matrix = np.asarray(penalty_matrix, dtype=np.float64)
        if penalty_matrix.shape != (p, p):
            raise ValueError(f"penalty_matrix must be {(p, p)}, got {penalty_matrix.shape}")
        penalty_matrix = np.clip(0.5 * (penalty_matrix + penalty_matrix.T), 0.0, None)
    penalty = penalty_matrix if penalty_matrix is not None else lambda1
    G_sym: np.ndarray | None = None
    g_lmax = 0.0
    if smoothness_operator is not None:
        smoothness_operator = np.asarray(smoothness_operator, dtype=np.float64)
        if smoothness_operator.shape != (p, p):
            raise ValueError(f"smoothness_operator must be {(p, p)}, got {smoothness_operator.shape}")
        G_sym = 0.5 * (smoothness_operator + smoothness_operator.T)  # symmetrize; §III.4(5)
        g_lmax = float(max(0.0, np.linalg.eigvalsh(G_sym).max()))

    def _rho_invariants(rho_: float):
        # All rho-dependent step constants, recomputed together so an adaptive rho update stays
        # coherent with the initial (fixed-rho) values — at a fixed rho this reproduces them exactly.
        tau1_ = penalty / rho_
        # S-subproblem Lipschitz constant ρ + λ_max(G); step 1/lipschitz is contractive (the crude
        # 1/ρ linearization diverges when λ_max(G) is not ≪ ρ). G=None ⇒ lipschitz=ρ, tau_pg=tau1.
        lipschitz_ = rho_ + g_lmax if G_sym is not None else rho_
        tau_pg_ = penalty / lipschitz_
        return tau1_, lipschitz_, tau_pg_, C / rho_, lambda2 / rho_

    tau1, lipschitz, tau_pg, C_over_rho, l2_over_rho = _rho_invariants(rho)
    S = np.eye(p, dtype=work_dtype)
    L = np.zeros((p, p), dtype=work_dtype)
    U = np.zeros((p, p), dtype=work_dtype)
    R = np.eye(p, dtype=work_dtype)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        Z_prev = (S - L).astype(np.float64, copy=False) if adaptive_rho else None  # dual resid s = ρ(Z−Z_prev)
        # R-step: prox of -logdet + linear term. The eigh runs in float64 (reduction); cast
        # the result back to the working dtype so the stored iterate stays float32 when asked.
        R = _prox_neg_logdet(S - L - U - C_over_rho, rho).astype(work_dtype, copy=False)
        # S-step: soft-threshold off-diagonal (scalar or disease-informed per-pair penalty).
        if G_sym is None:
            S = _soft_threshold_offdiag(R + L + U, tau1).astype(work_dtype, copy=False)
        else:
            # §III.4(5) quadratic prior-regularizers: one proximal-gradient step on the smooth
            # part q(S)=(1/2)tr(SᵀG S)+(ρ/2)‖S−(R+L+U)‖² (∇q = G·S + ρ(S−M)), step 1/lipschitz,
            # then soft-threshold. At the fixed point ∇q=0 so the ADMM converges to the true
            # smoothed-objective solution; at G=0 (lipschitz=ρ) this reduces exactly to the line
            # above. tr(SᵀG S) smooths precision rows across adjacent lags + related diseases.
            grad = G_sym @ S + rho * (S - (R + L + U))
            S = _soft_threshold_offdiag(S - grad / lipschitz, tau_pg).astype(work_dtype, copy=False)
        # L-step: PSD projection with trace shrink (eigh in float64; store in work dtype).
        L = _psd_project_shifted(S - R - U, l2_over_rho).astype(work_dtype, copy=False)
        # Dual update.
        primal = R - (S - L)
        U = U + primal
        # §V.1: the convergence norms are reductions → accumulate in float64.
        pn = float(np.linalg.norm(primal.astype(np.float64, copy=False)))
        rn = max(1.0, float(np.linalg.norm(R.astype(np.float64, copy=False))))
        if pn / rn < tol:
            converged = True
            break
        # Adaptive rho (Boyd §3.4.1 residual balancing): the fixed point is rho-independent (rho
        # only scales the dual step), so balancing primal r=R−(S−L) against dual s=ρ(Z−Z_prev)
        # reaches the SAME converged S/L in far fewer iterations. OPT-IN (default off): at the
        # loose production tol the two rho schedules stop at different pre-convergence iterates, so
        # the readout is only result-preserving in the tight-tol limit — enable it with a tolerance
        # tight enough to certify convergence. Rescale the scaled dual U by rho_old/rho_new; cap rho
        # to a bounded window. adaptive_rho=False = the exact fixed-rho path (byte-identical).
        if adaptive_rho:
            dual = rho * float(np.linalg.norm((S - L).astype(np.float64, copy=False) - Z_prev))
            new_rho = rho
            if pn > 10.0 * dual:
                new_rho = min(rho * 2.0, 1e6)
            elif dual > 10.0 * pn:
                new_rho = max(rho / 2.0, 1e-6)
            if new_rho != rho:
                U = (U * (rho / new_rho)).astype(work_dtype, copy=False)
                rho = new_rho
                tau1, lipschitz, tau_pg, C_over_rho, l2_over_rho = _rho_invariants(rho)

    # Promote the converged iterates to float64 for the edge/factor readout so float32
    # round-off cannot flip an edge near the selection threshold (§V.1: reductions in f64).
    S = S.astype(np.float64, copy=False)
    L = L.astype(np.float64, copy=False)
    precision = S - L
    # Direct edges: partial correlations from S.
    d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    partial = -S / np.outer(d, d)
    np.fill_diagonal(partial, 1.0)
    direct_edges = [
        (i, j, float(partial[i, j]))
        for i in range(p) for j in range(i + 1, p)
        if abs(partial[i, j]) >= edge_threshold
    ]
    direct_edges.sort(key=lambda e: abs(e[2]), reverse=True)

    # Latent factors: eigendecomposition of L (randomized SVD at scale, §V.3).
    use_randomized = randomized_factors if randomized_factors is not None else (p > 2 * factor_rank_cap)
    vals, vecs = _low_rank_factors(L, rank_cap=factor_rank_cap, randomized=use_randomized, seed=seed)
    # Keep only factors clearly above the noise floor (2% of the top eigenvalue). Shared
    # drivers are few and large; the fat tail of small PSD-projection eigenvalues emitted
    # spurious latent_shared pairs and made the exact/randomized readouts disagree — dropping
    # it cleans the readout so the exact and randomized low-rank factors track each other. (The
    # §V.6 exact-certifies-approximate CONTRACT is verified separately, whole-pipeline, by
    # validation.holdout.certify_approximation_on_slice — not by this intra-fit cleanup.)
    keep = vals > max(1e-6, 0.02 * vals.max() if vals.size else 0.0)
    factor_values = vals[keep]
    factor_loadings = vecs[:, keep]
    # §V.6(2) numerical error: an exact eigh has none; a randomized-SVD readout truncates the
    # tail, so the largest OMITTED (kept-out) eigenvalue relative to the top one bounds the
    # readout error — propagated into the affected (latent_shared) edges' uncertainty.
    dropped = vals[~keep]
    numerical_error = (
        float(dropped.max() / vals.max()) if use_randomized and dropped.size and vals.size and vals.max() > 0 else 0.0
    )

    # latent_shared pairs: variables both loading strongly on a common factor.
    # CPW incoherence gate: a factor supported on < min_factor_support variables is a
    # concentrated (coherent) component — a direct edge, not a shared driver — so it is
    # not reported as latent_shared (it is already recovered in S / direct_edges).
    latent_shared: list[tuple[int, int, float]] = []
    for f in range(factor_loadings.shape[1]):
        load = factor_loadings[:, f]
        strong = [i for i in range(p) if abs(load[i]) >= loading_threshold]
        if len(strong) < min_factor_support:
            continue
        for a_i in range(len(strong)):
            for b_i in range(a_i + 1, len(strong)):
                i, j = strong[a_i], strong[b_i]
                latent_shared.append((i, j, float(min(abs(load[i]), abs(load[j])))))
    # keep the strongest shared loading per pair
    best: dict[tuple[int, int], float] = {}
    for i, j, v in latent_shared:
        key = (i, j)
        best[key] = max(best.get(key, 0.0), v)
    # Mutual exclusion (§III.4.2): a pair reported as a direct edge in S is not also a
    # confounded (latent_shared) pair — one pair, one role. S (net of the shared driver)
    # is authoritative for direct links, so drop any direct-edge pair from latent_shared.
    direct_pairs = {(i, j) for i, j, _ in direct_edges}
    latent_shared = sorted(
        ((i, j, v) for (i, j), v in best.items() if (i, j) not in direct_pairs),
        key=lambda e: e[2], reverse=True,
    )

    # CPW identifiability diagnostic: score the recovered split (does NOT alter S/L). A low
    # score means S's mass sits inside L's span — the split is arbitrary, so a downstream
    # readout should treat latent_shared edges as suspect. Result-preserving annotation only.
    incoherence, _mu, _deg = _cpw_incoherence(S, factor_loadings, edge_threshold=edge_threshold)
    well_identified = incoherence is None or incoherence >= incoherence_threshold

    return SparseLowRankFit(
        S=S, L=L, precision=precision,
        factor_loadings=factor_loadings, factor_values=factor_values,
        converged=converged, iterations=it, numerical_error=numerical_error,
        work_dtype=str(work_dtype), cond_escalated=cond_escalated,
        direct_edges=direct_edges, latent_shared=latent_shared,
        incoherence=incoherence, well_identified=well_identified,
    )


def lag_chain_laplacian(K: int) -> np.ndarray:
    """``(K+1)×(K+1)`` path-graph Laplacian over lag positions 0..K (adjacent lags linked).

    The §III.4(5) temporal-smoothness prior ties a variable's precision row across
    neighbouring lags: lag ``a`` ~ lag ``a±1``. ``L = D − W`` for the chain graph.
    """
    n = K + 1
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    idx = np.arange(n - 1)
    W[idx, idx + 1] = 1.0
    W[idx + 1, idx] = 1.0
    return np.diag(W.sum(axis=1)) - W


def build_smoothness_operator(
    p: int, K: int, *, disease_laplacian: np.ndarray | None = None,
    gamma_temporal: float = 0.0, gamma_disease: float = 0.0,
) -> np.ndarray | None:
    """The §III.4(5) combined smoothness operator ``G = γ_t·(L_lag⊗I_p) + γ_d·(I_{K+1}⊗L_D)``
    in the ``p·(K+1)`` lag-extended feature space (feature index ``f = lag·p + var``).

    ``L_lag`` is the temporal chain Laplacian (:func:`lag_chain_laplacian`); ``L_D`` is the
    disease-graph Laplacian over the ``p`` base variables. Returns ``None`` when both weights
    are 0 (or their operands absent) so the caller passes no smoothing and the CPW fit is
    byte-identical. ``tr(Sᵀ G S)`` then smooths precision rows across adjacent lags and across
    structurally-close diseases.
    """
    terms: list[np.ndarray] = []
    if gamma_temporal > 0.0 and K >= 1:
        terms.append(gamma_temporal * np.kron(lag_chain_laplacian(K), np.eye(p)))
    if gamma_disease > 0.0 and disease_laplacian is not None:
        L_D = np.asarray(disease_laplacian, dtype=np.float64)
        if L_D.shape == (p, p):
            terms.append(gamma_disease * np.kron(np.eye(K + 1), L_D))
    if not terms:
        return None
    G = terms[0]
    for t in terms[1:]:
        G = G + t
    return G


__all__ = [
    "SparseLowRankFit", "fit_sparse_plus_lowrank",
    "lag_chain_laplacian", "build_smoothness_operator",
]
