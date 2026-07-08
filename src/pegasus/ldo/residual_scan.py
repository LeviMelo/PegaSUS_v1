"""LDO residual nonlinear-edge layer (MSD-II §II.6.3, MII-LDO-05).

After the structured linear backbone is fit, compute the joint-model residual
field ``e = Z - Ẑ`` (the Gaussian-graphical conditional residual
``e_j = (Ω Z)_j / Ω_jj``) and run the **existing** HSIC scanner (§6.7) on residual
pairs under the §6.8 nulls with §6.9 FDR. HSIC here is a *targeted nonlinear-edge
detector on what the structured backbone cannot explain* — a strict
generalization of MSD §6 that reuses ``pirs/{hsic,nulls,fdr}.py`` unchanged.

A significant residual HSIC between variables the linear precision left
unconnected is a ``nonlinear_residual`` link.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import random as _random

from pegasus.ldo.envelope import (
    ScaleExceedsEnvelopeError,
    estimate_residual_scan_bytes,
    load_compute_envelope,
)
from pegasus.ldo.fdr import correct_p_values
from pegasus.ldo.hsic import (
    build_hsic_representation,
    hsic_mode_for_n,
    hsic_pair_stat_and_null,
    hsic_pair_stat_and_null_gpu,
)
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.nulls import (
    descriptive_only_when_insufficient_blocks,
    generate_null_indices,
    select_null_regime,
)
from pegasus.ldo.records import LinkRecord


class ResidualScanUnderpowered(RuntimeError):
    """Raised when the complete-case residual sample is too small to run a verified scan.

    A typed skip (not a bug): the run RECORDS it in ``residual_scan_error`` rather than the scan
    silently returning no edges (which reads as 'scanned, found nothing'). §6.7 power gate."""


_MIN_N_EFF = 100
_MIN_NULL_BLOCKS = 5  # < this many spatial blocks → certify descriptive only (§6.8/§6.9)


def _spatial_block_labels(space_ids: tuple[str, ...], cell_space_idx) -> np.ndarray:
    """Spatial-block label per cell for the §6.8 structured null (MSD-II §II.4 line 201: the HSIC
    null uses the SHARED SpatialWeightGraph's ``blocks()``, not an ad-hoc UF adjacency). Contiguity
    blocks respect the actual disease spatial structure across administrative borders — two adjacent
    municipalities in different states stay in one block, so their correlation is correctly treated
    as spatial clustering (null), not surprise. Falls back to the UF (state) prefix when the
    structural graph is unavailable or does not cover the field."""
    n_muni = len(space_ids)
    cell_space_idx = np.asarray(cell_space_idx)
    try:
        from pegasus.geo.spatial_graph import load_spatial_graph, structural_graph_available
        if not structural_graph_available():
            raise RuntimeError("no structural graph")
        graph = load_spatial_graph()
        n_blocks = max(_MIN_NULL_BLOCKS, n_muni // 200)  # ~UF-count granularity at national S
        block_of_node = {
            str(node): f"B{bi}" for bi, members in enumerate(graph.blocks(n_blocks)) for node in members
        }
        labels = np.array([
            block_of_node.get(str(space_ids[s]) if s < n_muni else "00", "") for s in cell_space_idx
        ])
        if np.mean(labels != "") >= 0.5:  # graph meaningfully covers the field → use contiguity blocks
            uf = np.array([f"UF{str(space_ids[s])[:2]}" if s < n_muni else "UF00" for s in cell_space_idx])
            return np.where(labels != "", labels, uf)  # graph-absent munis keep a UF label
    except Exception:
        pass
    return np.array([str(space_ids[s])[:2] if s < n_muni else "00" for s in cell_space_idx])


def _residual_scan_memory_budget() -> int:
    """Bytes the residual-scan representation cache may use before it must refuse.

    Prefer 60% of the *currently available* system RAM (the scan is a CPU/numpy
    allocation, not a VRAM one), leaving headroom for its transient per-variable
    ``n×n`` kernel builds and the rest of the process. Falls back to the compute
    envelope (config/compute.yaml) if psutil is unavailable.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.60)
    except Exception:
        return int(load_compute_envelope().max_bytes)


@dataclass
class ResidualEdge:
    source: str
    target: str
    hsic: float
    p_value: float
    q_value: float


def joint_model_residuals(Z_matrix: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Gaussian-graphical conditional residuals ``e_j = (Ω Z)_j / Ω_jj`` (p × n)."""
    diag = np.clip(np.diag(precision), 1e-12, None)
    return (precision @ Z_matrix) / diag[:, None]


# --- Per-variable HSIC representation cache (residual-scan fast path, §V.3) -----------
# The exhaustive p²/2 scan re-derived each variable's kernel/feature representation once per pair
# (O(p²) rebuilds of a per-variable object). Cache each once via the shared hsic.py builder
# (O(p) builds); a pair is then a cheap product-sum. On CUDA the feature-map modes run the
# statistic + permutation null in float32 on the GPU (O(N·D), no dense N×N kernel).


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_var_reprs(E: np.ndarray, *, seed: int, kernel: str, budget: str, max_exact: int):
    n = int(E.shape[1])
    mode = hsic_mode_for_n(n=n, budget=budget, max_exact=max_exact)
    reprs = [
        build_hsic_representation(E[v], mode=mode, kernel=kernel, seed=seed, budget=budget, n=n)
        for v in range(E.shape[0])
    ]
    return mode, reprs


def _pair_stat_and_null(ri, rj, *, mode, n, perms, use_gpu, seed):
    if mode != "exact" and use_gpu and perms is not None:
        return hsic_pair_stat_and_null_gpu(ri["fx"], rj["fy"], n=n, perms=perms, seed=seed)
    return hsic_pair_stat_and_null(ri, rj, n=n, perms=perms)


def _infer_panel_type(field: GaussianField) -> str:
    """Map the field geometry+cadence to a §6.8 null-registry panel type.

    Monthly cadence with whole 12-month years → ``monthly_seasonal_panel``; a single time
    slice → ``cross_sectional_census``; otherwise a multi-period municipal panel →
    ``annual_municipal_panel`` (facility-stock/sparse-stratified regimes are declared by the
    caller, not inferred from a standard space×time panel)."""
    T = int(field.shape[2])
    res = str(getattr(field, "resolution", "") or "").lower()
    if T <= 1:
        return "cross_sectional_census"
    if "month" in res and T >= 12 and T % 12 == 0:
        return "monthly_seasonal_panel"
    return "annual_municipal_panel"


def _temporal_bucket(time_of_cell: np.ndarray, panel_type: str) -> np.ndarray:
    """Per-cell temporal-stratum label. Season (month) for monthly panels so a null swap
    preserves seasonality; exact time index for annual panels; a single bucket for a
    cross-section. The null permutes only WITHIN (spatial-block × this bucket)."""
    if panel_type == "monthly_seasonal_panel":
        return np.array([str(int(t) % 12) for t in time_of_cell])
    if panel_type == "cross_sectional_census":
        return np.array(["0"] * len(time_of_cell))
    return np.array([str(int(t)) for t in time_of_cell])


def _coarsen_residuals(
    E: np.ndarray, uf_of_cell: np.ndarray, bucket_of_cell: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """§II.7 multi-resolution: aggregate observed cells into (spatial-block × temporal-bucket)
    groups and average the residual within each, so the §III.6 HSIC layer runs at a coarser
    grain when the fine-cell kernel cache would exceed the memory budget (rather than being
    silently skipped). Returns the coarse residual (p×G), one spatial-block label per coarse
    cell, and a resolution note. At the coarse grain the null is a within-spatial-block swap."""
    keys = np.array([f"{u}|{b}" for u, b in zip(uf_of_cell.tolist(), bucket_of_cell.tolist())])
    groups = sorted(set(keys.tolist()))
    gpos = {g: i for i, g in enumerate(groups)}
    p = E.shape[0]
    E_coarse = np.zeros((p, len(groups)), dtype=np.float64)
    counts = np.zeros(len(groups), dtype=np.float64)
    uf_coarse = np.empty(len(groups), dtype=object)
    for cell, key in enumerate(keys.tolist()):
        g = gpos[key]
        E_coarse[:, g] += E[:, cell]
        counts[g] += 1.0
        uf_coarse[g] = key.split("|", 1)[0]
    E_coarse /= np.maximum(counts, 1.0)
    return E_coarse, uf_coarse.astype(str), np.array(groups), "coarsened_spatialblock_x_timebucket"


def scan_residual_nonlinear_edges(
    field: GaussianField,
    precision: np.ndarray,
    *,
    budget: str = "standard",
    permutations: int = 200,
    seed: int = 0,
    alpha: float = 0.1,
) -> list[LinkRecord]:
    """Detect nonlinear residual dependence the linear backbone missed."""
    p_all, S, T = field.shape
    Z_all = field.Z.reshape(p_all, S * T)
    # §III.6 shared residual sample needs cells where the modelled variables are jointly
    # observed. Requiring ALL p variables observed lets a few sparse national context variables
    # collapse the shared sample to nothing — a SILENT no-op (the old ``return []``). Instead,
    # trim the sparsest variables until the complete-case sample is powered, keeping the full set
    # whenever it already is (no behaviour change on dense panels). If even the trimmed set is
    # underpowered, RAISE a typed skip so the run RECORDS it (via ``residual_scan_error``) rather
    # than silently returning no edges.
    observed_full = np.isfinite(Z_all).all(axis=0)
    kept = np.arange(p_all)
    if int(observed_full.sum()) < _MIN_N_EFF and p_all > 2:
        order = np.argsort(-np.isfinite(Z_all).mean(axis=1))  # most-covered variables first
        for cut in range(p_all, 1, -1):
            sub = np.sort(order[:cut])
            if int(np.isfinite(Z_all[sub]).all(axis=0).sum()) >= _MIN_N_EFF:
                kept = sub
                break
    variables = [field.variables[k] for k in kept]
    p = len(kept)
    precision = np.asarray(precision)[np.ix_(kept, kept)]
    Z = Z_all[kept]
    observed = np.isfinite(Z).all(axis=0)
    n_eff = int(observed.sum())
    if n_eff < _MIN_N_EFF:
        raise ResidualScanUnderpowered(
            f"residual_scan_underpowered: complete-case n_eff={n_eff} < {_MIN_N_EFF} over "
            f"{p} covered variables (§6.7 power gate); nonlinear residual scan skipped."
        )
    Zc = Z[:, observed]
    Zc = np.where(np.isfinite(Zc), Zc, 0.0)
    E = joint_model_residuals(Zc, precision)

    # §6.8 panel-aware STRUCTURED null. The residuals retain BOTH spatial autocorrelation and
    # temporal/seasonal structure, so an iid — or spatial-only — permutation is anticonservative
    # (variables that merely co-cluster in space AND/OR co-vary seasonally test as dependent).
    # Select the null regime from the panel support (nulls.NULL_REGIMES) and permute cells only
    # WITHIN their (spatial-block × temporal-bucket) stratum, preserving both dependences so
    # HSIC tests structure BEYOND shared spatio-temporal clustering. Reuse the same structural
    # permutations across pairs (the null is structural, not data-dependent).
    panel_type = _infer_panel_type(field)
    regime = select_null_regime(panel_type)
    permutations = int(regime.permutations)
    observed_idx = np.where(observed)[0]
    # §II.4 line 201: the null's spatial block is the SHARED graph's contiguity blocks() (falls
    # back to the UF prefix when the graph is unavailable) — not an ad-hoc UF adjacency.
    uf_of_cell = _spatial_block_labels(field.space_ids, observed_idx // T)
    bucket_of_cell = _temporal_bucket(observed_idx % T, panel_type)

    # §II.7 multi-resolution continuation: if the fine-cell HSIC representation cache would
    # exceed the memory budget, COARSEN to (spatial-block × temporal-bucket) groups (averaging
    # the residual) so the §III.6 layer runs at a coarser grain instead of being skipped; the
    # coarse null is then a within-spatial-block swap. Only if even the coarsest grain does not
    # fit do we refuse (§II.10 refuse-don't-degrade).
    scan_resolution = str(getattr(field, "resolution", "cell"))
    within_block_only = False
    if estimate_residual_scan_bytes(p=p, n_eff=E.shape[1], budget=budget) > _residual_scan_memory_budget():
        E, uf_of_cell, bucket_of_cell, scan_resolution = _coarsen_residuals(E, uf_of_cell, bucket_of_cell)
        within_block_only = True  # coarse cells are one-per-(block,bucket): swap within block
        if estimate_residual_scan_bytes(p=p, n_eff=E.shape[1], budget=budget) > _residual_scan_memory_budget():
            raise ScaleExceedsEnvelopeError(
                "residual_hsic_scan_exceeds_memory: even the coarsest (spatial-block × temporal-"
                f"bucket) grain (n={E.shape[1]}, p={p}) exceeds the available-memory budget; "
                "nonlinear residual edges not computed (§II.7/§II.10)."
            )

    n = int(E.shape[1])
    n_spatial_blocks = len(set(uf_of_cell.tolist()))
    n_temporal_blocks = len(set(bucket_of_cell.tolist()))
    # Fine grain: swap within (spatial-block × temporal-bucket) — preserves both. Coarse grain:
    # one cell per (block,bucket), so swap within spatial-block (across buckets) — preserves the
    # spatial block, tests beyond it.
    if within_block_only:
        strata = uf_of_cell.tolist()
    else:
        strata = [f"{u}|{b}" for u, b in zip(uf_of_cell.tolist(), bucket_of_cell.tolist())]
    # The executed null is a restricted permutation WITHIN (spatial-block × temporal-bucket)
    # strata: it preserves both the spatial-block dependence and the temporal/seasonal bucket
    # (a monthly cell only permutes to a same-month cell in its block — so it IS season-
    # preserving by construction). This is the null MSD-III §III.6 was amended to endorse.
    # CRITICAL (§III.6/§6.8 provenance): ``null_strategy`` names the permutation that ACTUALLY
    # RAN — never the regime's designated circular-shift generator, which (a) is not what
    # executes here and (b) cannot even be applied to the ragged complete-case sample. The named
    # ``nulls.py`` generators remain available for a dense-panel entry point.
    if n_spatial_blocks >= 2 or n_temporal_blocks >= 2:
        rng = _random.Random(seed)
        perm_list: list[list[int]] | None = [
            generate_null_indices(strategy="restricted_intra_uf_spatial_swap",
                                  n=n, support={"uf_strata": strata}, rng=rng)
            for _ in range(permutations)
        ]
        null_strategy = (
            "restricted_within_spatial_block_swap" if within_block_only
            else "restricted_within_spatial_block_temporal_bucket_swap"
        ) + f"[regime:{panel_type}]"
    else:
        perm_list = None  # no spatial/temporal structure → iid fallback
        null_strategy = "iid_permutation"

    # §6.8 descriptive gate (the DISJUNCTION): fewer than five spatial OR five temporal blocks
    # → no valid structured null → surface descriptive, do not certify.
    sufficient_blocks = not descriptive_only_when_insufficient_blocks(
        spatial_blocks=n_spatial_blocks, temporal_blocks=n_temporal_blocks
    )

    # Precompute each variable's HSIC representation ONCE (O(p) builds), then score every pair
    # from the cache (bit-identical to numpy_kernel_hsic_permutation_test: same seeds, kernels,
    # centering, feature maps, shared permutation set).
    mode, reprs = _build_var_reprs(E, seed=seed, kernel="rbf", budget=budget, max_exact=5000)

    if perm_list is not None:
        perms: list[np.ndarray] | None = [np.asarray(pm, dtype=int) for pm in perm_list if len(pm) == n]
    else:
        _rng = np.random.default_rng(seed)
        perms = [_rng.permutation(n) for _ in range(max(permutations, 1))]

    # Feature-map modes run the statistic + null in float32 on the GPU when CUDA is present
    # (memory-bound O(N·D) matmuls); exact mode and the CPU-only case use the numpy reference.
    use_gpu = mode != "exact" and _cuda_available()
    pairs: list[tuple[int, int]] = [(i, j) for i in range(p) for j in range(i + 1, p)]
    stats: list[float] = []
    pvals: list[float] = []
    for i, j in pairs:
        stat, null_arr = _pair_stat_and_null(
            reprs[i], reprs[j], mode=mode, n=n, perms=perms, use_gpu=use_gpu, seed=seed
        )
        pval = float((1 + int((null_arr >= stat).sum())) / (1 + null_arr.size)) if null_arr.size else 1.0
        stats.append(stat)
        pvals.append(pval)

    # §6.9 FDR: the panel-support-selected method (BY for cross-fitted annual/monthly panels,
    # BH for cross-sectional, Storey-q for facility stock), not a hardcoded BH.
    qvals = correct_p_values(pvals, method=regime.fdr_method).q_values if pvals else []
    edge_warnings: tuple[str, ...] = (f"panel_type:{panel_type}", *regime.warnings)
    if within_block_only:
        edge_warnings = (*edge_warnings, f"multiresolution_coarsened:{scan_resolution}")
    records: list[LinkRecord] = []
    for (i, j), stat, pv, qv in zip(pairs, stats, pvals, qvals):
        if qv is not None and qv <= alpha:
            records.append(
                LinkRecord(
                    source_var=variables[i],
                    target_var=variables[j],
                    edge_type="nonlinear_residual",
                    weight=float(stat),
                    uncertainty=float(qv),
                    null_strategy=null_strategy,
                    fdr_method=regime.fdr_method,
                    certification_status="selected" if sufficient_blocks else "descriptive",
                    warnings=edge_warnings,
                )
            )
    return records


__all__ = ["ResidualEdge", "ResidualScanUnderpowered", "joint_model_residuals", "scan_residual_nonlinear_edges"]
