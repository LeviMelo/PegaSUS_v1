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

from pegasus.ldo.fdr import correct_p_values
from pegasus.ldo.hsic import numpy_kernel_hsic_permutation_test
from pegasus.ldo.margins import GaussianField
from pegasus.ldo.records import LinkRecord

_MIN_N_EFF = 100


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
    p, S, T = field.shape
    Z = field.Z.reshape(p, S * T)
    # Complete-case cells (all variables observed) for a shared residual sample.
    observed = np.isfinite(Z).all(axis=0)
    n_eff = int(observed.sum())
    if n_eff < _MIN_N_EFF:
        return []  # §6.7 gate: underpowered → no verified nonlinear scan
    Zc = Z[:, observed]
    Zc = np.where(np.isfinite(Zc), Zc, 0.0)
    E = joint_model_residuals(Zc, precision)

    pairs: list[tuple[int, int]] = [(i, j) for i in range(p) for j in range(i + 1, p)]
    stats: list[float] = []
    pvals: list[float] = []
    for i, j in pairs:
        stat, null, _ = numpy_kernel_hsic_permutation_test(
            covariate=E[i].tolist(),
            residuals=E[j].tolist(),
            permutations=permutations,
            seed=seed,
            budget=budget,
        )
        null_arr = np.asarray(null, dtype=np.float64)
        pval = float((1 + int((null_arr >= stat).sum())) / (1 + null_arr.size)) if null_arr.size else 1.0
        stats.append(stat)
        pvals.append(pval)

    qvals = correct_p_values(pvals, method="BH").q_values if pvals else []
    records: list[LinkRecord] = []
    for (i, j), stat, pv, qv in zip(pairs, stats, pvals, qvals):
        if qv is not None and qv <= alpha:
            records.append(
                LinkRecord(
                    source_var=field.variables[i],
                    target_var=field.variables[j],
                    edge_type="nonlinear_residual",
                    weight=float(stat),
                    uncertainty=float(qv),
                    null_strategy="iid_permutation",
                    fdr_method="benjamini_hochberg",
                    certification_status="selected",
                )
            )
    return records


__all__ = ["ResidualEdge", "joint_model_residuals", "scan_residual_nonlinear_edges"]
