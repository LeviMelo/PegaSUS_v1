"""Investigate workflow — compile → CommonPanel → LDO → LinkRecords (MSD-II §II.6, MII-OUT-01).

The single ExecutionStage=investigate entrypoint that runs the Lattice Dependency
Operator over a compiled run's CommonPanel and writes typed link records to the
Hypotheses key. This replaces the PIRS slice-zoo's design/selection/candidate/
scan chain with one in-memory orchestrator call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.pirs.ldo.field_selection import analytical_variable_ids
from pegasus.pirs.ldo.orchestrator import run_ldo
from pegasus.pirs.ldo.output import write_hypotheses
from pegasus.she.panel import Resolution, compile_common_panel


@dataclass
class InvestigateResult:
    run_dir: str
    n_link_records: int
    n_selected: int
    hypotheses_path: str
    panel_cells: int
    panel_fields: int
    diagnostics: dict[str, Any]


def _geography_prefixes(intent: Any) -> frozenset[str] | None:
    from pegasus.efg.executor import _scope_prefixes_from_intent

    return _scope_prefixes_from_intent(intent)


def run_investigate(
    run_dir: str | Path,
    *,
    intent: Any = None,
    resolution: Resolution = "year",
    K: int = 3,
    lambda1: float = 0.1,
    lambda2: float = 1.0,
    write: bool = True,
    **ldo_kwargs,
) -> InvestigateResult:
    """Run the LDO over a compiled run and (optionally) write the Hypotheses key.

    Production defaults: ``K=3`` (adaptively capped by the panel's time span / size,
    see ``run_ldo``) and ``lambda2=1.0`` — the raw ``run_ldo`` default of 0.1 lets the
    low-rank layer over-absorb on noisy real panels (nearly full rank, collapsing
    every relationship into latent_shared), so a stronger nuclear-norm penalty is the
    right production default; callers may override.
    """
    run_dir = Path(run_dir)
    prefixes = _geography_prefixes(intent)
    panel = compile_common_panel(run_dir, resolution=resolution, geography_prefixes=prefixes)

    # Restrict the LDO to analytical fields (drop raw source-column passthroughs and
    # support axes) unless the caller overrides — smaller precision solve, cleaner graph.
    keep_variables = ldo_kwargs.pop("keep_variables", None)
    if keep_variables is None:
        keep_variables = analytical_variable_ids(run_dir)

    ldo_run = run_ldo(panel, K=K, lambda1=lambda1, lambda2=lambda2, keep_variables=keep_variables, **ldo_kwargs)

    hypotheses_path = run_dir / "Hypotheses.parquet"
    if write:
        write_hypotheses(ldo_run.link_records, hypotheses_path)

    return InvestigateResult(
        run_dir=str(run_dir),
        n_link_records=len(ldo_run.link_records),
        n_selected=ldo_run.diagnostics.get("n_selected", 0),
        hypotheses_path=str(hypotheses_path),
        panel_cells=panel.index.height,
        panel_fields=len(panel.fields),
        diagnostics=ldo_run.diagnostics,
    )


__all__ = ["InvestigateResult", "run_investigate"]
