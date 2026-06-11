from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from pegasus.pirs.schemas import PIRSSelectionResult


@dataclass(frozen=True)
class DesignMatrixResult:
    status: str
    design_matrix_path: str
    outcome_vector_path: str
    support_index_path: str
    rows: int
    covariate_field_ids: tuple[str, ...]
    offset_field_id: str | None
    warnings: tuple[str, ...]


def _read_observations(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        return pl.DataFrame(payload["observations"])
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported PIRS fixture path: {path}")


def build_design_matrix(*, observations_path: str | Path, selection: PIRSSelectionResult, output_dir: str | Path) -> DesignMatrixResult:
    if selection.selected_outcome is None:
        raise ValueError("PIRS selection has no legal outcome")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = _read_observations(observations_path)
    columns = ["support_id"]
    outcome = selection.selected_outcome.field_id
    if outcome not in df.columns:
        raise ValueError(f"Outcome field missing from observations: {outcome}")
    for covariate in selection.selected_covariates:
        if covariate.field_id not in df.columns:
            raise ValueError(f"Covariate field missing from observations: {covariate.field_id}")
        columns.append(covariate.field_id)
    if selection.selected_offset is not None:
        if selection.selected_offset.field_id not in df.columns:
            raise ValueError(f"Offset field missing from observations: {selection.selected_offset.field_id}")
        columns.append(selection.selected_offset.field_id)
    design_path = output_dir / "pirs_design_matrix.parquet"
    outcome_path = output_dir / "pirs_outcome_vector.parquet"
    support_path = output_dir / "pirs_support_index.parquet"
    df.select(columns).write_parquet(design_path)
    df.select(["support_id", outcome]).write_parquet(outcome_path)
    support_cols = [c for c in ["support_id", "municipality_cod6", "year"] if c in df.columns]
    df.select(support_cols).write_parquet(support_path)
    return DesignMatrixResult(
        status="materialized",
        design_matrix_path=str(design_path),
        outcome_vector_path=str(outcome_path),
        support_index_path=str(support_path),
        rows=df.height,
        covariate_field_ids=tuple(c.field_id for c in selection.selected_covariates),
        offset_field_id=selection.selected_offset.field_id if selection.selected_offset else None,
        warnings=(),
    )
