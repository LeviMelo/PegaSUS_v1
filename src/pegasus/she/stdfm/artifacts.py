from __future__ import annotations
from pegasus.storage import write_table

from dataclasses import replace
from pathlib import Path

import polars as pl

from pegasus.she.stdfm.schema import STDFMFitResult, STDFMProblem


def materialize_stdfm_fit(
    result: STDFMFitResult,
    problem: STDFMProblem,
    *,
    output_dir: str | Path,
) -> STDFMFitResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    space, time, fields = problem.shape
    factors = problem.n_factors
    factor_path = output_dir / "stdfm_latent_factors.parquet"
    loading_path = output_dir / "stdfm_loadings.parquet"
    reconstruction_path = output_dir / "stdfm_reconstructed_fields.parquet"
    uncertainty_path = output_dir / "stdfm_uncertainty.parquet"
    certification_path = output_dir / "stdfm_certification.parquet"
    objective_trace_path = output_dir / "stdfm_objective_trace.parquet"

    write_table(factor_path, pl.DataFrame([{'space_index': s, 'time_index': t, 'factor_index': k, 'value': result.latent_factors[(s * time + t) * factors + k]} for s in range(space) for t in range(time) for k in range(factors)]).to_arrow())
    write_table(loading_path, pl.DataFrame([{'field_id': problem.field_ids[q], 'factor_index': k, 'loading': result.loadings[q * factors + k]} for q in range(fields) for k in range(factors)]).to_arrow())
    rows = [
        {
            "space_index": s,
            "time_index": t,
            "field_id": problem.field_ids[q],
            "observed": problem.observed_mask[(s * time + t) * fields + q],
            "reconstructed": result.reconstructed[(s * time + t) * fields + q],
        }
        for s in range(space)
        for t in range(time)
        for q in range(fields)
    ]
    write_table(reconstruction_path, pl.DataFrame(rows).to_arrow())
    write_table(uncertainty_path, pl.DataFrame([dict(row, uncertainty=result.uncertainty[index]) for index, row in enumerate(rows)]).to_arrow())
    write_table(certification_path, pl.DataFrame([result.certification]).to_arrow())
    write_table(objective_trace_path, pl.DataFrame([{'iteration': 0, 'objective': result.telemetry.initial_objective}, {'iteration': result.telemetry.iterations, 'objective': result.telemetry.final_objective}]).to_arrow())

    output = replace(
        result.output,
        latent_factor_path=str(factor_path),
        loading_matrix_path=str(loading_path),
        reconstructed_fields_path=str(reconstruction_path),
        certification_table_path=str(certification_path),
        uncertainty_path=str(uncertainty_path),
        objective_trace_path=str(objective_trace_path),
    )
    return replace(result, output=output)
