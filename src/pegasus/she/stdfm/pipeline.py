"""Production ST-DFM execution and certification boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from pegasus.she.stdfm.artifacts import materialize_stdfm_fit
from pegasus.she.stdfm.certification import (
    STDFMCertificationPolicy,
    STDFMCertificationRow,
    certify_stdfm_metrics,
)
from pegasus.she.stdfm.schema import STDFMFitResult, STDFMInputSchema, STDFMOutputSchema, STDFMProblem
from pegasus.she.stdfm.torch_solver import solve_stdfm


@dataclass(frozen=True)
class STDFMPipelineResult:
    output: STDFMOutputSchema
    certification: STDFMCertificationRow
    fit: STDFMFitResult | None


def run_stdfm_pipeline(
    input_schema: STDFMInputSchema,
    problem: STDFMProblem,
    *,
    output_dir: str | Path,
    policy: STDFMCertificationPolicy | None = None,
    require_cuda: bool = False,
    prefer_cuda: bool = False,
    max_iterations: int = 2_000,
) -> STDFMPipelineResult:
    observed_fraction = sum(problem.observed_mask) / problem.n_cells
    support_certification = certify_stdfm_metrics(
        field_id=input_schema.field_id,
        metrics={},
        observed_fraction=observed_fraction,
        periods=problem.shape[1],
        localities=problem.shape[0],
        spatial_penalty=problem.gamma_spatial,
        multi_starts=problem.multi_starts,
        warnings=input_schema.warnings,
        policy=policy,
    )
    if support_certification.status == "blocked_invalid_support":
        output = STDFMOutputSchema(
            field_id=input_schema.field_id,
            status="blocked_invalid_support",
            solver_backend="not_executed",
            certification_id=support_certification.certification_id,
            uncertainty=None,
            warnings=support_certification.warnings,
            reason=";".join(support_certification.warnings),
        )
        return STDFMPipelineResult(output, support_certification, None)
    fitted = solve_stdfm(
        input_schema,
        problem=problem,
        allow_uncertified=True,
        require_cuda=require_cuda,
        prefer_cuda=prefer_cuda,
        max_iterations=max_iterations,
    )
    if not isinstance(fitted, STDFMFitResult):
        certification = replace(support_certification, status="failed_certification", warnings=fitted.warnings)
        return STDFMPipelineResult(fitted, certification, None)
    metrics = dict(fitted.certification)
    metrics["factor_stability"] = fitted.telemetry.factor_stability
    certification = certify_stdfm_metrics(
        field_id=input_schema.field_id,
        metrics=metrics,
        observed_fraction=observed_fraction,
        periods=problem.shape[1],
        localities=problem.shape[0],
        spatial_penalty=problem.gamma_spatial,
        multi_starts=problem.multi_starts,
        warnings=fitted.output.warnings,
        policy=policy,
    )
    output = replace(
        fitted.output,
        status=certification.status,
        certification_id=certification.certification_id,
        warnings=tuple(dict.fromkeys([*fitted.output.warnings, *certification.warnings])),
    )
    fitted = replace(fitted, output=output, certification=certification.as_manifest())
    fitted = materialize_stdfm_fit(fitted, problem, output_dir=output_dir)
    return STDFMPipelineResult(fitted.output, certification, fitted)
