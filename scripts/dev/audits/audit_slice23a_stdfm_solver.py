from __future__ import annotations

import json

from pegasus.she.stdfm.schema import STDFMFitResult, STDFMProblem, build_stdfm_input_schema
from pegasus.she.stdfm.torch_solver import solve_stdfm


def main() -> int:
    schema = build_stdfm_input_schema(
        field_id="audit_stdfm",
        concept_id="audit_low_rank",
        support={"periods": ["1", "2", "3", "4", "5"]},
        periods=["1", "2", "3", "4", "5"],
        localities=["x"],
        transform="identity",
        dynamics="continuous",
        projection_matrix_id=None,
        stitch_metadata={"status": "single_segment"},
    )
    problem = STDFMProblem(
        field_ids=("a", "b"),
        shape=(1, 5, 2),
        observations=(1.0, 2.0, 2.0, 4.0, 1e9, -1e9, 4.0, 8.0, 5.0, 10.0),
        observed_mask=(True, True, True, True, False, False, True, True, True, True),
        link_function_by_field=("identity", "identity"),
        n_factors=1,
        gamma_temporal=0.1,
        gamma_transition=0.1,
        gamma_spatial=0.0,
        multi_starts=2,
        seed=23,
    )
    result = solve_stdfm(schema, problem=problem, allow_uncertified=True, max_iterations=1500)
    checks = {
        "fit_result": isinstance(result, STDFMFitResult),
        "converged": isinstance(result, STDFMFitResult) and result.telemetry.converged,
        "masked_cell_reconstructed": isinstance(result, STDFMFitResult) and max(abs(result.reconstructed[4]), abs(result.reconstructed[5])) < 20.0,
        "identified_loading": isinstance(result, STDFMFitResult) and result.loadings[0] > 0.0,
        "multi_start_telemetry": isinstance(result, STDFMFitResult) and result.telemetry.starts_completed == 2,
    }
    payload = {"slice": "23A", "checks": checks}
    if isinstance(result, STDFMFitResult):
        payload["telemetry"] = result.telemetry.as_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
