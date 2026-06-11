from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    run = Path(parser.parse_args().run)
    result = validate_output_bundle(run_dir=str(run))
    if not result.ok:
        raise SystemExit("Output validation failed: " + "; ".join(result.errors))

    v = pl.read_parquet(run / "V_fields.parquet")
    q = pl.read_parquet(run / "Q_tensor.parquet")
    failed = pl.read_parquet(run / "FailedBranches.parquet")
    manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))

    names = set(v["name"].to_list())
    if "PopulationTensorOfficialSIDRAIndependent" not in names:
        raise SystemExit("Missing PopulationTensorOfficialSIDRAIndependent field")

    pop_rows = [row for row in v.to_dicts() if row["name"] == "PopulationTensorOfficialSIDRAIndependent"]
    support = json.loads(pop_rows[0]["support_json"])
    required = {
        "PopulationTensorMode": "independent_denominator",
        "SolverBackend": "algebraic_sidra_anchor_identity",
        "SparseJacobian": True,
        "DenominatorFeedbackWarning": False,
    }
    for key, expected in required.items():
        if support.get(key) != expected:
            raise SystemExit(f"Population tensor support metadata mismatch for {key}: {support.get(key)!r}")
    if "population_tensor_diagnostics" not in support:
        raise SystemExit("Population tensor diagnostics missing from support_json")

    if pop_rows[0]["field_id"] not in set(q["field_id"].to_list()):
        raise SystemExit("Population tensor field missing from Q_tensor")
    if "population_tensor" not in manifest or "population_tensor" not in run_config:
        raise SystemExit("Population tensor metadata missing from manifest/run config")
    if manifest["telemetry"]["stage_status"].get("population_solver") != "success":
        raise SystemExit("population_solver telemetry stage not successful")
    if "failed_dense_national_population_tensor_above_threshold" not in set(failed["failed_branch_id"].to_list()):
        raise SystemExit("Dense national population tensor abort branch missing")

    print("AUDIT PASSED: Slice 6A population tensor independent denominator, solver registry metadata, diagnostics, SIM-feedback warning surface, and dense-scale abort gate validated.")


if __name__ == "__main__":
    main()
