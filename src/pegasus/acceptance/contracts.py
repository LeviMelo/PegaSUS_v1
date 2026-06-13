
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle

FORBIDDEN_DASHBOARD_COMPUTE_STAGES: tuple[str, ...] = (
    "datasus_acquire",
    "datasus_decode",
    "sidra_fetch",
    "sidra_extract",
    "sidra_normalize",
    "she_build",
    "efg_build",
    "population_solver",
    "race_bridge",
    "stdfm",
    "pirs_model",
    "pirs_hsic",
)

CANONICAL_ACCEPTANCE_SURFACES: tuple[str, ...] = (
    "compile_smoke",
    "race_bridge_compile",
    "cnes_sih_compile",
    "population_tensor_compile",
    "sidra_stdfm_standalone",
    "pirs_parametric_standalone",
    "hsic_standalone",
    "dashboard_read_only",
)

REQUIRED_OUTPUT_TABLES: tuple[str, ...] = (
    "V_fields",
    "E_DAG",
    "Q_tensor",
    "Warnings",
    "ModelAssociations",
    "ResidualAssociations",
    "Hypotheses",
    "VariableDictionary",
    "FailedBranches",
    "QuarantinedFields",
    "ForcedFields",
)


@dataclass(frozen=True)
class AcceptanceRunSummary:
    run_dir: str
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    first_class_keys: tuple[str, ...]
    field_count: int
    q_count: int
    warning_count: int
    failed_branch_count: int
    model_association_count: int
    residual_association_count: int
    hypothesis_count: int
    dashboard_safe_values: tuple[str, ...]
    compile_source_mode: str | None
    source_artifact_manifest_present: bool
    source_reality_production_candidate: bool | None
    substrate_present: bool
    substrate_source_reality_mode: str | None
    substrate_admissible_candidate_count: int | None
    substrate_excluded_field_count: int | None
    substrate_registry_backed: bool | None
    telemetry_stage_status: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "first_class_keys": list(self.first_class_keys),
            "field_count": self.field_count,
            "q_count": self.q_count,
            "warning_count": self.warning_count,
            "failed_branch_count": self.failed_branch_count,
            "model_association_count": self.model_association_count,
            "residual_association_count": self.residual_association_count,
            "hypothesis_count": self.hypothesis_count,
            "dashboard_safe_values": list(self.dashboard_safe_values),
            "compile_source_mode": self.compile_source_mode,
            "source_artifact_manifest_present": self.source_artifact_manifest_present,
            "source_reality_production_candidate": self.source_reality_production_candidate,
            "substrate_present": self.substrate_present,
            "substrate_source_reality_mode": self.substrate_source_reality_mode,
            "substrate_admissible_candidate_count": self.substrate_admissible_candidate_count,
            "substrate_excluded_field_count": self.substrate_excluded_field_count,
            "substrate_registry_backed": self.substrate_registry_backed,
            "telemetry_stage_status": self.telemetry_stage_status,
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_parquet(path: Path) -> int:
    if not path.exists():
        return 0
    return int(pq.read_table(path).num_rows)


def _string_values(path: Path, column: str) -> tuple[str, ...]:
    if not path.exists():
        return ()
    table = pq.read_table(path, columns=[column])
    values = sorted({str(v.as_py()) for v in table[column] if v.as_py() is not None})
    return tuple(values)


def exact_first_class_keys(run_dir: str | Path) -> tuple[str, ...]:
    root = Path(run_dir)
    return tuple(sorted(p.name if p.is_dir() else p.name for p in root.iterdir()))


def required_first_class_key_paths() -> tuple[str, ...]:
    return tuple(sorted(OUTPUT_BUNDLE_FILES.values()))


def acceptance_plan() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "slice": "11A",
        "purpose": "milestone acceptance hardening and canonical output-contract drift detection",
        "surfaces": list(CANONICAL_ACCEPTANCE_SURFACES),
        "required_output_keys": list(OUTPUT_BUNDLE_FILES.keys()),
        "required_output_paths": list(required_first_class_key_paths()),
        "required_tables": list(REQUIRED_OUTPUT_TABLES),
        "forbidden_dashboard_compute_stages": list(FORBIDDEN_DASHBOARD_COMPUTE_STAGES),
    }


def summarize_run(run_dir: str | Path, *, require_non_scaffold: bool = False) -> AcceptanceRunSummary:
    root = Path(run_dir)
    validation = validate_output_bundle(run_dir=str(root))
    errors = list(validation.errors)
    manifest = _load_json(root / "ReproducibilityManifest.json")
    run_config = _load_json(root / "RunConfig.json")
    source_reality = run_config.get("source_artifact_reality")
    if not isinstance(source_reality, dict):
        source_reality = manifest.get("source_artifact_reality")
    if not isinstance(source_reality, dict):
        source_reality = {}
    compile_source_mode = (
        run_config.get("compile_source_mode")
        or manifest.get("compile_source_mode")
        or source_reality.get("compile_source_mode")
    )
    source_artifact_manifest_present = bool(
        run_config.get("source_artifact_manifest_present")
        if "source_artifact_manifest_present" in run_config
        else source_reality.get("source_artifact_manifest_present", False)
    )
    source_reality_production_candidate = run_config.get("source_reality_production_candidate")
    if source_reality_production_candidate is None:
        source_reality_production_candidate = source_reality.get("production_candidate")
    telemetry = manifest.get("telemetry", {}) if isinstance(manifest.get("telemetry", {}), dict) else {}
    stage_status = telemetry.get("stage_status", {}) if isinstance(telemetry.get("stage_status", {}), dict) else {}

    substrate_gate = run_config.get("substrate_gate")
    if not isinstance(substrate_gate, dict):
        substrate_gate = manifest.get("substrate_gate")
    if not isinstance(substrate_gate, dict):
        p_vector = _load_json(root / "P_vector.json")
        substrate_gate = p_vector.get("substrate_gate") if isinstance(p_vector.get("substrate_gate"), dict) else {}
    if not isinstance(substrate_gate, dict):
        substrate_gate = {}
    substrate_present = substrate_gate.get("status") == "evaluated"
    substrate_source_reality_mode = substrate_gate.get("source_reality_mode") if substrate_gate else None
    substrate_admissible_candidate_count = substrate_gate.get("admissible_candidate_count") if substrate_gate else None
    substrate_excluded_field_count = substrate_gate.get("excluded_field_count") if substrate_gate else None
    substrate_registry_backed = substrate_gate.get("registry_backed") if substrate_gate else None

    v_path = root / "V_fields.parquet"
    q_path = root / "Q_tensor.parquet"
    fields = _count_parquet(v_path)
    q_rows = _count_parquet(q_path)
    if fields != q_rows:
        errors.append(f"V_fields/Q_tensor row-count mismatch: {fields} != {q_rows}")
    if require_non_scaffold and fields <= 1:
        errors.append("acceptance check required a non-scaffold run but V_fields has <=1 row")
    dashboard_values = _string_values(v_path, "dashboard_safe")
    if not dashboard_values:
        errors.append("V_fields.dashboard_safe has no values")
    bad_dashboard = sorted(v for v in dashboard_values if v not in {"True", "False", "warning"})
    if bad_dashboard:
        errors.append(f"invalid dashboard_safe values: {bad_dashboard}")

    return AcceptanceRunSummary(
        run_dir=str(root),
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(validation.warnings),
        first_class_keys=exact_first_class_keys(root),
        field_count=fields,
        q_count=q_rows,
        warning_count=_count_parquet(root / "Warnings.parquet"),
        failed_branch_count=_count_parquet(root / "FailedBranches.parquet"),
        model_association_count=_count_parquet(root / "ModelAssociations.parquet"),
        residual_association_count=_count_parquet(root / "ResidualAssociations.parquet"),
        hypothesis_count=_count_parquet(root / "Hypotheses.parquet"),
        dashboard_safe_values=dashboard_values,
        compile_source_mode=compile_source_mode,
        source_artifact_manifest_present=source_artifact_manifest_present,
        source_reality_production_candidate=(
            bool(source_reality_production_candidate)
            if source_reality_production_candidate is not None
            else None
        ),
        substrate_present=substrate_present,
        substrate_source_reality_mode=substrate_source_reality_mode,
        substrate_admissible_candidate_count=substrate_admissible_candidate_count,
        substrate_excluded_field_count=substrate_excluded_field_count,
        substrate_registry_backed=substrate_registry_backed,
        telemetry_stage_status=dict(stage_status),
    )


def assert_dashboard_did_not_compute(*, before: AcceptanceRunSummary, after: AcceptanceRunSummary) -> None:
    if before.as_manifest() != after.as_manifest():
        raise AssertionError("dashboard/read-only inspection changed the run acceptance summary")
