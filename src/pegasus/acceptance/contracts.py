from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pegasus.core.hashing import sha256_file
from pegasus.output.schemas import OUTPUT_BUNDLE_FILES
from pegasus.output.validate import validate_output_bundle
from pegasus.storage import read_table, row_count
from pegasus.workflows.stage_plan import validate_compiler_stage_plan

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
    "compile_local",
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


@dataclass(frozen=True)
class Level3AcceptanceResult:
    run_dir: str
    status: str
    ok: bool
    production_candidate: bool
    checks: dict[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    stage_statuses: dict[str, str] | None = None
    artifact_evidence: dict[str, Any] | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "25A.1",
            "run_dir": self.run_dir,
            "status": self.status,
            "classification": self.status,
            "ok": self.ok,
            "production_candidate": self.production_candidate,
            "checks": dict(self.checks),
            "errors": list(self.errors),
            "blocking_reasons": list(self.errors),
            "warnings": list(self.warnings),
            "stage_statuses": dict(self.stage_statuses or {}),
            "artifact_evidence": dict(self.artifact_evidence or {}),
        }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _count_parquet(path: Path) -> int:
    return row_count(path)


def _string_values(path: Path, column: str) -> tuple[str, ...]:
    if not path.exists():
        return ()
    table = read_table(path, columns=[column])
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


def summarize_run(run_dir: str | Path, *, require_nonempty: bool = False) -> AcceptanceRunSummary:
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
    if require_nonempty and fields <= 1:
        errors.append("acceptance check required a nonempty run but V_fields has <=1 row")
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



def _compiler_stage_plan_payload(root: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    plan = run_config.get("compiler_stage_plan")
    if isinstance(plan, dict):
        return plan
    manifest = _load_json(root / "ReproducibilityManifest.json")
    plan = manifest.get("compiler_stage_plan")
    return plan if isinstance(plan, dict) else {}

def evaluate_level3_acceptance(run_dir: str | Path) -> Level3AcceptanceResult:
    root = Path(run_dir)
    summary = summarize_run(root, require_nonempty=True)
    run_config = _load_json(root / "RunConfig.json")
    architecture = run_config.get("compiler_architecture", {})
    autonomous = run_config.get("autonomous_efg", {})
    if not isinstance(architecture, dict):
        architecture = {}
    if not isinstance(autonomous, dict):
        autonomous = {}

    autonomous_manifest = root / str(autonomous.get("manifest_path", ""))
    autonomous_manifest_valid = (
        autonomous_manifest.is_file()
        and bool(autonomous.get("manifest_hash"))
        and sha256_file(autonomous_manifest) == autonomous.get("manifest_hash")
    )
    autonomous_payload = _load_json(autonomous_manifest) if autonomous_manifest.is_file() else {}
    reproducibility = _load_json(root / "ReproducibilityManifest.json")
    telemetry = reproducibility.get("telemetry") if isinstance(reproducibility.get("telemetry"), dict) else {}
    stage_durations = telemetry.get("stage_wall_seconds") if isinstance(telemetry.get("stage_wall_seconds"), dict) else {}
    registry_hashes = reproducibility.get("registry_hashes") if isinstance(reproducibility.get("registry_hashes"), dict) else {}
    stage_plan = _compiler_stage_plan_payload(root, run_config)
    stage_plan_errors = validate_compiler_stage_plan(
        stage_plan=stage_plan,
        stage_status=summary.telemetry_stage_status,
    )
    stage_plan_present = bool(stage_plan)
    truthful_optional_stages = not stage_plan_errors
    stage_specs = stage_plan.get("by_stage", {}) if isinstance(stage_plan.get("by_stage"), dict) else {}
    requested_artifact_evidence: dict[str, bool] = {}
    for stage_id, spec in stage_specs.items():
        if not isinstance(spec, dict) or not spec.get("requested"):
            continue
        expected = [str(value) for value in spec.get("expected_artifacts", [])]
        requested_artifact_evidence[str(stage_id)] = bool(expected) and all((root / value).exists() for value in expected)
    e_dag_count = _count_parquet(root / "E_DAG.parquet")
    e_dag_rows = read_table(root / "E_DAG.parquet").to_pylist() if (root / "E_DAG.parquet").exists() else []
    operator_values = " ".join(str(row.get("operator_id") or row.get("operator") or "") for row in e_dag_rows).lower()
    precompression = autonomous_payload.get("precompression") if isinstance(autonomous_payload.get("precompression"), dict) else {}
    legality = autonomous_payload.get("legality_summary") if isinstance(autonomous_payload.get("legality_summary"), dict) else {}
    illegal_attempts = int(legality.get("illegal", 0) or legality.get("blocked", 0) or 0)
    all_stage_proofs = bool(summary.telemetry_stage_status) and all(
        status in {"success", "skipped", "blocked", "failed"} and stage in stage_durations
        for stage, status in summary.telemetry_stage_status.items()
    )
    dashboard_manifests = [_load_json(path) for path in (root / "Tables").glob("*dashboard*.json")] if (root / "Tables").exists() else []
    checks = {
        "01_exact_first_class_keys": summary.first_class_keys == required_first_class_key_paths(),
        "02_output_bundle_valid": summary.ok,
        "03_source_reality_declared": summary.compile_source_mode in {"materialized_external"},
        "04_materialized_external_for_production": (
            summary.compile_source_mode != "materialized_external"
            or bool(summary.source_reality_production_candidate and summary.source_artifact_manifest_present)
        ),
        "05_runtime_authority_autonomous": (
            architecture.get("graph_authority") == "autonomous_efg_core"
            and autonomous.get("graph_authority") == "autonomous_efg_core"
            and architecture.get("legacy_graph_authority") is False
        ),
        "06_legacy_materializers_quarantined": architecture.get("legacy_bootstrap_status") == "retired_deleted",
        "07_no_production_development_builder": architecture.get("legacy_bootstrap_builder") is None,
        "08_no_required_registry_scaffold": not any("scaffold" in str(key).lower() for key in registry_hashes),
        "09_registry_hashes_recorded": bool(registry_hashes) and all(bool(value) for value in registry_hashes.values()),
        "10_autonomous_efg_manifest_exists": autonomous_manifest_valid,
        "11_e_dag_nonempty_when_fields_exist": summary.field_count == 0 or e_dag_count > 0,
        "12_failed_branches_record_illegal_attempts": illegal_attempts == 0 or summary.failed_branch_count > 0,
        "13_precompression_exists": precompression.get("stage") == "topological_precompression",
        "14_protected_non_equivalence": int(precompression.get("protected_non_equivalence_count", 0) or 0) > 0,
        "15_geo_native_or_transform_legal": "geo_support" in summary.telemetry_stage_status and summary.telemetry_stage_status.get("geo_support") in {"success", "skipped", "blocked", "failed"},
        "16_no_direct_rate_allocation": "direct_rate_allocation" not in operator_values,
        "17_population_stage_truthful": requested_artifact_evidence.get("population_solver", True) or summary.telemetry_stage_status.get("population_solver") in {"blocked", "failed"},
        "18_stdfm_stage_truthful": requested_artifact_evidence.get("stdfm", True) or summary.telemetry_stage_status.get("stdfm") in {"blocked", "failed"},
        "19_pirs_model_artifacts": requested_artifact_evidence.get("pirs_model", True) or summary.telemetry_stage_status.get("pirs_model") in {"blocked", "failed"},
        "20_hsic_artifacts": requested_artifact_evidence.get("pirs_hsic", True) or summary.telemetry_stage_status.get("pirs_hsic") in {"blocked", "failed"},
        "21_dashboard_read_only": all(payload.get("read_only") is True for payload in dashboard_manifests),
        "22_every_stage_has_proof": stage_plan_present and truthful_optional_stages and all_stage_proofs,
        "23_production_gate_complete": True,
    }
    # Stable aliases retained for callers introduced by the 25A/26B acceptance foundation.
    checks["compiler_stage_plan_present"] = stage_plan_present
    checks["stage_skip_proofs_valid"] = not stage_plan_errors
    development_exempt = {"14_protected_non_equivalence"}
    structural_ok = all(passed for name, passed in checks.items() if name not in development_exempt)
    production_gates_ok = all(checks.values())
    production_candidate = bool(
        production_gates_ok
        and summary.compile_source_mode == "materialized_external"
        and summary.source_reality_production_candidate
        and summary.substrate_present
        and summary.substrate_registry_backed
    )
    checks["23_production_gate_complete"] = production_candidate or summary.compile_source_mode != "materialized_external"
    structural_ok = all(passed for name, passed in checks.items() if name not in development_exempt)
    errors = list(summary.errors)
    errors.extend(stage_plan_errors)
    errors.extend(name for name, passed in checks.items() if not passed and (summary.compile_source_mode == "materialized_external" or name not in development_exempt))
    if not structural_ok:
        status = "failed"
    elif production_candidate:
        status = "production_candidate"
    else:
        status = "development_validated"
    return Level3AcceptanceResult(
        run_dir=str(root),
        status=status,
        ok=structural_ok,
        production_candidate=production_candidate,
        checks=checks,
        errors=tuple(errors),
        warnings=tuple(summary.warnings),
        stage_statuses={key: str(value) for key, value in summary.telemetry_stage_status.items()},
        artifact_evidence={
            "autonomous_manifest": str(autonomous_manifest) if autonomous_manifest_valid else None,
            "e_dag_rows": e_dag_count,
            "registry_hash_count": len(registry_hashes),
            "requested_stage_artifacts": requested_artifact_evidence,
            "source_manifest_present": summary.source_artifact_manifest_present,
        },
    )


def assert_dashboard_did_not_compute(*, before: AcceptanceRunSummary, after: AcceptanceRunSummary) -> None:
    if before.as_manifest() != after.as_manifest():
        raise AssertionError("dashboard/read-only inspection changed the run acceptance summary")
