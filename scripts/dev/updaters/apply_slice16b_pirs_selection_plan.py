from __future__ import annotations

import py_compile
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

SLICE = "16B"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/pirs/selection_plan.py",
    "src/pegasus/workflows/pirs_selection.py",
    "tests/unit/test_slice16b_pirs_selection_plan.py",
    "tests/integration/test_slice16b_attach_pirs_selection_plan.py",
    "scripts/dev/audits/audit_slice16b_pirs_selection_plan.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/",
    "src/pegasus/output/",
    "src/pegasus/efg/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice16b] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if path.startswith(FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def preflight() -> None:
    required = (
        "src/pegasus/pirs/run_candidates.py",
        "src/pegasus/workflows/pirs_candidates.py",
        "src/pegasus/pirs/field_selection.py",
        "src/pegasus/pirs/crossfit.py",
        "src/pegasus/pirs/families.py",
        "src/pegasus/pirs/diagnostics.py",
        "src/pegasus/pirs/schemas.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if compile_text.count("def run_compile(") != 1 or "def _run_compile_impl(" not in compile_text:
        fail("compile.py public/private run_compile boundary is not in the Slice 13C shape")
    run_candidates = read("src/pegasus/pirs/run_candidates.py")
    for token in ("build_pirs_candidates_from_run", "write_pirs_candidate_manifest", "pirs_candidate_rejection_reason"):
        if token not in run_candidates:
            fail(f"PIRS candidate gate missing expected API: {token}")


def write_selection_plan_module() -> None:
    code = r'''
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pegasus.pirs.crossfit import assert_standard_deep_not_in_sample, fold_scheme_for_budget
from pegasus.pirs.diagnostics import build_pirs_diagnostics
from pegasus.pirs.families import exposure_offset_source, family_for_outcome
from pegasus.pirs.field_selection import select_fields_for_pirs
from pegasus.pirs.schemas import FieldCandidate, PIRSSelectionResult

Budget = Literal["fast", "standard", "deep"]


class PIRSSelectionPlanError(ValueError):
    """Raised when a PIRS selection plan cannot be built from a candidate gate."""


@dataclass(frozen=True)
class PIRSSelectionPlan:
    schema_version: str
    gate: str
    status: str
    budget: Budget
    top_k: int
    residual_mode: str
    fold_scheme: dict[str, Any]
    family: str | None
    exposure_offset_source: str | None
    selected_outcome: FieldCandidate | None
    selected_covariates: tuple[FieldCandidate, ...]
    selected_offset: FieldCandidate | None
    selection_rejected: tuple[dict[str, Any], ...]
    gate_rejected: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    candidate_manifest_path: str
    warnings: tuple[str, ...]

    @property
    def selected_outcome_field_id(self) -> str | None:
        return self.selected_outcome.field_id if self.selected_outcome else None

    @property
    def selected_covariate_field_ids(self) -> tuple[str, ...]:
        return tuple(candidate.field_id for candidate in self.selected_covariates)

    @property
    def selected_offset_field_id(self) -> str | None:
        return self.selected_offset.field_id if self.selected_offset else None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate": self.gate,
            "status": self.status,
            "budget": self.budget,
            "top_k": self.top_k,
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "family": self.family,
            "exposure_offset_source": self.exposure_offset_source,
            "selected_outcome_field_id": self.selected_outcome_field_id,
            "selected_covariate_field_ids": list(self.selected_covariate_field_ids),
            "selected_offset_field_id": self.selected_offset_field_id,
            "selected_outcome": asdict(self.selected_outcome) if self.selected_outcome else None,
            "selected_covariates": [asdict(candidate) for candidate in self.selected_covariates],
            "selected_offset": asdict(self.selected_offset) if self.selected_offset else None,
            "selection_rejected": list(self.selection_rejected),
            "gate_rejected": list(self.gate_rejected),
            "diagnostics": dict(self.diagnostics),
            "candidate_manifest_path": self.candidate_manifest_path,
            "warnings": list(self.warnings),
            "non_behavior": {
                "model_fitted": False,
                "design_matrix_materialized": False,
                "residuals_materialized": False,
                "hsic_triggered": False,
            },
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PIRSSelectionPlanError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8")


def _tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _candidate_from_mapping(payload: dict[str, Any]) -> FieldCandidate:
    return FieldCandidate(
        field_id=str(payload["field_id"]),
        role=str(payload.get("role", "covariate")),  # type: ignore[arg-type]
        utility=float(payload.get("utility", 0.0) or 0.0),
        q_state=str(payload.get("q_state", "verified")),  # type: ignore[arg-type]
        carrier=str(payload.get("carrier", "unknown")),
        unit=str(payload.get("unit", "unknown")),
        support=dict(payload.get("support") or {}),
        variance=None if payload.get("variance") is None else float(payload.get("variance")),
        warnings=_tuple_str(payload.get("warnings")),
        provenance=_tuple_str(payload.get("provenance")),
    )


def load_pirs_candidate_manifest(candidate_manifest: str | Path) -> tuple[list[FieldCandidate], tuple[dict[str, Any], ...], dict[str, Any]]:
    path = Path(candidate_manifest)
    payload = _load_json(path)
    if payload.get("gate") != "pirs_candidate_gate":
        raise PIRSSelectionPlanError(f"Not a PIRS candidate gate manifest: {path}")
    candidates = [_candidate_from_mapping(item) for item in payload.get("candidates", [])]
    rejected = tuple(dict(item) for item in payload.get("rejected", []))
    return candidates, rejected, payload


def _support_kind(candidates: list[FieldCandidate]) -> str:
    for candidate in candidates:
        support = candidate.support or {}
        if support.get("months") or support.get("month"):
            return "monthly_municipal_panel"
    return "annual_municipal_panel"


def build_pirs_selection_plan(
    *,
    candidate_manifest: str | Path,
    budget: Budget = "fast",
) -> PIRSSelectionPlan:
    candidate_manifest_path = Path(candidate_manifest)
    candidates, gate_rejected, raw_manifest = load_pirs_candidate_manifest(candidate_manifest_path)
    selection = select_fields_for_pirs(candidates, budget=budget)
    support_kind = _support_kind(candidates)
    fold = fold_scheme_for_budget(budget=budget, support_kind=support_kind)
    assert_standard_deep_not_in_sample(budget, fold.residual_mode)

    family = None
    offset_source = None
    warnings: list[str] = []
    if selection.selected_outcome is None:
        status = "blocked_no_outcome"
        warnings.append("pirs_selection_has_no_model_eligible_outcome")
    else:
        status = "planned"
        family = family_for_outcome(outcome=selection.selected_outcome, offset=selection.selected_offset)
        try:
            offset_source = exposure_offset_source(family=family, offset=selection.selected_offset)
        except ValueError as exc:
            status = "blocked_invalid_offset"
            warnings.append(str(exc))
            offset_source = None
    diagnostics = build_pirs_diagnostics(
        selection=selection,
        fold_scheme=fold,
        exposure_offset_source=offset_source,
    ).as_manifest()
    if raw_manifest.get("rejected_count", 0):
        warnings.append("pirs_candidate_gate_rejected_fields_before_selection")
    if fold.residual_mode == "in_sample":
        warnings.append("fast_budget_in_sample_residuals_not_for_standard_hsic")
    return PIRSSelectionPlan(
        schema_version="1.0",
        gate="pirs_selection_plan",
        status=status,
        budget=budget,
        top_k=selection.top_k,
        residual_mode=fold.residual_mode,
        fold_scheme=fold.as_manifest(),
        family=family,
        exposure_offset_source=offset_source,
        selected_outcome=selection.selected_outcome,
        selected_covariates=selection.selected_covariates,
        selected_offset=selection.selected_offset,
        selection_rejected=selection.rejected,
        gate_rejected=gate_rejected,
        diagnostics=diagnostics,
        candidate_manifest_path=str(candidate_manifest_path),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def selection_plan_summary(plan: PIRSSelectionPlan, *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "gate": "pirs_selection_plan",
        "status": plan.status,
        "budget": plan.budget,
        "residual_mode": plan.residual_mode,
        "candidate_manifest_path": plan.candidate_manifest_path,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "selected_outcome_field_id": plan.selected_outcome_field_id,
        "selected_covariate_count": len(plan.selected_covariates),
        "selected_offset_field_id": plan.selected_offset_field_id,
        "gate_rejected_count": len(plan.gate_rejected),
        "selection_rejected_count": len(plan.selection_rejected),
        "model_fitted": False,
        "design_matrix_materialized": False,
        "residuals_materialized": False,
        "hsic_triggered": False,
    }


def write_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    root = Path(run_dir)
    candidate_path = Path(candidate_manifest) if candidate_manifest is not None else root / "Tables" / "pirs_field_candidates.json"
    output_path = Path(output) if output is not None else root / "Tables" / "pirs_selection_plan.json"
    plan = build_pirs_selection_plan(candidate_manifest=candidate_path, budget=budget)
    manifest = plan.as_manifest()
    manifest["manifest_path"] = str(output_path)
    manifest["summary"] = selection_plan_summary(plan, manifest_path=output_path)
    _write_json(output_path, manifest)
    return manifest


def attach_pirs_selection_plan_to_run(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = write_pirs_selection_plan(run_dir=root, candidate_manifest=candidate_manifest, output=output, budget=budget)
    summary = dict(manifest["summary"])
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["pirs_selection_gate"] = summary
        _write_json(path, payload)
    return summary
'''
    write("src/pegasus/pirs/selection_plan.py", code)


def write_workflow_module() -> None:
    code = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, write_pirs_selection_plan

Budget = Literal["fast", "standard", "deep"]


def run_write_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    return write_pirs_selection_plan(run_dir=run_dir, candidate_manifest=candidate_manifest, output=output, budget=budget)


def run_attach_pirs_selection_plan(
    *,
    run_dir: str | Path,
    candidate_manifest: str | Path | None = None,
    output: str | Path | None = None,
    budget: Budget = "fast",
) -> dict[str, Any]:
    return attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, output=output, budget=budget)
'''
    write("src/pegasus/workflows/pirs_selection.py", code)


def write_tests() -> None:
    unit = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.pirs.selection_plan import build_pirs_selection_plan, load_pirs_candidate_manifest


def _candidate(field_id: str, role: str, utility: float) -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": "Deaths",
        "unit": "counts",
        "support": {"years": [2020, 2021]},
        "variance": 0.2,
        "warnings": [],
        "provenance": ["fixture"],
    }


def test_slice16b_builds_selection_plan_from_candidate_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "pirs_field_candidates.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 3,
            "rejected_count": 1,
            "candidates": [
                _candidate("outcome_low", "outcome", 1.0),
                _candidate("outcome_high", "outcome", 10.0),
                _candidate("covariate_a", "covariate", 3.0),
            ],
            "rejected": [{"field_id": "metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }),
        encoding="utf-8",
    )
    candidates, rejected, payload = load_pirs_candidate_manifest(manifest)
    assert len(candidates) == 3
    assert rejected[0]["field_id"] == "metadata_only"
    assert payload["gate"] == "pirs_candidate_gate"

    plan = build_pirs_selection_plan(candidate_manifest=manifest, budget="standard")
    assert plan.status == "planned"
    assert plan.selected_outcome_field_id == "outcome_high"
    assert plan.selected_covariate_field_ids == ("covariate_a",)
    assert plan.residual_mode == "cross_fitted"
    assert plan.fold_scheme["n_folds"] == 5
    assert len(plan.gate_rejected) == 1
    assert "pirs_candidate_gate_rejected_fields_before_selection" in plan.warnings


def test_slice16b_blocks_plan_without_outcome(tmp_path: Path) -> None:
    manifest = tmp_path / "pirs_field_candidates.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 1,
            "rejected_count": 0,
            "candidates": [_candidate("covariate_only", "covariate", 5.0)],
            "rejected": [],
        }),
        encoding="utf-8",
    )
    plan = build_pirs_selection_plan(candidate_manifest=manifest, budget="fast")
    assert plan.status == "blocked_no_outcome"
    assert plan.selected_outcome_field_id is None
    assert "pirs_selection_has_no_model_eligible_outcome" in plan.warnings
'''
    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, write_pirs_selection_plan


def _candidate(field_id: str, role: str, utility: float, carrier: str = "Deaths") -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": carrier,
        "unit": "counts",
        "support": {"years": [2020], "municipalities": ["270430"]},
        "variance": 0.5,
        "warnings": [],
        "provenance": ["fixture"],
    }


def test_slice16b_attaches_selection_plan_to_run_without_mutating_first_class_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    candidate_manifest = run_dir / "Tables" / "pirs_field_candidates.json"
    candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
    candidate_manifest.write_text(
        json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 2,
            "rejected_count": 1,
            "candidates": [
                _candidate("verified_outcome", "outcome", 8.0),
                _candidate("verified_covariate", "covariate", 4.0),
            ],
            "rejected": [{"field_id": "efg_metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }),
        encoding="utf-8",
    )

    manifest = write_pirs_selection_plan(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
    assert manifest["status"] == "planned"
    assert manifest["selected_outcome_field_id"] == "verified_outcome"
    assert (run_dir / "Tables" / "pirs_selection_plan.json").exists()

    summary = attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
    assert summary["status"] == "planned"
    assert summary["selected_covariate_count"] == 1
    p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
    assert p_vector["pirs_selection_gate"]["selected_outcome_field_id"] == "verified_outcome"
    assert p_vector["pirs_selection_gate"]["model_fitted"] is False
    assert p_vector["pirs_selection_gate"]["design_matrix_materialized"] is False
'''
    write("tests/unit/test_slice16b_pirs_selection_plan.py", unit)
    write("tests/integration/test_slice16b_attach_pirs_selection_plan.py", integration)


def write_audit() -> None:
    audit = r'''
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.pirs.selection_plan import attach_pirs_selection_plan_to_run, build_pirs_selection_plan


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _candidate(field_id: str, role: str, utility: float) -> dict:
    return {
        "field_id": field_id,
        "role": role,
        "utility": utility,
        "q_state": "verified",
        "carrier": "Deaths",
        "unit": "counts",
        "support": {"years": [2020]},
        "variance": 0.2,
        "warnings": [],
        "provenance": ["fixture"],
    }


def main() -> int:
    errors: list[str] = []
    if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
        errors.append("compile.py must still contain exactly one public run_compile")
    if _count_defs(Path("src/pegasus/pirs/selection_plan.py"), "build_pirs_selection_plan") != 1:
        errors.append("PIRS selection plan API missing")
    if _count_defs(Path("src/pegasus/pirs/run_candidates.py"), "build_pirs_candidates_from_run") != 1:
        errors.append("PIRS candidate gate API missing")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "run"
        create_empty_output_bundle(run_dir)
        candidate_manifest = run_dir / "Tables" / "pirs_field_candidates.json"
        candidate_manifest.parent.mkdir(parents=True, exist_ok=True)
        candidate_manifest.write_text(json.dumps({
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "candidate_count": 2,
            "rejected_count": 1,
            "candidates": [_candidate("audit_outcome", "outcome", 9.0), _candidate("audit_covariate", "covariate", 4.0)],
            "rejected": [{"field_id": "metadata_only", "reason": "metadata_only_field_not_model_eligible"}],
        }), encoding="utf-8")
        plan = build_pirs_selection_plan(candidate_manifest=candidate_manifest, budget="standard")
        if plan.status != "planned":
            errors.append("selection plan did not reach planned status")
        if plan.selected_outcome_field_id != "audit_outcome":
            errors.append("selection plan did not select expected outcome")
        if plan.residual_mode != "cross_fitted":
            errors.append("standard budget did not select cross_fitted residual mode")
        summary = attach_pirs_selection_plan_to_run(run_dir=run_dir, candidate_manifest=candidate_manifest, budget="standard")
        if summary.get("model_fitted") is not False or summary.get("design_matrix_materialized") is not False:
            errors.append("selection gate summary must remain non-mutating/model-free")
        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        if p_vector.get("pirs_selection_gate", {}).get("selected_outcome_field_id") != "audit_outcome":
            errors.append("pirs_selection_gate summary was not attached to P_vector")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16B PIRS selection plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice16b_pirs_selection_plan.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    sys.path.insert(0, str(ROOT / "src"))
    from pegasus.pirs.selection_plan import build_pirs_selection_plan
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "candidates.json"
        manifest.write_text(json.dumps({
            "gate": "pirs_candidate_gate",
            "candidates": [{
                "field_id": "x", "role": "outcome", "utility": 1.0,
                "q_state": "verified", "carrier": "Deaths", "unit": "counts", "support": {},
            }],
            "rejected": [],
        }), encoding="utf-8")
        plan = build_pirs_selection_plan(candidate_manifest=manifest, budget="fast")
        if plan.selected_outcome_field_id != "x":
            fail("self-validate selection plan did not select outcome")


def main() -> None:
    preflight()
    write_selection_plan_module()
    write_workflow_module()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 16B updater applied: PIRS selection plan boundary.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
