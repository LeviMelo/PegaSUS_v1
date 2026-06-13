from __future__ import annotations

import ast
import importlib
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "16C"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/pirs/design_plan.py",
    "src/pegasus/workflows/pirs_design.py",
    "tests/unit/test_slice16c_pirs_design_plan.py",
    "tests/integration/test_slice16c_attach_pirs_design_plan.py",
    "scripts/dev/audits/audit_slice16c_pirs_design_plan.py",
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
    raise SystemExit(f"[slice16c] {message}")


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


def _top_level_defs(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/pirs/run_candidates.py",
        "src/pegasus/pirs/selection_plan.py",
        "src/pegasus/workflows/pirs_candidates.py",
        "src/pegasus/workflows/pirs_selection.py",
        "src/pegasus/workflows/compile.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if _top_level_defs(compile_text, "run_compile") != 1:
        fail("compile.py must have exactly one public run_compile before Slice 16C")
    if _top_level_defs(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must have exactly one private _run_compile_impl before Slice 16C")
    selection_text = read("src/pegasus/pirs/selection_plan.py")
    for token in ("build_pirs_selection_plan", "write_pirs_selection_plan", "attach_pirs_selection_plan_to_run"):
        if token not in selection_text:
            fail(f"selection_plan.py missing required API: {token}")


def write_design_plan() -> None:
    module = r'''
from __future__ import annotations

"""Non-mutating PIRS design-plan boundary.

Slice 16C consumes a Slice 16B PIRS selection plan and emits an auditable
planned-only design contract.  It intentionally does not read observation
matrices, fit models, create residuals, or write first-class output-bundle keys.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DESIGN_PLAN_SCHEMA_VERSION = "1.0"
DESIGN_PLAN_ARTIFACT = "pirs_design_plan"
DESIGN_GATE_KEY = "pirs_design_gate"
DEFAULT_SELECTION_PLAN = Path("Tables/pirs_selection_plan.json")
DEFAULT_DESIGN_PLAN = Path("Tables/pirs_design_plan.json")
JSON_ATTACH_TARGETS: tuple[str, ...] = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)


class PIRSDesignPlanError(ValueError):
    """Raised when a PIRS selection plan cannot be converted to a design plan."""


@dataclass(frozen=True)
class PIRSDesignTerm:
    term_id: str
    role: str
    field_id: str | None
    transform: str
    dtype: str
    required: bool
    source: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "term_id": self.term_id,
            "role": self.role,
            "field_id": self.field_id,
            "transform": self.transform,
            "dtype": self.dtype,
            "required": self.required,
            "source": self.source,
        }


@dataclass(frozen=True)
class PIRSDesignPlan:
    status: str
    design_matrix_state: str
    budget: str
    residual_mode: str
    family: str
    fold_scheme: dict[str, Any]
    outcome_field_id: str | None
    covariate_field_ids: tuple[str, ...]
    offset_field_id: str | None
    terms: tuple[PIRSDesignTerm, ...]
    rejected: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]
    source_selection_plan_hash: str | None = None

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def covariate_count(self) -> int:
        return len(self.covariate_field_ids)

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": DESIGN_PLAN_SCHEMA_VERSION,
            "artifact": DESIGN_PLAN_ARTIFACT,
            "status": self.status,
            "design_matrix_state": self.design_matrix_state,
            "budget": self.budget,
            "residual_mode": self.residual_mode,
            "family": self.family,
            "fold_scheme": self.fold_scheme,
            "outcome_field_id": self.outcome_field_id,
            "covariate_field_ids": list(self.covariate_field_ids),
            "offset_field_id": self.offset_field_id,
            "terms": [term.as_manifest() for term in self.terms],
            "rejected": list(self.rejected),
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
            "source_selection_plan_hash": self.source_selection_plan_hash,
            "non_mutating": True,
            "model_fit_state": "not_started",
            "residual_state": "not_started",
            "hsic_state": "not_started",
            "created_at": _utc_now(),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return p


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> tuple[str, ...]:
    out: list[str] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            item = item.get("field_id") or item.get("id") or item.get("name")
        text = _string_or_none(item)
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _selection_payload(selection_plan: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(selection_plan, (str, Path)):
        return _load_json(selection_plan)
    if isinstance(selection_plan, Mapping):
        return dict(selection_plan)
    raise PIRSDesignPlanError(f"Unsupported selection plan type: {type(selection_plan)!r}")


def _selected_outcome(plan: Mapping[str, Any]) -> str | None:
    direct = _string_or_none(plan.get("selected_outcome_field_id"))
    if direct:
        return direct
    selection = _as_dict(plan.get("selection"))
    direct = _string_or_none(selection.get("selected_outcome_field_id") or selection.get("outcome_field_id"))
    if direct:
        return direct
    outcome = plan.get("selected_outcome") or selection.get("outcome")
    if isinstance(outcome, Mapping):
        return _string_or_none(outcome.get("field_id") or outcome.get("id"))
    return _string_or_none(outcome)


def _selected_covariates(plan: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("selected_covariate_field_ids", "covariate_field_ids", "selected_covariates"):
        values = _string_list(plan.get(key))
        if values:
            return values
    selection = _as_dict(plan.get("selection"))
    for key in ("selected_covariate_field_ids", "covariate_field_ids", "covariates"):
        values = _string_list(selection.get(key))
        if values:
            return values
    return ()


def _selected_offset(plan: Mapping[str, Any]) -> str | None:
    direct = _string_or_none(plan.get("selected_offset_field_id") or plan.get("offset_field_id"))
    if direct:
        return direct
    selection = _as_dict(plan.get("selection"))
    direct = _string_or_none(selection.get("selected_offset_field_id") or selection.get("offset_field_id"))
    if direct:
        return direct
    offset = plan.get("selected_offset") or selection.get("offset")
    if isinstance(offset, Mapping):
        return _string_or_none(offset.get("field_id") or offset.get("id"))
    return _string_or_none(offset)


def _rejected(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    rejected: list[dict[str, Any]] = []
    for key in ("gate_rejected", "selection_rejected", "rejected", "warnings_as_rejections"):
        for item in _as_list(plan.get(key)):
            if isinstance(item, Mapping):
                rejected.append(dict(item))
            elif item is not None:
                rejected.append({"reason": str(item)})
    return tuple(rejected)


def _family(plan: Mapping[str, Any], offset_field_id: str | None) -> str:
    direct = _string_or_none(plan.get("family"))
    if direct:
        return direct
    diagnostics = _as_dict(plan.get("diagnostics"))
    direct = _string_or_none(diagnostics.get("family"))
    if direct:
        return direct
    return "poisson_rate" if offset_field_id else "gaussian_identity"


def _fold_scheme(plan: Mapping[str, Any]) -> dict[str, Any]:
    fold = _as_dict(plan.get("fold_scheme"))
    if fold:
        return fold
    diagnostics = _as_dict(plan.get("diagnostics"))
    fold = _as_dict(diagnostics.get("fold_scheme"))
    if fold:
        return fold
    return {"mode": "not_declared", "fold_count": None, "source": "selection_plan_missing_fold_scheme"}


def _selection_hash(plan: Mapping[str, Any]) -> str | None:
    for key in ("selection_plan_hash", "source_selection_plan_hash", "manifest_hash", "registry_hash"):
        value = _string_or_none(plan.get(key))
        if value:
            return value
    return None


def _term(term_id: str, role: str, field_id: str | None, transform: str, *, required: bool, source: str) -> PIRSDesignTerm:
    return PIRSDesignTerm(
        term_id=term_id,
        role=role,
        field_id=field_id,
        transform=transform,
        dtype="float64",
        required=required,
        source=source,
    )


def build_pirs_design_plan(
    selection_plan: str | Path | Mapping[str, Any],
    *,
    budget: str | None = None,
) -> PIRSDesignPlan:
    """Build a non-mutating design plan from a PIRS selection plan."""

    payload = _selection_payload(selection_plan)
    outcome = _selected_outcome(payload)
    covariates = _selected_covariates(payload)
    offset = _selected_offset(payload)
    rejected = _rejected(payload)
    warnings = [str(w) for w in _as_list(payload.get("warnings")) if w]
    if not outcome:
        warnings.append("pirs_design_plan_missing_outcome")
    if not covariates:
        warnings.append("pirs_design_plan_missing_covariates")

    terms: list[PIRSDesignTerm] = [_term("intercept", "intercept", None, "constant_one", required=True, source="design_plan")]
    if outcome:
        terms.append(_term("response", "outcome", outcome, "identity", required=True, source="selection_plan"))
    for i, field_id in enumerate(covariates, start=1):
        terms.append(_term(f"covariate_{i:03d}", "covariate", field_id, "identity", required=True, source="selection_plan"))
    if offset:
        terms.append(_term("offset", "offset", offset, "log_exposure_offset", required=False, source="selection_plan"))

    status = "planned" if outcome and covariates else "blocked"
    return PIRSDesignPlan(
        status=status,
        design_matrix_state="planned_only",
        budget=str(budget or payload.get("budget") or "fast"),
        residual_mode=str(payload.get("residual_mode") or "in_sample"),
        family=_family(payload, offset),
        fold_scheme=_fold_scheme(payload),
        outcome_field_id=outcome,
        covariate_field_ids=covariates,
        offset_field_id=offset,
        terms=tuple(terms),
        rejected=rejected,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=("pirs_selection_plan", "slice16c_design_plan_boundary"),
        source_selection_plan_hash=_selection_hash(payload),
    )


def pirs_design_plan_summary(plan: PIRSDesignPlan | Mapping[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
    payload = plan.as_manifest() if isinstance(plan, PIRSDesignPlan) else dict(plan)
    covariates = _string_list(payload.get("covariate_field_ids"))
    warnings = _as_list(payload.get("warnings"))
    rejected = _as_list(payload.get("rejected"))
    return {
        "status": payload.get("status"),
        "design_matrix_state": payload.get("design_matrix_state"),
        "budget": payload.get("budget"),
        "residual_mode": payload.get("residual_mode"),
        "family": payload.get("family"),
        "fold_scheme": payload.get("fold_scheme"),
        "outcome_field_id": payload.get("outcome_field_id"),
        "covariate_count": len(covariates),
        "term_count": len(_as_list(payload.get("terms"))),
        "rejected_count": len(rejected),
        "warnings_count": len(warnings),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "non_mutating": True,
        "model_fit_state": payload.get("model_fit_state", "not_started"),
        "residual_state": payload.get("residual_state", "not_started"),
        "hsic_state": payload.get("hsic_state", "not_started"),
    }


def write_pirs_design_plan(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    selection_plan = selection_plan if selection_plan is not None else run_path / DEFAULT_SELECTION_PLAN
    output_path = Path(output) if output is not None else run_path / DEFAULT_DESIGN_PLAN
    plan = build_pirs_design_plan(selection_plan, budget=budget)
    payload = plan.as_manifest()
    payload["summary"] = pirs_design_plan_summary(plan, manifest_path=output_path)
    _write_json(output_path, payload)
    return payload


def _attach_summary(run_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(run_dir)
    for rel in JSON_ATTACH_TARGETS:
        path = root / rel
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        payload[DESIGN_GATE_KEY] = dict(summary)
        _write_json(path, payload)


def attach_pirs_design_plan_to_run(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | Mapping[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    payload = write_pirs_design_plan(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )
    summary = pirs_design_plan_summary(payload, manifest_path=Path(output) if output is not None else Path(run_dir) / DEFAULT_DESIGN_PLAN)
    _attach_summary(run_dir, summary)
    payload["summary"] = summary
    return payload
'''
    write("src/pegasus/pirs/design_plan.py", module)


def write_workflow() -> None:
    workflow = r'''
from __future__ import annotations

"""Workflow wrappers for Slice 16C PIRS design-plan artifacts."""

from pathlib import Path
from typing import Any

from pegasus.pirs.design_plan import attach_pirs_design_plan_to_run, write_pirs_design_plan


def run_plan_pirs_design(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | dict[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    return write_pirs_design_plan(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )


def run_attach_pirs_design_plan_to_run(
    *,
    run_dir: str | Path,
    selection_plan: str | Path | dict[str, Any] | None = None,
    output: str | Path | None = None,
    budget: str | None = None,
) -> dict[str, Any]:
    return attach_pirs_design_plan_to_run(
        run_dir=run_dir,
        selection_plan=selection_plan,
        output=output,
        budget=budget,
    )
'''
    write("src/pegasus/workflows/pirs_design.py", workflow)


def write_tests() -> None:
    unit = r'''
from __future__ import annotations

from pathlib import Path

from pegasus.pirs.design_plan import build_pirs_design_plan, pirs_design_plan_summary, write_pirs_design_plan


def _selection_payload() -> dict:
    return {
        "budget": "standard",
        "selected_outcome_field_id": "field:outcome",
        "selected_covariate_field_ids": ["field:cov_a", "field:cov_b"],
        "selected_offset_field_id": "field:population",
        "residual_mode": "cross_fitted",
        "family": "poisson_rate",
        "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
        "selection_rejected": [{"field_id": "field:bad", "reason": "zero_variance"}],
        "warnings": ["selection_warning"],
    }


def test_slice16c_builds_non_mutating_design_plan_from_selection_payload() -> None:
    plan = build_pirs_design_plan(_selection_payload())
    manifest = plan.as_manifest()

    assert manifest["artifact"] == "pirs_design_plan"
    assert manifest["status"] == "planned"
    assert manifest["design_matrix_state"] == "planned_only"
    assert manifest["model_fit_state"] == "not_started"
    assert manifest["residual_state"] == "not_started"
    assert manifest["hsic_state"] == "not_started"
    assert manifest["outcome_field_id"] == "field:outcome"
    assert manifest["covariate_field_ids"] == ["field:cov_a", "field:cov_b"]
    assert manifest["offset_field_id"] == "field:population"
    assert {term["role"] for term in manifest["terms"]} == {"intercept", "outcome", "covariate", "offset"}
    assert manifest["rejected"][0]["reason"] == "zero_variance"


def test_slice16c_blocks_design_plan_without_outcome_or_covariates() -> None:
    plan = build_pirs_design_plan({"budget": "fast", "selected_covariate_field_ids": []})
    manifest = plan.as_manifest()
    assert manifest["status"] == "blocked"
    assert "pirs_design_plan_missing_outcome" in manifest["warnings"]
    assert "pirs_design_plan_missing_covariates" in manifest["warnings"]


def test_slice16c_writes_design_plan_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    out = run_dir / "Tables" / "pirs_design_plan.json"
    payload = write_pirs_design_plan(run_dir=run_dir, selection_plan=_selection_payload(), output=out)
    assert out.exists()
    summary = pirs_design_plan_summary(payload, manifest_path=out)
    assert summary["status"] == "planned"
    assert summary["covariate_count"] == 2
    assert summary["non_mutating"] is True
'''
    integration = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run


def test_slice16c_attaches_design_gate_without_changing_first_class_bundle_keys(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    create_empty_output_bundle(run_dir)
    selection_path = run_dir / "Tables" / "pirs_selection_plan.json"
    selection_path.write_text(
        json.dumps(
            {
                "budget": "standard",
                "selected_outcome_field_id": "field:outcome",
                "selected_covariate_field_ids": ["field:cov_a"],
                "selected_offset_field_id": None,
                "residual_mode": "cross_fitted",
                "family": "gaussian_identity",
                "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
                "gate_rejected": [{"field_id": "field:quarantined", "reason": "metadata_only"}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = run_attach_pirs_design_plan_to_run(run_dir=run_dir, selection_plan=selection_path)
    manifest_path = run_dir / "Tables" / "pirs_design_plan.json"

    assert payload["status"] == "planned"
    assert manifest_path.exists()
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        attached = json.loads((run_dir / rel).read_text(encoding="utf-8"))
        assert attached["pirs_design_gate"]["status"] == "planned"
        assert attached["pirs_design_gate"]["design_matrix_state"] == "planned_only"
        assert attached["pirs_design_gate"]["model_fit_state"] == "not_started"
    assert validate_output_bundle(run_dir=str(run_dir)).ok
'''
    write("tests/unit/test_slice16c_pirs_design_plan.py", unit)
    write("tests/integration/test_slice16c_attach_pirs_design_plan.py", integration)


def write_audit() -> None:
    audit = r'''
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.pirs.design_plan import build_pirs_design_plan
from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run


def _count_defs(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def main() -> int:
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    design_path = Path("src/pegasus/pirs/design_plan.py")
    design_text = design_path.read_text(encoding="utf-8")

    if _count_defs(compile_path, "run_compile") != 1:
        errors.append("compile.py must retain exactly one public run_compile")
    if _count_defs(compile_path, "_run_compile_impl") != 1:
        errors.append("compile.py must retain exactly one private _run_compile_impl")
    for forbidden in ("fit_parametric_model", "ModelAssociations", "ResidualAssociations", "Hypotheses", "run_hsic"):
        if forbidden in design_text:
            errors.append(f"design_plan.py must not invoke or mention {forbidden}")

    plan = build_pirs_design_plan(
        {
            "selected_outcome_field_id": "field:outcome",
            "selected_covariate_field_ids": ["field:cov"],
            "residual_mode": "cross_fitted",
            "fold_scheme": {"mode": "cross_fitted", "fold_count": 5},
        }
    ).as_manifest()
    if plan.get("design_matrix_state") != "planned_only":
        errors.append("design plan must remain planned_only")
    if plan.get("model_fit_state") != "not_started":
        errors.append("design plan must not start model fitting")
    if plan.get("residual_state") != "not_started":
        errors.append("design plan must not create residuals")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        selection = run_dir / "Tables" / "pirs_selection_plan.json"
        selection.write_text(
            json.dumps(
                {
                    "selected_outcome_field_id": "field:outcome",
                    "selected_covariate_field_ids": ["field:cov"],
                    "residual_mode": "in_sample",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        attached = run_attach_pirs_design_plan_to_run(run_dir=run_dir, selection_plan=selection)
        if attached.get("status") != "planned":
            errors.append("attached design plan did not reach planned status")
        if not (run_dir / "Tables" / "pirs_design_plan.json").exists():
            errors.append("pirs_design_plan.json was not written")
        if not validate_output_bundle(run_dir=str(run_dir)).ok:
            errors.append("output bundle validation failed after design gate attach")

    payload = {"ok": not errors, "errors": errors}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 16C PIRS design plan boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    write("scripts/dev/audits/audit_slice16c_pirs_design_plan.py", audit)


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    mod = importlib.import_module("pegasus.pirs.design_plan")
    plan = mod.build_pirs_design_plan(
        {
            "selected_outcome_field_id": "field:outcome",
            "selected_covariate_field_ids": ["field:cov"],
            "residual_mode": "in_sample",
        }
    )
    manifest = plan.as_manifest()
    if manifest.get("status") != "planned":
        fail("self-validate: design plan did not reach planned status")
    if manifest.get("design_matrix_state") != "planned_only":
        fail("self-validate: design plan is not planned_only")


def main() -> None:
    preflight()
    write_design_plan()
    write_workflow()
    write_tests()
    write_audit()
    self_validate()
    print("Slice 16C updater applied: PIRS design plan boundary added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
