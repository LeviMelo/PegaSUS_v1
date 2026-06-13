from __future__ import annotations

import ast
import py_compile
import sys
import zipfile
from pathlib import Path
from textwrap import dedent

SLICE = "16F"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/workflows/pirs_pipeline.py",
    "tests/unit/test_slice16f_pirs_planning_pipeline.py",
    "tests/integration/test_slice16f_pirs_planning_pipeline.py",
    "scripts/dev/audits/audit_slice16f_pirs_planning_pipeline.py",
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
    raise SystemExit(f"[slice16f] {message}")


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


def top_level_function_count(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/workflows/pirs_candidates.py",
        "src/pegasus/workflows/pirs_selection.py",
        "src/pegasus/workflows/pirs_design.py",
        "src/pegasus/workflows/pirs_readiness.py",
        "src/pegasus/pirs/run_candidates.py",
        "src/pegasus/pirs/selection_plan.py",
        "src/pegasus/pirs/design_plan.py",
        "src/pegasus/pirs/design_readiness.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if top_level_function_count(compile_text, "run_compile") != 1:
        fail("compile.py must still contain exactly one public run_compile")
    if top_level_function_count(compile_text, "_run_compile_impl") != 1:
        fail("compile.py must still contain exactly one private _run_compile_impl")
    checks = {
        "src/pegasus/workflows/pirs_candidates.py": ("run_attach_pirs_candidate_gate",),
        "src/pegasus/workflows/pirs_selection.py": ("run_attach_pirs_selection_plan",),
        "src/pegasus/workflows/pirs_design.py": ("run_attach_pirs_design_plan_to_run",),
        "src/pegasus/workflows/pirs_readiness.py": ("run_attach_pirs_design_readiness_to_run",),
    }
    for path, tokens in checks.items():
        text = read(path)
        for token in tokens:
            if token not in text:
                fail(f"{path} missing expected workflow API: {token}")


def workflow_module() -> str:
    return dedent(r'''
    """PIRS planning-pipeline orchestrator for run bundles.

    Slice 16F composes the existing 16A-16D workflow services.  It is still a
    planning boundary: it does not fit models, build numerical design matrices,
    create residuals, run HSIC, or mutate compile outputs beyond attaching a
    compact summary to existing JSON surfaces.
    """

    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    from pegasus.workflows.pirs_candidates import run_attach_pirs_candidate_gate
    from pegasus.workflows.pirs_selection import run_attach_pirs_selection_plan
    from pegasus.workflows.pirs_design import run_attach_pirs_design_plan_to_run
    from pegasus.workflows.pirs_readiness import run_attach_pirs_design_readiness_to_run


    DEFAULT_PIPELINE_MANIFEST = Path("Tables") / "pirs_planning_pipeline.json"
    JSON_SURFACES: tuple[str, ...] = (
        "RunConfig.json",
        "P_vector.json",
        "UserIntent.json",
        "ReproducibilityManifest.json",
    )


    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}


    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        return path


    def _gate_from_payload(payload: Any, key: str) -> dict[str, Any]:
        if isinstance(payload, dict):
            gate = payload.get(key)
            if isinstance(gate, dict):
                return gate
            summary = payload.get("summary")
            if isinstance(summary, dict):
                return summary
        return {}


    def _manifest_path(value: dict[str, Any], fallback: Path) -> str:
        path = value.get("manifest_path")
        if path is not None:
            return str(path)
        return str(fallback)


    def pirs_planning_pipeline_summary(payload: dict[str, Any], *, manifest_path: str | Path | None = None) -> dict[str, Any]:
        candidate_gate = payload.get("candidate_gate") if isinstance(payload.get("candidate_gate"), dict) else {}
        selection_gate = payload.get("selection_gate") if isinstance(payload.get("selection_gate"), dict) else {}
        design_gate = payload.get("design_gate") if isinstance(payload.get("design_gate"), dict) else {}
        readiness_gate = payload.get("design_readiness_gate") if isinstance(payload.get("design_readiness_gate"), dict) else {}
        return {
            "schema_version": "1.0",
            "slice": "16F",
            "gate": "pirs_planning_pipeline_gate",
            "status": str(payload.get("status", "blocked")),
            "ready": bool(readiness_gate.get("ready", False)),
            "budget": payload.get("budget"),
            "candidate_count": candidate_gate.get("candidate_count"),
            "candidate_rejected_count": candidate_gate.get("rejected_count"),
            "selection_status": selection_gate.get("status"),
            "selected_outcome_field_id": selection_gate.get("selected_outcome_field_id"),
            "selected_covariate_count": selection_gate.get("selected_covariate_count"),
            "design_status": design_gate.get("status"),
            "design_matrix_state": readiness_gate.get("design_matrix_state", design_gate.get("design_matrix_state")),
            "readiness_status": readiness_gate.get("status"),
            "ready_field_count": readiness_gate.get("ready_field_count"),
            "blocked_field_count": readiness_gate.get("blocked_field_count"),
            "model_fit_state": "not_started",
            "residual_state": "not_started",
            "hsic_state": "not_started",
            "manifest_path": str(manifest_path) if manifest_path is not None else payload.get("manifest_path"),
            "non_mutating_planning_only": True,
        }


    def attach_pirs_planning_pipeline_summary_to_run(
        *,
        run_dir: str | Path,
        summary: dict[str, Any],
    ) -> None:
        root = Path(run_dir)
        for rel in JSON_SURFACES:
            path = root / rel
            if not path.exists():
                continue
            payload = _load_json(path)
            payload["pirs_planning_pipeline_gate"] = summary
            _write_json(path, payload)


    def run_pirs_planning_pipeline(
        *,
        run_dir: str | Path,
        budget: str = "fast",
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run the 16A-16D PIRS planning chain and attach a compact summary.

        The chain is intentionally non-executing.  It stops at design readiness
        and records whether a future numerical design-matrix builder may proceed.
        """
        root = Path(run_dir)
        out = Path(output) if output is not None else root / DEFAULT_PIPELINE_MANIFEST

        candidate_gate = run_attach_pirs_candidate_gate(run_dir=root)
        selection_gate = run_attach_pirs_selection_plan(run_dir=root, budget=budget)
        design_payload = run_attach_pirs_design_plan_to_run(run_dir=root, budget=budget)
        design_gate = _gate_from_payload(design_payload, "pirs_design_gate")
        readiness_payload = run_attach_pirs_design_readiness_to_run(run_dir=root)
        readiness_gate = _gate_from_payload(readiness_payload, "pirs_design_readiness_gate")

        status = "ready" if readiness_gate.get("status") == "ready" else "blocked"
        payload: dict[str, Any] = {
            "schema_version": "1.0",
            "slice": "16F",
            "pipeline": "pirs_planning_pipeline",
            "status": status,
            "budget": budget,
            "candidate_gate": candidate_gate,
            "selection_gate": selection_gate,
            "design_gate": design_gate,
            "design_readiness_gate": readiness_gate,
            "artifacts": {
                "pirs_field_candidates": _manifest_path(candidate_gate, root / "Tables" / "pirs_field_candidates.json"),
                "pirs_selection_plan": _manifest_path(selection_gate, root / "Tables" / "pirs_selection_plan.json"),
                "pirs_design_plan": _manifest_path(design_gate, root / "Tables" / "pirs_design_plan.json"),
                "pirs_design_readiness": _manifest_path(readiness_gate, root / "Tables" / "pirs_design_readiness.json"),
            },
            "model_execution_state": "not_started",
            "residual_state": "not_started",
            "hsic_state": "not_started",
            "non_mutating_planning_only": True,
        }
        _write_json(out, payload)
        summary = pirs_planning_pipeline_summary(payload, manifest_path=out)
        payload["summary"] = summary
        payload["manifest_path"] = str(out)
        _write_json(out, payload)
        attach_pirs_planning_pipeline_summary_to_run(run_dir=root, summary=summary)
        return {"manifest_path": str(out), "pirs_planning_pipeline_gate": summary, "pipeline": payload}


    def inspect_pirs_planning_pipeline_manifest(manifest: str | Path) -> dict[str, Any]:
        payload = _load_json(Path(manifest))
        if not payload:
            raise FileNotFoundError(f"missing or invalid PIRS planning pipeline manifest: {manifest}")
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return summary
        return pirs_planning_pipeline_summary(payload, manifest_path=manifest)
    ''')


def unit_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    from pegasus.workflows import pirs_pipeline


    def _json_surfaces(run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
            (run_dir / name).write_text("{}", encoding="utf-8")
        (run_dir / "Tables").mkdir(exist_ok=True)


    def test_slice16f_pipeline_composes_gates_and_attaches_summary(tmp_path: Path, monkeypatch) -> None:
        run_dir = tmp_path / "run"
        _json_surfaces(run_dir)

        monkeypatch.setattr(
            pirs_pipeline,
            "run_attach_pirs_candidate_gate",
            lambda **kwargs: {"status": "evaluated", "candidate_count": 2, "rejected_count": 1, "manifest_path": str(run_dir / "Tables" / "pirs_field_candidates.json")},
        )
        monkeypatch.setattr(
            pirs_pipeline,
            "run_attach_pirs_selection_plan",
            lambda **kwargs: {"status": "planned", "selected_outcome_field_id": "outcome", "selected_covariate_count": 1, "manifest_path": str(run_dir / "Tables" / "pirs_selection_plan.json")},
        )
        monkeypatch.setattr(
            pirs_pipeline,
            "run_attach_pirs_design_plan_to_run",
            lambda **kwargs: {"pirs_design_gate": {"status": "planned", "design_matrix_state": "planned_only", "manifest_path": str(run_dir / "Tables" / "pirs_design_plan.json")}},
        )
        monkeypatch.setattr(
            pirs_pipeline,
            "run_attach_pirs_design_readiness_to_run",
            lambda **kwargs: {"pirs_design_readiness_gate": {"status": "blocked", "ready": False, "ready_field_count": 0, "blocked_field_count": 2, "design_matrix_state": "blocked_until_tensor_backed_fields", "manifest_path": str(run_dir / "Tables" / "pirs_design_readiness.json")}},
        )

        result = pirs_pipeline.run_pirs_planning_pipeline(run_dir=run_dir, budget="standard")
        gate = result["pirs_planning_pipeline_gate"]
        assert gate["status"] == "blocked"
        assert gate["budget"] == "standard"
        assert gate["candidate_count"] == 2
        assert gate["selected_outcome_field_id"] == "outcome"
        assert gate["blocked_field_count"] == 2
        assert gate["model_fit_state"] == "not_started"
        assert Path(result["manifest_path"]).exists()
        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        assert run_config["pirs_planning_pipeline_gate"]["status"] == "blocked"
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.workflows.pirs_pipeline import inspect_pirs_planning_pipeline_manifest, run_pirs_planning_pipeline


    def test_slice16f_pipeline_runs_on_empty_bundle_as_blocked_planning_chain(tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        create_empty_output_bundle(run_dir)

        result = run_pirs_planning_pipeline(run_dir=run_dir, budget="fast")
        gate = result["pirs_planning_pipeline_gate"]
        assert gate["status"] == "blocked"
        assert gate["ready"] is False
        assert gate["model_fit_state"] == "not_started"
        assert gate["residual_state"] == "not_started"
        assert gate["hsic_state"] == "not_started"
        assert (run_dir / "Tables" / "pirs_field_candidates.json").exists()
        assert (run_dir / "Tables" / "pirs_selection_plan.json").exists()
        assert (run_dir / "Tables" / "pirs_design_plan.json").exists()
        assert (run_dir / "Tables" / "pirs_design_readiness.json").exists()
        assert (run_dir / "Tables" / "pirs_planning_pipeline.json").exists()

        p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
        assert p_vector["pirs_planning_pipeline_gate"]["status"] == "blocked"
        inspected = inspect_pirs_planning_pipeline_manifest(run_dir / "Tables" / "pirs_planning_pipeline.json")
        assert inspected["status"] == "blocked"
    ''')


def audit_script() -> str:
    return dedent(r'''
    from __future__ import annotations

    import ast
    import json
    import tempfile
    from pathlib import Path

    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.workflows.pirs_pipeline import run_pirs_planning_pipeline


    def _count_defs(path: Path, name: str) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


    def main() -> int:
        errors: list[str] = []
        if _count_defs(Path("src/pegasus/workflows/compile.py"), "run_compile") != 1:
            errors.append("compile.py must still contain exactly one public run_compile")
        if _count_defs(Path("src/pegasus/workflows/pirs_pipeline.py"), "run_pirs_planning_pipeline") != 1:
            errors.append("pirs_pipeline workflow API missing")
        for path in (
            "src/pegasus/workflows/pirs_candidates.py",
            "src/pegasus/workflows/pirs_selection.py",
            "src/pegasus/workflows/pirs_design.py",
            "src/pegasus/workflows/pirs_readiness.py",
        ):
            if not Path(path).exists():
                errors.append(f"missing planning dependency: {path}")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            create_empty_output_bundle(run_dir)
            result = run_pirs_planning_pipeline(run_dir=run_dir, budget="fast")
            gate = result.get("pirs_planning_pipeline_gate", {})
            if gate.get("status") != "blocked":
                errors.append("empty run bundle should produce a blocked PIRS planning pipeline")
            for name in (
                "pirs_field_candidates.json",
                "pirs_selection_plan.json",
                "pirs_design_plan.json",
                "pirs_design_readiness.json",
                "pirs_planning_pipeline.json",
            ):
                if not (run_dir / "Tables" / name).exists():
                    errors.append(f"pipeline did not write Tables/{name}")
            p_vector = json.loads((run_dir / "P_vector.json").read_text(encoding="utf-8"))
            if "pirs_planning_pipeline_gate" not in p_vector:
                errors.append("P_vector.json missing pirs_planning_pipeline_gate")

        payload = {"ok": not errors, "errors": errors}
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        if errors:
            return 1
        print("AUDIT PASSED: Slice 16F PIRS planning pipeline orchestrator")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    ''')


def write_files() -> None:
    write("src/pegasus/workflows/pirs_pipeline.py", workflow_module())
    write("tests/unit/test_slice16f_pirs_planning_pipeline.py", unit_test())
    write("tests/integration/test_slice16f_pirs_planning_pipeline.py", integration_test())
    write("scripts/dev/audits/audit_slice16f_pirs_planning_pipeline.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pegasus.workflows.pirs_pipeline import run_pirs_planning_pipeline, pirs_planning_pipeline_summary

    if not callable(run_pirs_planning_pipeline) or not callable(pirs_planning_pipeline_summary):
        fail("post-validate: pirs planning pipeline APIs not callable")


def main() -> None:
    preflight()
    write_files()
    self_validate()
    print("Slice 16F updater applied: PIRS planning pipeline orchestrator added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
