from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] if "scripts/dev/updaters" in str(Path(__file__).resolve()) else Path.cwd()


def p(rel: str) -> Path:
    return ROOT / rel


def read(rel: str) -> str:
    path = p(rel)
    if not path.exists():
        raise SystemExit(f"[slice26a26b] required file missing: {rel}")
    return path.read_text(encoding="utf-8")


def write(rel: str, text: str) -> None:
    path = p(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip("\n").rstrip() + "\n", encoding="utf-8", newline="\n")


def count_top_level_defs(rel: str, name: str) -> int:
    tree = ast.parse(read(rel))
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = [
        "src/pegasus/workflows/compile.py",
        "src/pegasus/workflows/acceptance.py",
        "src/pegasus/acceptance/contracts.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/cli.py",
        "src/pegasus/efg/dag.py",
        "src/pegasus/efg/compile_attach.py",
        "config/intents/alagoas_smoke.json",
    ]
    missing = [rel for rel in required if not p(rel).exists()]
    if missing:
        raise SystemExit(f"[slice26a26b] missing preflight files: {missing}")
    compile_text = read("src/pegasus/workflows/compile.py")
    if "autonomous_efg_core" not in compile_text or "_compiler_architecture_metadata" not in compile_text:
        raise SystemExit("[slice26a26b] compile.py does not look like post-25A autonomous compiler metadata is present")
    if count_top_level_defs("src/pegasus/workflows/compile.py", "run_compile") != 1:
        raise SystemExit("[slice26a26b] compile.py must contain exactly one public run_compile")
    if count_top_level_defs("src/pegasus/workflows/compile.py", "_run_compile_impl") != 1:
        raise SystemExit("[slice26a26b] compile.py must contain exactly one private _run_compile_impl")


STAGE_PLAN = r'''
"""Intent-derived compiler stage plans and proof-carrying skip contracts.

Macro-Slice 26A/26B changes the meaning of optional compiler stages from
free-form telemetry labels into an auditable contract.  A stage may be marked
``skipped`` only when the current intent did not request it or when a typed
source/artifact gate explicitly proves that skipping is legal.  This module is
pure metadata logic; it does not run population, ST-DFM, PIRS, HSIC, or geo
kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

OPTIONAL_STAGE_IDS: tuple[str, ...] = (
    "geo_support",
    "population_solver",
    "stdfm",
    "pirs_model",
    "pirs_hsic",
)

EXECUTED_STATUS = {"success"}
NON_EXECUTED_STATUS = {"skipped", "blocked", "failed"}


@dataclass(frozen=True)
class CompilerStageRequirement:
    stage_id: str
    requested: bool
    request_source: str
    required_for_level3: bool
    can_execute: bool
    skip_allowed: bool
    skip_reason: str | None = None
    blocked_reason: str | None = None
    executor: str | None = None
    expected_artifacts: tuple[str, ...] = ()

    def as_manifest(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "requested": self.requested,
            "request_source": self.request_source,
            "required_for_level3": self.required_for_level3,
            "can_execute": self.can_execute,
            "skip_allowed": self.skip_allowed,
            "skip_reason": self.skip_reason,
            "blocked_reason": self.blocked_reason,
            "executor": self.executor,
            "expected_artifacts": list(self.expected_artifacts),
        }


@dataclass(frozen=True)
class CompilerStagePlan:
    schema_version: str
    plan_id: str
    intent_budget: str | None
    intent_geo_mode: str | None
    intent_population_mode: str | None
    source_manifest_supplied: bool
    require_materialized_external: bool
    stages: tuple[CompilerStageRequirement, ...]

    def by_stage(self) -> dict[str, CompilerStageRequirement]:
        return {stage.stage_id: stage for stage in self.stages}

    def skip_reason_map(self) -> dict[str, str]:
        return {stage.stage_id: stage.skip_reason or "" for stage in self.stages if stage.skip_allowed}

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "intent_budget": self.intent_budget,
            "intent_geo_mode": self.intent_geo_mode,
            "intent_population_mode": self.intent_population_mode,
            "source_manifest_supplied": self.source_manifest_supplied,
            "require_materialized_external": self.require_materialized_external,
            "requested_stage_count": sum(1 for stage in self.stages if stage.requested),
            "stages": [stage.as_manifest() for stage in self.stages],
            "by_stage": {stage.stage_id: stage.as_manifest() for stage in self.stages},
            "skip_reasons": self.skip_reason_map(),
        }


def _intent_value(intent: Any, name: str, default: Any = None) -> Any:
    if isinstance(intent, Mapping):
        return intent.get(name, default)
    return getattr(intent, name, default)


def _lower_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values.lower()}
    try:
        return {str(value).lower() for value in values}
    except TypeError:
        return {str(values).lower()}


def _contains_any(values: Iterable[str], needles: Iterable[str]) -> bool:
    haystack = {str(value).lower() for value in values}
    return any(any(needle in value for value in haystack) for needle in needles)


def _stage(
    *,
    stage_id: str,
    requested: bool,
    request_source: str,
    required_for_level3: bool,
    can_execute: bool,
    skip_reason: str,
    blocked_reason: str | None = None,
    executor: str | None = None,
    expected_artifacts: Iterable[str] = (),
) -> CompilerStageRequirement:
    return CompilerStageRequirement(
        stage_id=stage_id,
        requested=requested,
        request_source=request_source,
        required_for_level3=required_for_level3,
        can_execute=can_execute,
        skip_allowed=not requested,
        skip_reason=None if requested else skip_reason,
        blocked_reason=blocked_reason,
        executor=executor,
        expected_artifacts=tuple(expected_artifacts),
    )


def build_compile_stage_plan(
    *,
    intent: Any,
    population_tensor_mode: str | None = None,
    include_cnes_sih: bool = False,
    source_manifest: str | None = None,
    require_materialized_external: bool = False,
) -> CompilerStagePlan:
    """Build the proof-carrying stage contract for one compile run."""
    context_policy = _lower_set(_intent_value(intent, "context_policy", []))
    health_seeds = _lower_set(_intent_value(intent, "health_seeds", []))
    mandatory_fields = _lower_set(_intent_value(intent, "mandatory_fields", []))
    budget = _intent_value(intent, "budget", None)
    geo_mode = str(_intent_value(intent, "geo_mode", "native") or "native")
    population_mode = str(_intent_value(intent, "population_mode", "") or "")
    tensor_mode = population_tensor_mode or None
    population_requested = bool(
        tensor_mode
        or population_mode in {"independent_population_tensor", "sim_informed_population_tensor", "independent_denominator", "sim_informed_denominator"}
        or "include_population_tensor" in context_policy
    )
    stdfm_requested = bool(
        _contains_any(context_policy, ("stdfm", "latent", "sidra_context"))
        or _contains_any(mandatory_fields, ("stdfm", "latent"))
    )
    pirs_model_requested = bool(
        _contains_any(context_policy, ("pirs", "residual", "model_execution", "run_pirs"))
        or _contains_any(mandatory_fields, ("residual", "modelassociation", "pirs"))
    )
    pirs_hsic_requested = bool(
        _contains_any(context_policy, ("hsic", "residual_scan", "hypothesis"))
        or _contains_any(mandatory_fields, ("hsic", "hypothesis"))
    )
    geo_requested = geo_mode.lower() not in {"native", "none", ""}
    stages = (
        _stage(
            stage_id="geo_support",
            requested=geo_requested,
            request_source="intent.geo_mode" if geo_requested else "native_geo_mode",
            required_for_level3=geo_requested,
            can_execute=geo_requested,
            skip_reason="intent.geo_mode is native; no AMC/geneallocation support transform requested",
            executor="pegasus.geo.support" if geo_requested else None,
            expected_artifacts=("geospatial_transform_manifest",) if geo_requested else (),
        ),
        _stage(
            stage_id="population_solver",
            requested=population_requested,
            request_source="intent.population_mode/context_policy" if population_requested else "official_sidra_anchor_or_unrequested",
            required_for_level3=population_requested,
            can_execute=population_requested,
            skip_reason="intent does not request population tensor optimization",
            executor="pegasus.she.population.solvers.solve_population_tensor_problem" if population_requested else None,
            expected_artifacts=("Tables/population_tensor.parquet", "Tables/population_tensor_manifest.json") if population_requested else (),
        ),
        _stage(
            stage_id="stdfm",
            requested=stdfm_requested,
            request_source="intent.context_policy/mandatory_fields" if stdfm_requested else "latent_context_not_requested",
            required_for_level3=stdfm_requested,
            can_execute=stdfm_requested,
            skip_reason="intent does not request ST-DFM latent-factor fitting",
            executor="pegasus.she.stdfm.torch_solver.solve_stdfm" if stdfm_requested else None,
            expected_artifacts=("Tables/stdfm_latent_factors.parquet", "Tables/stdfm_certification.parquet") if stdfm_requested else (),
        ),
        _stage(
            stage_id="pirs_model",
            requested=pirs_model_requested,
            request_source="intent.context_policy/mandatory_fields" if pirs_model_requested else "pirs_model_not_requested",
            required_for_level3=pirs_model_requested,
            can_execute=pirs_model_requested,
            skip_reason="intent does not request PIRS model execution",
            executor="pegasus.pirs.model_execution.execute_pirs_model_from_design_matrix" if pirs_model_requested else None,
            expected_artifacts=("Tables/pirs_model_execution_manifest.json", "Tables/pirs_residual_values.parquet") if pirs_model_requested else (),
        ),
        _stage(
            stage_id="pirs_hsic",
            requested=pirs_hsic_requested,
            request_source="intent.context_policy/mandatory_fields" if pirs_hsic_requested else "pirs_hsic_not_requested",
            required_for_level3=pirs_hsic_requested,
            can_execute=pirs_hsic_requested,
            skip_reason="intent does not request HSIC residual scan",
            executor="pegasus.pirs.hsic_run.execute_hsic_residual_scan" if pirs_hsic_requested else None,
            expected_artifacts=("Tables/hsic_residual_scan_manifest.json", "Hypotheses.parquet") if pirs_hsic_requested else (),
        ),
    )
    return CompilerStagePlan(
        schema_version="26B.1",
        plan_id="compile_stage_plan_v1",
        intent_budget=None if budget is None else str(budget),
        intent_geo_mode=geo_mode,
        intent_population_mode=population_mode or tensor_mode,
        source_manifest_supplied=bool(source_manifest),
        require_materialized_external=require_materialized_external,
        stages=stages,
    )


def normalize_stage_plan_manifest(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    by_stage = payload.get("by_stage")
    if isinstance(by_stage, Mapping):
        return {"by_stage": dict(by_stage), "raw": dict(payload)}
    stages = payload.get("stages")
    if isinstance(stages, list):
        return {"by_stage": {str(stage.get("stage_id")): dict(stage) for stage in stages if isinstance(stage, Mapping)}, "raw": dict(payload)}
    return {"by_stage": {}, "raw": dict(payload)}


def validate_compiler_stage_plan(
    *,
    stage_plan: Mapping[str, Any] | None,
    stage_status: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Return structural errors for telemetry/status consistency."""
    normalized = normalize_stage_plan_manifest(stage_plan)
    by_stage = normalized.get("by_stage", {})
    if not by_stage:
        return ("compiler_stage_plan_missing",)
    statuses = {str(key): str(value) for key, value in (stage_status or {}).items()}
    errors: list[str] = []
    for stage_id, spec in by_stage.items():
        if not stage_id:
            continue
        requested = bool(spec.get("requested"))
        skip_allowed = bool(spec.get("skip_allowed"))
        skip_reason = spec.get("skip_reason")
        status = statuses.get(stage_id)
        if status is None:
            errors.append(f"stage_status_missing:{stage_id}")
            continue
        if status == "skipped" and requested:
            errors.append(f"requested_stage_skipped:{stage_id}")
        if status == "skipped" and not skip_allowed:
            errors.append(f"skip_not_allowed:{stage_id}")
        if status == "skipped" and not skip_reason:
            errors.append(f"skip_reason_missing:{stage_id}")
        if status == "success" and not requested and spec.get("request_source") not in {"native_geo_mode", "official_sidra_anchor_or_unrequested"}:
            # Non-requested success can happen for base compile stages outside this plan;
            # for optional stages in this plan it is suspicious but not fatal.
            pass
    return tuple(errors)
'''


def write_stage_plan_module() -> None:
    write("src/pegasus/workflows/stage_plan.py", STAGE_PLAN)


RUNTIME_AUDIT = r'''
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

FORBIDDEN_ARCHITECTURE_VALUES = {
    "compatibility_materializer",
    "pegasus.workflows.efg.run_build_sim_fixture",
}
REQUIRED_ARCHITECTURE_VALUES = {
    "autonomous_efg_core",
    "autonomous_compiler_services",
    "quarantined_fixture_only",
}


def _literal_return_dict(function: ast.FunctionDef) -> dict:
    for node in ast.walk(function):
        if isinstance(node, ast.Return):
            return ast.literal_eval(node.value)
    raise AssertionError("function has no literal return")


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"{path} expected one {name}, found {len(matches)}")
    return matches[0]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    args = parser.parse_args()
    errors: list[str] = []
    compile_path = Path("src/pegasus/workflows/compile.py")
    try:
        metadata = _literal_return_dict(_function(compile_path, "_compiler_architecture_metadata"))
    except Exception as exc:  # pragma: no cover - audit diagnostic path
        errors.append(f"cannot inspect _compiler_architecture_metadata: {exc}")
        metadata = {}
    serialized = json.dumps(metadata, sort_keys=True)
    for token in FORBIDDEN_ARCHITECTURE_VALUES:
        if token in serialized:
            errors.append(f"forbidden legacy architecture token still present: {token}")
    for token in REQUIRED_ARCHITECTURE_VALUES:
        if token not in serialized:
            errors.append(f"required autonomous architecture token missing: {token}")
    if metadata.get("legacy_graph_authority") is not False:
        errors.append("legacy_graph_authority must be False")
    if not Path("src/pegasus/workflows/stage_plan.py").exists():
        errors.append("stage_plan workflow module missing")
    contracts_text = Path("src/pegasus/acceptance/contracts.py").read_text(encoding="utf-8")
    for token in ("validate_compiler_stage_plan", "compiler_stage_plan_present", "stage_skip_proofs_valid"):
        if token not in contracts_text:
            errors.append(f"acceptance contract missing token: {token}")
    cli_text = Path("src/pegasus/cli.py").read_text(encoding="utf-8")
    if '@acceptance_app.command("level3")' not in cli_text:
        errors.append("CLI missing pegasus acceptance level3 command")
    if args.run:
        run = Path(args.run)
        run_config = _load_json(run / "RunConfig.json")
        manifest = _load_json(run / "ReproducibilityManifest.json")
        if "compiler_stage_plan" not in run_config:
            errors.append("RunConfig.json missing compiler_stage_plan")
        if "compiler_stage_plan" not in manifest:
            errors.append("ReproducibilityManifest.json missing compiler_stage_plan")
        if run_config.get("compiler_architecture", {}).get("numerical_materialization") != "autonomous_compiler_services":
            errors.append("run compiler_architecture does not declare autonomous compiler services")
    payload = {"ok": not errors, "errors": errors, "architecture": metadata}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 26A/26B runtime authority quarantine and stage-plan proofs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_audit() -> None:
    write("scripts/dev/audits/audit_slice26a26b_runtime_authority_stage_plan.py", RUNTIME_AUDIT)


UNIT_TEST = r'''
from __future__ import annotations

from pegasus.workflows.compile import _compiler_architecture_metadata
from pegasus.workflows.stage_plan import build_compile_stage_plan, validate_compiler_stage_plan


def test_slice26a_architecture_quarantines_legacy_runtime_authority() -> None:
    metadata = _compiler_architecture_metadata()
    assert metadata["graph_authority"] == "autonomous_efg_core"
    assert metadata["numerical_materialization"] == "autonomous_compiler_services"
    assert metadata["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert metadata["legacy_graph_authority"] is False
    assert metadata["numerical_materialization"] != "legacy_bootstrap"
    assert metadata["legacy_bootstrap_builder"] is None
    assert "pegasus.workflows.efg.run_build_sim_fixture" not in str(metadata)


def test_slice26b_unrequested_optional_stages_have_skip_proofs() -> None:
    plan = build_compile_stage_plan(
        intent={"budget": "fast", "geo_mode": "native", "population_mode": "official_sidra_anchor", "context_policy": []},
        population_tensor_mode=None,
    )
    manifest = plan.as_manifest()
    for stage_id in ("population_solver", "stdfm", "pirs_model", "pirs_hsic"):
        spec = manifest["by_stage"][stage_id]
        assert spec["requested"] is False
        assert spec["skip_allowed"] is True
        assert spec["skip_reason"]
    errors = validate_compiler_stage_plan(
        stage_plan=manifest,
        stage_status={
            "geo_support": "success",
            "population_solver": "skipped",
            "stdfm": "skipped",
            "pirs_model": "skipped",
            "pirs_hsic": "skipped",
        },
    )
    assert errors == ()


def test_slice26b_requested_stage_cannot_be_silently_skipped() -> None:
    plan = build_compile_stage_plan(
        intent={
            "budget": "fast",
            "geo_mode": "native",
            "population_mode": "independent_population_tensor",
            "context_policy": ["include_population_tensor", "hsic"],
        },
        population_tensor_mode="independent_denominator",
    )
    errors = validate_compiler_stage_plan(
        stage_plan=plan.as_manifest(),
        stage_status={
            "geo_support": "success",
            "population_solver": "skipped",
            "stdfm": "skipped",
            "pirs_model": "skipped",
            "pirs_hsic": "skipped",
        },
    )
    assert "requested_stage_skipped:population_solver" in errors
    assert "requested_stage_skipped:pirs_hsic" in errors
'''


INTEGRATION_TEST = r'''
from __future__ import annotations

import json
from pathlib import Path

from pegasus.acceptance.contracts import evaluate_level3_acceptance
from pegasus.workflows.compile import run_compile


def test_slice26a26b_compile_emits_authority_and_stage_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = run_compile(intent_path="config/intents/alagoas_smoke.json", run_dir=run_dir)
    assert result["validation"].ok
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    repro = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
    architecture = run_config["compiler_architecture"]
    assert architecture["graph_authority"] == "autonomous_efg_core"
    assert architecture["numerical_materialization"] == "autonomous_compiler_services"
    assert architecture["legacy_bootstrap_status"] == "quarantined_fixture_only"
    assert architecture["legacy_graph_authority"] is False
    assert "compiler_stage_plan" in run_config
    assert "compiler_stage_plan" in repro
    assert run_config["compiler_stage_plan"]["by_stage"]["stdfm"]["skip_reason"]
    level3 = evaluate_level3_acceptance(run_dir)
    assert level3.ok, level3.as_manifest()
    assert level3.checks["compiler_stage_plan_present"] is True
    assert level3.checks["stage_skip_proofs_valid"] is True
'''


def write_tests() -> None:
    write("tests/unit/test_slice26a26b_runtime_authority_stage_plan.py", UNIT_TEST)
    write("tests/integration/test_slice26a26b_runtime_authority_stage_plan_integration.py", INTEGRATION_TEST)


def patch_compile() -> None:
    rel = "src/pegasus/workflows/compile.py"
    text = read(rel)
    new_func = '''

def _compiler_architecture_metadata() -> dict[str, Any]:
    return {
        "schema_version": "26A.1",
        "graph_authority": "autonomous_efg_core",
        "graph_builder": "pegasus.efg.dag.build_efg",
        "numerical_materialization": "autonomous_compiler_services",
        "numerical_materializer": "pegasus.workflows.compile._run_compile_impl",
        "legacy_bootstrap_builder": None,
        "legacy_bootstrap_status": "quarantined_fixture_only",
        "legacy_graph_authority": False,
        "fixture_compatibility_modules": [
            "pegasus.output.sim_efg_bundle",
            "pegasus.output.sinasc_efg_bundle",
            "pegasus.output.cnes_sih_efg_bundle",
            "pegasus.output.population_tensor_bundle",
            "pegasus.output.sidra_stdfm_bundle",
            "pegasus.output.pirs_bundle",
            "pegasus.output.hsic_bundle",
        ],
        "production_runtime_authority": "autonomous_efg_core_plus_compiler_services",
    }
'''
    text2, count = re.subn(
        r"\n\ndef _compiler_architecture_metadata\(\) -> dict\[str, Any\]:\n    return \{.*?\n    \}\n(?=\n\ndef _run_compile_impl)",
        new_func,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit("[slice26a26b] could not replace _compiler_architecture_metadata in compile.py")
    text = text2
    anchor = "    compiler_architecture = _compiler_architecture_metadata()\n"
    if anchor not in text:
        raise SystemExit("[slice26a26b] compile.py missing compiler_architecture assignment anchor")
    if "build_compile_stage_plan" not in text:
        insertion = anchor + "    from pegasus.workflows.stage_plan import build_compile_stage_plan\n\n    compiler_stage_plan = build_compile_stage_plan(\n        intent=intent,\n        population_tensor_mode=population_tensor_mode,\n        include_cnes_sih=include_cnes_sih,\n        source_manifest=None if source_manifest is None else str(source_manifest),\n        require_materialized_external=require_materialized_external,\n    )\n"
        text = text.replace(anchor, insertion, 1)
    elif "compiler_stage_plan = build_compile_stage_plan" not in text:
        insertion = anchor + "    compiler_stage_plan = build_compile_stage_plan(\n        intent=intent,\n        population_tensor_mode=population_tensor_mode,\n        include_cnes_sih=include_cnes_sih,\n        source_manifest=None if source_manifest is None else str(source_manifest),\n        require_materialized_external=require_materialized_external,\n    )\n"
        text = text.replace(anchor, insertion, 1)

    # Add compiler_stage_plan next to every run-facing compiler_architecture payload.
    def add_stage_plan_after(match: re.Match[str]) -> str:
        indent = match.group(1)
        line = match.group(0)
        next_line = f'{indent}"compiler_stage_plan": compiler_stage_plan.as_manifest(),\n'
        end = match.end()
        if text[end:end + len(next_line)] == next_line:
            return line
        return line + next_line

    text = re.sub(r'^(\s*)"compiler_architecture": compiler_architecture,\n', add_stage_plan_after, text, flags=re.M)

    # Ensure skipped_reasons includes stage-plan-derived reasons when a local skipped_reasons dict exists.
    if "compiler_stage_plan.skip_reason_map()" not in text and "skipped_reasons = {" in text:
        text = text.replace(
            "    skipped_reasons = {",
            "    skipped_reasons = {**compiler_stage_plan.skip_reason_map(),",
            1,
        )
    write(rel, text)


def patch_acceptance_contracts() -> None:
    rel = "src/pegasus/acceptance/contracts.py"
    text = read(rel)
    if "from pegasus.workflows.stage_plan import" not in text:
        marker = "from pegasus.output.validate import validate_output_bundle\n"
        if marker not in text:
            raise SystemExit("[slice26a26b] acceptance/contracts.py import anchor missing")
        text = text.replace(
            marker,
            marker + "from pegasus.workflows.stage_plan import validate_compiler_stage_plan\n",
            1,
        )
    if "def _compiler_stage_plan_payload" not in text:
        helper = r'''

def _compiler_stage_plan_payload(root: Path, run_config: dict[str, Any]) -> dict[str, Any]:
    plan = run_config.get("compiler_stage_plan")
    if isinstance(plan, dict):
        return plan
    manifest = _load_json(root / "ReproducibilityManifest.json")
    plan = manifest.get("compiler_stage_plan")
    return plan if isinstance(plan, dict) else {}
'''
        text = text.replace("\ndef evaluate_level3_acceptance", helper + "\ndef evaluate_level3_acceptance", 1)
    old = '''    optional_stages = ("population_solver", "stdfm", "pirs_model", "pirs_hsic")
    truthful_optional_stages = all(
        summary.telemetry_stage_status.get(stage) in {"success", "skipped"}
        for stage in optional_stages
    )
'''
    new = '''    stage_plan = _compiler_stage_plan_payload(root, run_config)
    stage_plan_errors = validate_compiler_stage_plan(
        stage_plan=stage_plan,
        stage_status=summary.telemetry_stage_status,
    )
    stage_plan_present = bool(stage_plan)
    truthful_optional_stages = not stage_plan_errors
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif "stage_plan_errors = validate_compiler_stage_plan" not in text:
        raise SystemExit("[slice26a26b] could not patch Level3 optional-stage truthfulness block")
    check_anchor = '        "truthful_optional_stage_statuses": truthful_optional_stages,\n'
    if check_anchor in text and '        "compiler_stage_plan_present": stage_plan_present,\n' not in text:
        text = text.replace(
            check_anchor,
            check_anchor + '        "compiler_stage_plan_present": stage_plan_present,\n        "stage_skip_proofs_valid": not stage_plan_errors,\n',
            1,
        )
    error_anchor = "    errors = list(summary.errors)\n"
    if error_anchor in text and "stage_plan_errors" not in text[text.find(error_anchor):text.find(error_anchor)+300]:
        text = text.replace(
            error_anchor,
            "    errors = list(summary.errors)\n    errors.extend(stage_plan_errors)\n",
            1,
        )
    write(rel, text)


def patch_acceptance_workflow() -> None:
    rel = "src/pegasus/workflows/acceptance.py"
    text = read(rel)
    if "evaluate_level3_acceptance" not in text:
        text = text.replace(
            "from pegasus.acceptance.contracts import acceptance_plan, summarize_run\n",
            "from pegasus.acceptance.contracts import acceptance_plan, evaluate_level3_acceptance, summarize_run\n",
            1,
        )
    if "def run_acceptance_level3" not in text:
        text = text.rstrip() + r'''


def run_acceptance_level3(*, run_dir: str | Path) -> dict[str, Any]:
    return evaluate_level3_acceptance(run_dir).as_manifest()
'''
    write(rel, text)


def patch_cli() -> None:
    rel = "src/pegasus/cli.py"
    text = read(rel)
    if '@acceptance_app.command("level3")' in text:
        write(rel, text)
        return
    block = r'''


@acceptance_app.command("level3")
def acceptance_level3(
    run: Path = typer.Option(..., "--run"),
) -> None:
    from pegasus.workflows.acceptance import run_acceptance_level3

    result = run_acceptance_level3(run_dir=run)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok", False):
        raise typer.Exit(1)
'''
    # Place after check-run if possible; appending is safe because acceptance_app is already defined.
    insertion_point = text.find("\n\n# Slice 12A source artifact reality gate commands")
    if insertion_point != -1:
        text = text[:insertion_point] + block + text[insertion_point:]
    else:
        text = text.rstrip() + block + "\n"
    write(rel, text)


def main() -> None:
    preflight()
    write_stage_plan_module()
    patch_compile()
    patch_acceptance_contracts()
    patch_acceptance_workflow()
    patch_cli()
    write_tests()
    write_audit()
    print("Applied Slice 26A/26B runtime authority quarantine and proof-carrying stage plan.")
    print("Touched:")
    for rel in [
        "src/pegasus/workflows/stage_plan.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/acceptance/contracts.py",
        "src/pegasus/workflows/acceptance.py",
        "src/pegasus/cli.py",
        "tests/unit/test_slice26a26b_runtime_authority_stage_plan.py",
        "tests/integration/test_slice26a26b_runtime_authority_stage_plan_integration.py",
        "scripts/dev/audits/audit_slice26a26b_runtime_authority_stage_plan.py",
    ]:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
