from __future__ import annotations

import ast
import textwrap
import zipfile
from pathlib import Path

ROOT = Path.cwd()
PATCH = [
    "src/pegasus/output/reproducibility.py",
    "src/pegasus/output/race_bridge_attach.py",
]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    rel(path).write_text(content, encoding="utf-8", newline="\n")


def _replace_node_segment(text: str, node: ast.AST, replacement: str) -> str:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise RuntimeError("AST node lacks line-span metadata; Python 3.11+ expected.")
    lines = text.splitlines(keepends=True)
    start = int(node.lineno) - 1
    end = int(node.end_lineno)
    if start < 0 or end < start:
        raise RuntimeError("Invalid AST line span while patching.")
    replacement = textwrap.dedent(replacement).lstrip("\n").rstrip() + "\n"
    return "".join(lines[:start] + [replacement] + lines[end:])


def _find_top_assignment(tree: ast.Module, name: str) -> ast.Assign | ast.AnnAssign:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node
    raise RuntimeError(f"Could not find top-level assignment for {name}.")


def _find_top_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Could not find top-level function {name}.")


def preflight() -> None:
    required = [
        "pyproject.toml",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/output/race_bridge_attach.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/workflows/race_bridge.py",
        "src/pegasus/registries/race_bridge.py",
        "config/intents/alagoas_smoke_race_bridge.json",
        "config/registries/demographic/race_bridge_priors.yaml",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 4B repair v3 preflight failed; missing expected files: {missing}")

    repro = read("src/pegasus/output/reproducibility.py")
    tree = ast.parse(repro)
    assignment = _find_top_assignment(tree, "COMPILE_TELEMETRY_STAGES")
    value_node = assignment.value if isinstance(assignment, (ast.Assign, ast.AnnAssign)) else None
    try:
        stages = list(ast.literal_eval(value_node))
    except Exception as exc:
        raise RuntimeError("Could not parse COMPILE_TELEMETRY_STAGES as a Python literal tuple/list.") from exc
    if "geo_support" not in stages:
        raise RuntimeError(f"geo_support is absent from COMPILE_TELEMETRY_STAGES; found={stages}")

    attach = read("src/pegasus/output/race_bridge_attach.py")
    attach_tree = ast.parse(attach)
    _find_top_assignment(attach_tree, "RACE_BRIDGE_WARNING_IDS")
    _find_top_function(attach_tree, "_warning_rows")


def patch_reproducibility() -> None:
    path = "src/pegasus/output/reproducibility.py"
    text = read(path)
    tree = ast.parse(text)
    assignment = _find_top_assignment(tree, "COMPILE_TELEMETRY_STAGES")
    value_node = assignment.value if isinstance(assignment, (ast.Assign, ast.AnnAssign)) else None
    stages = list(ast.literal_eval(value_node))

    if "race_bridge" not in stages:
        insert_at = stages.index("geo_support") if "geo_support" in stages else len(stages)
        stages.insert(insert_at, "race_bridge")
        print("Patched RunTelemetry: inserted race_bridge into COMPILE_TELEMETRY_STAGES before geo_support.")
    else:
        print("RunTelemetry already declares race_bridge; stage list unchanged except formatting normalization.")

    replacement = "COMPILE_TELEMETRY_STAGES = (\n" + "".join(f'    "{stage}",\n' for stage in stages) + ")\n"
    write(path, _replace_node_segment(text, assignment, replacement))


def patch_race_bridge_warnings() -> None:
    path = "src/pegasus/output/race_bridge_attach.py"
    text = read(path)
    tree = ast.parse(text)

    ids_node = _find_top_assignment(tree, "RACE_BRIDGE_WARNING_IDS")
    ids_replacement = '''
    RACE_BRIDGE_WARNING_IDS = {
        "race_bridge_admin_axis_preserved",
        "race_bridge_bayesian_ecological_warning",
        "race_bridge_missing_preserved",
        "race_bridge_sensitivity_metadata",
        "race_bridge_posterior_sensitivity",
    }
    '''
    text = _replace_node_segment(text, ids_node, ids_replacement)

    tree = ast.parse(text)
    warning_rows_node = _find_top_function(tree, "_warning_rows")
    warning_rows_replacement = '''
    def _warning_rows(posterior: RaceBridgePosterior) -> list[dict[str, Any]]:
        return [
            {
                "warning_id": "race_bridge_admin_axis_preserved",
                "field_id": "run",
                "severity": "warning",
                "message": "Raw SIM administrative race/color counts are preserved as raw declaration-process counts; they are not overwritten by IBGE self-declared race.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_bayesian_ecological_warning",
                "field_id": "run",
                "severity": "downgrade",
                "message": "Posterior race counts are bridge-derived observer fields, not direct self-declared measurements; Bridge_R uses a fixed-C emission prior and sensitivity bounds.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_missing_preserved",
                "field_id": "SIMRaceBridgeMissingRaceObserver",
                "severity": "warning",
                "message": f"Missing/invalid administrative race is preserved as an observer count: n={posterior.missing_count}, share={posterior.missing_share:.6f}.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_sensitivity_metadata",
                "field_id": "run",
                "severity": "downgrade",
                "message": f"Bridge_R posterior fields carry sensitivity interval width={posterior.sensitivity_width:.6f}; dashboard safety remains downgraded unless calibration later justifies promotion.",
                "created_at": _now(),
                "inherited_from": None,
            },
        ]
    '''
    text = _replace_node_segment(text, warning_rows_node, warning_rows_replacement)
    write(path, text)
    print("Patched Race Bridge warnings: restored 4A-compatible raw-admin and Bayesian observer messages.")


def main() -> None:
    preflight()
    print("Slice 4B repair v3 preflight passed.")
    print("PATCH:")
    for p in PATCH:
        print(f"  {p}")
    patch_reproducibility()
    patch_race_bridge_warnings()
    print("Applied Slice 4B repair v3.")


if __name__ == "__main__":
    main()
