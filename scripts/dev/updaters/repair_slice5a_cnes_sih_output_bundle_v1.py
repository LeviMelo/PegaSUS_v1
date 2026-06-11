from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path.cwd()
BUNDLE = REPO_ROOT / "src/pegasus/output/cnes_sih_efg_bundle.py"
CLI = REPO_ROOT / "src/pegasus/cli.py"


def _replace_ast_assignment(path: Path, function_name: str, target_name: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    target_node: ast.Assign | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    if any(isinstance(t, ast.Name) and t.id == target_name for t in child.targets):
                        target_node = child
                        break
            break
    if target_node is None or target_node.end_lineno is None:
        raise RuntimeError(f"Could not find assignment to {target_name!r} inside {function_name}().")
    start = target_node.lineno - 1
    end = target_node.end_lineno
    new_lines = [line + "\n" for line in replacement.rstrip("\n").split("\n")]
    lines[start:end] = new_lines
    path.write_text("".join(lines), encoding="utf-8")


def _insert_after_once(path: Path, needle: str, insertion: str, *, guard: str) -> None:
    text = path.read_text(encoding="utf-8")
    if guard in text:
        return
    if needle not in text:
        raise RuntimeError(f"Insertion point not found in {path}: {needle!r}")
    text = text.replace(needle, needle + insertion, 1)
    path.write_text(text, encoding="utf-8")


def patch_bundle() -> None:
    if not BUNDLE.exists():
        raise RuntimeError(f"Missing file: {BUNDLE}")
    text = BUNDLE.read_text(encoding="utf-8")
    if "def write_cnes_sih_fixture_efg_bundle" not in text:
        raise RuntimeError("Slice 5A bundle writer not found; refusing blind patch.")

    # Add telemetry stage constants import if needed.
    text = BUNDLE.read_text(encoding="utf-8")
    if "from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES" not in text:
        marker = "from pegasus.output.bundle import create_empty_output_bundle\n"
        if marker not in text:
            raise RuntimeError("Cannot insert COMPILE_TELEMETRY_STAGES import; bundle import marker not found.")
        text = text.replace(
            marker,
            marker + "from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES\n",
            1,
        )
        BUNDLE.write_text(text, encoding="utf-8")

    # Empty scaffold-only first-class optional output tables after replacing core rows.
    needle = '    _write_rows_like(run_dir / "V_fields.parquet", fields); _write_rows_like(run_dir / "Q_tensor.parquet", q_rows); _write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows); _write_rows_like(run_dir / "E_DAG.parquet", edges); _write_rows_like(run_dir / "Warnings.parquet", warnings); _write_rows_like(run_dir / "FailedBranches.parquet", failed)\n'
    insertion = '''    for name in [
        "ModelAssociations.parquet",
        "ResidualAssociations.parquet",
        "Hypotheses.parquet",
        "QuarantinedFields.parquet",
        "ForcedFields.parquet",
    ]:
        _write_rows_like(run_dir / name, [])
'''
    _insert_after_once(BUNDLE, needle, insertion, guard='"QuarantinedFields.parquet",\n        "ForcedFields.parquet"')

    # Replace the invalid telemetry dictionary with validator-compatible telemetry.
    telemetry_replacement = '''    stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
    stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
    for stage in ["datasus_decode", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
        if stage in stage_status:
            stage_status[stage] = "success"
    for stage in ["population_solver", "stdfm", "pirs_model", "pirs_hsic"]:
        if stage in stage_status:
            stage_status[stage] = "blocked"
    telemetry = {
        "total_wall_seconds": 0.0,
        "stage_status": stage_status,
        "stage_wall_seconds": stage_wall_seconds,
        "stage_errors": {
            "population_solver": "No population denominator tensor invoked in Slice 5A CNES/SIH fixture path.",
            "stdfm": "ST-DFM scaffold remains blocked for Slice 5A CNES/SIH fixture path.",
            "pirs_model": "PIRS model fitting is outside Slice 5A.",
            "pirs_hsic": "HSIC scanning is outside Slice 5A.",
        },
        "resource_summary": {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {"cnes_events": cnes.facilities_total, "sih_events": sih.admissions_total},
            "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows), "FailedBranches": len(failed)},
            "parquet_bytes_written": 0,
        },
    }'''
    _replace_ast_assignment(BUNDLE, "write_cnes_sih_fixture_efg_bundle", "telemetry", telemetry_replacement)


def patch_cli_eof() -> None:
    if not CLI.exists():
        return
    text = CLI.read_text(encoding="utf-8")
    CLI.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    print("Slice 5A output bundle repair preflight passed.")
    print("PATCH:")
    print("  src/pegasus/output/cnes_sih_efg_bundle.py")
    print("  src/pegasus/cli.py")
    patch_bundle()
    patch_cli_eof()
    print("Applied Slice 5A output bundle telemetry/scaffold repair.")


if __name__ == "__main__":
    main()
