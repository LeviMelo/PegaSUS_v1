from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path.cwd()

OLD_16F = ROOT / "tests/integration/test_slice16f_pirs_planning_pipeline.py"
NEW_16F = ROOT / "tests/integration/test_slice16f_pirs_planning_pipeline_integration.py"
OLD_18A = ROOT / "tests/integration/test_slice18a_hsic_residual_scan_integration.py"
NEW_18A = ROOT / "tests/integration/test_slice18a_hsic_residual_scan_integration_integration.py"
HSIC_RUN = ROOT / "src/pegasus/pirs/hsic_run.py"
AUDIT = ROOT / "scripts/dev/audits/audit_slice18a_hsic_residual_scan_integration.py"
UNIT_TEST = ROOT / "tests/unit/test_slice18a_hsic_residual_scan_integration.py"


def fail(message: str) -> None:
    raise SystemExit(f"[repair18a] {message}")


def read(path: Path) -> str:
    if not path.exists():
        fail(f"missing required file: {path}")
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"expected one occurrence for {label}; found {count}")
    return text.replace(old, new, 1)


def cleanup_test_filenames() -> None:
    # Keep only the collision-free Slice 16F integration filename in the working tree.
    # If OLD_16F is tracked, git will record this as a deletion after `git add -u`.
    if NEW_16F.exists() and OLD_16F.exists():
        OLD_16F.unlink()

    # Rename 18A integration test to collision-free basename if the old file still exists.
    if OLD_18A.exists():
        if NEW_18A.exists():
            OLD_18A.unlink()
        else:
            OLD_18A.rename(NEW_18A)


def patch_hsic_run() -> None:
    text = read(HSIC_RUN)

    old_hypothesis = '''def _hypothesis_row(row: Mapping[str, Any]) -> dict[str, Any]:
    warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
    return {
        "hypothesis_id": row.get("hypothesis_id"),
        "outcome_field_id": row.get("residual_field_id"),
        "exposure_field_id": row.get("covariate_field_id"),
        "residual_field_id": row.get("residual_field_id"),
        "method": "HSIC",
        "statistic": row.get("statistic"),
        "p_value": row.get("p_value"),
        "q_value": row.get("q_value"),
        "state": row.get("state", "fragile"),
        "warnings": _compact(warnings),
        "metadata_json": _compact(dict(row)),
        "created_at": row.get("created_at") or _now(),
    }
'''
    new_hypothesis = '''def _hypothesis_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Render a scan row into the canonical Hypotheses.parquet schema.

    The output bundle Hypotheses schema is fixed and does not contain generic
    Slice 18A convenience columns such as ``method``, ``metadata_json``, or
    ``created_at``. Those details must be carried through the canonical HSIC
    columns and approximation_diagnostics_json.
    """
    warnings = row.get("warnings") if isinstance(row.get("warnings"), list) else []
    diagnostics = {
        "model_id": row.get("model_id"),
        "budget": row.get("budget"),
        "kernel": row.get("kernel"),
        "covariate_column": row.get("covariate_column"),
        "permutations": row.get("permutations"),
        "seed": row.get("seed"),
        "created_at": row.get("created_at") or _now(),
    }
    return {
        "hypothesis_id": row.get("hypothesis_id"),
        "outcome_field_id": row.get("outcome_field_id") or row.get("residual_field_id"),
        "covariate_field_id": row.get("covariate_field_id"),
        "residual_field_id": row.get("residual_field_id"),
        "statistic": row.get("statistic"),
        "p_value": row.get("p_value"),
        "q_value": row.get("q_value"),
        "hsic_mode": row.get("hsic_mode"),
        "residual_mode": row.get("residual_mode"),
        "fold_scheme": row.get("fold_scheme"),
        "bootstrap_count": row.get("bootstrap_count"),
        "residual_uncertainty": row.get("residual_uncertainty"),
        "null_strategy": row.get("null_strategy"),
        "fdr_method": row.get("fdr_method"),
        "n_eff": row.get("n_eff"),
        "state": row.get("state", "fragile"),
        "warnings": _compact(warnings),
        "approximation_diagnostics_json": _compact(diagnostics),
    }
'''
    text = replace_once(text, old_hypothesis, new_hypothesis, "canonical _hypothesis_row")

    old_q = '''    q_values = _fdr_bh([row.get("p_value") for row in scan_rows])
    for row, q_value in zip(scan_rows, q_values):
        row["q_value"] = q_value

    _write_rows(root / DEFAULT_HSIC_SCORES, scan_rows)
'''
    new_q = '''    q_values = _fdr_bh([row.get("p_value") for row in scan_rows])
    outcome_specs = [spec for spec in _field_specs(design_manifest) if spec.get("role") == "outcome"]
    outcome_field_id = (
        model_manifest.get("outcome_field_id")
        or (outcome_specs[0].get("field_id") if outcome_specs else None)
        or str(residual_field_id)
    )
    residual_mode = model_manifest.get("residual_mode") or design_manifest.get("residual_mode") or "in_sample"
    fold_scheme = model_manifest.get("fold_scheme") or design_manifest.get("fold_scheme") or "none_in_sample_fast_budget"
    for row, q_value in zip(scan_rows, q_values):
        row["q_value"] = q_value
        row.setdefault("outcome_field_id", outcome_field_id)
        row.setdefault("residual_mode", residual_mode)
        row.setdefault("fold_scheme", fold_scheme)
        row.setdefault("bootstrap_count", None)
        row.setdefault("residual_uncertainty", "in_sample_descriptive")
        row.setdefault("null_strategy", "permutation_linear_centered")
        row.setdefault("fdr_method", "BH")

    _write_rows(root / DEFAULT_HSIC_SCORES, scan_rows)
'''
    text = replace_once(text, old_q, new_q, "scan-row canonical hypothesis metadata")
    write(HSIC_RUN, text)


def patch_integration_test() -> None:
    path = NEW_18A if NEW_18A.exists() else OLD_18A
    text = read(path)
    old = '    assert {row["method"] for row in hypotheses} == {"HSIC"}\n'
    new = '''    assert {row["hsic_mode"] for row in hypotheses} == {"exact_linear"}
    assert {row["covariate_field_id"] for row in hypotheses} == {"capacity_a", "capacity_b"}
    assert {row["residual_field_id"] for row in hypotheses} == {gate["residual_field_id"]}
    assert {row["fdr_method"] for row in hypotheses} == {"BH"}
    assert all(row["n_eff"] == 4.0 for row in hypotheses)
    assert all(row["approximation_diagnostics_json"] for row in hypotheses)
'''
    text = replace_once(text, old, new, "integration method assertion")
    write(path, text)


def patch_audit() -> None:
    text = read(AUDIT)
    old = '''        hypotheses = pq.read_table(run_dir / "Hypotheses.parquet").to_pylist()
        if not hypotheses or hypotheses[0].get("method") != "HSIC":
            errors.append("HSIC hypothesis rows missing")
'''
    new = '''        hypotheses = pq.read_table(run_dir / "Hypotheses.parquet").to_pylist()
        if not hypotheses:
            errors.append("HSIC hypothesis rows missing")
        else:
            row = hypotheses[0]
            required = ("hypothesis_id", "covariate_field_id", "residual_field_id", "hsic_mode", "fdr_method", "n_eff", "approximation_diagnostics_json")
            missing = [name for name in required if row.get(name) in (None, "")]
            if missing:
                errors.append("HSIC hypothesis row missing canonical schema values: " + ", ".join(missing))
            if row.get("hsic_mode") not in {"exact_linear", "disabled"}:
                errors.append("HSIC hypothesis row has unexpected hsic_mode")
'''
    text = replace_once(text, old, new, "audit method assertion")
    write(AUDIT, text)


def self_validate() -> None:
    for path in (HSIC_RUN, AUDIT, UNIT_TEST, NEW_18A):
        if path.exists():
            py_compile.compile(str(path), doraise=True)
        else:
            fail(f"post-patch file missing: {path}")


def main() -> None:
    cleanup_test_filenames()
    patch_hsic_run()
    patch_integration_test()
    patch_audit()
    self_validate()
    print("Repair Slice 18A v1 applied: canonical HSIC Hypotheses schema, collision-free tests, stale 16F file cleanup.")
    print("Touched/affected files:")
    for path in (HSIC_RUN, AUDIT, NEW_18A, OLD_16F):
        print(f"  - {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
