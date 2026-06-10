import json
import shutil
from pathlib import Path

import polars as pl

from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.compile import run_compile


def _compile_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "compile_run"
    result = run_compile(
        intent_path="config/intents/alagoas_smoke.json",
        run_dir=run_dir,
        data_root=tmp_path / "data",
    )
    assert result["status"] == "success"
    assert validate_output_bundle(run_dir=str(run_dir)).ok
    return run_dir


def _copy_run(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_slice2e_validator_accepts_compile_smoke_bundle(tmp_path: Path):
    run = _compile_run(tmp_path)
    result = validate_output_bundle(run_dir=str(run))
    assert result.ok, result.errors


def test_slice2e_validator_rejects_extra_first_class_artifact(tmp_path: Path):
    run = _copy_run(_compile_run(tmp_path), tmp_path / "with_extra")
    (run / "leak.txt").write_text("not a first-class bundle key", encoding="utf-8")

    result = validate_output_bundle(run_dir=str(run))

    assert not result.ok
    assert any("extra first-class artifact" in error and "leak.txt" in error for error in result.errors)


def test_slice2e_validator_rejects_missing_q_tensor_coverage(tmp_path: Path):
    run = _copy_run(_compile_run(tmp_path), tmp_path / "missing_q")
    q_path = run / "Q_tensor.parquet"
    q = pl.read_parquet(q_path)
    v = pl.read_parquet(run / "V_fields.parquet")
    removed_field = v["field_id"].to_list()[0]
    q = q.filter(pl.col("field_id") != removed_field)
    q.write_parquet(q_path)

    result = validate_output_bundle(run_dir=str(run))

    assert not result.ok
    assert any("Q_tensor does not cover all V_fields" in error for error in result.errors)


def test_slice2e_validator_rejects_invalid_compile_telemetry(tmp_path: Path):
    run = _copy_run(_compile_run(tmp_path), tmp_path / "bad_telemetry")
    manifest_path = run / "ReproducibilityManifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["telemetry"]["stage_status"]["pirs_model"] = "not_terminal"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    result = validate_output_bundle(run_dir=str(run))

    assert not result.ok
    assert any("invalid telemetry stage status" in error for error in result.errors)


def test_slice2e_validator_rejects_missing_compile_source_hashes(tmp_path: Path):
    run = _copy_run(_compile_run(tmp_path), tmp_path / "missing_hashes")
    manifest_path = run / "ReproducibilityManifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_hashes"] = {}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    result = validate_output_bundle(run_dir=str(run))

    assert not result.ok
    assert any("source_hashes" in error for error in result.errors)
