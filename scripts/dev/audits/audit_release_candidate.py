from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pegasus.acceptance.contracts import evaluate_level3_acceptance, required_first_class_key_paths
from pegasus.core.config import validate_config_tree
from pegasus.dashboard.read_only import bundle_overview
from pegasus.datasus.cache import DatasusCache
from pegasus.datasus.manifests import build_datasus_request_manifest
from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
from pegasus.output.validate import validate_output_bundle
from pegasus.pirs.hsic import run_hsic_scan
from pegasus.registries.validators import validate_registry_tree
from pegasus.workflows.compile import run_compile


ROOT = Path(__file__).resolve().parents[3]


def _boundary_status() -> tuple[bool, str]:
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts/dev/audits/audit_boundary_closure.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        payload = json.loads(process.stdout)
        status = str(payload.get("status", "failed"))
    except json.JSONDecodeError:
        status = "invalid_output"
    return process.returncode == 0 and status == "passed", status


def _hsic_acceptance() -> dict[str, bool]:
    x = [float(index) / 30.0 for index in range(120)]
    residuals = [value * value + 0.01 * (index % 3) for index, value in enumerate(x)]
    null = list(x)
    random.Random(91).shuffle(null)
    kwargs: dict[str, Any] = {
        "outcome_residual_field_id": "release_residual",
        "residuals": residuals,
        "support_intersection": {"n_eff": 120},
        "budget": "standard",
        "null_strategy": "unrestricted_permutation",
        "fdr_method": "BY",
        "permutations": 19,
        "seed": 91,
    }
    signal = run_hsic_scan(covariate_field_id="signal", covariate=x, **kwargs)
    null_result = run_hsic_scan(covariate_field_id="null", covariate=null, **kwargs)
    repeat = run_hsic_scan(covariate_field_id="signal", covariate=x, **kwargs)
    return {
        "deterministic": signal.statistic == repeat.statistic and signal.p_value == repeat.p_value,
        "signal_above_null": bool(
            signal.statistic is not None
            and null_result.statistic is not None
            and signal.statistic > null_result.statistic
        ),
        "null_diagnostics": signal.approximation_diagnostics.get("permutations_executed") == 19,
        "compute_boundary": "compute_plan" in signal.approximation_diagnostics,
    }


def run_audit(work_root: str | Path | None = None) -> dict[str, Any]:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if work_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="pegasus_release_")
        root = Path(temporary.name)
    else:
        root = Path(work_root)
        root.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = root / "fixture_run"
        result = run_compile(
            intent_path=ROOT / "config/intents/alagoas_smoke.json",
            run_dir=run_dir,
            data_root=root / "data",
        )
        validation = validate_output_bundle(run_dir=str(run_dir))
        level3 = evaluate_level3_acceptance(run_dir)
        dashboard = bundle_overview(run_dir=run_dir, limit=2)
        request = build_datasus_request_manifest(
            system="SIM-DO", uf="AL", year_start=2022, year_end=2022,
            config={"rscript_path": "__pegasus_release_missing_Rscript__"}, data_root=root / "datasus",
        )
        unavailable = fetch_datasus_chunk(
            request,
            config=DatasusConfig(rscript_path="__pegasus_release_missing_Rscript__", r_timeout_seconds=1, heartbeat_timeout_seconds=1),
            cache=DatasusCache(root / "cache"),
            timeout_seconds=1,
            heartbeat_timeout_seconds=1,
        )
        try:
            unavailable_payload = json.loads(unavailable.error_message or "{}")
        except json.JSONDecodeError:
            unavailable_payload = {}
        boundary_ok, boundary_status = _boundary_status()
        hsic = _hsic_acceptance()
        checks = {
            "config_valid": not validate_config_tree(ROOT),
            "registries_valid": not validate_registry_tree(ROOT / "config/registries"),
            "fixture_compile_success": bool(result["validation"].ok),
            "exact_17_key_bundle": tuple(sorted(path.name for path in run_dir.iterdir())) == required_first_class_key_paths(),
            "mandatory_validator": bool(validation.ok),
            "fixture_classification": level3.status == "fixture_validated" and not level3.production_candidate,
            "source_reality_explicit": dashboard["source_reality"].get("compile_source_mode") == "fixture_only",
            "datasus_unavailable_explicit": unavailable_payload.get("source") == "DATASUS" and unavailable_payload.get("status") == "unavailable",
            "efg_nonempty": dashboard["efg"]["fields"]["row_count"] > 0 and dashboard["efg"]["edges"]["row_count"] > 0,
            "dashboard_read_only": dashboard["read_only"] is True and dashboard["validation_ok"] is True,
            "boundary_closure": boundary_ok,
            "hsic_acceptance": all(hsic.values()),
        }
        return {
            "audit": "release_candidate",
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "classification": level3.status,
            "boundary_status": boundary_status,
            "hsic": hsic,
            "error_count": sum(not value for value in checks.values()),
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    payload = run_audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
