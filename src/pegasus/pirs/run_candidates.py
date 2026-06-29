
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from pegasus.pirs.schemas import FieldCandidate


class PIRSCandidateGateError(ValueError):
    """Raised when PIRS run-candidate extraction cannot inspect a run bundle."""


@dataclass(frozen=True)
class PIRSRunCandidateGateResult:
    run_dir: str
    candidate_count: int
    rejected_count: int
    candidates: tuple[FieldCandidate, ...]
    rejected: tuple[dict[str, Any], ...]

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "gate": "pirs_candidate_gate",
            "run_dir": self.run_dir,
            "candidate_count": self.candidate_count,
            "rejected_count": self.rejected_count,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "rejected": list(self.rejected),
            "model_boundary": {
                "metadata_only_fields_model_eligible": False,
                "quarantined_descriptive_fields_model_eligible": False,
                "dashboard_unsafe_fields_model_eligible": False,
                "unmaterialized_candidate_q_tensor_fields_model_eligible": False,
            },
        }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(payload), encoding="utf-8")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _parse_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return default
        return parsed
    return default


def _as_list(value: Any) -> list[Any]:
    parsed = _parse_json_cell(value, [])
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    if parsed in (None, ""):
        return []
    return [parsed]


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _parse_json_cell(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy_false(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"false", "0", "no", "n", "blocked"}


def _role_for_field(field: dict[str, Any]) -> Literal["outcome", "covariate", "offset"]:
    roles = {str(value) for value in _as_list(field.get("role"))}
    name = str(field.get("name") or field.get("field_id") or "").lower()
    unit = str(field.get("unit") or "").lower()
    kind = str(field.get("kind") or "")
    carrier = str(field.get("carrier") or "").lower()
    if "offset" in roles or "denominator" in roles or carrier == "population" or "population" in name or unit in {"person", "persons", "population"}:
        return "offset"
    # Rates/densities are the modelled outcomes; explicit outcome-roled fields too.
    # Counts, capacity, costs, context and observer measures serve as covariates —
    # otherwise every measure becomes an "outcome" and PIRS has nothing to regress
    # against (0 covariates → blocked design).
    if kind == "intensive_density" or "outcome" in roles:
        return "outcome"
    return "covariate"


def _candidate_utility(field: dict[str, Any], q: dict[str, Any]) -> float:
    n_eff = _safe_float(q.get("n_eff"), 0.0) or 0.0
    missingness = _safe_float(q.get("missingness"), 0.0) or 0.0
    provenance_risk = _safe_float(q.get("provenance_risk"), 0.0) or 0.0
    utility = n_eff * max(0.0, 1.0 - min(1.0, missingness)) * max(0.0, 1.0 - min(1.0, provenance_risk))
    if str(field.get("kind")) == "intensive_density":
        utility *= 1.05
    return float(utility)


def pirs_candidate_rejection_reason(field: dict[str, Any], q: dict[str, Any] | None) -> str | None:
    field_id = str(field.get("field_id") or "")
    state = str(field.get("state") or "")
    materialization_state = str(field.get("materialization_state") or "")
    warnings = {str(value) for value in _as_list(field.get("warnings"))}
    roles = {str(value) for value in _as_list(field.get("role"))}
    if not field_id:
        return "missing_field_id"
    if materialization_state == "metadata_only":
        return "metadata_only_field_not_model_eligible"
    if "geography_axis" in roles or "time_axis_candidate" in roles:
        return "raw_axis_field_not_model_covariate"
    if "source_field" in roles and "covariate" not in roles and "outcome" not in roles:
        return "raw_source_field_not_model_covariate"
    # MSD §3.13: model eligibility is governed by the Q-state class, NOT by
    # dashboard safety. verified/fragile/forced_fragile fields are model-eligible
    # even when dashboard_safe is False/"warning"; only illegal/blocked/no-children
    # classes are excluded. (The previous dashboard_safe rejection wrongly killed
    # every real epidemiological field, leaving PIRS with no outcome.)
    model_ineligible = {"illegal_excluded", "blocked", "quarantined_nochildren"}
    if state in model_ineligible:
        return f"q_state_{state}_not_model_eligible"
    if q is None:
        return "missing_q_tensor_row"
    q_state = str(q.get("state") or state)
    if q_state in model_ineligible:
        return f"q_state_{q_state}_not_model_eligible"
    q_warnings = {str(value) for value in _as_list(q.get("warnings"))}
    if "q_tensor_unmaterialized_candidate_no_numerical_tensor" in q_warnings or "efg_promotion_metadata_only" in q_warnings:
        return "unmaterialized_candidate_q_tensor_not_model_eligible"
    n_eff = _safe_float(q.get("n_eff"), 0.0) or 0.0
    if n_eff <= 0.0:
        return "nonpositive_n_eff_not_model_eligible"
    missingness = _safe_float(q.get("missingness"), 0.0)
    if missingness is not None and missingness >= 1.0:
        return "complete_missingness_not_model_eligible"
    if "efg_promotion_metadata_only" in warnings:
        return "metadata_only_warning_not_model_eligible"
    return None


def pirs_candidate_from_rows(field: dict[str, Any], q: dict[str, Any]) -> FieldCandidate:
    return FieldCandidate(
        field_id=str(field["field_id"]),
        role=_role_for_field(field),
        utility=_candidate_utility(field, q),
        q_state=str(q.get("state") or field.get("state") or "verified"),  # type: ignore[arg-type]
        carrier=str(field.get("carrier") or "unknown"),
        unit=str(field.get("unit") or "unknown"),
        support=_as_dict(field.get("support_json") or field.get("support")),
        variance=_safe_float(q.get("cv"), None),
        warnings=tuple(str(value) for value in sorted(set(_as_list(field.get("warnings")) + _as_list(q.get("warnings"))))),
        provenance=tuple(str(value) for value in _as_list(field.get("provenance"))),
    )


def build_pirs_candidates_from_run(run_dir: str | Path) -> PIRSRunCandidateGateResult:
    root = Path(run_dir)
    v_rows = _read_rows(root / "V_fields.parquet")
    q_rows = _read_rows(root / "Q_tensor.parquet")
    q_by_id = {str(row.get("field_id")): row for row in q_rows if row.get("field_id") is not None}
    candidates: list[FieldCandidate] = []
    rejected: list[dict[str, Any]] = []
    for field in v_rows:
        field_id = str(field.get("field_id") or "")
        q = q_by_id.get(field_id)
        reason = pirs_candidate_rejection_reason(field, q)
        if reason is not None:
            rejected.append({
                "field_id": field_id,
                "reason": reason,
                "state": field.get("state"),
                "materialization_state": field.get("materialization_state"),
                "dashboard_safe": field.get("dashboard_safe"),
            })
            continue
        assert q is not None
        candidates.append(pirs_candidate_from_rows(field, q))
    return PIRSRunCandidateGateResult(
        run_dir=str(root),
        candidate_count=len(candidates),
        rejected_count=len(rejected),
        candidates=tuple(candidates),
        rejected=tuple(rejected),
    )


def write_pirs_candidate_manifest(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    result = build_pirs_candidates_from_run(root)
    output_path = Path(output) if output is not None else root / "Tables" / "pirs_field_candidates.json"
    manifest = result.as_manifest()
    manifest["manifest_path"] = str(output_path)
    _write_json(output_path, manifest)
    return manifest


def attach_pirs_candidate_gate_to_run(
    *,
    run_dir: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = write_pirs_candidate_manifest(run_dir=root, output=output)
    summary = {
        "schema_version": "1.0",
        "gate": "pirs_candidate_gate",
        "status": "evaluated",
        "candidate_count": manifest["candidate_count"],
        "rejected_count": manifest["rejected_count"],
        "manifest_path": manifest["manifest_path"],
        "metadata_only_fields_model_eligible": False,
        "quarantined_descriptive_fields_model_eligible": False,
    }
    for rel in ("RunConfig.json", "P_vector.json", "UserIntent.json", "ReproducibilityManifest.json"):
        path = root / rel
        if not path.exists():
            continue
        payload = _load_json(path)
        payload["pirs_candidate_gate"] = summary
        _write_json(path, payload)
    return summary
