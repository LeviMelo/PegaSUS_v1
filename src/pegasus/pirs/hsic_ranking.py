from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

RUN_SUMMARY_FILES = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)

STATE_PRIORITY = {
    "verified": 0,
    "supported": 1,
    "fragile": 2,
    "exploratory": 3,
    "blocked": 9,
    "error": 99,
}


@dataclass(frozen=True)
class RankedHSICCard:
    rank: int
    hypothesis_id: str
    residual_field_id: str | None
    covariate_field_id: str | None
    statistic: float | None
    p_value: float | None
    q_value: float | None
    n_eff: float | None
    state: str
    evidence_tier: str
    decision: str
    warnings: list[str]
    hsic_mode: str | None = None
    residual_mode: str | None = None
    fold_scheme: str | None = None
    dashboard_safe: bool = True

    def as_row(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "hypothesis_id": self.hypothesis_id,
            "residual_field_id": self.residual_field_id,
            "covariate_field_id": self.covariate_field_id,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "n_eff": self.n_eff,
            "state": self.state,
            "evidence_tier": self.evidence_tier,
            "decision": self.decision,
            "warnings": json.dumps(self.warnings, ensure_ascii=False, sort_keys=True),
            "hsic_mode": self.hsic_mode,
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "dashboard_safe": self.dashboard_safe,
        }

    def as_dashboard_card(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "hypothesis_id": self.hypothesis_id,
            "title": f"HSIC residual signal: {self.covariate_field_id or 'unknown covariate'}",
            "residual_field_id": self.residual_field_id,
            "covariate_field_id": self.covariate_field_id,
            "metrics": {
                "statistic": self.statistic,
                "p_value": self.p_value,
                "q_value": self.q_value,
                "n_eff": self.n_eff,
            },
            "state": self.state,
            "evidence_tier": self.evidence_tier,
            "decision": self.decision,
            "warnings": list(self.warnings),
            "dashboard_safe": self.dashboard_safe,
            "read_only": True,
        }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _write_parquet_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        pq.write_table(pa.table({
            "rank": pa.array([], type=pa.int64()),
            "hypothesis_id": pa.array([], type=pa.string()),
            "residual_field_id": pa.array([], type=pa.string()),
            "covariate_field_id": pa.array([], type=pa.string()),
            "statistic": pa.array([], type=pa.float64()),
            "p_value": pa.array([], type=pa.float64()),
            "q_value": pa.array([], type=pa.float64()),
            "n_eff": pa.array([], type=pa.float64()),
            "state": pa.array([], type=pa.string()),
            "evidence_tier": pa.array([], type=pa.string()),
            "decision": pa.array([], type=pa.string()),
            "warnings": pa.array([], type=pa.string()),
            "hsic_mode": pa.array([], type=pa.string()),
            "residual_mode": pa.array([], type=pa.string()),
            "fold_scheme": pa.array([], type=pa.string()),
            "dashboard_safe": pa.array([], type=pa.bool_()),
        }), path)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    return [str(value)]


def _state(row: dict[str, Any]) -> str:
    state = str(row.get("state") or "exploratory")
    return state if state else "exploratory"


def _evidence_tier(row: dict[str, Any]) -> str:
    state = _state(row)
    if state == "blocked":
        return "blocked"
    q = _float_or_none(row.get("q_value"))
    n = _float_or_none(row.get("n_eff"))
    if state == "verified" and q is not None and q <= 0.05 and n is not None and n >= 100:
        return "verified"
    if q is not None and q <= 0.10 and n is not None and n >= 20:
        return "supported_descriptive"
    if state == "fragile":
        return "fragile_descriptive"
    return "exploratory_descriptive"


def _decision(row: dict[str, Any]) -> str:
    tier = _evidence_tier(row)
    if tier == "verified":
        return "prioritize_for_review"
    if tier == "supported_descriptive":
        return "retain_for_review"
    if tier == "fragile_descriptive":
        return "show_as_fragile"
    if tier == "blocked":
        return "do_not_surface"
    return "show_as_exploratory"


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    state = _state(row)
    q = _float_or_none(row.get("q_value"))
    p = _float_or_none(row.get("p_value"))
    stat = _float_or_none(row.get("statistic"))
    n_eff = _float_or_none(row.get("n_eff"))
    return (
        STATE_PRIORITY.get(state, 50),
        1.0 if q is None else q,
        1.0 if p is None else p,
        -abs(stat or 0.0),
        -(n_eff or 0.0),
        str(row.get("hypothesis_id") or ""),
    )


def _scan_rows_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = manifest.get("scan_rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _merge_hypothesis_schema_values(row: dict[str, Any], hypothesis_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hypothesis_id = str(row.get("hypothesis_id") or "")
    hyp = hypothesis_by_id.get(hypothesis_id, {})
    merged = dict(hyp)
    merged.update({k: v for k, v in row.items() if v is not None})
    return merged


def rank_hsic_rows(rows: Iterable[dict[str, Any]], hypothesis_rows: Iterable[dict[str, Any]] = ()) -> list[RankedHSICCard]:
    hypothesis_by_id = {str(row.get("hypothesis_id")): dict(row) for row in hypothesis_rows if row.get("hypothesis_id")}
    merged_rows = [_merge_hypothesis_schema_values(dict(row), hypothesis_by_id) for row in rows]
    ranked: list[RankedHSICCard] = []
    for rank, row in enumerate(sorted(merged_rows, key=_sort_key), start=1):
        state = _state(row)
        warnings = _warnings(row.get("warnings"))
        dashboard_safe = state != "blocked" and "dashboard_unsafe" not in warnings
        ranked.append(RankedHSICCard(
            rank=rank,
            hypothesis_id=str(row.get("hypothesis_id") or f"hsic_unidentified_{rank}"),
            residual_field_id=row.get("residual_field_id"),
            covariate_field_id=row.get("covariate_field_id"),
            statistic=_float_or_none(row.get("statistic")),
            p_value=_float_or_none(row.get("p_value")),
            q_value=_float_or_none(row.get("q_value")),
            n_eff=_float_or_none(row.get("n_eff")),
            state=state,
            evidence_tier=_evidence_tier(row),
            decision=_decision(row),
            warnings=warnings,
            hsic_mode=row.get("hsic_mode"),
            residual_mode=row.get("residual_mode"),
            fold_scheme=row.get("fold_scheme"),
            dashboard_safe=dashboard_safe,
        ))
    return ranked


def _attach_summary_to_run(run_dir: Path, key: str, summary: dict[str, Any]) -> None:
    for name in RUN_SUMMARY_FILES:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except Exception:
            continue
        payload[key] = summary
        _write_json(path, payload)


def build_hsic_ranking_artifacts(
    *,
    run_dir: str | Path,
    scan_manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    tables = run_dir / "Tables"
    manifest_path = Path(scan_manifest) if scan_manifest is not None else tables / "hsic_residual_scan_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"HSIC scan manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)

    scores_path = tables / "hsic_residual_scan_scores.parquet"
    score_rows = _read_parquet_rows(scores_path)
    scan_rows = score_rows or _scan_rows_from_manifest(manifest)
    hypothesis_rows = _read_parquet_rows(run_dir / "Hypotheses.parquet")
    ranked = rank_hsic_rows(scan_rows, hypothesis_rows)

    ranking_path = Path(output) if output is not None else tables / "hsic_residual_scan_ranking.parquet"
    _write_parquet_rows(ranking_path, [card.as_row() for card in ranked])

    dashboard_path = tables / "dashboard_hsic_cards.json"
    dashboard_payload = {
        "schema_version": "1.0",
        "slice": "18B",
        "read_only": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_path": str(manifest_path),
        "ranking_path": str(ranking_path),
        "card_count": len(ranked),
        "cards": [card.as_dashboard_card() for card in ranked if card.dashboard_safe],
    }
    _write_json(dashboard_path, dashboard_payload)

    summary = {
        "gate": "hsic_ranking_gate",
        "schema_version": "1.0",
        "slice": "18B",
        "status": "ranked" if ranked else "no_scores",
        "rank_count": len(ranked),
        "dashboard_card_count": len(dashboard_payload["cards"]),
        "verified_count": sum(1 for card in ranked if card.evidence_tier == "verified"),
        "supported_count": sum(1 for card in ranked if card.evidence_tier == "supported_descriptive"),
        "fragile_count": sum(1 for card in ranked if card.evidence_tier == "fragile_descriptive"),
        "blocked_count": sum(1 for card in ranked if card.evidence_tier == "blocked"),
        "top_hypothesis_id": ranked[0].hypothesis_id if ranked else None,
        "manifest_path": str(tables / "hsic_residual_scan_ranking_manifest.json"),
        "ranking_path": str(ranking_path),
        "dashboard_path": str(dashboard_path),
        "read_only_dashboard": True,
    }

    try:
        from pegasus.output.validate import validate_output_bundle

        validation = validate_output_bundle(run_dir=str(run_dir))
        summary["output_validation"] = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    except Exception as exc:  # pragma: no cover - defensive telemetry only
        summary["output_validation"] = {"ok": False, "errors": [str(exc)]}

    ranking_manifest = {
        **summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hsic_scan_manifest": str(manifest_path),
        "source_score_path": str(scores_path),
        "ranked_hypotheses": [card.as_dashboard_card() for card in ranked],
    }
    _write_json(tables / "hsic_residual_scan_ranking_manifest.json", ranking_manifest)
    _attach_summary_to_run(run_dir, "hsic_ranking_gate", summary)
    return ranking_manifest


def inspect_hsic_ranking_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    return {
        "schema_version": payload.get("schema_version"),
        "slice": payload.get("slice"),
        "status": payload.get("status"),
        "rank_count": payload.get("rank_count"),
        "dashboard_card_count": payload.get("dashboard_card_count"),
        "top_hypothesis_id": payload.get("top_hypothesis_id"),
        "ranking_path": payload.get("ranking_path"),
        "dashboard_path": payload.get("dashboard_path"),
        "read_only_dashboard": payload.get("read_only_dashboard"),
    }
