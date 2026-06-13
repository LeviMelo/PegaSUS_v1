from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

RUN_SUMMARY_FILES = (
    "RunConfig.json",
    "P_vector.json",
    "UserIntent.json",
    "ReproducibilityManifest.json",
)

REPORT_MARKER = "<!-- PegaSUS HSIC evidence report: read-only generated artifact -->"


@dataclass(frozen=True)
class HSICEvidenceItem:
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

    def as_json(self) -> dict[str, Any]:
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
            "warnings": list(self.warnings),
            "hsic_mode": self.hsic_mode,
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "dashboard_safe": True,
            "read_only": True,
        }

    def as_csv(self) -> dict[str, Any]:
        payload = self.as_json()
        payload["warnings"] = ";".join(self.warnings)
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


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


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cards_from_dashboard(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    cards = payload.get("cards")
    if not isinstance(cards, list):
        return []
    return [dict(card) for card in cards if isinstance(card, dict) and card.get("dashboard_safe", True)]


def _cards_from_ranking_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = _read_json(path)
    rows = payload.get("ranked_hypotheses")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, dict) and row.get("dashboard_safe", True)]
    return []


def _cards_from_ranking_parquet(path: Path) -> list[dict[str, Any]]:
    rows = _read_parquet_rows(path)
    return [dict(row) for row in rows if row.get("dashboard_safe", True)]


def _metrics_value(row: dict[str, Any], key: str) -> Any:
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics.get(key)
    return row.get(key)


def _evidence_items(cards: Iterable[dict[str, Any]]) -> list[HSICEvidenceItem]:
    items: list[HSICEvidenceItem] = []
    for idx, row in enumerate(cards, start=1):
        hypothesis_id = str(row.get("hypothesis_id") or f"hsic_report_item_{idx}")
        warnings = _warnings(row.get("warnings"))
        state = str(row.get("state") or "exploratory")
        evidence_tier = str(row.get("evidence_tier") or "exploratory_descriptive")
        decision = str(row.get("decision") or "show_as_exploratory")
        items.append(HSICEvidenceItem(
            rank=int(row.get("rank") or idx),
            hypothesis_id=hypothesis_id,
            residual_field_id=row.get("residual_field_id"),
            covariate_field_id=row.get("covariate_field_id"),
            statistic=_number(_metrics_value(row, "statistic")),
            p_value=_number(_metrics_value(row, "p_value")),
            q_value=_number(_metrics_value(row, "q_value")),
            n_eff=_number(_metrics_value(row, "n_eff")),
            state=state,
            evidence_tier=evidence_tier,
            decision=decision,
            warnings=warnings,
            hsic_mode=row.get("hsic_mode"),
            residual_mode=row.get("residual_mode"),
            fold_scheme=row.get("fold_scheme"),
        ))
    return sorted(items, key=lambda item: (item.rank, item.hypothesis_id))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_hsic_markdown_report(items: list[HSICEvidenceItem], *, generated_at: str, source: str | None = None) -> str:
    lines: list[str] = [
        REPORT_MARKER,
        "# PegaSUS HSIC residual evidence report",
        "",
        f"Generated at: {generated_at}",
        "",
        "This report is a read-only descriptive export. It does not refit PIRS models, rerun HSIC, or mutate the output bundle.",
        "",
    ]
    if source:
        lines += [f"Source ranking manifest: `{source}`", ""]
    lines += [
        f"Evidence items: {len(items)}",
        "",
        "| Rank | Hypothesis | Covariate | State | Evidence tier | q | p | statistic | n_eff | Decision |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for item in items:
        lines.append(
            "| "
            + " | ".join([
                str(item.rank),
                item.hypothesis_id,
                item.covariate_field_id or "",
                item.state,
                item.evidence_tier,
                _fmt(item.q_value),
                _fmt(item.p_value),
                _fmt(item.statistic),
                _fmt(item.n_eff),
                item.decision,
            ])
            + " |"
        )
    lines += ["", "## Warnings", ""]
    if not items:
        lines.append("No dashboard-safe HSIC evidence items were available.")
    else:
        for item in items:
            if item.warnings:
                lines.append(f"- {item.hypothesis_id}: {', '.join(item.warnings)}")
        if not any(item.warnings for item in items):
            lines.append("No warnings recorded for exported evidence items.")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, items: list[HSICEvidenceItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank", "hypothesis_id", "residual_field_id", "covariate_field_id", "statistic", "p_value",
        "q_value", "n_eff", "state", "evidence_tier", "decision", "warnings", "hsic_mode",
        "residual_mode", "fold_scheme", "dashboard_safe", "read_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for item in items:
            writer.writerow(item.as_csv())


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


def build_hsic_report_artifacts(
    *,
    run_dir: str | Path,
    ranking_manifest: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    tables = run_dir / "Tables"
    ranking_manifest_path = Path(ranking_manifest) if ranking_manifest is not None else tables / "hsic_residual_scan_ranking_manifest.json"
    if not ranking_manifest_path.exists():
        raise FileNotFoundError(f"HSIC ranking manifest not found: {ranking_manifest_path}")

    out_dir = Path(output_dir) if output_dir is not None else tables
    out_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = tables / "dashboard_hsic_cards.json"
    ranking_parquet_path = tables / "hsic_residual_scan_ranking.parquet"

    cards = _cards_from_dashboard(dashboard_path)
    source_kind = "dashboard_cards"
    if not cards:
        cards = _cards_from_ranking_manifest(ranking_manifest_path)
        source_kind = "ranking_manifest"
    if not cards:
        cards = _cards_from_ranking_parquet(ranking_parquet_path)
        source_kind = "ranking_parquet"

    items = _evidence_items(cards)
    generated_at = datetime.now(timezone.utc).isoformat()

    report_json_path = out_dir / "hsic_evidence_report.json"
    report_md_path = out_dir / "hsic_evidence_report.md"
    report_csv_path = out_dir / "hsic_evidence_report.csv"
    manifest_path = out_dir / "hsic_evidence_report_manifest.json"

    report_payload = {
        "schema_version": "1.0",
        "slice": "18C",
        "read_only": True,
        "generated_at": generated_at,
        "source_kind": source_kind,
        "source_ranking_manifest": str(ranking_manifest_path),
        "source_dashboard_cards": str(dashboard_path),
        "evidence_count": len(items),
        "items": [item.as_json() for item in items],
    }
    _write_json(report_json_path, report_payload)
    report_md_path.write_text(
        render_hsic_markdown_report(items, generated_at=generated_at, source=str(ranking_manifest_path)),
        encoding="utf-8",
    )
    _write_csv(report_csv_path, items)

    summary = {
        "gate": "hsic_report_gate",
        "schema_version": "1.0",
        "slice": "18C",
        "status": "exported" if items else "empty",
        "read_only": True,
        "evidence_count": len(items),
        "source_kind": source_kind,
        "top_hypothesis_id": items[0].hypothesis_id if items else None,
        "manifest_path": str(manifest_path),
        "report_json_path": str(report_json_path),
        "report_md_path": str(report_md_path),
        "report_csv_path": str(report_csv_path),
    }
    try:
        from pegasus.output.validate import validate_output_bundle
        validation = validate_output_bundle(run_dir=str(run_dir))
        summary["output_validation"] = {"ok": bool(validation.ok), "errors": list(validation.errors)}
    except Exception as exc:  # pragma: no cover
        summary["output_validation"] = {"ok": False, "errors": [str(exc)]}

    manifest = {
        **summary,
        "generated_at": generated_at,
        "source_ranking_manifest": str(ranking_manifest_path),
        "source_dashboard_cards": str(dashboard_path),
        "items": [item.as_json() for item in items],
    }
    _write_json(manifest_path, manifest)
    _attach_summary_to_run(run_dir, "hsic_report_gate", summary)
    return manifest


def inspect_hsic_report_manifest(path: str | Path) -> dict[str, Any]:
    payload = _read_json(Path(path))
    return {
        "schema_version": payload.get("schema_version"),
        "slice": payload.get("slice"),
        "status": payload.get("status"),
        "read_only": payload.get("read_only"),
        "evidence_count": payload.get("evidence_count"),
        "top_hypothesis_id": payload.get("top_hypothesis_id"),
        "report_json_path": payload.get("report_json_path"),
        "report_md_path": payload.get("report_md_path"),
        "report_csv_path": payload.get("report_csv_path"),
    }


__all__ = [
    "HSICEvidenceItem",
    "build_hsic_report_artifacts",
    "inspect_hsic_report_manifest",
    "render_hsic_markdown_report",
]
