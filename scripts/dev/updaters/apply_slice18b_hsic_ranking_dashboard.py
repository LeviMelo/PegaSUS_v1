from __future__ import annotations

import re
from pathlib import Path

ROOT = Path.cwd()


def write(path: str, text: str) -> None:
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")


def patch_cli() -> None:
    path = ROOT / "src" / "pegasus" / "cli.py"
    if not path.exists():
        raise SystemExit("Missing src/pegasus/cli.py")
    text = path.read_text(encoding="utf-8")
    if "pirs_app" not in text:
        raise SystemExit("src/pegasus/cli.py does not expose pirs_app; cannot attach Slice 18B commands safely")
    if "import json" not in text:
        text = "import json\n" + text
    if "from pathlib import Path" not in text:
        text = "from pathlib import Path\n" + text

    marker_start = "# BEGIN SLICE18B HSIC RANKING DASHBOARD CLI"
    marker_end = "# END SLICE18B HSIC RANKING DASHBOARD CLI"
    pattern = re.compile(rf"\n?{re.escape(marker_start)}.*?{re.escape(marker_end)}\n?", re.S)
    text = pattern.sub("\n", text)

    block = r'''
# BEGIN SLICE18B HSIC RANKING DASHBOARD CLI
@pirs_app.command("rank-hsic-scan")
def pirs_rank_hsic_scan(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True),
    scan_manifest: Path | None = typer.Option(None, "--scan-manifest", exists=False, file_okay=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", exists=False, file_okay=True, dir_okay=False),
) -> None:
    """Rank HSIC residual-scan results and write read-only dashboard cards."""
    from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan

    payload = run_rank_hsic_residual_scan(
        run_dir=run_dir,
        scan_manifest=scan_manifest,
        output=output,
    )
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("inspect-hsic-ranking")
def pirs_inspect_hsic_ranking(
    manifest: Path = typer.Option(..., "--manifest", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Inspect an existing HSIC ranking manifest without mutating a run."""
    from pegasus.workflows.hsic_rank import inspect_hsic_ranking_manifest

    payload = inspect_hsic_ranking_manifest(manifest)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))


@pirs_app.command("inspect-hsic-dashboard")
def pirs_inspect_hsic_dashboard(
    dashboard: Path = typer.Option(..., "--dashboard", exists=True, file_okay=True, dir_okay=False),
) -> None:
    """Inspect read-only HSIC dashboard cards without mutating a run."""
    from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards

    payload = inspect_hsic_dashboard_cards(dashboard)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, default=str))
# END SLICE18B HSIC RANKING DASHBOARD CLI
'''
    text = text.rstrip() + "\n\n" + block.strip() + "\n"
    path.write_text(text, encoding="utf-8")


HSIC_RANKING = r'''
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
    if state == "verified":
        return "verified"
    if state == "fragile":
        return "fragile_descriptive"
    if state == "blocked":
        return "blocked"
    q = _float_or_none(row.get("q_value"))
    n = _float_or_none(row.get("n_eff"))
    if q is not None and q <= 0.10 and n is not None and n >= 20:
        return "supported_descriptive"
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
'''

HSIC_WORKFLOW = r'''
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.pirs.hsic_ranking import build_hsic_ranking_artifacts, inspect_hsic_ranking_manifest


def run_rank_hsic_residual_scan(
    *,
    run_dir: str | Path,
    scan_manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_ranking_artifacts(run_dir=run_dir, scan_manifest=scan_manifest, output=output)


def run_attach_hsic_ranking_to_run(
    *,
    run_dir: str | Path,
    scan_manifest: str | Path | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    return build_hsic_ranking_artifacts(run_dir=run_dir, scan_manifest=scan_manifest, output=output)


__all__ = [
    "run_rank_hsic_residual_scan",
    "run_attach_hsic_ranking_to_run",
    "inspect_hsic_ranking_manifest",
]
'''

DASHBOARD = r'''
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_hsic_dashboard_cards(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_hsic_dashboard_cards(path: str | Path) -> dict[str, Any]:
    payload = load_hsic_dashboard_cards(path)
    cards = payload.get("cards") or []
    return {
        "schema_version": payload.get("schema_version"),
        "slice": payload.get("slice"),
        "read_only": bool(payload.get("read_only")),
        "card_count": len(cards),
        "top_hypothesis_id": cards[0].get("hypothesis_id") if cards else None,
        "source_manifest_path": payload.get("source_manifest_path"),
        "ranking_path": payload.get("ranking_path"),
    }


__all__ = ["load_hsic_dashboard_cards", "inspect_hsic_dashboard_cards"]
'''

UNIT_TEST = r'''
from __future__ import annotations

from pegasus.pirs.hsic_ranking import rank_hsic_rows


def test_slice18b_ranks_by_state_q_value_and_statistic() -> None:
    rows = [
        {"hypothesis_id": "h_fragile", "state": "fragile", "q_value": 0.02, "p_value": 0.01, "statistic": 1.0, "n_eff": 4, "covariate_field_id": "b"},
        {"hypothesis_id": "h_verified", "state": "verified", "q_value": 0.50, "p_value": 0.20, "statistic": 0.1, "n_eff": 30, "covariate_field_id": "a"},
        {"hypothesis_id": "h_supported", "state": "exploratory", "q_value": 0.05, "p_value": 0.01, "statistic": 3.0, "n_eff": 40, "covariate_field_id": "c"},
    ]
    ranked = rank_hsic_rows(rows)
    assert [card.hypothesis_id for card in ranked] == ["h_verified", "h_fragile", "h_supported"]
    assert ranked[0].evidence_tier == "verified"
    assert ranked[1].evidence_tier == "fragile_descriptive"
    assert ranked[2].evidence_tier == "supported_descriptive"
    assert all(card.dashboard_safe for card in ranked)


def test_slice18b_merges_hypothesis_schema_values_when_scores_are_thin() -> None:
    rows = [{"hypothesis_id": "h1", "statistic": 2.0, "q_value": 0.2, "state": "exploratory"}]
    hypotheses = [{"hypothesis_id": "h1", "covariate_field_id": "cov_a", "residual_field_id": "resid", "n_eff": 11.0}]
    ranked = rank_hsic_rows(rows, hypotheses)
    assert ranked[0].covariate_field_id == "cov_a"
    assert ranked[0].residual_field_id == "resid"
    assert ranked[0].n_eff == 11.0
'''

INTEGRATION_TEST = r'''
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from pegasus.cli import app
from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def _seed_scanned_run(run_dir: Path) -> None:
    create_empty_output_bundle(run_dir)
    tables = run_dir / "Tables"
    rows = [
        {
            "hypothesis_id": "hsic__resid__cov_a",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_a",
            "statistic": 0.7,
            "p_value": 0.03,
            "q_value": 0.06,
            "n_eff": 28.0,
            "state": "exploratory",
            "hsic_mode": "exact_linear",
            "residual_mode": "in_sample",
            "fold_scheme": "none",
            "warnings": [],
        },
        {
            "hypothesis_id": "hsic__resid__cov_b",
            "residual_field_id": "resid",
            "covariate_field_id": "cov_b",
            "statistic": 0.1,
            "p_value": 0.7,
            "q_value": 0.9,
            "n_eff": 28.0,
            "state": "fragile",
            "hsic_mode": "exact_linear",
            "residual_mode": "in_sample",
            "fold_scheme": "none",
            "warnings": ["hsic_descriptive_small_support"],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tables / "hsic_residual_scan_scores.parquet")
    _write_json(tables / "hsic_residual_scan_manifest.json", {
        "schema_version": "1.0",
        "slice": "18A",
        "status": "scanned",
        "score_count": 2,
        "hypothesis_count": 2,
        "scan_rows": rows,
    })


def test_slice18b_ranks_hsic_scan_writes_dashboard_and_preserves_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_scanned_run(run_dir)
    payload = run_rank_hsic_residual_scan(run_dir=run_dir)
    assert payload["status"] == "ranked"
    assert payload["rank_count"] == 2
    assert payload["dashboard_card_count"] == 2
    assert (run_dir / "Tables" / "hsic_residual_scan_ranking.parquet").exists()
    assert (run_dir / "Tables" / "dashboard_hsic_cards.json").exists()
    assert validate_output_bundle(run_dir=str(run_dir)).ok
    cards = inspect_hsic_dashboard_cards(run_dir / "Tables" / "dashboard_hsic_cards.json")
    assert cards["read_only"] is True
    assert cards["card_count"] == 2
    run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
    assert run_config["hsic_ranking_gate"]["status"] == "ranked"


def test_slice18b_cli_ranks_and_inspects_hsic_dashboard(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _seed_scanned_run(run_dir)
    runner = CliRunner()
    result = runner.invoke(app, ["pirs", "rank-hsic-scan", "--run-dir", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "hsic_ranking_gate" in result.output or "ranked_hypotheses" in result.output
    manifest = run_dir / "Tables" / "hsic_residual_scan_ranking_manifest.json"
    dashboard = run_dir / "Tables" / "dashboard_hsic_cards.json"
    inspect_result = runner.invoke(app, ["pirs", "inspect-hsic-ranking", "--manifest", str(manifest)])
    assert inspect_result.exit_code == 0, inspect_result.output
    assert "ranked" in inspect_result.output
    dash_result = runner.invoke(app, ["pirs", "inspect-hsic-dashboard", "--dashboard", str(dashboard)])
    assert dash_result.exit_code == 0, dash_result.output
    assert "read_only" in dash_result.output
'''

AUDIT = r'''
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pegasus.dashboard.hsic_readonly import inspect_hsic_dashboard_cards
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.workflows.hsic_rank import run_rank_hsic_residual_scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


def main() -> int:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        create_empty_output_bundle(run_dir)
        tables = run_dir / "Tables"
        rows = [
            {
                "hypothesis_id": "hsic__resid__cov_a",
                "residual_field_id": "resid",
                "covariate_field_id": "cov_a",
                "statistic": 1.2,
                "p_value": 0.01,
                "q_value": 0.04,
                "n_eff": 30.0,
                "state": "exploratory",
                "hsic_mode": "exact_linear",
                "residual_mode": "in_sample",
                "fold_scheme": "none",
                "warnings": [],
            }
        ]
        pq.write_table(pa.Table.from_pylist(rows), tables / "hsic_residual_scan_scores.parquet")
        _write_json(tables / "hsic_residual_scan_manifest.json", {
            "schema_version": "1.0",
            "slice": "18A",
            "status": "scanned",
            "score_count": 1,
            "hypothesis_count": 1,
            "scan_rows": rows,
        })
        payload = run_rank_hsic_residual_scan(run_dir=run_dir)
        if payload.get("status") != "ranked":
            errors.append("ranking status was not ranked")
        if payload.get("rank_count") != 1:
            errors.append("rank count mismatch")
        ranking_path = run_dir / "Tables" / "hsic_residual_scan_ranking.parquet"
        dashboard_path = run_dir / "Tables" / "dashboard_hsic_cards.json"
        if not ranking_path.exists():
            errors.append("ranking parquet missing")
        if not dashboard_path.exists():
            errors.append("dashboard cards json missing")
        cards = inspect_hsic_dashboard_cards(dashboard_path)
        if cards.get("read_only") is not True:
            errors.append("dashboard cards are not read-only")
        if cards.get("card_count") != 1:
            errors.append("dashboard card count mismatch")
        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            errors.extend(validation.errors)
        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        if run_config.get("hsic_ranking_gate", {}).get("status") != "ranked":
            errors.append("run summary missing hsic_ranking_gate")
    print(json.dumps({"ok": not errors, "errors": errors}, ensure_ascii=False, sort_keys=True, indent=2))
    if errors:
        return 1
    print("AUDIT PASSED: Slice 18B HSIC ranking and dashboard read-only exposure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def main() -> int:
    required = [
        ROOT / "src" / "pegasus" / "pirs" / "hsic_run.py",
        ROOT / "src" / "pegasus" / "workflows" / "hsic_execute.py",
        ROOT / "src" / "pegasus" / "cli.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Slice 18B requires Slice 18A to be present. Missing:\n" + "\n".join(missing))

    write("src/pegasus/pirs/hsic_ranking.py", HSIC_RANKING)
    write("src/pegasus/workflows/hsic_rank.py", HSIC_WORKFLOW)
    write("src/pegasus/dashboard/hsic_readonly.py", DASHBOARD)
    write("tests/unit/test_slice18b_hsic_ranking_dashboard.py", UNIT_TEST)
    write("tests/integration/test_slice18b_hsic_ranking_dashboard_integration.py", INTEGRATION_TEST)
    write("scripts/dev/audits/audit_slice18b_hsic_ranking_dashboard.py", AUDIT)
    patch_cli()

    print("Slice 18B updater applied: HSIC ranking, dashboard read-only exposure, and CLI added.")
    print("Touched files:")
    for rel in [
        "src/pegasus/pirs/hsic_ranking.py",
        "src/pegasus/workflows/hsic_rank.py",
        "src/pegasus/dashboard/hsic_readonly.py",
        "src/pegasus/cli.py",
        "tests/unit/test_slice18b_hsic_ranking_dashboard.py",
        "tests/integration/test_slice18b_hsic_ranking_dashboard_integration.py",
        "scripts/dev/audits/audit_slice18b_hsic_ranking_dashboard.py",
    ]:
        print(f"  - {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
