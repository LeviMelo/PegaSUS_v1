"""RunCatalog — make compiled run bundles discoverable by what they ARE, not by path-guessing.

A run bundle's identity (scope, time window, health seeds, stage, intent) already lives in its
``UserIntent.json`` + ``ReproducibilityManifest.json``; there was just no index over them, so finding
"the national C25 investigate run" meant enumerating directories and hand-reading JSON. This is that
index: a read-side scan of the canonical ``data/runs`` root (test-fixture roots are excluded per the
data-lifecycle contract) into typed :class:`RunRecord`s, with ``find_runs`` / ``latest_run`` queries.

It reads only the small identity JSONs (never the multi-GB tensors), so it stays cheap at national
scale. It is a pure discovery layer — it does not modify the compile/flush path.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    path: str
    intent_stem: str | None = None
    intent_hash: str | None = None
    created_at: str | None = None            # ISO8601 from ReproducibilityManifest.generated_at
    code_version: str | None = None
    execution_stage: str | None = None       # "investigate" | "full" | ...
    execution_scale: str | None = None       # "national" | "state" | "municipality"
    run_profile: str | None = None
    geo_level: str | None = None
    ufs: tuple[str, ...] = ()
    year_start: int | None = None
    year_end: int | None = None
    health_seeds: tuple[str, ...] = ()

    def matches(
        self,
        *,
        uf: str | None = None,
        year: int | None = None,
        execution_stage: str | None = None,
        execution_scale: str | None = None,
        run_profile: str | None = None,
        seed: str | None = None,
        intent_contains: str | None = None,
    ) -> bool:
        if uf is not None and uf not in self.ufs:
            return False
        if year is not None and not (
            (self.year_start is None or year >= self.year_start)
            and (self.year_end is None or year <= self.year_end)
        ):
            return False
        if execution_stage is not None and self.execution_stage != execution_stage:
            return False
        if execution_scale is not None and self.execution_scale != execution_scale:
            return False
        if run_profile is not None and self.run_profile != run_profile:
            return False
        if seed is not None and seed not in self.health_seeds:
            return False
        if intent_contains is not None and intent_contains not in (self.intent_stem or ""):
            return False
        return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def extract_record(run_dir: str | Path) -> RunRecord | None:
    """Build a :class:`RunRecord` from a bundle's identity JSONs, or None if it isn't a run bundle."""
    run_dir = Path(run_dir)
    repro = _read_json(run_dir / "ReproducibilityManifest.json")
    intent = _read_json(run_dir / "UserIntent.json")
    if not repro and not intent:
        return None

    geography = intent.get("geography") if isinstance(intent.get("geography"), dict) else {}
    time = intent.get("time") if isinstance(intent.get("time"), dict) else {}
    seeds = intent.get("health_seeds") or []
    ufs = geography.get("uf") or []
    intent_path = repro.get("intent_path") or ""
    intent_stem = Path(str(intent_path).replace("\\", "/")).stem or None

    return RunRecord(
        run_id=str(repro.get("run_id") or run_dir.name),
        path=str(run_dir),
        intent_stem=intent_stem,
        intent_hash=repro.get("intent_hash"),
        created_at=repro.get("generated_at"),
        code_version=repro.get("code_version"),
        execution_stage=intent.get("execution_stage"),
        execution_scale=intent.get("execution_scale"),
        run_profile=intent.get("run_profile") or repro.get("run_profile"),
        geo_level=geography.get("level"),
        ufs=tuple(str(x) for x in ufs),
        year_start=time.get("start_year"),
        year_end=time.get("end_year"),
        health_seeds=tuple(str(x) for x in seeds),
    )


def scan_runs(runs_root: str | Path = "data/runs") -> list[RunRecord]:
    """Scan the canonical runs root into records. Stage-workspace siblings (``*_stage_workspace``)
    and non-bundle dirs are skipped. Sorted newest-first by ``created_at``."""
    runs_root = Path(runs_root)
    records: list[RunRecord] = []
    if not runs_root.exists():
        return records
    for child in runs_root.iterdir():
        if not child.is_dir() or child.name.endswith("_stage_workspace"):
            continue
        rec = extract_record(child)
        if rec is not None:
            records.append(rec)
    records.sort(key=lambda r: r.created_at or "", reverse=True)
    return records


def find_runs(records: list[RunRecord], **filters: Any) -> list[RunRecord]:
    """Filter records by any RunRecord.matches criteria (uf, year, execution_stage/scale, seed, …)."""
    return [r for r in records if r.matches(**filters)]


def latest_run(records: list[RunRecord], **filters: Any) -> RunRecord | None:
    """The newest run matching the filters (records are already sorted newest-first)."""
    matches = find_runs(records, **filters)
    return matches[0] if matches else None


def build_catalog(
    runs_root: str | Path = "data/runs",
    out_path: str | Path = "data/manifests/runs/catalog.json",
) -> list[RunRecord]:
    """Scan and persist the catalog (metadata role; never GC'd). Returns the records."""
    records = scan_runs(runs_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": "1.0", "runs_root": str(runs_root), "records": [asdict(r) for r in records]}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_catalog(path: str | Path = "data/manifests/runs/catalog.json") -> list[RunRecord]:
    payload = _read_json(Path(path))
    out: list[RunRecord] = []
    for row in payload.get("records", []):
        if not isinstance(row, dict):
            continue
        row = dict(row)
        row["ufs"] = tuple(row.get("ufs") or ())
        row["health_seeds"] = tuple(row.get("health_seeds") or ())
        out.append(RunRecord(**{k: v for k, v in row.items() if k in RunRecord.__annotations__}))
    return out


__all__ = [
    "RunRecord",
    "extract_record",
    "scan_runs",
    "find_runs",
    "latest_run",
    "build_catalog",
    "load_catalog",
]
