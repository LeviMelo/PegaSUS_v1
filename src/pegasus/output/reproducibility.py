from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

TERMINAL_STAGE_STATUSES = {"success", "skipped", "blocked", "failed"}

COMPILE_TELEMETRY_STAGES = (
    "datasus_manifest",
    "datasus_acquire",
    "datasus_decode",
    "sidra_metadata",
    "sidra_plan",
    "sidra_fetch",
    "sidra_normalize",
    "race_bridge",
    "geo_support",
    "she_build",
    "population_solver",
    "stdfm",
    "efg_build",
    "q_tensor",
    "pirs_model",
    "pirs_hsic",
    "output_serialization",
    "output_validation",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


@dataclass
class RunTelemetry:
    "Incremental compile telemetry for TDD-compliant run reproducibility."

    run_id: str
    diagnostic_path: Path
    run_dir: Path | None = None
    started_at: float = field(default_factory=time.perf_counter)
    stage_status: dict[str, str] = field(default_factory=lambda: {s: "skipped" for s in COMPILE_TELEMETRY_STAGES})
    stage_wall_seconds: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in COMPILE_TELEMETRY_STAGES})
    resource_summary: dict[str, Any] = field(
        default_factory=lambda: {
            "peak_rss_mb": None,
            "peak_vram_mb": None,
            "duckdb_temp_bytes": None,
            "rows_read": {},
            "rows_written": {},
            "parquet_bytes_written": 0,
        }
    )

    def model(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_wall_seconds": max(0.0, time.perf_counter() - self.started_at),
            "stage_status": dict(self.stage_status),
            "stage_wall_seconds": dict(self.stage_wall_seconds),
            "resource_summary": self.resource_summary,
        }

    def set_stage(self, stage: str, status: str, duration: float = 0.0) -> None:
        if stage not in self.stage_status:
            raise KeyError(f"Unknown telemetry stage: {stage}")
        if status not in TERMINAL_STAGE_STATUSES:
            raise ValueError(f"Invalid telemetry status: {status}")
        self.stage_status[stage] = status
        self.stage_wall_seconds[stage] = max(0.0, float(duration))

    def block(self, stage: str, *, reason: str | None = None) -> None:
        self.set_stage(stage, "blocked", 0.0)
        if reason:
            blocked = self.resource_summary.setdefault("blocked_reasons", {})
            blocked[stage] = reason
        self.flush()

    @contextmanager
    def stage(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        self.flush()
        try:
            yield
        except Exception:
            self.set_stage(stage, "failed", time.perf_counter() - started)
            self.flush()
            raise
        else:
            self.set_stage(stage, "success", time.perf_counter() - started)
            self.flush()

    def flush(self) -> None:
        self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        self.diagnostic_path.write_text(stable_json({"telemetry": self.model()}), encoding="utf-8")
        if self.run_dir is not None:
            manifest_path = self.run_dir / "ReproducibilityManifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    manifest = {}
                manifest["telemetry"] = self.model()
                manifest_path.write_text(stable_json(manifest), encoding="utf-8")


def write_reproducibility_manifest(
    *,
    run_dir: str | Path,
    run_id: str,
    intent_hash: str,
    source_hashes: dict[str, str],
    registry_hashes: dict[str, str],
    telemetry: RunTelemetry,
    extras: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "generated_at": utc_now(),
        "code_version": "0.1.0",
        "intent_hash": intent_hash,
        "source_hashes": source_hashes,
        "registry_hashes": registry_hashes,
        "random_seed": 0,
        "telemetry": telemetry.model(),
    }
    if extras:
        manifest.update(extras)
    path = run_dir / "ReproducibilityManifest.json"
    path.write_text(stable_json(manifest), encoding="utf-8")
    telemetry.run_dir = run_dir
    telemetry.flush()
    return path
