from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.dashboard.read_only import inspect_run, read_table_head


def run_dashboard_inspect_run(*, run_dir: str | Path) -> dict[str, Any]:
    return inspect_run(run_dir=run_dir, validate=True)


def run_dashboard_table_head(*, run_dir: str | Path, table_name: str, limit: int = 10) -> dict[str, Any]:
    return read_table_head(run_dir=run_dir, table_name=table_name, limit=limit)

