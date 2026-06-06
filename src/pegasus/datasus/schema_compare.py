from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compare_profiles(
    raw_profile: dict[str, Any],
    processed_profile: dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    raw_cols = {c["column"]: c for c in raw_profile.get("columns", [])}
    proc_cols = {c["column"]: c for c in processed_profile.get("columns", [])}

    raw_names = set(raw_cols)
    proc_names = set(proc_cols)

    common = sorted(raw_names & proc_names)
    changed: list[dict[str, Any]] = []

    for col in common:
        r = raw_cols[col]
        p = proc_cols[col]
        diffs = {}
        for key in ["dtype", "missing_count", "unique_count", "numeric_parse_count"]:
            if r.get(key) != p.get(key):
                diffs[key] = {"raw": r.get(key), "processed": p.get(key)}
        if diffs:
            changed.append({"column": col, "differences": diffs})

    result = {
        "raw_path": raw_profile.get("path"),
        "processed_path": processed_profile.get("path"),
        "raw_only_columns": sorted(raw_names - proc_names),
        "processed_only_columns": sorted(proc_names - raw_names),
        "common_columns_changed": changed,
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
