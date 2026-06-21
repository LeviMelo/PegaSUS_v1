from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def yn(value: object) -> str:
    if value is True:
        return "YES"
    if value is False:
        return "NO"
    if value is None:
        return "NA"
    return str(value)


def iteration_of(row: dict[str, Any]) -> Any:
    return row.get("iteration", row.get("repeat"))


def summarize(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        by_stage[str(row.get("stage"))].append(row)

    iterations = sorted({
        iteration_of(row)
        for row in rows
        if iteration_of(row) is not None
    })

    matrix: list[dict[str, Any]] = []
    for i in iterations:
        row_summary: dict[str, Any] = {"iteration": i}

        for stage in [
            "rcurl_url_exists_bare_ftp_host",
            "rcurl_url_exists_exact_file",
            "direct_download",
            "direct_read_dbc",
            "process_direct_decoded",
            "microdatasus_fetch",
            "process_microdatasus_fetched",
        ]:
            hits = [
                item
                for item in by_stage.get(stage, [])
                if iteration_of(item) == i
            ]

            if not hits:
                row_summary[stage] = "NA"
                continue

            hit = hits[-1]
            if stage == "microdatasus_fetch":
                row_summary[stage] = (
                    f"ok={yn(hit.get('ok'))}; "
                    f"null={yn(hit.get('is_null'))}; "
                    f"rows={hit.get('rows')}; "
                    f"cols={hit.get('cols')}; "
                    f"err={hit.get('error')}"
                )
            elif stage in {
                "direct_read_dbc",
                "process_direct_decoded",
                "process_microdatasus_fetched",
            }:
                row_summary[stage] = (
                    f"ok={yn(hit.get('ok'))}; "
                    f"rows={hit.get('rows')}; "
                    f"cols={hit.get('cols')}; "
                    f"err={hit.get('error')}"
                )
            elif stage == "direct_download":
                row_summary[stage] = (
                    f"ok={yn(hit.get('ok'))}; "
                    f"bytes={hit.get('bytes')}; "
                    f"err={hit.get('error')}"
                )
            else:
                row_summary[stage] = (
                    f"ok={yn(hit.get('ok'))}; "
                    f"value={hit.get('value')}; "
                    f"err={hit.get('error')}"
                )

        matrix.append(row_summary)

    runtime = by_stage.get("runtime", [{}])[-1]
    exports = by_stage.get("microdatasus_exports", [{}])[-1]
    target = by_stage.get("target", [{}])[-1]

    return {
        "path": str(path),
        "target": target,
        "runtime": {
            "R_version": runtime.get("R_version"),
            "libPaths": runtime.get("libPaths"),
            "packages": runtime.get("packages"),
        },
        "exports": exports,
        "matrix": matrix,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: python summarize_microdatasus_probe.py <probe.jsonl> ...",
            file=sys.stderr,
        )
        return 2

    for arg in sys.argv[1:]:
        summary = summarize(Path(arg))
        print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())