from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
BANNED = (
    r"\bpl\.read_parquet\b",
    r"\.write_parquet\b",
    r"\bpyarrow\.parquet\b",
    r"\bpq\.read_table\b",
    r"\bpq\.write_table\b",
)


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    errors: list[str] = []
    for pattern in BANNED:
        if re.search(pattern, text):
            errors.append(f"storage bypass remains in {TARGET.relative_to(ROOT)}: {pattern}")
    if "from pegasus.output.table_io import" not in text:
        errors.append("sidra_denominator_anchor.py does not import output.table_io")
    if "import polars as pl" in text:
        errors.append("sidra_denominator_anchor.py still imports Polars directly")
    payload = {
        "audit": "slice28za_sidra_anchor_storage_boundary",
        "target": str(TARGET.relative_to(ROOT)),
        "errors": errors,
        "status": "failed" if errors else "passed",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
