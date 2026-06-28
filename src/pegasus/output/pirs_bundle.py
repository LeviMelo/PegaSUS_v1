"""PIRS fixture loading compatibility boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pegasus.pirs.schemas import FieldCandidate


def candidates_from_fixture(path: str | Path) -> list[FieldCandidate]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = []
    for row in payload.get("candidates", []):
        candidates.append(
            FieldCandidate(
                field_id=str(row["field_id"]),
                role=row["role"],
                utility=float(row.get("utility", 0.0)),
                q_state=row.get("q_state", "verified"),
                carrier=str(row.get("carrier", "")),
                unit=str(row.get("unit", "")),
                support=dict(row.get("support") or {}),
                variance=None if row.get("variance") is None else float(row["variance"]),
                warnings=tuple(str(x) for x in row.get("warnings", []) or []),
                provenance=tuple(str(x) for x in row.get("provenance", []) or []),
            )
        )
    return candidates


__all__ = ["candidates_from_fixture"]
