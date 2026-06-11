"""False-discovery correction utilities for PIRS HSIC outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FDRResult:
    method: str
    q_values: list[float | None]
    diagnostics: dict[str, Any]

    def as_manifest(self) -> dict[str, Any]:
        return {"method": self.method, "q_values": self.q_values, "diagnostics": self.diagnostics}


def _harmonic(n: int) -> float:
    return sum(1.0 / i for i in range(1, n + 1)) if n > 0 else 1.0


def correct_p_values(p_values: list[float | None], *, method: str) -> FDRResult:
    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    q: list[float | None] = [None for _ in p_values]
    m = len(indexed)
    if m == 0:
        return FDRResult(method=method, q_values=q, diagnostics={"n_tests": 0})
    factor = _harmonic(m) if method.upper() == "BY" else 1.0
    ordered = sorted(indexed, key=lambda item: item[1])
    prev = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(ordered), start=1):
        rank = m - rank_from_end + 1
        val = min(prev, p * m * factor / rank)
        q[idx] = min(1.0, max(0.0, val))
        prev = q[idx] if q[idx] is not None else prev
    return FDRResult(method=method, q_values=q, diagnostics={"n_tests": m, "dependency_factor": factor})
