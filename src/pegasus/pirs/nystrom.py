"""Nyström approximation contract for PIRS HSIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NystromDiagnostics:
    n_landmarks: int
    landmark_policy: str
    seed: int
    approximation_rank: int
    warning: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "approximation": "nystrom",
            "n_landmarks": self.n_landmarks,
            "landmark_policy": self.landmark_policy,
            "seed": self.seed,
            "approximation_rank": self.approximation_rank,
            "warning": self.warning,
        }


def nystrom_diagnostics(*, n_eff: int, budget: str, seed: int = 20260611) -> NystromDiagnostics:
    n_landmarks = min(max(32, int(n_eff ** 0.5)), 512)
    if budget == "deep":
        n_landmarks = min(max(n_landmarks, 128), 1024)
    return NystromDiagnostics(
        n_landmarks=n_landmarks,
        landmark_policy="uniform_seeded",
        seed=seed,
        approximation_rank=n_landmarks,
        warning="hsic_approximation_diagnostics_emitted",
    )
