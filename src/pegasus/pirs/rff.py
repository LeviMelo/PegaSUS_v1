"""Random Fourier feature approximation contract for PIRS HSIC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RFFDiagnostics:
    n_features: int
    seed: int
    bandwidth_policy: str
    warning: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "approximation": "rff",
            "n_features": self.n_features,
            "seed": self.seed,
            "bandwidth_policy": self.bandwidth_policy,
            "warning": self.warning,
        }


def rff_diagnostics(*, n_eff: int, budget: str, seed: int = 20260611) -> RFFDiagnostics:
    n_features = 128 if budget == "fast" else 256
    return RFFDiagnostics(
        n_features=min(max(n_features, 64), max(64, n_eff)),
        seed=seed,
        bandwidth_policy="median_subsample",
        warning="hsic_approximation_diagnostics_emitted",
    )
