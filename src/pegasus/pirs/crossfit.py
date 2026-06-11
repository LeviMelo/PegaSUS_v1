from __future__ import annotations

from dataclasses import dataclass

from pegasus.pirs.schemas import ResidualMode


@dataclass(frozen=True)
class FoldScheme:
    residual_mode: ResidualMode
    fold_scheme: str
    n_folds: int
    preserves_spatial_blocks: bool
    preserves_temporal_blocks: bool
    bootstrap_count: int

    def as_manifest(self) -> dict[str, object]:
        return {
            "residual_mode": self.residual_mode,
            "fold_scheme": self.fold_scheme,
            "n_folds": self.n_folds,
            "preserves_spatial_blocks": self.preserves_spatial_blocks,
            "preserves_temporal_blocks": self.preserves_temporal_blocks,
            "bootstrap_count": self.bootstrap_count,
        }


def residual_mode_for_budget(budget: str) -> ResidualMode:
    if budget == "fast":
        return "in_sample"
    if budget == "standard":
        return "cross_fitted"
    if budget == "deep":
        return "parametric_bootstrap"
    raise ValueError(f"Unsupported PIRS budget: {budget!r}")


def fold_scheme_for_budget(*, budget: str, support_kind: str = "annual_municipal_panel") -> FoldScheme:
    mode = residual_mode_for_budget(budget)
    if mode == "in_sample":
        return FoldScheme(mode, "none_in_sample_fast_budget", 1, False, False, 0)
    if mode == "cross_fitted":
        return FoldScheme(mode, f"{support_kind}_blocked_kfold", 5, True, True, 0)
    return FoldScheme(mode, f"{support_kind}_blocked_kfold_plus_parametric_bootstrap", 5, True, True, 200)


def assert_standard_deep_not_in_sample(budget: str, mode: ResidualMode) -> None:
    if budget in {"standard", "deep"} and mode == "in_sample":
        raise ValueError("standard_and_deep_pirs_must_not_use_in_sample_residuals")
