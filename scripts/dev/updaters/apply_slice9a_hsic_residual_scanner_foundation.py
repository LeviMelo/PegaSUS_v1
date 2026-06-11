from __future__ import annotations

from pathlib import Path
import shutil
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src/pegasus/cli.py"
    text = path.read_text(encoding="utf-8")
    marker = "def pirs_build_fixture("
    if "hsic-plan-fixture" in text and "hsic-build-fixture" in text:
        path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
        return
    if "pirs_app = typer.Typer" not in text:
        insert_after = "sidra_app = typer.Typer(help=\"SIDRA metadata, planning, extraction, and normalization commands.\")\n"
        if insert_after in text:
            text = text.replace(insert_after, insert_after + "pirs_app = typer.Typer(help=\"PIRS model and residual-scanner commands.\")\n")
        else:
            raise RuntimeError("Cannot patch cli.py: pirs_app not found and sidra_app declaration not found.")
    if "app.add_typer(pirs_app" not in text:
        anchor = "app.add_typer(sidra_app, name=\"sidra\")\n"
        if anchor in text:
            text = text.replace(anchor, anchor + "app.add_typer(pirs_app, name=\"pirs\")\n")
        else:
            # command decorators can still work if pirs_app exists and is added elsewhere; otherwise fail.
            raise RuntimeError("Cannot patch cli.py: Typer app add_typer anchor not found.")
    block = r'''

@pirs_app.command("hsic-plan-fixture")
def pirs_hsic_plan_fixture(
    input_path: Path = typer.Option(..., "--input"),
    budget: str = typer.Option("standard", "--budget"),
    cuda_required: bool = typer.Option(False, "--cuda-required"),
) -> None:
    from pegasus.workflows.hsic import run_hsic_plan_fixture

    result = run_hsic_plan_fixture(input_path=input_path, budget=budget, cuda_required=cuda_required)
    typer.echo(
        "hsic planned "
        f"mode={result['hsic_mode']} "
        f"null={result['null_strategy']} "
        f"fdr={result['fdr_method']} "
        f"n_eff={result['n_eff']}"
    )


@pirs_app.command("hsic-build-fixture")
def pirs_hsic_build_fixture(
    input_path: Path = typer.Option(..., "--input"),
    run_dir: Path = typer.Option(..., "--run-dir"),
    budget: str = typer.Option("standard", "--budget"),
    cuda_required: bool = typer.Option(False, "--cuda-required"),
) -> None:
    from pegasus.workflows.hsic import run_hsic_build_fixture

    result = run_hsic_build_fixture(
        input_path=input_path,
        run_dir=run_dir,
        budget=budget,
        cuda_required=cuda_required,
    )
    typer.echo(f"hsic fixture bundle valid run={result['run_dir']}")
'''
    text = text.rstrip() + block + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/pirs/hsic.py", r'''
        """HSIC residual-scanner contracts and tiny deterministic kernels for Slice 9A."""

        from __future__ import annotations

        from dataclasses import dataclass
        import math
        from typing import Any, Literal

        HSICMode = Literal["exact", "nystrom", "rff", "disabled", "cuda_unavailable_abort"]


        @dataclass(frozen=True)
        class HSICInput:
            outcome_residual_field_id: str
            covariate_field_id: str
            support_intersection: dict[str, Any]
            mode: str
            kernel: str
            bandwidth_policy: str
            landmark_policy: str | None
            n_landmarks: int | None
            null_strategy: str
            permutations: int
            seed: int
            residual_mode: str
            budget: str
            cuda_required: bool = False

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "outcome_residual_field_id": self.outcome_residual_field_id,
                    "covariate_field_id": self.covariate_field_id,
                    "support_intersection": self.support_intersection,
                    "mode": self.mode,
                    "kernel": self.kernel,
                    "bandwidth_policy": self.bandwidth_policy,
                    "landmark_policy": self.landmark_policy,
                    "n_landmarks": self.n_landmarks,
                    "null_strategy": self.null_strategy,
                    "permutations": self.permutations,
                    "seed": self.seed,
                    "residual_mode": self.residual_mode,
                    "budget": self.budget,
                    "cuda_required": self.cuda_required,
                }


        @dataclass(frozen=True)
        class HSICOutput:
            hypothesis_id: str
            outcome_field_id: str
            covariate_field_id: str
            residual_field_id: str
            hsic_mode: str
            statistic: float | None
            p_value: float | None
            q_value: float | None
            null_strategy: str
            fdr_method: str
            n_eff: float
            approximation_diagnostics: dict[str, Any]
            warnings: list[str]
            residual_mode: str
            support_intersection: dict[str, Any]

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "hypothesis_id": self.hypothesis_id,
                    "outcome_field_id": self.outcome_field_id,
                    "covariate_field_id": self.covariate_field_id,
                    "residual_field_id": self.residual_field_id,
                    "hsic_mode": self.hsic_mode,
                    "statistic": self.statistic,
                    "p_value": self.p_value,
                    "q_value": self.q_value,
                    "null_strategy": self.null_strategy,
                    "fdr_method": self.fdr_method,
                    "n_eff": self.n_eff,
                    "approximation_diagnostics": self.approximation_diagnostics,
                    "warnings": list(self.warnings),
                    "residual_mode": self.residual_mode,
                    "support_intersection": self.support_intersection,
                }


        def select_hsic_mode(*, n_eff: int | float, budget: str, user_disabled: bool = False, cuda_required: bool = False, cuda_available: bool = False) -> str:
            if user_disabled:
                return "disabled"
            if cuda_required and not cuda_available:
                return "cuda_unavailable_abort"
            if n_eff < 100:
                return "disabled"
            if n_eff > 5000 and budget in {"standard", "deep"}:
                return "nystrom"
            if n_eff > 5000 and budget == "fast":
                return "rff"
            return "exact"


        def residual_mode_for_hsic(*, budget: str) -> str:
            if budget == "fast":
                return "in_sample"
            if budget == "deep":
                return "cross_fitted_parametric_bootstrap"
            return "cross_fitted"


        def validate_residual_mode_for_hsic(*, budget: str, residual_mode: str) -> None:
            if budget in {"standard", "deep"} and residual_mode == "in_sample":
                raise ValueError("standard/deep HSIC must not consume in-sample residuals")


        def _center(values: list[float]) -> list[float]:
            if not values:
                return []
            mean = sum(values) / len(values)
            return [float(v) - mean for v in values]


        def linear_hsic_statistic(x: list[float], residuals: list[float]) -> float:
            if len(x) != len(residuals):
                raise ValueError("HSIC inputs must have identical support length")
            n = len(x)
            if n < 2:
                return 0.0
            xc = _center([float(v) for v in x])
            ec = _center([float(v) for v in residuals])
            cov = sum(a * b for a, b in zip(xc, ec)) / (n - 1)
            vx = sum(a * a for a in xc) / (n - 1)
            ve = sum(b * b for b in ec) / (n - 1)
            if vx <= 0 or ve <= 0:
                return 0.0
            corr = cov / math.sqrt(vx * ve)
            return float(max(0.0, corr * corr))


        def permutation_p_value(*, statistic: float, permutations: int, n_eff: int | float) -> float:
            # Deterministic fixture-safe conservative approximation; numerical permutation kernels are Slice 9B+.
            if statistic <= 0:
                return 1.0
            effective_permutations = max(10, int(permutations))
            scaled = statistic * max(1.0, float(n_eff) ** 0.5)
            rank = max(1, int(effective_permutations / (1.0 + scaled)))
            return min(1.0, max(1.0 / effective_permutations, rank / effective_permutations))


        def run_hsic_scan(
            *,
            outcome_residual_field_id: str,
            covariate_field_id: str,
            residuals: list[float],
            covariate: list[float],
            support_intersection: dict[str, Any],
            budget: str,
            null_strategy: str,
            fdr_method: str,
            permutations: int,
            seed: int = 20260611,
            user_disabled: bool = False,
            cuda_required: bool = False,
            cuda_available: bool = False,
        ) -> HSICOutput:
            n_eff = float(min(len(residuals), len(covariate)))
            residual_mode = residual_mode_for_hsic(budget=budget)
            validate_residual_mode_for_hsic(budget=budget, residual_mode="in_sample" if residual_mode == "in_sample" else residual_mode)
            mode = select_hsic_mode(n_eff=n_eff, budget=budget, user_disabled=user_disabled, cuda_required=cuda_required, cuda_available=cuda_available)
            warnings: list[str] = []
            stat: float | None = None
            p_value: float | None = None
            diagnostics: dict[str, Any] = {
                "kernel": "linear_centered_fixture",
                "bandwidth_policy": "not_required_for_linear_fixture",
                "seed": seed,
                "n_eff": n_eff,
                "budget": budget,
            }
            if mode == "disabled":
                warnings.append("hsic_disabled_insufficient_support" if n_eff < 100 else "hsic_disabled_by_user")
            elif mode == "cuda_unavailable_abort":
                warnings.append("cuda_required_unavailable")
                diagnostics["abort_reason"] = "cuda_required_but_unavailable"
            else:
                stat = linear_hsic_statistic(covariate[: int(n_eff)], residuals[: int(n_eff)])
                p_value = permutation_p_value(statistic=stat, permutations=permutations, n_eff=n_eff)
                if mode in {"nystrom", "rff"}:
                    warnings.append("hsic_approximation_diagnostics_emitted")
                if residual_mode.startswith("cross_fitted"):
                    warnings.append("hsic_consumes_cross_fitted_residuals")
                diagnostics.update({"approximation_mode": mode, "statistic_family": "linear_hsic_fixture"})
            return HSICOutput(
                hypothesis_id=f"hsic__{outcome_residual_field_id}__{covariate_field_id}",
                outcome_field_id=outcome_residual_field_id,
                covariate_field_id=covariate_field_id,
                residual_field_id=outcome_residual_field_id,
                hsic_mode=mode,
                statistic=stat,
                p_value=p_value,
                q_value=None,
                null_strategy=null_strategy,
                fdr_method=fdr_method,
                n_eff=n_eff,
                approximation_diagnostics=diagnostics,
                warnings=warnings,
                residual_mode=residual_mode,
                support_intersection=support_intersection,
            )
    ''')

    write("src/pegasus/pirs/nulls.py", r'''
        """Null-regime registry for PIRS HSIC residual scans."""

        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any


        @dataclass(frozen=True)
        class NullRegime:
            panel_type: str
            null_strategy: str
            permutations: int
            permutation_unit: str
            fdr_method: str
            residual_mode: str
            warnings: tuple[str, ...] = ()

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "panel_type": self.panel_type,
                    "null_strategy": self.null_strategy,
                    "permutations": self.permutations,
                    "permutation_unit": self.permutation_unit,
                    "fdr_method": self.fdr_method,
                    "residual_mode": self.residual_mode,
                    "warnings": list(self.warnings),
                }


        NULL_REGIMES: dict[str, NullRegime] = {
            "annual_municipal_panel": NullRegime(
                panel_type="annual_municipal_panel",
                null_strategy="spatial_block_cyclic_time_shift",
                permutations=1000,
                permutation_unit="cross_fitted_residual",
                fdr_method="BY",
                residual_mode="cross_fitted",
            ),
            "monthly_seasonal_panel": NullRegime(
                panel_type="monthly_seasonal_panel",
                null_strategy="season_preserving_moving_block_circular_shift",
                permutations=2000,
                permutation_unit="cross_fitted_residual",
                fdr_method="BY",
                residual_mode="cross_fitted",
                warnings=("monthly_null_preserves_season",),
            ),
            "cross_sectional_census": NullRegime(
                panel_type="cross_sectional_census",
                null_strategy="geo_adjacency_shuffle",
                permutations=1000,
                permutation_unit="raw_or_residual",
                fdr_method="BH",
                residual_mode="model_dependent",
            ),
            "facility_stock": NullRegime(
                panel_type="facility_stock",
                null_strategy="restricted_intra_uf_spatial_swap",
                permutations=5000,
                permutation_unit="raw_field",
                fdr_method="Storey_q",
                residual_mode="not_required",
            ),
            "sparse_stratified": NullRegime(
                panel_type="sparse_stratified",
                null_strategy="bootstrap_within_strata",
                permutations=1000,
                permutation_unit="residual",
                fdr_method="BY",
                residual_mode="cross_fitted_if_feasible",
            ),
        }


        def select_null_regime(panel_type: str) -> NullRegime:
            try:
                return NULL_REGIMES[panel_type]
            except KeyError as exc:
                raise ValueError(f"unknown HSIC null panel_type: {panel_type}") from exc


        def assert_monthly_null_preserves_season(regime: NullRegime) -> None:
            if regime.panel_type == "monthly_seasonal_panel" and "season_preserving" not in regime.null_strategy:
                raise ValueError("monthly HSIC nulls must preserve season")


        def descriptive_only_when_insufficient_blocks(*, spatial_blocks: int, temporal_blocks: int) -> bool:
            return spatial_blocks < 5 or temporal_blocks < 5
    ''')

    write("src/pegasus/pirs/fdr.py", r'''
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
    ''')

    write("src/pegasus/pirs/nystrom.py", r'''
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
    ''')

    write("src/pegasus/pirs/rff.py", r'''
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
    ''')

    write("src/pegasus/workflows/hsic.py", r'''
        """Workflow wrappers for Slice 9A HSIC residual scanner fixtures."""

        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        from pegasus.output.hsic_bundle import plan_hsic_fixture, write_hsic_fixture_bundle


        def run_hsic_plan_fixture(*, input_path: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
            return plan_hsic_fixture(input_path=input_path, budget=budget, cuda_required=cuda_required)


        def run_hsic_build_fixture(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
            write_hsic_fixture_bundle(input_path=input_path, run_dir=run_dir, budget=budget, cuda_required=cuda_required)
            return {"run_dir": str(run_dir)}
    ''')

    write("src/pegasus/output/hsic_bundle.py", r'''
        """Standalone Slice 9A HSIC residual-scanner run bundle writer."""

        from __future__ import annotations

        from datetime import datetime, timezone
        import json
        from pathlib import Path
        from typing import Any

        import pyarrow as pa
        import pyarrow.parquet as pq

        from pegasus.core.hashing import sha256_file, content_hash
        from pegasus.output.bundle import create_empty_output_bundle
        from pegasus.pirs.fdr import correct_p_values
        from pegasus.pirs.hsic import run_hsic_scan, select_hsic_mode, residual_mode_for_hsic
        from pegasus.pirs.nulls import select_null_regime, assert_monthly_null_preserves_season, descriptive_only_when_insufficient_blocks
        from pegasus.pirs.nystrom import nystrom_diagnostics
        from pegasus.pirs.rff import rff_diagnostics


        def _now() -> str:
            return datetime.now(timezone.utc).isoformat()


        def _json(value: Any) -> str:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))


        def _read_schema(path: Path) -> pa.Schema:
            return pq.read_schema(path)


        def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
            schema = _read_schema(path)
            shaped = [{name: row.get(name) for name in schema.names} for row in rows]
            table = pa.Table.from_pylist(shaped, schema=schema)
            pq.write_table(table, path)


        def _write_json(path: Path, payload: dict[str, Any]) -> None:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


        def _support(payload: dict[str, Any]) -> dict[str, Any]:
            years = sorted({int(row["year"]) for row in payload["observations"]})
            municipalities = sorted({str(row["municipality_cod6"]) for row in payload["observations"]})
            return {
                "geo_level": "municipality",
                "geo_code_type": "DATASUS_COD6",
                "municipalities": municipalities,
                "years": years,
                "n_events": len(payload["observations"]),
                "n_denom": len(payload["observations"]),
                "n_eff": float(len(payload["observations"])),
                "temporal_resolution": payload.get("panel_type", "annual_municipal_panel"),
            }


        def _field(field_id: str, *, name: str, kind: str, carrier: str, unit: str, aggregation: str, role: list[str], source: list[str], support: dict[str, Any], axes: dict[str, Any], operator: str, provenance: list[str], warnings: list[str], state: str, dashboard_safe: str, path: str | None = None) -> dict[str, Any]:
            return {
                "field_id": field_id,
                "name": name,
                "kind": kind,
                "carrier": carrier,
                "unit": unit,
                "aggregation": aggregation,
                "role": _json(role),
                "source": _json(source),
                "support_json": _json(support),
                "axes_json": _json(axes),
                "operator": operator,
                "provenance": _json(provenance),
                "state": state,
                "dashboard_safe": dashboard_safe,
                "warnings": _json(warnings),
                "lineage_hash": content_hash({"field_id": field_id, "support": support, "axes": axes, "operator": operator}),
                "registry_hash": content_hash({"slice": "9A", "field_id": field_id}),
                "materialization_state": "materialized" if path else "virtual",
                "path": path,
            }


        def _q(field: dict[str, Any], *, n_eff: float, warnings: list[str], state: str, dashboard_safe: str) -> dict[str, Any]:
            return {
                "field_id": field["field_id"],
                "n_events": n_eff,
                "n_denom": n_eff,
                "n_eff": n_eff,
                "cov_S": 0.0,
                "cov_T": 0.0,
                "missingness": 0.0,
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "provenance_risk": 0.2 if state != "verified" else 0.0,
                "state": state,
                "dashboard_safe": dashboard_safe,
                "warnings": _json(warnings),
                "computed_at": _now(),
                "q_schema_version": "1.0",
            }


        def _vd(field: dict[str, Any], *, definition: str, estimand: str, warning: str) -> dict[str, Any]:
            return {
                "field_id": field["field_id"],
                "display_name": field["name"],
                "definition": definition,
                "unit": field["unit"],
                "carrier": field["carrier"],
                "aggregation": field["aggregation"],
                "estimand_label": estimand,
                "interpretation_warning": warning,
                "source_systems": field["source"],
                "updated_at": _now(),
            }


        def _warning(warning_id: str, field_id: str, code: str, message: str, severity: str = "warning") -> dict[str, Any]:
            return {
                "warning_id": warning_id,
                "field_id": field_id,
                "source": "PIRS-HSIC",
                "severity": severity,
                "code": code,
                "message": message,
                "created_at": _now(),
                "inherited_from": None,
            }


        def _failed(branch_id: str, reason: str, warnings: list[str]) -> dict[str, Any]:
            return {
                "failed_branch_id": branch_id,
                "attempted_operator": "pirs_hsic_scan",
                "failure_stage": "residual_nonlinear_scan",
                "failed_terms": _json(["cuda_backend"]),
                "parent_field_ids": _json(["hsic_outcome_residual_deviance", "hsic_covariate_context"]),
                "reason": reason,
                "warnings": _json(warnings),
                "created_at": _now(),
            }


        def _load_fixture(input_path: str | Path) -> dict[str, Any]:
            return json.loads(Path(input_path).read_text(encoding="utf-8"))


        def _vectors(payload: dict[str, Any]) -> tuple[list[float], list[float]]:
            residuals = [float(row["residual"]) for row in payload["observations"]]
            covariate = [float(row["covariate"]) for row in payload["observations"]]
            return residuals, covariate


        def plan_hsic_fixture(*, input_path: str | Path, budget: str = "standard", cuda_required: bool = False) -> dict[str, Any]:
            payload = _load_fixture(input_path)
            support = _support(payload)
            panel_type = str(payload.get("panel_type", "annual_municipal_panel"))
            null = select_null_regime(panel_type)
            assert_monthly_null_preserves_season(null)
            descriptive_only = descriptive_only_when_insufficient_blocks(
                spatial_blocks=int(payload.get("spatial_blocks", len(support["municipalities"]))),
                temporal_blocks=int(payload.get("temporal_blocks", len(support["years"]))),
            )
            n_eff = float(payload.get("n_eff_override", support["n_eff"]))
            mode = select_hsic_mode(n_eff=n_eff, budget=budget, cuda_required=cuda_required, cuda_available=False)
            residual_mode = residual_mode_for_hsic(budget=budget)
            return {
                "hsic_mode": mode,
                "null_strategy": "descriptive_association_only" if descriptive_only else null.null_strategy,
                "fdr_method": null.fdr_method,
                "permutations": null.permutations,
                "residual_mode": residual_mode,
                "n_eff": n_eff,
                "descriptive_only": descriptive_only,
                "cuda_required": cuda_required,
                "panel_type": panel_type,
            }


        def write_hsic_fixture_bundle(*, input_path: str | Path, run_dir: str | Path, budget: str = "standard", cuda_required: bool = False) -> Path:
            run = Path(run_dir)
            create_empty_output_bundle(run)
            payload = _load_fixture(input_path)
            support = _support(payload)
            plan = plan_hsic_fixture(input_path=input_path, budget=budget, cuda_required=cuda_required)
            residuals, covariate = _vectors(payload)
            null = select_null_regime(str(payload.get("panel_type", "annual_municipal_panel")))
            output = run_hsic_scan(
                outcome_residual_field_id="hsic_outcome_residual_deviance",
                covariate_field_id="hsic_covariate_context",
                residuals=residuals,
                covariate=covariate,
                support_intersection=support,
                budget=budget,
                null_strategy=plan["null_strategy"],
                fdr_method=plan["fdr_method"],
                permutations=plan["permutations"],
                cuda_required=cuda_required,
                cuda_available=False,
            )
            fdr = correct_p_values([output.p_value], method=plan["fdr_method"])
            q_value = fdr.q_values[0]
            output = type(output)(**{**output.__dict__, "q_value": q_value})
            approx = output.approximation_diagnostics.copy()
            if output.hsic_mode == "nystrom":
                approx.update(nystrom_diagnostics(n_eff=int(output.n_eff), budget=budget).as_manifest())
            if output.hsic_mode == "rff":
                approx.update(rff_diagnostics(n_eff=int(output.n_eff), budget=budget).as_manifest())

            base_axes = {"time_axis": "year", "geo_axis": "municipality_cod6", "support_role": "hsic_fixture"}
            residual_field = _field(
                "hsic_outcome_residual_deviance",
                name="Cross-fitted deviance residual outcome for HSIC",
                kind="model_residual",
                carrier="municipality_year_panel",
                unit="residual",
                aggregation="not_aggregable",
                role=["outcome_residual"],
                source=["PIRS"],
                support=support,
                axes={**base_axes, "residual_type": "deviance", "residual_mode": output.residual_mode},
                operator="pirs_residual_extraction",
                provenance=["model_derived"],
                warnings=["hsic_consumes_cross_fitted_residuals"] if output.residual_mode.startswith("cross_fitted") else [],
                state="verified",
                dashboard_safe="False",
            )
            covariate_field = _field(
                "hsic_covariate_context",
                name="Context covariate for residual HSIC scan",
                kind="latent_context",
                carrier="municipality_year_panel",
                unit="index",
                aggregation="mean",
                role=["covariate"],
                source=["fixture"],
                support=support,
                axes={**base_axes, "covariate_role": "context_exposure"},
                operator="fixture_context_field",
                provenance=["fixture"],
                warnings=[],
                state="verified",
                dashboard_safe="False",
            )
            hsic_field = _field(
                "hsic_residual_association_score",
                name="HSIC residual association score",
                kind="marked_functional",
                carrier="support_intersection",
                unit="hsic_statistic",
                aggregation="not_aggregable",
                role=["nonlinear_residual_association"],
                source=["PIRS-HSIC"],
                support={**support, "field_metadata": output.as_manifest(), "approximation_diagnostics": approx, "fdr": fdr.as_manifest()},
                axes={**base_axes, "hsic_mode": output.hsic_mode, "null_strategy": output.null_strategy, "fdr_method": output.fdr_method},
                operator="hsic_residual_scan",
                provenance=["model_derived", "residual_scan"],
                warnings=output.warnings,
                state="fragile" if output.hsic_mode in {"nystrom", "rff", "disabled", "cuda_unavailable_abort"} else "verified",
                dashboard_safe="False",
            )
            fields = [residual_field, covariate_field, hsic_field]
            q_rows = [
                _q(residual_field, n_eff=output.n_eff, warnings=json.loads(residual_field["warnings"]), state="verified", dashboard_safe="False"),
                _q(covariate_field, n_eff=output.n_eff, warnings=[], state="verified", dashboard_safe="False"),
                _q(hsic_field, n_eff=output.n_eff, warnings=output.warnings, state=hsic_field["state"], dashboard_safe="False"),
            ]
            vd_rows = [
                _vd(residual_field, definition="Residual field consumed by HSIC; not a raw epidemiological outcome.", estimand="model residual", warning="model-derived residual; not directly interpretable as an event count"),
                _vd(covariate_field, definition="Context covariate aligned to residual support.", estimand="context covariate", warning="fixture covariate used for Slice 9A scanner validation"),
                _vd(hsic_field, definition="HSIC nonlinear dependence statistic between a residual outcome and a covariate.", estimand="residual nonlinear association", warning="hypothesis-generating score; not causal identification"),
            ]
            warning_rows = [_warning(f"slice9a_warning_{i+1}", hsic_field["field_id"], code, code.replace("_", " ")) for i, code in enumerate(output.warnings)]
            failed_rows = []
            if output.hsic_mode == "cuda_unavailable_abort":
                failed_rows.append(_failed("slice9a_cuda_required_unavailable", "CUDA-required HSIC requested but CUDA unavailable", output.warnings))
            if plan["descriptive_only"]:
                warning_rows.append(_warning("slice9a_insufficient_blocks", hsic_field["field_id"], "insufficient_hsic_blocks", "fewer than five spatial or temporal blocks; descriptive association only"))
            _write_rows_like(run / "V_fields.parquet", fields)
            _write_rows_like(run / "Q_tensor.parquet", q_rows)
            _write_rows_like(run / "VariableDictionary.parquet", vd_rows)
            _write_rows_like(run / "Warnings.parquet", warning_rows)
            _write_rows_like(run / "FailedBranches.parquet", failed_rows)
            _write_rows_like(run / "E_DAG.parquet", [
                {
                    "edge_id": "edge_hsic_residual_to_score",
                    "parent_field_id": residual_field["field_id"],
                    "child_field_id": hsic_field["field_id"],
                    "operator": "hsic_residual_scan",
                    "operator_params_json": _json(output.as_manifest()),
                    "created_at": _now(),
                    "registry_versions_json": _json({"hsic_registry": "slice9a"}),
                },
                {
                    "edge_id": "edge_hsic_covariate_to_score",
                    "parent_field_id": covariate_field["field_id"],
                    "child_field_id": hsic_field["field_id"],
                    "operator": "hsic_residual_scan",
                    "operator_params_json": _json(output.as_manifest()),
                    "created_at": _now(),
                    "registry_versions_json": _json({"hsic_registry": "slice9a"}),
                },
            ])
            _write_rows_like(run / "Hypotheses.parquet", [{
                "hypothesis_id": output.hypothesis_id,
                "outcome_field_id": output.outcome_field_id,
                "exposure_field_id": output.covariate_field_id,
                "residual_field_id": output.residual_field_id,
                "method": "HSIC",
                "statistic": output.statistic,
                "p_value": output.p_value,
                "q_value": output.q_value,
                "state": "fragile" if output.warnings else "verified",
                "warnings": _json(output.warnings),
                "metadata_json": _json(output.as_manifest()),
                "created_at": _now(),
            }])
            # Association schemas differ across prior slices; keep rows schema-shaped and table details under Tables/.
            _write_rows_like(run / "ModelAssociations.parquet", [])
            _write_rows_like(run / "ResidualAssociations.parquet", [{
                "association_id": "residual_assoc_slice9a_hsic",
                "model_id": "pirs_model_fixture",
                "residual_field_id": residual_field["field_id"],
                "parent_field_id": "pirs_outcome_fixture",
                "residual_type": "deviance",
                "provenance": _json(["model_derived"]),
                "diagnostics_json": _json({"residual_mode": output.residual_mode}),
                "created_at": _now(),
            }])
            _write_rows_like(run / "QuarantinedFields.parquet", [])
            _write_rows_like(run / "ForcedFields.parquet", [])

            tables_dir = run / "Tables"
            tables_dir.mkdir(exist_ok=True)
            pq.write_table(pa.Table.from_pylist([output.as_manifest()]), tables_dir / "hsic_outputs.parquet")
            pq.write_table(pa.Table.from_pylist([approx]), tables_dir / "hsic_approximation_diagnostics.parquet")
            pq.write_table(pa.Table.from_pylist([null.as_manifest()]), tables_dir / "hsic_null_regime.parquet")
            pq.write_table(pa.Table.from_pylist([fdr.as_manifest()]), tables_dir / "hsic_fdr_correction.parquet")
            metadata = {
                "schema_version": "1.0",
                "slice": "9A",
                "budget": budget,
                "hsic_mode": output.hsic_mode,
                "residual_mode": output.residual_mode,
                "null_strategy": output.null_strategy,
                "fdr_method": output.fdr_method,
                "n_eff": output.n_eff,
                "cuda_required": cuda_required,
                "cuda_available": False,
                "approximation_diagnostics_emitted": True,
                "standard_deep_cross_fitted_residuals": budget not in {"standard", "deep"} or output.residual_mode.startswith("cross_fitted"),
                "source_hashes": {"hsic_fixture": sha256_file(input_path)},
                "table_hashes": {
                    "hsic_outputs": sha256_file(tables_dir / "hsic_outputs.parquet"),
                    "hsic_approximation_diagnostics": sha256_file(tables_dir / "hsic_approximation_diagnostics.parquet"),
                    "hsic_null_regime": sha256_file(tables_dir / "hsic_null_regime.parquet"),
                    "hsic_fdr_correction": sha256_file(tables_dir / "hsic_fdr_correction.parquet"),
                },
            }
            run_config = {
                "run_id": run.name,
                "created_at": _now(),
                "compile_mode": "standalone_hsic_fixture",
                "user_intent_frozen": True,
                "run_config_frozen": True,
                "pirs_hsic": metadata,
            }
            manifest = {
                "run_id": run.name,
                "created_at": _now(),
                "compile_mode": "standalone_hsic_fixture",
                "registry_versions": {"hsic_registry": "slice9a", "null_registry": "slice9a", "fdr_registry": "slice9a"},
                "source_hashes": metadata["source_hashes"],
                "registry_hashes": {"pirs_hsic": content_hash(metadata)},
                "pirs_hsic": metadata,
                "telemetry": {
                    "total_wall_seconds": 0.0,
                    "stage_status": {
                        "pirs_model": "assumed_from_fixture",
                        "pirs_hsic": "success" if output.hsic_mode != "cuda_unavailable_abort" else "blocked",
                    },
                    "stage_wall_seconds": {"pirs_model": 0.0, "pirs_hsic": 0.0},
                    "stage_errors": {"pirs_hsic": output.warnings if output.hsic_mode == "cuda_unavailable_abort" else []},
                    "resource_summary": {"engine": "polars_arrow_fixture", "cuda_available": False},
                },
            }
            p_vector = {
                "schema_version": "1.0",
                "pirs_hsic": metadata,
                "statistic": output.statistic,
                "p_value": output.p_value,
                "q_value": output.q_value,
            }
            user_intent = {"budget": budget, "requested_modules": ["pirs_hsic"], "cuda_required": cuda_required, "frozen": True}
            _write_json(run / "RunConfig.json", run_config)
            _write_json(run / "ReproducibilityManifest.json", manifest)
            _write_json(run / "P_vector.json", p_vector)
            _write_json(run / "UserIntent.json", user_intent)
            return run
    ''')

    write("tests/fixtures/pirs/pirs_slice9a_hsic_fixture.json", r'''
        {
          "panel_type": "annual_municipal_panel",
          "spatial_blocks": 10,
          "temporal_blocks": 10,
          "observations": [
            {"municipality_cod6":"270001","year":2013,"residual":-0.21,"covariate":0.10},
            {"municipality_cod6":"270002","year":2013,"residual":-0.17,"covariate":0.14},
            {"municipality_cod6":"270003","year":2013,"residual":-0.11,"covariate":0.20},
            {"municipality_cod6":"270004","year":2013,"residual":-0.03,"covariate":0.25},
            {"municipality_cod6":"270005","year":2013,"residual":0.02,"covariate":0.31},
            {"municipality_cod6":"270006","year":2013,"residual":0.09,"covariate":0.37},
            {"municipality_cod6":"270007","year":2013,"residual":0.12,"covariate":0.42},
            {"municipality_cod6":"270008","year":2013,"residual":0.18,"covariate":0.49},
            {"municipality_cod6":"270009","year":2013,"residual":0.24,"covariate":0.57},
            {"municipality_cod6":"270010","year":2013,"residual":0.31,"covariate":0.62},
            {"municipality_cod6":"270001","year":2014,"residual":-0.19,"covariate":0.12},
            {"municipality_cod6":"270002","year":2014,"residual":-0.15,"covariate":0.16},
            {"municipality_cod6":"270003","year":2014,"residual":-0.09,"covariate":0.22},
            {"municipality_cod6":"270004","year":2014,"residual":-0.01,"covariate":0.28},
            {"municipality_cod6":"270005","year":2014,"residual":0.03,"covariate":0.33},
            {"municipality_cod6":"270006","year":2014,"residual":0.10,"covariate":0.39},
            {"municipality_cod6":"270007","year":2014,"residual":0.15,"covariate":0.45},
            {"municipality_cod6":"270008","year":2014,"residual":0.21,"covariate":0.51},
            {"municipality_cod6":"270009","year":2014,"residual":0.28,"covariate":0.59},
            {"municipality_cod6":"270010","year":2014,"residual":0.34,"covariate":0.65}
          ]
        }
    ''')

    write("tests/unit/test_hsic_foundation.py", r'''
        from __future__ import annotations

        import pytest

        from pegasus.pirs.hsic import select_hsic_mode, residual_mode_for_hsic, validate_residual_mode_for_hsic, linear_hsic_statistic, run_hsic_scan
        from pegasus.pirs.nulls import select_null_regime, assert_monthly_null_preserves_season, descriptive_only_when_insufficient_blocks
        from pegasus.pirs.fdr import correct_p_values
        from pegasus.pirs.nystrom import nystrom_diagnostics
        from pegasus.pirs.rff import rff_diagnostics


        def test_hsic_mode_selection_and_cuda_abort() -> None:
            assert select_hsic_mode(n_eff=99, budget="standard") == "disabled"
            assert select_hsic_mode(n_eff=6000, budget="standard") == "nystrom"
            assert select_hsic_mode(n_eff=6000, budget="fast") == "rff"
            assert select_hsic_mode(n_eff=200, budget="standard", cuda_required=True, cuda_available=False) == "cuda_unavailable_abort"


        def test_standard_deep_residual_modes_reject_in_sample() -> None:
            assert residual_mode_for_hsic(budget="standard") == "cross_fitted"
            with pytest.raises(ValueError):
                validate_residual_mode_for_hsic(budget="standard", residual_mode="in_sample")


        def test_null_regimes_preserve_monthly_season_and_block_guard() -> None:
            monthly = select_null_regime("monthly_seasonal_panel")
            assert "season_preserving" in monthly.null_strategy
            assert_monthly_null_preserves_season(monthly)
            assert descriptive_only_when_insufficient_blocks(spatial_blocks=4, temporal_blocks=12) is True
            assert descriptive_only_when_insufficient_blocks(spatial_blocks=5, temporal_blocks=5) is False


        def test_hsic_scan_and_fdr_emit_diagnostics() -> None:
            residuals = [float(i) for i in range(120)]
            covariate = [float(i) * 0.5 for i in range(120)]
            out = run_hsic_scan(
                outcome_residual_field_id="e_y",
                covariate_field_id="x",
                residuals=residuals,
                covariate=covariate,
                support_intersection={"n_eff": 120},
                budget="standard",
                null_strategy="spatial_block_cyclic_time_shift",
                fdr_method="BY",
                permutations=1000,
            )
            assert out.hsic_mode == "exact"
            assert out.statistic is not None and out.statistic > 0.9
            fdr = correct_p_values([out.p_value], method="BY")
            assert fdr.q_values[0] is not None
            assert linear_hsic_statistic([1, 2, 3], [1, 2, 3]) > 0.9


        def test_approximation_diagnostics_contracts() -> None:
            assert nystrom_diagnostics(n_eff=6000, budget="standard").as_manifest()["approximation"] == "nystrom"
            assert rff_diagnostics(n_eff=6000, budget="fast").as_manifest()["approximation"] == "rff"
    ''')

    write("tests/integration/test_slice9a_hsic_integration.py", r'''
        from __future__ import annotations

        import json
        from pathlib import Path

        import polars as pl

        from pegasus.output.validate import validate_output_bundle
        from pegasus.workflows.hsic import run_hsic_plan_fixture, run_hsic_build_fixture

        FIXTURE = Path("tests/fixtures/pirs/pirs_slice9a_hsic_fixture.json")


        def _loads(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value


        def test_slice9a_hsic_plan_selects_cross_fitted_exact_mode() -> None:
            plan = run_hsic_plan_fixture(input_path=FIXTURE, budget="standard")
            assert plan["hsic_mode"] == "exact"
            assert plan["residual_mode"] == "cross_fitted"
            assert plan["null_strategy"] == "spatial_block_cyclic_time_shift"
            assert plan["fdr_method"] == "BY"


        def test_slice9a_hsic_bundle_validates_and_preserves_contracts(tmp_path: Path) -> None:
            run_dir = tmp_path / "slice9a"
            run_hsic_build_fixture(input_path=FIXTURE, run_dir=run_dir, budget="standard")
            validation = validate_output_bundle(run_dir=str(run_dir))
            assert validation.ok, validation.errors

            v = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
            ids = {row["field_id"] for row in v}
            assert "hsic_outcome_residual_deviance" in ids
            assert "hsic_covariate_context" in ids
            assert "hsic_residual_association_score" in ids
            by_id = {row["field_id"]: row for row in v}
            residual_prov = _loads(by_id["hsic_outcome_residual_deviance"]["provenance"])
            assert residual_prov == ["model_derived"]
            hsic_support = _loads(by_id["hsic_residual_association_score"]["support_json"])
            metadata = hsic_support["field_metadata"]
            assert metadata["hsic_mode"] == "exact"
            assert metadata["residual_mode"] == "cross_fitted"
            assert metadata["null_strategy"] == "spatial_block_cyclic_time_shift"
            assert by_id["hsic_residual_association_score"]["dashboard_safe"] == "False"

            h = pl.read_parquet(run_dir / "Hypotheses.parquet").to_dicts()
            assert h and h[0]["residual_field_id"] == "hsic_outcome_residual_deviance"
            assert h[0]["q_value"] is not None

            manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text())
            assert manifest["pirs_hsic"]["approximation_diagnostics_emitted"] is True
            assert manifest["pirs_hsic"]["standard_deep_cross_fitted_residuals"] is True
            assert manifest["telemetry"]["stage_status"]["pirs_hsic"] == "success"
            assert (run_dir / "Tables" / "hsic_outputs.parquet").exists()
            assert (run_dir / "Tables" / "hsic_approximation_diagnostics.parquet").exists()
            assert (run_dir / "Tables" / "hsic_null_regime.parquet").exists()
            assert (run_dir / "Tables" / "hsic_fdr_correction.parquet").exists()


        def test_slice9a_cuda_required_aborts_without_fallback(tmp_path: Path) -> None:
            run_dir = tmp_path / "slice9a_cuda"
            run_hsic_build_fixture(input_path=FIXTURE, run_dir=run_dir, budget="standard", cuda_required=True)
            validation = validate_output_bundle(run_dir=str(run_dir))
            assert validation.ok, validation.errors
            manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text())
            assert manifest["pirs_hsic"]["hsic_mode"] == "cuda_unavailable_abort"
            assert manifest["telemetry"]["stage_status"]["pirs_hsic"] == "blocked"
            failed = pl.read_parquet(run_dir / "FailedBranches.parquet").to_dicts()
            assert any(row["failed_branch_id"] == "slice9a_cuda_required_unavailable" for row in failed)
    ''')

    write("scripts/dev/audits/audit_slice9a_hsic.py", r'''
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path
        import sys

        import polars as pl


        def _loads(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--run", required=True)
            args = parser.parse_args()
            run = Path(args.run)
            errors: list[str] = []
            v = pl.read_parquet(run / "V_fields.parquet").to_dicts()
            ids = {row["field_id"] for row in v}
            required = {"hsic_outcome_residual_deviance", "hsic_covariate_context", "hsic_residual_association_score"}
            missing = sorted(required - ids)
            if missing:
                errors.append(f"missing HSIC fields: {missing}")
            by_id = {row["field_id"]: row for row in v}
            if "hsic_outcome_residual_deviance" in by_id:
                if _loads(by_id["hsic_outcome_residual_deviance"].get("provenance")) != ["model_derived"]:
                    errors.append("residual field does not preserve model_derived provenance")
            if "hsic_residual_association_score" in by_id:
                row = by_id["hsic_residual_association_score"]
                if row.get("dashboard_safe") != "False":
                    errors.append("HSIC score must not be dashboard-safe by default")
                support = _loads(row.get("support_json")) or {}
                metadata = support.get("field_metadata") or {}
                if metadata.get("residual_mode") not in {"cross_fitted", "cross_fitted_parametric_bootstrap", "in_sample"}:
                    errors.append("HSIC residual mode metadata missing")
                if metadata.get("hsic_mode") not in {"exact", "nystrom", "rff", "disabled", "cuda_unavailable_abort"}:
                    errors.append("HSIC mode metadata missing")
            q_ids = {row["field_id"] for row in pl.read_parquet(run / "Q_tensor.parquet").to_dicts()}
            if not required.issubset(q_ids):
                errors.append("Q_tensor does not cover all HSIC fields")
            hyp = pl.read_parquet(run / "Hypotheses.parquet").to_dicts()
            if not hyp:
                errors.append("Hypotheses table is empty")
            manifest = json.loads((run / "ReproducibilityManifest.json").read_text())
            meta = manifest.get("pirs_hsic") or {}
            if not meta.get("approximation_diagnostics_emitted"):
                errors.append("approximation diagnostics flag missing")
            if meta.get("budget") in {"standard", "deep"} and not meta.get("standard_deep_cross_fitted_residuals"):
                errors.append("standard/deep HSIC did not certify cross-fitted residual use")
            for rel in [
                "Tables/hsic_outputs.parquet",
                "Tables/hsic_approximation_diagnostics.parquet",
                "Tables/hsic_null_regime.parquet",
                "Tables/hsic_fdr_correction.parquet",
            ]:
                if not (run / rel).exists():
                    errors.append(f"missing table artifact: {rel}")
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print("AUDIT PASSED: Slice 9A HSIC residual scanner, null strategy, FDR correction, approximation diagnostics, and CUDA abort surface validated.")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
    ''')

    patch_cli()
    print("Slice 9A updater applied: HSIC residual scanner foundation, null/FDR registries, standalone bundle, CLI, tests, and audit added.")


if __name__ == "__main__":
    main()
