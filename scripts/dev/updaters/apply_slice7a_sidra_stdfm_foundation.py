from __future__ import annotations

import json
import shutil
import textwrap
import zipfile
from pathlib import Path

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def patch_cli() -> None:
    path = ROOT / "src/pegasus/cli.py"
    text = path.read_text(encoding="utf-8")
    marker = "# Slice 7A SIDRA context/ST-DFM commands"
    if marker in text:
        return
    block = r'''

# Slice 7A SIDRA context/ST-DFM commands
@sidra_app.command("context-plan-fixture")
def sidra_context_plan_fixture(
    input_path: Path = typer.Option(..., "--input"),
) -> None:
    from pegasus.workflows.sidra_context import run_sidra_context_plan_fixture

    result = run_sidra_context_plan_fixture(input_path=input_path)
    print(
        "[green]sidra context plan[/green] "
        f"segments={result['segments']} projection_status={result['projection_status']} "
        f"stdfm_status={result['stdfm_status']}"
    )


@sidra_app.command("context-build-fixture")
def sidra_context_build_fixture(
    input_path: Path = typer.Option(..., "--input"),
    run_dir: Path = typer.Option(..., "--run-dir"),
) -> None:
    from pegasus.workflows.sidra_context import run_sidra_context_build_fixture

    result = run_sidra_context_build_fixture(input_path=input_path, run_dir=run_dir)
    print(f"[green]sidra context fixture bundle built[/green] run={result['run_dir']}")
'''
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def main() -> None:
    print("Slice 7A updater preflight passed.")
    print("CREATE/REPLACE:")
    for rel in [
        "src/pegasus/sidra/stitching.py",
        "src/pegasus/sidra/projection.py",
        "src/pegasus/sidra/regime.py",
        "src/pegasus/she/high_dimensional.py",
        "src/pegasus/she/stdfm/schema.py",
        "src/pegasus/she/stdfm/blocked.py",
        "src/pegasus/she/stdfm/certification.py",
        "src/pegasus/she/stdfm/objective.py",
        "src/pegasus/she/stdfm/regime.py",
        "src/pegasus/she/stdfm/torch_solver.py",
        "src/pegasus/output/sidra_stdfm_bundle.py",
        "src/pegasus/workflows/sidra_context.py",
        "tests/fixtures/sidra/sidra_slice7_context_fixture.json",
        "tests/unit/test_sidra_stdfm_foundation.py",
        "tests/integration/test_slice7a_sidra_stdfm_integration.py",
        "scripts/dev/audits/audit_slice7a_sidra_stdfm.py",
    ]:
        print(f"  {rel}")
    print("PATCH:")
    print("  src/pegasus/cli.py")

    write("src/pegasus/sidra/stitching.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any


        class SIDRAStitchingError(ValueError):
            """Raised when SIDRA longitudinal segments cannot be legally stitched."""


        @dataclass(frozen=True)
        class SIDRASegment:
            concept_id: str
            table_id: str
            variable_id: str
            segment_id: str
            periods: tuple[str, ...]
            unit: str
            classification_version: str
            source_hash: str | None = None

            @classmethod
            def from_mapping(cls, payload: dict[str, Any]) -> "SIDRASegment":
                periods = tuple(str(p) for p in payload.get("periods", []))
                if not periods:
                    raise SIDRAStitchingError("SIDRA segment must declare at least one period.")
                return cls(
                    concept_id=str(payload["concept_id"]),
                    table_id=str(payload["table_id"]),
                    variable_id=str(payload["variable_id"]),
                    segment_id=str(payload.get("segment_id") or f"{payload['table_id']}:{payload['variable_id']}"),
                    periods=periods,
                    unit=str(payload.get("unit") or "unknown"),
                    classification_version=str(payload.get("classification_version") or "unclassified"),
                    source_hash=str(payload.get("source_hash")) if payload.get("source_hash") is not None else None,
                )

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "concept_id": self.concept_id,
                    "table_id": self.table_id,
                    "variable_id": self.variable_id,
                    "segment_id": self.segment_id,
                    "periods": list(self.periods),
                    "unit": self.unit,
                    "classification_version": self.classification_version,
                    "source_hash": self.source_hash,
                }


        @dataclass(frozen=True)
        class SIDRAStitchingResult:
            concept_id: str
            status: str
            stitched_periods: tuple[str, ...]
            segment_provenance: tuple[dict[str, Any], ...]
            warnings: tuple[str, ...]
            reason: str | None = None

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "concept_id": self.concept_id,
                    "status": self.status,
                    "stitched_periods": list(self.stitched_periods),
                    "segment_provenance": list(self.segment_provenance),
                    "warnings": list(self.warnings),
                    "reason": self.reason,
                }


        def stitch_sidra_longitudinal_segments(segments: list[SIDRASegment]) -> SIDRAStitchingResult:
            if not segments:
                raise SIDRAStitchingError("Cannot stitch empty SIDRA segment list.")

            concept_ids = {s.concept_id for s in segments}
            if len(concept_ids) != 1:
                raise SIDRAStitchingError("SIDRA stitching cannot merge different concept_id values.")
            concept_id = next(iter(concept_ids))

            units = {s.unit for s in segments}
            if len(units) != 1:
                return SIDRAStitchingResult(
                    concept_id=concept_id,
                    status="blocked",
                    stitched_periods=tuple(),
                    segment_provenance=tuple(s.as_manifest() for s in segments),
                    warnings=("sidra_stitch_unit_mismatch",),
                    reason="SIDRA longitudinal stitching requires unit compatibility.",
                )

            by_period: dict[str, SIDRASegment] = {}
            warnings: list[str] = []
            for segment in segments:
                for period in segment.periods:
                    if period in by_period and by_period[period].segment_id != segment.segment_id:
                        warnings.append("sidra_stitch_overlap_segment_priority")
                    by_period.setdefault(period, segment)

            if len({s.table_id for s in segments}) > 1 or len({s.variable_id for s in segments}) > 1:
                warnings.append("sidra_table_identity_not_concept_identity")
                warnings.append("sidra_stitch_segment_provenance")

            if len({s.classification_version for s in segments}) > 1:
                warnings.append("sidra_stitch_classification_version_change")

            return SIDRAStitchingResult(
                concept_id=concept_id,
                status="stitched" if len(segments) > 1 else "direct",
                stitched_periods=tuple(sorted(by_period.keys())),
                segment_provenance=tuple(s.as_manifest() for s in segments),
                warnings=tuple(dict.fromkeys(warnings)),
            )


        def stitch_from_payload(payload: dict[str, Any]) -> SIDRAStitchingResult:
            return stitch_sidra_longitudinal_segments([SIDRASegment.from_mapping(x) for x in payload.get("segments", [])])
    ''')

    write("src/pegasus/sidra/projection.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any

        from pegasus.sidra.schemas import ProjectionResult


        class SIDRAProjectionError(ValueError):
            """Raised when a SIDRA classification projection matrix is malformed."""


        @dataclass(frozen=True)
        class ClassificationProjectionMatrix:
            matrix_id: str
            source_axis: str
            target_axis: str
            weights: dict[str, dict[str, float]]
            fractional: bool

            def as_rows(self) -> list[dict[str, Any]]:
                rows: list[dict[str, Any]] = []
                for source, targets in self.weights.items():
                    for target, weight in targets.items():
                        rows.append(
                            {
                                "projection_matrix_id": self.matrix_id,
                                "source_axis": self.source_axis,
                                "target_axis": self.target_axis,
                                "source_category": source,
                                "target_category": target,
                                "weight": float(weight),
                                "fractional": bool(weight not in {0.0, 1.0}),
                            }
                        )
                return rows

            def warnings(self) -> list[str]:
                warnings: list[str] = []
                if self.fractional:
                    warnings.append("sidra_fractional_classification_projection")
                return warnings


        def project_classification_to_axis(
            *,
            measure_kind: str,
            has_denominator: bool,
            projection_matrix_id: str | None,
        ) -> ProjectionResult:
            if projection_matrix_id is None:
                return ProjectionResult(
                    status="blocked",
                    projection_matrix_id=None,
                    warnings=["projection_matrix_missing"],
                    reason="No registered SIDRA classification projection matrix was supplied.",
                )

            if measure_kind in {"rate", "proportion", "percentage"} and not has_denominator:
                return ProjectionResult(
                    status="blocked",
                    projection_matrix_id=projection_matrix_id,
                    warnings=["ratio_projection_requires_denominator_recovery"],
                    reason="Direct projection of rates/proportions is illegal without numerator/denominator recovery.",
                )

            return ProjectionResult(
                status="projected",
                projection_matrix_id=projection_matrix_id,
                warnings=[],
            )


        def load_projection_matrix(payload: dict[str, Any]) -> ClassificationProjectionMatrix:
            matrix_id = str(payload.get("matrix_id") or payload.get("projection_matrix_id") or "projection_unset")
            source_axis = str(payload.get("source_axis") or "source")
            target_axis = str(payload.get("target_axis") or "target")
            raw_entries = payload.get("entries", [])
            weights: dict[str, dict[str, float]] = {}
            for entry in raw_entries:
                source = str(entry["source"])
                target = str(entry["target"])
                weight = float(entry["weight"])
                if weight < 0:
                    raise SIDRAProjectionError("Projection matrix weights must be non-negative.")
                weights.setdefault(source, {})[target] = weights.setdefault(source, {}).get(target, 0.0) + weight
            for source, targets in weights.items():
                total = sum(targets.values())
                if abs(total - 1.0) > 1e-9:
                    raise SIDRAProjectionError(f"Projection matrix row {source!r} sums to {total}, not 1.")
            fractional = any(weight not in {0.0, 1.0} for targets in weights.values() for weight in targets.values())
            return ClassificationProjectionMatrix(
                matrix_id=matrix_id,
                source_axis=source_axis,
                target_axis=target_axis,
                weights=weights,
                fractional=fractional,
            )


        def projection_metadata(
            *,
            measure_kind: str,
            has_denominator: bool,
            matrix: ClassificationProjectionMatrix | None,
        ) -> dict[str, Any]:
            result = project_classification_to_axis(
                measure_kind=measure_kind,
                has_denominator=has_denominator,
                projection_matrix_id=matrix.matrix_id if matrix else None,
            )
            warnings = list(result.warnings)
            if matrix is not None:
                warnings.extend(matrix.warnings())
            return {
                "status": result.status,
                "projection_matrix_id": result.projection_matrix_id,
                "source_axis": matrix.source_axis if matrix else None,
                "target_axis": matrix.target_axis if matrix else None,
                "fractional": matrix.fractional if matrix else False,
                "warnings": list(dict.fromkeys(warnings)),
                "reason": result.reason,
            }
    ''')

    write("src/pegasus/sidra/regime.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any, Literal

        SIDRARegime = Literal[
            "direct",
            "harmonize",
            "deflate",
            "bounded_interpolate",
            "cross_sectional",
            "do_not_reconstruct",
        ]


        @dataclass(frozen=True)
        class SIDRAContextRegimeResult:
            regime: SIDRARegime
            stdfm_gate: bool
            warnings: tuple[str, ...]
            reason: str

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "regime": self.regime,
                    "stdfm_gate": self.stdfm_gate,
                    "warnings": list(self.warnings),
                    "reason": self.reason,
                }


        def classify_sidra_context_regime(
            *,
            missing_t: bool,
            schema_stable: bool,
            schema_mismatch: bool,
            projectable: bool,
            unit: str,
            anchors_bounded: bool,
            concept_compatible: bool,
            temporal_points: int,
            dynamics: str,
        ) -> SIDRAContextRegimeResult:
            unit_norm = unit.strip().lower()
            if not missing_t and schema_stable:
                return SIDRAContextRegimeResult("direct", False, tuple(), "Complete stable SIDRA segment.")
            if schema_mismatch and projectable:
                return SIDRAContextRegimeResult("harmonize", False, ("sidra_schema_harmonization",), "Projectable schema mismatch.")
            if unit_norm in {"r$", "milr$", "sm"}:
                return SIDRAContextRegimeResult("deflate", False, ("sidra_deflation_required",), "Monetary or salary-minimum SIDRA unit.")
            gate = anchors_bounded and concept_compatible and temporal_points >= 3 and dynamics in {
                "continuous",
                "semi-continuous",
                "proportion",
                "positive",
            }
            if gate:
                return SIDRAContextRegimeResult(
                    "bounded_interpolate",
                    True,
                    ("stdfm_candidate_requires_certification",),
                    "SIDRA field meets ST-DFM gate but solver is certification-bound.",
                )
            if temporal_points >= 1:
                return SIDRAContextRegimeResult(
                    "cross_sectional",
                    False,
                    ("sidra_cross_sectional_only",),
                    "SIDRA field lacks enough certified longitudinal structure for ST-DFM.",
                )
            return SIDRAContextRegimeResult("do_not_reconstruct", False, ("sidra_do_not_reconstruct",), "SIDRA field lacks reconstructable support.")
    ''')

    write("src/pegasus/she/high_dimensional.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any

        from pegasus.sidra.category_maps import bounded_pushforward_scaffold


        class HighDimensionalExposureError(ValueError):
            """Raised when high-dimensional SIDRA exposure is requested without bounded pushforward."""


        @dataclass(frozen=True)
        class HighDimensionalBound:
            status: str
            raw_axes: tuple[str, ...]
            exposed_axes: tuple[str, ...]
            axes_dropped: tuple[str, ...]
            estimated_cells_raw: int
            estimated_cells_bounded: int
            warnings: tuple[str, ...]
            reason: str | None = None

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "status": self.status,
                    "raw_axes": list(self.raw_axes),
                    "exposed_axes": list(self.exposed_axes),
                    "axes_dropped": list(self.axes_dropped),
                    "estimated_cells_raw": self.estimated_cells_raw,
                    "estimated_cells_bounded": self.estimated_cells_bounded,
                    "warnings": list(self.warnings),
                    "reason": self.reason,
                }


        def _estimate_cells(axis_cardinalities: dict[str, int], axes: list[str]) -> int:
            cells = 1
            for axis in axes:
                cells *= int(axis_cardinalities.get(axis, 1))
            return cells


        def bound_high_dimensional_sidra_exposure(
            *,
            raw_axes: list[str],
            demanded_axes: list[str],
            axis_cardinalities: dict[str, int],
            aggregation: str,
            high_dimensional: bool,
        ) -> HighDimensionalBound:
            result = bounded_pushforward_scaffold(
                raw_axes=raw_axes,
                demanded_axes=demanded_axes,
                aggregation=aggregation,
                high_dimensional=high_dimensional,
            )
            raw_cells = _estimate_cells(axis_cardinalities, raw_axes)
            bounded_cells = _estimate_cells(axis_cardinalities, result.axes_kept)
            return HighDimensionalBound(
                status=result.status,
                raw_axes=tuple(raw_axes),
                exposed_axes=tuple(result.axes_kept),
                axes_dropped=tuple(result.axes_dropped),
                estimated_cells_raw=raw_cells,
                estimated_cells_bounded=bounded_cells,
                warnings=tuple(result.warnings),
                reason=result.reason,
            )


        def require_bounded_pushforward(bound: HighDimensionalBound) -> None:
            if bound.status == "blocked":
                raise HighDimensionalExposureError(bound.reason or "High-dimensional SIDRA exposure is not legally bounded.")
    ''')

    write("src/pegasus/she/stdfm/schema.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass, field
        from typing import Any, Literal

        STDFMState = Literal["blocked_solver_pending", "certification_pending", "verified"]


        @dataclass(frozen=True)
        class STDFMInputSchema:
            field_id: str
            concept_id: str
            support: dict[str, Any]
            observation_shape: tuple[int, int]
            transform: str
            dynamics: str
            projection_matrix_id: str | None
            stitch_metadata: dict[str, Any]
            warnings: tuple[str, ...] = field(default_factory=tuple)

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "field_id": self.field_id,
                    "concept_id": self.concept_id,
                    "support": self.support,
                    "observation_shape": list(self.observation_shape),
                    "transform": self.transform,
                    "dynamics": self.dynamics,
                    "projection_matrix_id": self.projection_matrix_id,
                    "stitch_metadata": self.stitch_metadata,
                    "warnings": list(self.warnings),
                }


        @dataclass(frozen=True)
        class STDFMOutputSchema:
            field_id: str
            status: STDFMState
            solver_backend: str
            certification_id: str | None
            uncertainty: float | None
            warnings: tuple[str, ...]
            reason: str

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "field_id": self.field_id,
                    "status": self.status,
                    "solver_backend": self.solver_backend,
                    "certification_id": self.certification_id,
                    "uncertainty": self.uncertainty,
                    "warnings": list(self.warnings),
                    "reason": self.reason,
                }


        def build_stdfm_input_schema(
            *,
            field_id: str,
            concept_id: str,
            support: dict[str, Any],
            periods: list[str],
            localities: list[str],
            transform: str,
            dynamics: str,
            projection_matrix_id: str | None,
            stitch_metadata: dict[str, Any],
            warnings: list[str] | tuple[str, ...] | None = None,
        ) -> STDFMInputSchema:
            return STDFMInputSchema(
                field_id=field_id,
                concept_id=concept_id,
                support=support,
                observation_shape=(len(localities), len(periods)),
                transform=transform,
                dynamics=dynamics,
                projection_matrix_id=projection_matrix_id,
                stitch_metadata=stitch_metadata,
                warnings=tuple(warnings or ()),
            )
    ''')

    write("src/pegasus/she/stdfm/blocked.py", r'''
        from __future__ import annotations

        from pegasus.she.stdfm.schema import STDFMOutputSchema


        def blocked_solver_pending(*, field_id: str, reason: str | None = None) -> STDFMOutputSchema:
            return STDFMOutputSchema(
                field_id=field_id,
                status="blocked_solver_pending",
                solver_backend="pytorch_cuda_pending_calibration",
                certification_id=None,
                uncertainty=None,
                warnings=("blocked_solver_pending", "stdfm_certification_required"),
                reason=reason or "ST-DFM solver is architecturally declared but blocked until calibration/certification is available.",
            )
    ''')

    write("src/pegasus/she/stdfm/certification.py", r'''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import Any


        class STDFMCertificationError(ValueError):
            """Raised when a ST-DFM field is promoted without certification."""


        @dataclass(frozen=True)
        class STDFMCertificationRow:
            certification_id: str
            field_id: str
            status: str
            denominator_uncertainty_declared: bool
            survey_uncertainty_declared: bool
            calibration_dataset_hash: str | None
            warnings: tuple[str, ...]

            def as_manifest(self) -> dict[str, Any]:
                return {
                    "certification_id": self.certification_id,
                    "field_id": self.field_id,
                    "status": self.status,
                    "denominator_uncertainty_declared": self.denominator_uncertainty_declared,
                    "survey_uncertainty_declared": self.survey_uncertainty_declared,
                    "calibration_dataset_hash": self.calibration_dataset_hash,
                    "warnings": list(self.warnings),
                }


        def blocked_certification_row(*, field_id: str) -> STDFMCertificationRow:
            return STDFMCertificationRow(
                certification_id=f"cert_pending::{field_id}",
                field_id=field_id,
                status="blocked_solver_pending",
                denominator_uncertainty_declared=False,
                survey_uncertainty_declared=False,
                calibration_dataset_hash=None,
                warnings=("stdfm_certification_required", "blocked_solver_pending"),
            )


        def assert_verified_promotion_allowed(row: STDFMCertificationRow) -> None:
            if row.status != "verified":
                raise STDFMCertificationError("ST-DFM verified promotion requires certification status='verified'.")
            if not (row.denominator_uncertainty_declared or row.survey_uncertainty_declared):
                raise STDFMCertificationError(
                    "ST-DFM proportion/latent field promotion requires denominator or survey uncertainty."
                )
    ''')

    write("src/pegasus/she/stdfm/objective.py", r'''
        from __future__ import annotations

        from typing import Any


        def stdfm_objective_pseudocode_contract() -> dict[str, Any]:
            """Typed formula-to-code contract for the gated ST-DFM objective.

            This is not a numerical solver. It is the required disambiguated bridge
            between the MSD formula and future production PyTorch implementation.
            """
            return {
                "inputs": {
                    "Y": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64", "missing": "mask"},
                    "M": {"shape": "[n_space, n_time, n_fields]", "dtype": "bool", "meaning": "observed mask"},
                    "Z": {"shape": "[n_space, n_time, n_covariates]", "dtype": "float64"},
                    "support": "municipality/year support aligned before invocation",
                },
                "outputs": {
                    "latent_factor": {"shape": "[n_space, n_time, k]", "dtype": "float64"},
                    "reconstruction": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
                    "uncertainty": {"shape": "[n_space, n_time, n_fields]", "dtype": "float64"},
                },
                "epsilon_stabilization": "variance and bounded-link denominators clamp at eps=1e-9",
                "warnings": ["stdfm_certification_required", "blocked_solver_pending"],
                "failure_modes": [
                    "insufficient_temporal_points",
                    "concept_incompatibility",
                    "uncertified_verified_promotion",
                ],
            }
    ''')

    write("src/pegasus/she/stdfm/regime.py", r'''
        from __future__ import annotations

        from pegasus.sidra.regime import SIDRAContextRegimeResult, classify_sidra_context_regime


        def stdfm_gate_for_sidra_context(
            *,
            anchors_bounded: bool,
            concept_compatible: bool,
            temporal_points: int,
            dynamics: str,
        ) -> SIDRAContextRegimeResult:
            return classify_sidra_context_regime(
                missing_t=True,
                schema_stable=False,
                schema_mismatch=False,
                projectable=False,
                unit="index",
                anchors_bounded=anchors_bounded,
                concept_compatible=concept_compatible,
                temporal_points=temporal_points,
                dynamics=dynamics,
            )
    ''')

    write("src/pegasus/she/stdfm/torch_solver.py", r'''
        from __future__ import annotations

        from pegasus.she.stdfm.blocked import blocked_solver_pending
        from pegasus.she.stdfm.schema import STDFMInputSchema, STDFMOutputSchema


        def solve_stdfm(input_schema: STDFMInputSchema, *, allow_uncertified: bool = False) -> STDFMOutputSchema:
            if not allow_uncertified:
                return blocked_solver_pending(
                    field_id=input_schema.field_id,
                    reason="PyTorch ST-DFM solver is blocked until calibration and certification are supplied.",
                )
            return blocked_solver_pending(
                field_id=input_schema.field_id,
                reason="Uncertified ST-DFM execution is not available in Slice 7A.",
            )
    ''')

    write("src/pegasus/output/sidra_stdfm_bundle.py", r'''
        from __future__ import annotations

        import json
        import sys
        from datetime import datetime, timezone
        from pathlib import Path
        from typing import Any

        import polars as pl

        from pegasus.core.hashing import sha256_file, sha256_text, stable_json
        from pegasus.output.bundle import create_empty_output_bundle
        from pegasus.output.cnes_sih_efg_bundle import write_rows_like
        from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES
        from pegasus.sidra.projection import load_projection_matrix, projection_metadata
        from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
        from pegasus.she.high_dimensional import bound_high_dimensional_sidra_exposure
        from pegasus.she.stdfm.blocked import blocked_solver_pending
        from pegasus.she.stdfm.certification import blocked_certification_row
        from pegasus.she.stdfm.objective import stdfm_objective_pseudocode_contract
        from pegasus.she.stdfm.schema import build_stdfm_input_schema


        def _now() -> str:
            return datetime.now(timezone.utc).isoformat()


        def _json(value: Any) -> str:
            return json.dumps(value, sort_keys=True, ensure_ascii=False)


        def _support(periods: list[str], localities: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "years": [int(p) for p in periods if str(p).isdigit()],
                "periods": [str(p) for p in periods],
                "municipalities_ibge_cod7": [str(x) for x in localities],
                "n_events": None,
                "n_denom": None,
                "missingness": 0.0,
                "denom_fragility": 0.0,
            }
            if extra:
                payload.update(extra)
            return payload


        def _field(
            *,
            field_id: str,
            name: str,
            kind: str,
            carrier: str,
            unit: str,
            aggregation: str,
            role: list[str],
            source: list[str],
            support: dict[str, Any],
            axes: dict[str, Any],
            provenance: list[str],
            warnings: list[str],
            state: str,
            dashboard_safe: bool,
            materialization_state: str,
            operator: str,
            metadata: dict[str, Any],
        ) -> dict[str, Any]:
            lineage = {
                "field_id": field_id,
                "operator": operator,
                "support": support,
                "axes": axes,
                "metadata": metadata,
            }
            return {
                "field_id": field_id,
                "name": name,
                "technical_name": field_id,
                "kind": kind,
                "carrier": carrier,
                "unit": unit,
                "aggregation": aggregation,
                "role": _json(role),
                "role_json": _json(role),
                "source": _json(source),
                "source_json": _json(source),
                "support": _json(support),
                "support_json": _json(support),
                "axes": _json(axes),
                "axes_json": _json(axes),
                "lineage": _json(lineage),
                "lineage_json": _json(lineage),
                "lineage_hash": sha256_text(stable_json(lineage)),
                "provenance": _json(provenance),
                "provenance_json": _json(provenance),
                "warnings": _json(warnings),
                "warnings_json": _json(warnings),
                "state": state,
                "dashboard_safe": dashboard_safe,
                "materialization_state": materialization_state,
                "operator": operator,
                "metadata": _json(metadata),
                "metadata_json": _json(metadata),
            }


        def _q(field: dict[str, Any], *, risk_score: float, warnings: list[str]) -> dict[str, Any]:
            return {
                "field_id": field["field_id"],
                "state": field["state"],
                "dashboard_safe": field["dashboard_safe"],
                "n_eff": None,
                "denom_fragility": 0.0,
                "missingness": 0.0,
                "risk_score": risk_score,
                "warnings": _json(warnings),
                "warnings_json": _json(warnings),
                "q_json": _json({"risk_score": risk_score, "warnings": warnings}),
            }


        def _vd(field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
            return {
                "field_id": field["field_id"],
                "name": field["name"],
                "technical_name": field["technical_name"],
                "definition": definition,
                "estimand": estimand,
                "source_systems": field["source"],
                "carrier": field["carrier"],
                "unit": field["unit"],
                "support_description": field["support"],
                "axis_description": field["axes"],
                "provenance_description": field["provenance"],
                "state": field["state"],
                "dashboard_safe": field["dashboard_safe"],
                "interpretation_warning": warning,
            }


        def _warning(warning_id: str, field_id: str, code: str, message: str, severity: str = "warning") -> dict[str, Any]:
            return {
                "warning_id": warning_id,
                "field_id": field_id,
                "source": "slice7a_sidra_stdfm",
                "severity": severity,
                "code": code,
                "message": message,
            }


        def _failed(branch_id: str, operator: str, parents: list[str], reason: str, warnings: list[str]) -> dict[str, Any]:
            return {
                "branch_id": branch_id,
                "operator": operator,
                "parents": _json(parents),
                "parent_field_ids": _json(parents),
                "stage": "SHE",
                "terms": _json(parents),
                "reason": reason,
                "warnings": _json(warnings),
            }


        def _empty_optional_tables(run_dir: Path) -> None:
            for name in [
                "ModelAssociations.parquet",
                "ResidualAssociations.parquet",
                "Hypotheses.parquet",
                "QuarantinedFields.parquet",
                "ForcedFields.parquet",
            ]:
                write_rows_like(run_dir / name, [])


        def build_sidra_stdfm_fixture_bundle(*, input_path: str | Path, run_dir: str | Path) -> Path:
            input_path = Path(input_path)
            run_dir = Path(run_dir)
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            create_empty_output_bundle(run_dir)

            segments = [SIDRASegment.from_mapping(x) for x in payload["stitching"]["segments"]]
            stitch = stitch_sidra_longitudinal_segments(segments)
            projection_matrix = load_projection_matrix(payload["projection"])
            projection = projection_metadata(measure_kind="additive", has_denominator=False, matrix=projection_matrix)
            high_dim = bound_high_dimensional_sidra_exposure(
                raw_axes=payload["high_dimensional"]["raw_axes"],
                demanded_axes=payload["high_dimensional"]["demanded_axes"],
                axis_cardinalities={k: int(v) for k, v in payload["high_dimensional"]["axis_cardinalities"].items()},
                aggregation=payload["high_dimensional"].get("aggregation", "additive"),
                high_dimensional=True,
            )
            periods = list(stitch.stitched_periods)
            localities = [str(x) for x in payload.get("localities", ["2704302"])]
            base_support = _support(periods, localities)
            stitch_metadata = stitch.as_manifest()

            stdfm_input = build_stdfm_input_schema(
                field_id="sidra_stdfm_blocked_candidate",
                concept_id=stitch.concept_id,
                support=base_support,
                periods=periods,
                localities=localities,
                transform="identity",
                dynamics="continuous",
                projection_matrix_id=projection_matrix.matrix_id,
                stitch_metadata=stitch_metadata,
                warnings=["stdfm_certification_required"],
            )
            stdfm_output = blocked_solver_pending(field_id=stdfm_input.field_id)
            certification = blocked_certification_row(field_id=stdfm_input.field_id)

            fields: list[dict[str, Any]] = []
            fields.append(
                _field(
                    field_id="sidra_stitched_gdp_context",
                    name="SIDRA stitched GDP context",
                    kind="latent_context",
                    carrier="municipality_year",
                    unit="R$",
                    aggregation="additive",
                    role=["context", "sidra_stitched"],
                    source=["SIDRA"],
                    support=base_support,
                    axes={"locality": "IBGE7", "time": "year", "concept": stitch.concept_id},
                    provenance=["SIDRA", "sidra_longitudinal_stitching"],
                    warnings=list(stitch.warnings),
                    state="fragile" if stitch.warnings else "verified",
                    dashboard_safe=not bool(stitch.warnings),
                    materialization_state="materialized",
                    operator="sidra_longitudinal_stitch",
                    metadata={"stitch_metadata": stitch_metadata},
                )
            )
            fields.append(
                _field(
                    field_id="sidra_projected_labor_context",
                    name="SIDRA projected labor context",
                    kind="latent_context",
                    carrier="municipality_year_sector",
                    unit="count",
                    aggregation="additive",
                    role=["context", "sidra_projection"],
                    source=["SIDRA"],
                    support=base_support,
                    axes={"locality": "IBGE7", "time": "year", "projection_axis": projection_matrix.target_axis},
                    provenance=["SIDRA", "sidra_classification_projection"],
                    warnings=list(projection["warnings"]),
                    state="fragile" if projection["warnings"] else "verified",
                    dashboard_safe=False if projection["warnings"] else True,
                    materialization_state="materialized",
                    operator="sidra_classification_projection",
                    metadata={"projection_matrix_id": projection_matrix.matrix_id, "projection": projection},
                )
            )
            fields.append(
                _field(
                    field_id="sidra_highdim_bounded_context",
                    name="SIDRA high-dimensional bounded context",
                    kind="latent_context",
                    carrier="municipality_year_bounded_context",
                    unit="index",
                    aggregation="additive",
                    role=["context", "sidra_high_dimensional_bounded"],
                    source=["SIDRA"],
                    support=base_support,
                    axes={"raw_axes": high_dim.raw_axes, "exposed_axes": high_dim.exposed_axes},
                    provenance=["SIDRA", "high_dimensional_bounded_pushforward"],
                    warnings=list(high_dim.warnings),
                    state="fragile",
                    dashboard_safe=False,
                    materialization_state="materialized",
                    operator="high_dimensional_bounded_pushforward",
                    metadata={"high_dimensional_bound": high_dim.as_manifest()},
                )
            )
            fields.append(
                _field(
                    field_id=stdfm_input.field_id,
                    name="ST-DFM blocked latent reconstruction candidate",
                    kind="latent_context",
                    carrier="municipality_year_context",
                    unit="index",
                    aggregation="model_based",
                    role=["context", "stdfm_candidate"],
                    source=["SIDRA"],
                    support=base_support,
                    axes={"locality": "IBGE7", "time": "year", "latent_factor": "F"},
                    provenance=["SIDRA", "ST-DFM", "blocked_solver_pending"],
                    warnings=list(stdfm_output.warnings),
                    state="blocked",
                    dashboard_safe=False,
                    materialization_state="blocked",
                    operator="ST-DFM",
                    metadata={"stdfm_input": stdfm_input.as_manifest(), "stdfm_output": stdfm_output.as_manifest()},
                )
            )

            q_rows = [
                _q(fields[0], risk_score=0.35 if fields[0]["state"] == "fragile" else 0.05, warnings=json.loads(fields[0]["warnings"])),
                _q(fields[1], risk_score=0.45 if json.loads(fields[1]["warnings"]) else 0.05, warnings=json.loads(fields[1]["warnings"])),
                _q(fields[2], risk_score=0.55, warnings=json.loads(fields[2]["warnings"])),
                _q(fields[3], risk_score=0.95, warnings=json.loads(fields[3]["warnings"])),
            ]
            vd_rows = [
                _vd(fields[0], "SIDRA field stitched across table/variable segments with explicit segment provenance.", "stitched SIDRA contextual field", "Use with segment-provenance warning when table identity changes."),
                _vd(fields[1], "SIDRA field projected across classification axes with registered projection matrix.", "projected SIDRA contextual field", "Fractional projections are fragile unless externally validated."),
                _vd(fields[2], "High-dimensional SIDRA context exposed only after bounded pushforward.", "bounded high-dimensional context", "Dropped axes must remain visible in metadata."),
                _vd(fields[3], "ST-DFM reconstruction candidate blocked pending solver calibration and certification.", "blocked latent reconstruction", "No direct fallback or verified promotion is allowed."),
            ]
            warnings = [
                _warning("w_sidra_stitch", fields[0]["field_id"], "sidra_stitch_segment_provenance", "SIDRA stitching preserved table/variable segment provenance."),
                _warning("w_sidra_projection", fields[1]["field_id"], "sidra_fractional_classification_projection", "Fractional classification projection emits fragile state."),
                _warning("w_sidra_highdim", fields[2]["field_id"], "high_dimensional_bounded_pushforward", "High-dimensional SIDRA field was bounded before EFG exposure."),
                _warning("w_stdfm_blocked", fields[3]["field_id"], "blocked_solver_pending", "ST-DFM candidate did not fall back to direct interpolation."),
            ]
            failed = [
                _failed(
                    "fb_unbounded_high_dimensional_sidra",
                    "direct_high_dimensional_efg_exposure",
                    [fields[2]["field_id"]],
                    "High-dimensional SIDRA exposure without bounded pushforward is illegal.",
                    ["high_dimensional_bounded_pushforward"],
                ),
                _failed(
                    "fb_stdfm_solver_pending",
                    "ST-DFM",
                    [fields[3]["field_id"]],
                    "ST-DFM solver pending emits blocked_solver_pending, not direct fallback.",
                    ["blocked_solver_pending", "stdfm_certification_required"],
                ),
            ]

            write_rows_like(run_dir / "V_fields.parquet", fields)
            write_rows_like(run_dir / "Q_tensor.parquet", q_rows)
            write_rows_like(run_dir / "VariableDictionary.parquet", vd_rows)
            write_rows_like(run_dir / "Warnings.parquet", warnings)
            write_rows_like(run_dir / "FailedBranches.parquet", failed)
            write_rows_like(run_dir / "E_DAG.parquet", [])
            _empty_optional_tables(run_dir)

            tables = run_dir / "Tables"
            tables.mkdir(parents=True, exist_ok=True)
            pl.DataFrame([x for x in stitch.segment_provenance]).write_parquet(tables / "sidra_stitching_segments.parquet")
            pl.DataFrame(projection_matrix.as_rows()).write_parquet(tables / "sidra_projection_matrix.parquet")
            pl.DataFrame([high_dim.as_manifest()]).write_parquet(tables / "sidra_high_dimensional_bounds.parquet")
            pl.DataFrame([certification.as_manifest()]).write_parquet(tables / "stdfm_certification.parquet")
            pl.DataFrame([stdfm_objective_pseudocode_contract()]).write_parquet(tables / "stdfm_objective_contract.parquet")

            stage_status = {stage: "skipped" for stage in COMPILE_TELEMETRY_STAGES}
            stage_wall_seconds = {stage: 0.0 for stage in COMPILE_TELEMETRY_STAGES}
            for stage in ["sidra_fetch", "she_build", "efg_build", "q_tensor", "output_serialization", "output_validation"]:
                if stage in stage_status:
                    stage_status[stage] = "success"
            if "stdfm" in stage_status:
                stage_status["stdfm"] = "blocked"
            telemetry = {
                "total_wall_seconds": 0.0,
                "stage_status": stage_status,
                "stage_wall_seconds": stage_wall_seconds,
                "stage_errors": {"stdfm": "blocked_solver_pending"},
                "resource_summary": {
                    "peak_rss_mb": None,
                    "peak_vram_mb": None,
                    "duckdb_temp_bytes": None,
                    "rows_read": {"sidra_context_fixture": 1},
                    "rows_written": {"V_fields": len(fields), "Q_tensor": len(q_rows), "VariableDictionary": len(vd_rows)},
                    "parquet_bytes_written": 0,
                },
            }
            source_hash = sha256_file(input_path)
            sidra_context = {
                "schema_version": "1.0",
                "stitching": stitch.as_manifest(),
                "projection": projection,
                "high_dimensional_bound": high_dim.as_manifest(),
                "stdfm_input": stdfm_input.as_manifest(),
                "stdfm_output": stdfm_output.as_manifest(),
                "certification": certification.as_manifest(),
                "table_paths": {
                    "stitching": "Tables/sidra_stitching_segments.parquet",
                    "projection": "Tables/sidra_projection_matrix.parquet",
                    "high_dimensional": "Tables/sidra_high_dimensional_bounds.parquet",
                    "stdfm_certification": "Tables/stdfm_certification.parquet",
                    "stdfm_objective_contract": "Tables/stdfm_objective_contract.parquet",
                },
            }
            run_config = {
                "schema_version": "1.0",
                "workflow": "slice7a_sidra_stdfm_context_fixture",
                "source_systems": ["SIDRA"],
                "source_hashes": {"sidra_context_fixture": source_hash},
                "registry_hashes": {"sidra_stdfm": "slice7a_contract_v1"},
                "sidra_context": sidra_context,
            }
            manifest = {
                "schema_version": "1.0",
                "run_id": run_dir.name,
                "generated_at": _now(),
                "workflow": "slice7a_sidra_stdfm_context_fixture",
                "source_hashes": run_config["source_hashes"],
                "registry_hashes": run_config["registry_hashes"],
                "telemetry": telemetry,
                "sidra_context": sidra_context,
                "environment": {"python": sys.version.split()[0]},
            }
            (run_dir / "UserIntent.json").write_text(
                json.dumps({"workflow": "slice7a_sidra_stdfm_context_fixture", "source_systems": ["SIDRA"]}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (run_dir / "RunConfig.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
            (run_dir / "ReproducibilityManifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            (run_dir / "P_vector.json").write_text(json.dumps({"sidra_context": sidra_context}, indent=2, sort_keys=True), encoding="utf-8")
            return run_dir
    ''')

    write("src/pegasus/workflows/sidra_context.py", r'''
        from __future__ import annotations

        import json
        from pathlib import Path
        from typing import Any

        from pegasus.output.sidra_stdfm_bundle import build_sidra_stdfm_fixture_bundle
        from pegasus.sidra.projection import load_projection_matrix, projection_metadata
        from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
        from pegasus.she.high_dimensional import bound_high_dimensional_sidra_exposure
        from pegasus.she.stdfm.blocked import blocked_solver_pending


        def run_sidra_context_plan_fixture(*, input_path: str | Path) -> dict[str, Any]:
            payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
            stitch = stitch_sidra_longitudinal_segments([SIDRASegment.from_mapping(x) for x in payload["stitching"]["segments"]])
            matrix = load_projection_matrix(payload["projection"])
            projection = projection_metadata(measure_kind="additive", has_denominator=False, matrix=matrix)
            bound = bound_high_dimensional_sidra_exposure(
                raw_axes=payload["high_dimensional"]["raw_axes"],
                demanded_axes=payload["high_dimensional"]["demanded_axes"],
                axis_cardinalities={k: int(v) for k, v in payload["high_dimensional"]["axis_cardinalities"].items()},
                aggregation=payload["high_dimensional"].get("aggregation", "additive"),
                high_dimensional=True,
            )
            stdfm = blocked_solver_pending(field_id="sidra_stdfm_blocked_candidate")
            return {
                "segments": len(stitch.segment_provenance),
                "stitch_status": stitch.status,
                "projection_status": projection["status"],
                "projection_warnings": projection["warnings"],
                "high_dimensional_status": bound.status,
                "stdfm_status": stdfm.status,
            }


        def run_sidra_context_build_fixture(*, input_path: str | Path, run_dir: str | Path) -> dict[str, Any]:
            path = build_sidra_stdfm_fixture_bundle(input_path=input_path, run_dir=run_dir)
            return {"run_dir": str(path)}
    ''')

    write("tests/fixtures/sidra/sidra_slice7_context_fixture.json", r'''
        {
          "localities": ["2704302"],
          "stitching": {
            "segments": [
              {
                "concept_id": "municipal_gdp_current_brl",
                "table_id": "5938",
                "variable_id": "37",
                "segment_id": "gdp_2010_2019",
                "periods": ["2018", "2019"],
                "unit": "R$",
                "classification_version": "ibge_old"
              },
              {
                "concept_id": "municipal_gdp_current_brl",
                "table_id": "5938_v2",
                "variable_id": "37",
                "segment_id": "gdp_2020_2021",
                "periods": ["2020", "2021"],
                "unit": "R$",
                "classification_version": "ibge_new"
              }
            ]
          },
          "projection": {
            "matrix_id": "slice7_cnae_to_sector_fractional_v1",
            "source_axis": "CNAE_section",
            "target_axis": "economic_sector",
            "entries": [
              {"source": "A", "target": "agriculture", "weight": 1.0},
              {"source": "C", "target": "industry", "weight": 0.7},
              {"source": "C", "target": "services", "weight": 0.3}
            ]
          },
          "high_dimensional": {
            "raw_axes": ["municipality", "year", "sector", "occupation", "firm_size"],
            "demanded_axes": ["municipality", "year", "sector"],
            "axis_cardinalities": {
              "municipality": 1,
              "year": 4,
              "sector": 3,
              "occupation": 12,
              "firm_size": 5
            },
            "aggregation": "additive"
          }
        }
    ''')

    write("tests/unit/test_sidra_stdfm_foundation.py", r'''
        from pathlib import Path

        import pytest

        from pegasus.sidra.projection import load_projection_matrix, projection_metadata
        from pegasus.sidra.regime import classify_sidra_context_regime
        from pegasus.sidra.stitching import SIDRASegment, stitch_sidra_longitudinal_segments
        from pegasus.she.high_dimensional import HighDimensionalExposureError, bound_high_dimensional_sidra_exposure, require_bounded_pushforward
        from pegasus.she.stdfm.blocked import blocked_solver_pending
        from pegasus.she.stdfm.certification import STDFMCertificationError, assert_verified_promotion_allowed, blocked_certification_row
        from pegasus.she.stdfm.objective import stdfm_objective_pseudocode_contract
        from pegasus.she.stdfm.schema import build_stdfm_input_schema
        from pegasus.she.stdfm.torch_solver import solve_stdfm


        def test_sidra_stitching_preserves_segment_provenance() -> None:
            result = stitch_sidra_longitudinal_segments([
                SIDRASegment.from_mapping({"concept_id": "gdp", "table_id": "t1", "variable_id": "v", "segment_id": "a", "periods": ["2018"], "unit": "R$", "classification_version": "old"}),
                SIDRASegment.from_mapping({"concept_id": "gdp", "table_id": "t2", "variable_id": "v", "segment_id": "b", "periods": ["2019"], "unit": "R$", "classification_version": "new"}),
            ])
            assert result.status == "stitched"
            assert "sidra_table_identity_not_concept_identity" in result.warnings
            assert len(result.segment_provenance) == 2


        def test_fractional_projection_matrix_warns_and_preserves_rows() -> None:
            matrix = load_projection_matrix({
                "matrix_id": "m",
                "source_axis": "raw",
                "target_axis": "target",
                "entries": [
                    {"source": "x", "target": "a", "weight": 0.25},
                    {"source": "x", "target": "b", "weight": 0.75},
                ],
            })
            meta = projection_metadata(measure_kind="additive", has_denominator=False, matrix=matrix)
            assert meta["status"] == "projected"
            assert meta["fractional"] is True
            assert "sidra_fractional_classification_projection" in meta["warnings"]
            assert len(matrix.as_rows()) == 2


        def test_high_dimensional_bounding_blocks_illegal_aggregation() -> None:
            bound = bound_high_dimensional_sidra_exposure(
                raw_axes=["municipality", "year", "sector", "occupation"],
                demanded_axes=["municipality", "year", "sector"],
                axis_cardinalities={"municipality": 1, "year": 2, "sector": 2, "occupation": 10},
                aggregation="mean",
                high_dimensional=True,
            )
            assert bound.status == "blocked"
            with pytest.raises(HighDimensionalExposureError):
                require_bounded_pushforward(bound)


        def test_stdfm_gate_and_blocked_solver_contract() -> None:
            regime = classify_sidra_context_regime(
                missing_t=True,
                schema_stable=False,
                schema_mismatch=False,
                projectable=False,
                unit="index",
                anchors_bounded=True,
                concept_compatible=True,
                temporal_points=4,
                dynamics="continuous",
            )
            assert regime.regime == "bounded_interpolate"
            assert regime.stdfm_gate is True
            inp = build_stdfm_input_schema(
                field_id="f",
                concept_id="c",
                support={"years": [2018, 2019], "municipalities_ibge_cod7": ["2704302"]},
                periods=["2018", "2019"],
                localities=["2704302"],
                transform="identity",
                dynamics="continuous",
                projection_matrix_id="m",
                stitch_metadata={"status": "stitched"},
            )
            out = solve_stdfm(inp)
            assert out.status == "blocked_solver_pending"
            assert "blocked_solver_pending" in out.warnings
            assert blocked_solver_pending(field_id="x").status == "blocked_solver_pending"


        def test_stdfm_certification_blocks_verified_promotion_without_certification() -> None:
            row = blocked_certification_row(field_id="f")
            with pytest.raises(STDFMCertificationError):
                assert_verified_promotion_allowed(row)
            contract = stdfm_objective_pseudocode_contract()
            assert "Y" in contract["inputs"]
            assert "blocked_solver_pending" in contract["warnings"]
    ''')

    write("tests/integration/test_slice7a_sidra_stdfm_integration.py", r'''
        import json
        from pathlib import Path

        import polars as pl

        from pegasus.output.validate import validate_output_bundle
        from pegasus.workflows.sidra_context import run_sidra_context_build_fixture, run_sidra_context_plan_fixture


        FIXTURE = Path("tests/fixtures/sidra/sidra_slice7_context_fixture.json")


        def _loads(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value


        def test_slice7a_plan_reports_stitch_projection_highdim_and_stdfm() -> None:
            result = run_sidra_context_plan_fixture(input_path=FIXTURE)
            assert result["segments"] == 2
            assert result["stitch_status"] == "stitched"
            assert result["projection_status"] == "projected"
            assert "sidra_fractional_classification_projection" in result["projection_warnings"]
            assert result["high_dimensional_status"] == "bounded"
            assert result["stdfm_status"] == "blocked_solver_pending"


        def test_slice7a_bundle_validates_and_preserves_sidra_stdfm_contracts(tmp_path: Path) -> None:
            run_dir = tmp_path / "slice7a"
            run_sidra_context_build_fixture(input_path=FIXTURE, run_dir=run_dir)
            validation = validate_output_bundle(run_dir=str(run_dir))
            assert validation.ok, validation.errors

            v = pl.read_parquet(run_dir / "V_fields.parquet").to_dicts()
            ids = {row["field_id"] for row in v}
            assert "sidra_stitched_gdp_context" in ids
            assert "sidra_projected_labor_context" in ids
            assert "sidra_highdim_bounded_context" in ids
            assert "sidra_stdfm_blocked_candidate" in ids

            by_id = {row["field_id"]: row for row in v}
            projected_meta = _loads(by_id["sidra_projected_labor_context"].get("metadata_json") or by_id["sidra_projected_labor_context"].get("metadata"))
            assert projected_meta["projection_matrix_id"] == "slice7_cnae_to_sector_fractional_v1"
            assert projected_meta["projection"]["fractional"] is True

            highdim_meta = _loads(by_id["sidra_highdim_bounded_context"].get("metadata_json") or by_id["sidra_highdim_bounded_context"].get("metadata"))
            assert highdim_meta["high_dimensional_bound"]["status"] == "bounded"
            assert "occupation" in highdim_meta["high_dimensional_bound"]["axes_dropped"]

            stdfm = by_id["sidra_stdfm_blocked_candidate"]
            assert stdfm["state"] == "blocked"
            assert stdfm["dashboard_safe"] is False
            stdfm_meta = _loads(stdfm.get("metadata_json") or stdfm.get("metadata"))
            assert stdfm_meta["stdfm_output"]["status"] == "blocked_solver_pending"

            warnings = pl.read_parquet(run_dir / "Warnings.parquet").to_dicts()
            codes = {row["code"] for row in warnings}
            assert "sidra_stitch_segment_provenance" in codes
            assert "sidra_fractional_classification_projection" in codes
            assert "high_dimensional_bounded_pushforward" in codes
            assert "blocked_solver_pending" in codes

            failed = pl.read_parquet(run_dir / "FailedBranches.parquet").to_dicts()
            reasons = "\n".join(row["reason"] for row in failed)
            assert "bounded pushforward" in reasons
            assert "blocked_solver_pending" in reasons

            manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
            assert manifest["sidra_context"]["stdfm_output"]["status"] == "blocked_solver_pending"
            assert (run_dir / "Tables" / "sidra_stitching_segments.parquet").exists()
            assert (run_dir / "Tables" / "sidra_projection_matrix.parquet").exists()
            assert (run_dir / "Tables" / "sidra_high_dimensional_bounds.parquet").exists()
            assert (run_dir / "Tables" / "stdfm_certification.parquet").exists()
    ''')

    write("scripts/dev/audits/audit_slice7a_sidra_stdfm.py", r'''
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        import polars as pl


        def _loads(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value


        def main() -> None:
            parser = argparse.ArgumentParser()
            parser.add_argument("--run", required=True)
            args = parser.parse_args()
            run = Path(args.run)
            errors: list[str] = []

            v = pl.read_parquet(run / "V_fields.parquet").to_dicts()
            q = pl.read_parquet(run / "Q_tensor.parquet").to_dicts()
            warnings = pl.read_parquet(run / "Warnings.parquet").to_dicts()
            failed = pl.read_parquet(run / "FailedBranches.parquet").to_dicts()
            manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

            ids = {row["field_id"] for row in v}
            required = {
                "sidra_stitched_gdp_context",
                "sidra_projected_labor_context",
                "sidra_highdim_bounded_context",
                "sidra_stdfm_blocked_candidate",
            }
            missing = sorted(required - ids)
            if missing:
                errors.append(f"missing Slice 7A fields: {missing}")
            q_ids = {row["field_id"] for row in q}
            if not required.issubset(q_ids):
                errors.append("Q_tensor does not cover all Slice 7A fields")

            by_id = {row["field_id"]: row for row in v}
            if "sidra_projected_labor_context" in by_id:
                meta = _loads(by_id["sidra_projected_labor_context"].get("metadata_json") or by_id["sidra_projected_labor_context"].get("metadata"))
                if not meta or meta.get("projection", {}).get("fractional") is not True:
                    errors.append("fractional projection metadata missing")
            if "sidra_highdim_bounded_context" in by_id:
                meta = _loads(by_id["sidra_highdim_bounded_context"].get("metadata_json") or by_id["sidra_highdim_bounded_context"].get("metadata"))
                if not meta or meta.get("high_dimensional_bound", {}).get("status") != "bounded":
                    errors.append("high-dimensional bounded pushforward metadata missing")
            if "sidra_stdfm_blocked_candidate" in by_id:
                row = by_id["sidra_stdfm_blocked_candidate"]
                meta = _loads(row.get("metadata_json") or row.get("metadata"))
                if row.get("state") != "blocked" or row.get("dashboard_safe") is not False:
                    errors.append("ST-DFM candidate is not blocked/non-dashboard-safe")
                if not meta or meta.get("stdfm_output", {}).get("status") != "blocked_solver_pending":
                    errors.append("ST-DFM blocked_solver_pending metadata missing")

            codes = {row["code"] for row in warnings}
            for code in [
                "sidra_stitch_segment_provenance",
                "sidra_fractional_classification_projection",
                "high_dimensional_bounded_pushforward",
                "blocked_solver_pending",
            ]:
                if code not in codes:
                    errors.append(f"missing warning code: {code}")
            if not any("blocked_solver_pending" in row.get("reason", "") for row in failed):
                errors.append("FailedBranches missing ST-DFM blocked_solver_pending branch")
            if not any("bounded pushforward" in row.get("reason", "") for row in failed):
                errors.append("FailedBranches missing unbounded high-dimensional SIDRA rejection")

            sidra_context = manifest.get("sidra_context", {})
            if sidra_context.get("stdfm_output", {}).get("status") != "blocked_solver_pending":
                errors.append("ReproducibilityManifest missing ST-DFM blocked metadata")
            for table in [
                "sidra_stitching_segments.parquet",
                "sidra_projection_matrix.parquet",
                "sidra_high_dimensional_bounds.parquet",
                "stdfm_certification.parquet",
                "stdfm_objective_contract.parquet",
            ]:
                if not (run / "Tables" / table).exists():
                    errors.append(f"missing table: Tables/{table}")

            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                raise SystemExit(1)
            print("AUDIT PASSED: Slice 7A SIDRA stitching, projection, high-dimensional bounding, and ST-DFM blocked certification contracts validated.")


        if __name__ == "__main__":
            main()
    ''')

    patch_cli()

    # Persist updater copy into the repository when run from repo root.
    updater_target = ROOT / "scripts/dev/updaters/apply_slice7a_sidra_stdfm_foundation.py"
    updater_target.parent.mkdir(parents=True, exist_ok=True)
    if Path(__file__).resolve() != updater_target.resolve():
        shutil.copyfile(Path(__file__), updater_target)

    print("Applied Slice 7A SIDRA/ST-DFM foundation. Run the validation ladder supplied by the assistant.")


if __name__ == "__main__":
    main()
