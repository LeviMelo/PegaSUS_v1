from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path.cwd()

CREATE_OR_REPLACE = [
    "src/pegasus/efg/race_bridge.py",
    "src/pegasus/output/race_bridge_attach.py",
    "src/pegasus/workflows/race_bridge.py",
    "tests/fixtures/race_bridge/fixedC_valid.json",
    "tests/fixtures/race_bridge/fixedC_invalid_empty.json",
    "tests/fixtures/race_bridge/fixedC_invalid_rowsum.json",
    "tests/unit/test_race_bridge_fixedc.py",
    "tests/integration/test_slice4a_race_bridge_integration.py",
    "scripts/dev/audits/audit_slice4a_race_bridge.py",
]

PATCH = [
    "src/pegasus/cli.py",
]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = rel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def preflight() -> None:
    required = [
        "pyproject.toml",
        "src/pegasus/cli.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/output/maternal_child_compile_attach.py",
        "src/pegasus/she/maternal_child_linkage.py",
        "tests/fixtures/datasus/sim_do_fixture.csv",
        "tests/fixtures/datasus/sinasc_fixture.csv",
        "scripts/dev/audits/audit_slice3b_compile_maternal_child_linkage.py",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 4A preflight failed; missing expected files: {missing}")

    compile_py = read("src/pegasus/workflows/compile.py")
    for marker in [
        "def run_compile",
        "attach_maternal_child_compile_fields",
        "run_datasus_normalize_sinasc",
        "maternal_child_linkage",
    ]:
        if marker not in compile_py:
            raise RuntimeError(f"Slice 4A preflight failed; Slice 3B compile marker not found: {marker}")

    cli = read("src/pegasus/cli.py")
    for marker in ["efg_app", "def efg_attach_sidra_denominator", "def compile("]:
        if marker not in cli:
            raise RuntimeError(f"Slice 4A preflight failed; CLI marker not found: {marker}")

    validator = read("src/pegasus/output/validate.py")
    if "validate_output_bundle" not in validator or "Q_tensor does not cover all V_fields" not in validator:
        raise RuntimeError("Slice 4A preflight failed: hardened Slice 2E validator not detected.")


def patch_cli() -> None:
    path = rel("src/pegasus/cli.py")
    cli = path.read_text(encoding="utf-8")
    if "def efg_attach_race_bridge" in cli:
        path.write_text(cli, encoding="utf-8", newline="\n")
        return

    block = r'''

@efg_app.command("attach-race-bridge")
def efg_attach_race_bridge(
    run_dir: Path = typer.Option(..., "--run-dir"),
    sim_events: Path = typer.Option(..., "--sim-events"),
    bridge_prior: Path = typer.Option(..., "--bridge-prior"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    from pegasus.workflows.report.race_bridge import run_attach_race_bridge

    result = run_attach_race_bridge(
        run_dir=run_dir,
        sim_events_path=sim_events,
        bridge_prior_path=bridge_prior,
        municipality_cod6=municipality_cod6,
    )
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]race bridge fields attached and run bundle valid[/green] {result['run_dir']}")
'''
    # Append after existing command declarations. The decorator is valid because efg_app is created near module top.
    cli = cli.rstrip() + "\n" + textwrap.dedent(block) + "\n"
    path.write_text(cli, encoding="utf-8", newline="\n")


def main() -> None:
    preflight()
    print("Slice 4A updater preflight passed.")
    print("CREATE/REPLACE:")
    for p in CREATE_OR_REPLACE:
        print(f"  {p}")
    print("PATCH:")
    for p in PATCH:
        print(f"  {p}")

    write("src/pegasus/efg/race_bridge.py", r'''
    from __future__ import annotations

    import json
    import math
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash, sha256_file

    ADMIN_RACE_LABELS = {
        "1": "branca",
        "2": "preta",
        "3": "amarela",
        "4": "parda",
        "5": "indigena",
    }

    DEFAULT_TARGET_CATEGORIES = ["branca", "preta", "amarela", "parda", "indigena"]


    class RaceBridgeValidationError(ValueError):
        """Raised when a Bridge_R emission-prior object is invalid."""


    @dataclass(frozen=True)
    class RaceBridgePrior:
        bridge_id: str
        mode: str
        source_axis: str
        target_axis: str
        source_categories: list[str]
        target_categories: list[str]
        matrix: dict[str, dict[str, float]]
        sensitivity_width: float
        metadata: dict[str, Any]
        prior_hash: str


    @dataclass(frozen=True)
    class RaceBridgeCounts:
        raw_admin_counts: dict[str, int]
        missing_count: int
        total_count: int
        support: dict[str, Any]

        @property
        def missing_share(self) -> float:
            if self.total_count <= 0:
                return 0.0
            return self.missing_count / float(self.total_count)


    @dataclass(frozen=True)
    class RaceBridgePosterior:
        posterior_counts: dict[str, float]
        lower_counts: dict[str, float]
        upper_counts: dict[str, float]
        raw_admin_counts: dict[str, int]
        missing_count: int
        missing_share: float
        sensitivity_width: float
        race_bridge_cv: float
        prior: RaceBridgePrior
        support: dict[str, Any]

        def metadata(self) -> dict[str, Any]:
            return {
                "numerator_axis_source": self.prior.source_axis,
                "denominator_axis_target": self.prior.target_axis,
                "bridge_operator": "Bridge_R_fixedC_dynamic_weight",
                "emission_matrix_registry_version": self.prior.bridge_id,
                "bridge_mode": self.prior.mode,
                "missing_race_share": self.missing_share,
                "race_bridge_cv": self.race_bridge_cv,
                "sensitivity_width": self.sensitivity_width,
                "race_axis_warning": "Administrative race/color is declaration-process data and is not overwritten by IBGE self-declared race.",
                "bayesian_ecological_bridge_warning": "Posterior race counts are bridge-derived observer fields, not direct self-declared measurements.",
                "prior_hash": self.prior.prior_hash,
            }


    def _canonical_code(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "null", "nan"}:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits or None


    def load_race_bridge_prior(path: str | Path) -> RaceBridgePrior:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return validate_race_bridge_prior(payload, prior_hash=sha256_file(path))


    def validate_race_bridge_prior(payload: dict[str, Any], *, prior_hash: str | None = None) -> RaceBridgePrior:
        if not isinstance(payload, dict):
            raise RaceBridgeValidationError("Bridge_R prior payload must be a JSON object.")
        bridge_id = str(payload.get("bridge_id") or "")
        mode = str(payload.get("mode") or "")
        source_axis = str(payload.get("source_axis") or "")
        target_axis = str(payload.get("target_axis") or "")
        source_categories = [str(x) for x in payload.get("source_categories") or []]
        target_categories = [str(x) for x in payload.get("target_categories") or []]
        matrix = payload.get("matrix") or {}
        sensitivity_width = float(payload.get("sensitivity_width", 0.0))
        metadata = dict(payload.get("metadata") or {})

        if not bridge_id:
            raise RaceBridgeValidationError("Bridge_R prior missing bridge_id.")
        if mode != "fixedC_dynamic_weight":
            raise RaceBridgeValidationError(f"Unsupported Bridge_R mode: {mode!r}")
        if not source_axis or not target_axis:
            raise RaceBridgeValidationError("Bridge_R prior must declare source_axis and target_axis.")
        if not source_categories or not target_categories:
            raise RaceBridgeValidationError("Bridge_R prior must declare nonempty source and target categories.")
        if not isinstance(matrix, dict) or not matrix:
            raise RaceBridgeValidationError("Bridge_R prior matrix must be a nonempty object.")
        if sensitivity_width < 0:
            raise RaceBridgeValidationError("Bridge_R sensitivity_width must be nonnegative.")

        normalized: dict[str, dict[str, float]] = {}
        missing_rows = [src for src in source_categories if src not in matrix]
        if missing_rows:
            raise RaceBridgeValidationError(f"Bridge_R matrix missing source rows: {missing_rows}")

        for src in source_categories:
            row = matrix.get(src)
            if not isinstance(row, dict) or not row:
                raise RaceBridgeValidationError(f"Bridge_R matrix row is empty or invalid: {src}")
            clean_row: dict[str, float] = {}
            for tgt in target_categories:
                value = row.get(tgt, 0.0)
                try:
                    weight = float(value)
                except Exception as exc:
                    raise RaceBridgeValidationError(f"Bridge_R matrix value is nonnumeric at {src}->{tgt}: {value!r}") from exc
                if not math.isfinite(weight) or weight < 0:
                    raise RaceBridgeValidationError(f"Bridge_R matrix value is negative/nonfinite at {src}->{tgt}: {weight}")
                clean_row[tgt] = weight
            extra_targets = sorted(set(row) - set(target_categories))
            if extra_targets:
                raise RaceBridgeValidationError(f"Bridge_R matrix row has undeclared target categories at {src}: {extra_targets}")
            total = sum(clean_row.values())
            if abs(total - 1.0) > 1e-6:
                raise RaceBridgeValidationError(f"Bridge_R matrix row must sum to 1.0: source={src} sum={total}")
            normalized[src] = clean_row

        return RaceBridgePrior(
            bridge_id=bridge_id,
            mode=mode,
            source_axis=source_axis,
            target_axis=target_axis,
            source_categories=source_categories,
            target_categories=target_categories,
            matrix=normalized,
            sensitivity_width=sensitivity_width,
            metadata=metadata,
            prior_hash=prior_hash or content_hash(payload),
        )


    def summarize_sim_admin_race_counts(
        sim_events_path: str | Path,
        *,
        municipality_cod6: str | None = None,
        year: int | None = None,
    ) -> RaceBridgeCounts:
        df = pl.read_parquet(sim_events_path)
        if municipality_cod6 is not None and "mun_residence_cod6" in df.columns:
            df = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
        if year is not None and "year" in df.columns:
            df = df.filter(pl.col("year") == int(year))

        support = {
            "time": {"years": sorted(int(x) for x in df["year"].drop_nulls().unique().to_list()) if "year" in df.columns else []},
            "geography": {
                "municipality_cod6": sorted(str(x) for x in df["mun_residence_cod6"].drop_nulls().unique().to_list())
                if "mun_residence_cod6" in df.columns else []
            },
            "n_events": int(df.height),
        }

        raw_counts = {code: 0 for code in ADMIN_RACE_LABELS}
        missing = 0
        if "race_color_admin" not in df.columns:
            return RaceBridgeCounts(raw_admin_counts=raw_counts, missing_count=int(df.height), total_count=int(df.height), support=support)

        states = df["race_missingness_state"].to_list() if "race_missingness_state" in df.columns else [None] * df.height
        for code_value, state in zip(df["race_color_admin"].to_list(), states, strict=False):
            code = _canonical_code(code_value)
            if code in raw_counts and state == "valid_admin_race":
                raw_counts[code] += 1
            else:
                missing += 1
        return RaceBridgeCounts(raw_admin_counts=raw_counts, missing_count=missing, total_count=int(df.height), support=support)


    def fixedc_dynamic_weight_bridge(counts: RaceBridgeCounts, prior: RaceBridgePrior) -> RaceBridgePosterior:
        for source in counts.raw_admin_counts:
            if source not in prior.source_categories:
                raise RaceBridgeValidationError(f"Raw administrative race category not supported by prior: {source}")

        posterior = {target: 0.0 for target in prior.target_categories}
        for source, n in counts.raw_admin_counts.items():
            row = prior.matrix[source]
            for target, weight in row.items():
                posterior[target] += float(n) * weight

        # Width is deliberately conservative: explicit prior width plus unallocated missing-race share.
        width = min(1.0, max(prior.sensitivity_width, counts.missing_share))
        lower = {target: max(0.0, value * (1.0 - width)) for target, value in posterior.items()}
        upper = {target: value * (1.0 + width) for target, value in posterior.items()}
        mean = sum(posterior.values()) / len(posterior) if posterior else 0.0
        variance = sum((value - mean) ** 2 for value in posterior.values()) / len(posterior) if posterior else 0.0
        cv = math.sqrt(variance) / mean if mean > 0 else 0.0

        return RaceBridgePosterior(
            posterior_counts=posterior,
            lower_counts=lower,
            upper_counts=upper,
            raw_admin_counts=dict(counts.raw_admin_counts),
            missing_count=counts.missing_count,
            missing_share=counts.missing_share,
            sensitivity_width=width,
            race_bridge_cv=cv,
            prior=prior,
            support=counts.support,
        )
    ''')

    write("src/pegasus/output/race_bridge_attach.py", r'''
    from __future__ import annotations

    import json
    import platform
    import sys
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import sha256_file, sha256_text
    from pegasus.efg.race_bridge import (
        ADMIN_RACE_LABELS,
        RaceBridgePosterior,
        fixedc_dynamic_weight_bridge,
        load_race_bridge_prior,
        summarize_sim_admin_race_counts,
    )
    from pegasus.output.validate import validate_output_bundle


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


    def _append_rows(path: Path, rows: list[dict[str, Any]], *, id_column: str | None = None) -> None:
        if not rows:
            return
        existing = pl.read_parquet(path)
        if id_column and id_column in existing.columns:
            ids = {str(row[id_column]) for row in rows if row.get(id_column) is not None}
            if ids:
                existing = existing.filter(~pl.col(id_column).cast(pl.Utf8).is_in(sorted(ids)))
        new = pl.DataFrame(rows)
        for column in existing.columns:
            if column not in new.columns:
                new = new.with_columns(pl.lit(None).alias(column))
        for column in new.columns:
            if column not in existing.columns:
                new = new.drop(column)
        new = new.select(existing.columns)
        pl.concat([existing, new], how="vertical_relaxed").write_parquet(path)


    def _field_row(
        *,
        field_id: str,
        name: str,
        kind: str,
        carrier: str,
        unit: str,
        aggregation: str,
        support: dict[str, Any],
        axes: dict[str, Any],
        operator: str,
        provenance: list[str],
        warnings: list[str],
        state: str,
        dashboard_safe: bool,
        path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "field_id": field_id,
            "name": name,
            "kind": kind,
            "carrier": carrier,
            "unit": unit,
            "aggregation": aggregation,
            "role": _json(["observer", "demographic", "race_bridge"]),
            "source": _json(["SIM-DO", "Bridge_R"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": operator,
            "provenance": _json(provenance),
            "state": state,
            "dashboard_safe": "True" if dashboard_safe else "False",
            "warnings": _json(warnings),
            "lineage_hash": sha256_text(_json({"field_id": field_id, "support": support, "axes": axes, "operator": operator})),
            "registry_hash": axes.get("prior_hash") or "race_bridge_registry_unset",
            "materialization_state": "materialized",
            "path": path,
        }


    def _q_row(*, field: dict[str, Any], n_events: float | int | None, warnings: list[str], missingness: float, denom_fragility: float) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "n_events": float(n_events) if n_events is not None else None,
            "n_denom": None,
            "n_eff": float(n_events) if n_events is not None else None,
            "cov_S": None,
            "cov_T": None,
            "missingness": float(missingness),
            "zero_inflation": 0.0,
            "denom_fragility": float(denom_fragility),
            "provenance_risk": 0.65 if field["state"] != "verified" else 0.25,
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "warnings": _json(warnings),
            "computed_at": _now(),
            "q_schema_version": "v1",
        }


    def _vd_row(*, field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "display_name": field["name"],
            "technical_name": field["field_id"],
            "definition": definition,
            "estimand_label": estimand,
            "source_systems": _json(["SIM-DO", "Bridge_R"]),
            "carrier": field["carrier"],
            "unit": field["unit"],
            "support_description": "SIM death-event support after compile smoke municipality/time filtering.",
            "axis_description": "Administrative SIM race/color axis is preserved; Bridge_R emits posterior observer fields on the declared target axis.",
            "provenance_description": "Derived from normalized SIM administrative race counts and a validated fixed-C emission prior object.",
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "interpretation_warning": warning,
        }


    def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": edge_id,
            "parent_field_id": parent,
            "child_field_id": child,
            "operator": operator,
            "operator_params_json": _json(params),
            "registry_versions_json": _json({"Bridge_R": params.get("emission_matrix_registry_version", "unset")}),
            "created_at": _now(),
        }


    def _warning_rows(posterior: RaceBridgePosterior) -> list[dict[str, Any]]:
        metadata = posterior.metadata()
        return [
            {
                "warning_id": "race_bridge_admin_axis_preserved",
                "field_id": "run",
                "severity": "warning",
                "message": "Raw SIM administrative race/color counts are preserved and not overwritten by Bridge_R posterior fields.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_bayesian_ecological_warning",
                "field_id": "run",
                "severity": "warning",
                "message": metadata["bayesian_ecological_bridge_warning"],
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_sensitivity_metadata",
                "field_id": "run",
                "severity": "info",
                "message": _json({
                    "missing_race_share": posterior.missing_share,
                    "race_bridge_cv": posterior.race_bridge_cv,
                    "sensitivity_width": posterior.sensitivity_width,
                    "prior_hash": posterior.prior.prior_hash,
                }),
                "created_at": _now(),
                "inherited_from": None,
            },
        ]


    def _build_rows(posterior: RaceBridgePosterior, *, sim_events_path: Path, bridge_prior_path: Path) -> dict[str, list[dict[str, Any]]]:
        support = {
            **posterior.support,
            "n_missing_race": posterior.missing_count,
            "missing_race_share": posterior.missing_share,
        }
        metadata = posterior.metadata()
        common_warnings = [
            "race_bridge_admin_axis_preserved",
            "race_bridge_bayesian_ecological_warning",
            "race_bridge_sensitivity_metadata",
        ]
        fields: list[dict[str, Any]] = []
        q_rows: list[dict[str, Any]] = []
        vd_rows: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        table_path = "Tables/race_bridge_summary.parquet"
        for code, label in ADMIN_RACE_LABELS.items():
            field = _field_row(
                field_id=f"SIMRaceAdminRawCount_{code}",
                name=f"SIM Raw Administrative Race Count {label}",
                kind="extensive_measure",
                carrier="death_event",
                unit="deaths",
                aggregation="additive_count",
                support={**support, "race_color_admin": code, "n_events": posterior.raw_admin_counts.get(code, 0)},
                axes={"race_axis": posterior.prior.source_axis, "race_admin_code": code, "race_admin_label": label, "raw_admin_preserved": True},
                operator="raw_admin_race_count",
                provenance=["SIM-DO", str(sim_events_path)],
                warnings=["race_bridge_admin_axis_preserved"],
                state="verified",
                dashboard_safe=True,
                path=table_path,
            )
            fields.append(field)
            q_rows.append(_q_row(field=field, n_events=posterior.raw_admin_counts.get(code, 0), warnings=["race_bridge_admin_axis_preserved"], missingness=0.0, denom_fragility=0.0))
            vd_rows.append(_vd_row(field=field, definition="Raw SIM administrative race/color death count preserved before Bridge_R.", estimand="raw_administrative_race_count", warning="Raw administrative race axis; not equivalent to IBGE self-declared race."))

        missing_field = _field_row(
            field_id="SIMRaceBridgeMissingRaceObserver",
            name="SIM Missing Administrative Race Observer",
            kind="observer_proxy",
            carrier="death_event",
            unit="deaths",
            aggregation="missingness_count",
            support={**support, "n_events": posterior.missing_count},
            axes={"race_axis": posterior.prior.source_axis, "missing_category_preserved": True, **metadata},
            operator="missing_race_observer",
            provenance=["SIM-DO", str(sim_events_path)],
            warnings=common_warnings,
            state="fragile" if posterior.missing_count else "verified",
            dashboard_safe=False if posterior.missing_count else True,
            path=table_path,
        )
        fields.append(missing_field)
        q_rows.append(_q_row(field=missing_field, n_events=posterior.missing_count, warnings=common_warnings, missingness=posterior.missing_share, denom_fragility=posterior.sensitivity_width))
        vd_rows.append(_vd_row(field=missing_field, definition="Observer field for SIM records with missing, ignored, sentinel, or invalid administrative race/color.", estimand="missing_race_observer", warning="Missing race is preserved and is not imputed into posterior target categories."))

        for target, value in posterior.posterior_counts.items():
            field_id = f"SIMRaceBridgePosteriorCount_{target}"
            axes = {
                "race_axis": posterior.prior.target_axis,
                "race_target_category": target,
                "raw_admin_fields": [f"SIMRaceAdminRawCount_{code}" for code in posterior.raw_admin_counts],
                **metadata,
                "lower_count": posterior.lower_counts[target],
                "upper_count": posterior.upper_counts[target],
            }
            state = "fragile" if posterior.sensitivity_width > 0.05 or posterior.missing_share > 0 else "verified"
            field = _field_row(
                field_id=field_id,
                name=f"SIM Race Bridge Posterior Count {target}",
                kind="bridge_module",
                carrier="death_event",
                unit="deaths_posterior",
                aggregation="posterior_count_from_fixedC",
                support={**support, "n_events": value},
                axes=axes,
                operator="Bridge_R_fixedC_dynamic_weight",
                provenance=["SIM-DO", "Bridge_R", str(sim_events_path), str(bridge_prior_path)],
                warnings=common_warnings,
                state=state,
                dashboard_safe=False if state != "verified" else True,
                path=table_path,
            )
            fields.append(field)
            q_rows.append(_q_row(field=field, n_events=value, warnings=common_warnings, missingness=posterior.missing_share, denom_fragility=posterior.sensitivity_width))
            vd_rows.append(_vd_row(field=field, definition="Bridge_R posterior target-axis death count from raw SIM administrative race counts and a validated fixed-C emission prior.", estimand="race_bridge_posterior_count", warning="Bridge-derived observer field with sensitivity interval; not a direct measurement."))
            for source_code in posterior.raw_admin_counts:
                edges.append(_edge(
                    f"edge_SIMRaceAdminRawCount_{source_code}_to_{field_id}",
                    f"SIMRaceAdminRawCount_{source_code}",
                    field_id,
                    "Bridge_R_fixedC_dynamic_weight",
                    metadata,
                ))

        return {"fields": fields, "q_rows": q_rows, "vd_rows": vd_rows, "edges": edges, "warnings": _warning_rows(posterior)}


    def attach_race_bridge_to_run(
        *,
        run_dir: str | Path,
        sim_events_path: str | Path,
        bridge_prior_path: str | Path,
        municipality_cod6: str | None = None,
    ) -> Path:
        run_dir = Path(run_dir)
        sim_events_path = Path(sim_events_path)
        bridge_prior_path = Path(bridge_prior_path)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        if not sim_events_path.exists():
            raise FileNotFoundError(f"SIM events parquet does not exist: {sim_events_path}")
        if not bridge_prior_path.exists():
            raise FileNotFoundError(f"Bridge prior JSON does not exist: {bridge_prior_path}")

        prior = load_race_bridge_prior(bridge_prior_path)
        counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6)
        posterior = fixedc_dynamic_weight_bridge(counts, prior)
        rows = _build_rows(posterior, sim_events_path=sim_events_path, bridge_prior_path=bridge_prior_path)

        summary_rows = []
        for target in prior.target_categories:
            summary_rows.append({
                "race_target_category": target,
                "posterior_count": posterior.posterior_counts[target],
                "lower_count": posterior.lower_counts[target],
                "upper_count": posterior.upper_counts[target],
                "sensitivity_width": posterior.sensitivity_width,
                "race_bridge_cv": posterior.race_bridge_cv,
                "missing_race_share": posterior.missing_share,
            })
        summary_path = run_dir / "Tables" / "race_bridge_summary.parquet"
        pl.DataFrame(summary_rows).write_parquet(summary_path)

        _append_rows(run_dir / "V_fields.parquet", rows["fields"], id_column="field_id")
        _append_rows(run_dir / "Q_tensor.parquet", rows["q_rows"], id_column="field_id")
        _append_rows(run_dir / "VariableDictionary.parquet", rows["vd_rows"], id_column="field_id")
        _append_rows(run_dir / "E_DAG.parquet", rows["edges"], id_column="edge_id")
        _append_rows(run_dir / "Warnings.parquet", rows["warnings"], id_column="warning_id")

        run_config_path = run_dir / "RunConfig.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
        run_config["race_bridge"] = {
            "bridge_id": prior.bridge_id,
            "mode": prior.mode,
            "prior_hash": prior.prior_hash,
            "source_axis": prior.source_axis,
            "target_axis": prior.target_axis,
            "missing_race_share": posterior.missing_share,
            "sensitivity_width": posterior.sensitivity_width,
            "race_bridge_cv": posterior.race_bridge_cv,
            "raw_admin_counts_preserved": True,
            "missing_category_preserved": True,
        }
        source_hashes = dict(run_config.get("source_hashes") or {})
        source_hashes["race_bridge_prior"] = sha256_file(bridge_prior_path)
        source_hashes["race_bridge_summary"] = sha256_file(summary_path)
        run_config["source_hashes"] = source_hashes
        run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        manifest_path = run_dir / "ReproducibilityManifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("source_hashes", {}).update({
                "race_bridge_prior": sha256_file(bridge_prior_path),
                "race_bridge_summary": sha256_file(summary_path),
            })
            manifest["race_bridge"] = run_config["race_bridge"]
            manifest.setdefault("environment", {}).update({"python": sys.version.split()[0], "platform": platform.platform()})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        p_path = run_dir / "P_vector.json"
        if p_path.exists():
            p = json.loads(p_path.read_text(encoding="utf-8"))
            if isinstance(p, dict):
                p["race_bridge"] = run_config["race_bridge"]
                p.setdefault("source_hashes", {}).update({"race_bridge_prior": sha256_file(bridge_prior_path)})
                p_path.write_text(json.dumps(p, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            raise RuntimeError("Race bridge attachment produced invalid output bundle: " + "; ".join(validation.errors))
        return run_dir
    ''')

    write("src/pegasus/workflows/race_bridge.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.output.race_bridge_attach import attach_race_bridge_to_run
    from pegasus.output.validate import validate_output_bundle


    def run_attach_race_bridge(
        *,
        run_dir: str | Path,
        sim_events_path: str | Path,
        bridge_prior_path: str | Path,
        municipality_cod6: str | None = None,
    ) -> dict[str, Any]:
        output = attach_race_bridge_to_run(
            run_dir=run_dir,
            sim_events_path=sim_events_path,
            bridge_prior_path=bridge_prior_path,
            municipality_cod6=municipality_cod6,
        )
        validation = validate_output_bundle(run_dir=str(output))
        return {"run_dir": output, "validation": validation}
    ''')

    write("tests/fixtures/race_bridge/fixedC_valid.json", r'''
    {
      "bridge_id": "fixedC_sim_admin_to_ibge_selfdeclared_smoke_v1",
      "mode": "fixedC_dynamic_weight",
      "source_axis": "SIM_ADMIN_RACACOR",
      "target_axis": "IBGE_SELF_DECLARED_RACE",
      "source_categories": ["1", "2", "3", "4", "5"],
      "target_categories": ["branca", "preta", "amarela", "parda", "indigena"],
      "sensitivity_width": 0.08,
      "metadata": {
        "calibration_scope": "synthetic_smoke_fixture",
        "epistemic_status": "validation_fixture_only",
        "warning": "Not a real empirical emission matrix."
      },
      "matrix": {
        "1": {"branca": 0.96, "preta": 0.01, "amarela": 0.01, "parda": 0.02, "indigena": 0.00},
        "2": {"branca": 0.02, "preta": 0.88, "amarela": 0.00, "parda": 0.10, "indigena": 0.00},
        "3": {"branca": 0.05, "preta": 0.00, "amarela": 0.90, "parda": 0.05, "indigena": 0.00},
        "4": {"branca": 0.08, "preta": 0.10, "amarela": 0.01, "parda": 0.80, "indigena": 0.01},
        "5": {"branca": 0.01, "preta": 0.00, "amarela": 0.00, "parda": 0.04, "indigena": 0.95}
      }
    }
    ''')

    write("tests/fixtures/race_bridge/fixedC_invalid_empty.json", r'''
    {
      "bridge_id": "invalid_empty",
      "mode": "fixedC_dynamic_weight",
      "source_axis": "SIM_ADMIN_RACACOR",
      "target_axis": "IBGE_SELF_DECLARED_RACE",
      "source_categories": [],
      "target_categories": [],
      "matrix": {}
    }
    ''')

    write("tests/fixtures/race_bridge/fixedC_invalid_rowsum.json", r'''
    {
      "bridge_id": "invalid_rowsum",
      "mode": "fixedC_dynamic_weight",
      "source_axis": "SIM_ADMIN_RACACOR",
      "target_axis": "IBGE_SELF_DECLARED_RACE",
      "source_categories": ["1"],
      "target_categories": ["branca", "preta"],
      "sensitivity_width": 0.1,
      "matrix": {
        "1": {"branca": 0.6, "preta": 0.6}
      }
    }
    ''')

    write("tests/unit/test_race_bridge_fixedc.py", r'''
    import json
    from pathlib import Path

    import pytest

    from pegasus.efg.race_bridge import (
        RaceBridgeCounts,
        RaceBridgeValidationError,
        fixedc_dynamic_weight_bridge,
        load_race_bridge_prior,
    )


    def test_valid_fixedc_prior_bridges_counts_and_preserves_missing():
        prior = load_race_bridge_prior("tests/fixtures/race_bridge/fixedC_valid.json")
        counts = RaceBridgeCounts(
            raw_admin_counts={"1": 2, "2": 1, "3": 0, "4": 3, "5": 0},
            missing_count=1,
            total_count=7,
            support={"time": {"years": [2022]}, "geography": {"municipality_cod6": ["270430"]}, "n_events": 7},
        )
        posterior = fixedc_dynamic_weight_bridge(counts, prior)
        assert posterior.missing_count == 1
        assert posterior.missing_share == pytest.approx(1 / 7)
        assert posterior.raw_admin_counts["4"] == 3
        assert posterior.posterior_counts["parda"] > posterior.posterior_counts["preta"]
        assert posterior.sensitivity_width >= posterior.missing_share
        assert posterior.metadata()["bridge_mode"] == "fixedC_dynamic_weight"


    def test_empty_prior_blocks_posterior_output():
        with pytest.raises(RaceBridgeValidationError):
            load_race_bridge_prior("tests/fixtures/race_bridge/fixedC_invalid_empty.json")


    def test_prior_rows_must_sum_to_one():
        with pytest.raises(RaceBridgeValidationError):
            load_race_bridge_prior("tests/fixtures/race_bridge/fixedC_invalid_rowsum.json")
    ''')

    write("tests/integration/test_slice4a_race_bridge_integration.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.compile import run_compile
    from pegasus.workflows.report.race_bridge import run_attach_race_bridge


    def test_slice4a_attach_race_bridge_to_compile_run(tmp_path: Path):
        data_root = tmp_path / "data"
        run_dir = tmp_path / "run"
        compile_result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=run_dir,
            data_root=data_root,
        )
        assert compile_result["validation"].ok, compile_result["validation"].errors

        sim_events = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        result = run_attach_race_bridge(
            run_dir=run_dir,
            sim_events_path=sim_events,
            bridge_prior_path="tests/fixtures/race_bridge/fixedC_valid.json",
            municipality_cod6="270430",
        )
        assert result["validation"].ok, result["validation"].errors
        assert validate_output_bundle(run_dir=str(run_dir)).ok

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        field_ids = set(v["field_id"].to_list())
        assert "SIMRaceAdminRawCount_4" in field_ids
        assert "SIMRaceBridgeMissingRaceObserver" in field_ids
        assert "SIMRaceBridgePosteriorCount_parda" in field_ids
        assert "SIMRaceBridgePosteriorCount_branca" in field_ids

        posterior = v.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
        assert "Bridge_R_fixedC_dynamic_weight" in posterior["operator"]
        assert "missing_race_share" in posterior["axes_json"]
        assert "bayesian_ecological_bridge_warning" in posterior["axes_json"]

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        parda_q = q.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
        assert parda_q["state"] in {"fragile", "verified"}

        table = pl.read_parquet(run_dir / "Tables" / "race_bridge_summary.parquet")
        assert table.height == 5
        assert set(table["race_target_category"].to_list()) == {"branca", "preta", "amarela", "parda", "indigena"}
    ''')

    write("scripts/dev/audits/audit_slice4a_race_bridge.py", r'''
    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


    def _json_cell(value):
        if isinstance(value, (dict, list)):
            return value
        return json.loads(str(value))


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        args = parser.parse_args()
        run = Path(args.run)
        failures: list[dict] = []

        validation = validate_output_bundle(run_dir=str(run))
        if not validation.ok:
            failures.append({"kind": "validate_output_bundle", "errors": validation.errors})

        v = pl.read_parquet(run / "V_fields.parquet")
        field_ids = set(v["field_id"].to_list())
        required = {
            "SIMRaceAdminRawCount_1",
            "SIMRaceAdminRawCount_2",
            "SIMRaceAdminRawCount_3",
            "SIMRaceAdminRawCount_4",
            "SIMRaceAdminRawCount_5",
            "SIMRaceBridgeMissingRaceObserver",
            "SIMRaceBridgePosteriorCount_branca",
            "SIMRaceBridgePosteriorCount_preta",
            "SIMRaceBridgePosteriorCount_amarela",
            "SIMRaceBridgePosteriorCount_parda",
            "SIMRaceBridgePosteriorCount_indigena",
        }
        missing = sorted(required - field_ids)
        if missing:
            failures.append({"kind": "missing_race_bridge_fields", "missing": missing})

        for row in v.filter(pl.col("field_id").str.starts_with("SIMRaceBridgePosteriorCount_")).to_dicts():
            axes = _json_cell(row["axes_json"])
            required_metadata = [
                "numerator_axis_source",
                "denominator_axis_target",
                "bridge_operator",
                "emission_matrix_registry_version",
                "bridge_mode",
                "missing_race_share",
                "race_bridge_cv",
                "sensitivity_width",
                "race_axis_warning",
                "bayesian_ecological_bridge_warning",
                "prior_hash",
                "lower_count",
                "upper_count",
            ]
            absent = [key for key in required_metadata if key not in axes]
            if absent:
                failures.append({"kind": "posterior_missing_bridge_metadata", "field_id": row["field_id"], "missing": absent})
            if row["dashboard_safe"] == "True" and float(axes.get("sensitivity_width", 0.0)) > 0.05:
                failures.append({"kind": "posterior_dashboard_safety_not_downgraded", "field_id": row["field_id"], "sensitivity_width": axes.get("sensitivity_width")})

        warnings = pl.read_parquet(run / "Warnings.parquet")
        warning_text = "\n".join(str(x) for x in warnings["message"].to_list()) if "message" in warnings.columns else ""
        if "Raw SIM administrative race/color counts are preserved" not in warning_text:
            failures.append({"kind": "missing_raw_admin_preservation_warning"})
        if "Posterior race counts are bridge-derived observer fields" not in warning_text:
            failures.append({"kind": "missing_bayesian_bridge_warning"})

        q = pl.read_parquet(run / "Q_tensor.parquet")
        q_ids = set(q["field_id"].to_list())
        q_missing = sorted(required - q_ids)
        if q_missing:
            failures.append({"kind": "q_tensor_missing_race_bridge_fields", "missing": q_missing})

        table_path = run / "Tables" / "race_bridge_summary.parquet"
        if not table_path.exists():
            failures.append({"kind": "missing_race_bridge_summary_table", "path": str(table_path)})
        else:
            table = pl.read_parquet(table_path)
            if table.height != 5:
                failures.append({"kind": "invalid_race_bridge_summary_rows", "rows": table.height})
            for column in ["posterior_count", "lower_count", "upper_count", "sensitivity_width", "race_bridge_cv", "missing_race_share"]:
                if column not in table.columns:
                    failures.append({"kind": "missing_race_bridge_summary_column", "column": column})

        run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
        bridge = run_config.get("race_bridge")
        if not isinstance(bridge, dict):
            failures.append({"kind": "missing_run_config_race_bridge"})
        else:
            for key in ["bridge_id", "mode", "prior_hash", "missing_race_share", "sensitivity_width", "raw_admin_counts_preserved", "missing_category_preserved"]:
                if key not in bridge:
                    failures.append({"kind": "run_config_race_bridge_missing_key", "key": key})
            if bridge.get("raw_admin_counts_preserved") is not True:
                failures.append({"kind": "raw_admin_counts_not_marked_preserved"})
            if bridge.get("missing_category_preserved") is not True:
                failures.append({"kind": "missing_category_not_marked_preserved"})

        if failures:
            fail({"status": "failed", "failures": failures})
        print("AUDIT PASSED: Slice 4A race bridge preserves raw admin counts, missing race observer, posterior metadata, sensitivity intervals, and output validation.")


    if __name__ == "__main__":
        main()
    ''')

    patch_cli()

    print("Applied Slice 4A race bridge foundation.")
    print("Run the validation commands supplied by the assistant.")


if __name__ == "__main__":
    main()
