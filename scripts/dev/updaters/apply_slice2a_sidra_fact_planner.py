from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/sidra/schemas.py", r'''
    from __future__ import annotations

    from typing import Literal

    from pydantic import BaseModel, ConfigDict, Field


    class SIDRARequest(BaseModel):
        model_config = ConfigDict(extra="forbid")

        table_id: str
        variables: list[str]
        periods: list[str]
        locality_level: str
        localities: list[str]
        classifications: dict[str, list[str]] = Field(default_factory=dict)


    class SIDRATableMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid")

        table_id: str
        name: str
        variables: list[str]
        periods: list[str]
        locality_levels: list[str]
        localities_by_level: dict[str, list[str]]
        classifications: dict[str, list[str]] = Field(default_factory=dict)
        units_by_variable: dict[str, str | None] = Field(default_factory=dict)


    class SIDRAMetadata(BaseModel):
        model_config = ConfigDict(extra="forbid")

        tables: dict[str, SIDRATableMetadata]


    class SIDRAChunk(BaseModel):
        model_config = ConfigDict(extra="forbid")

        chunk_id: str
        table_id: str
        variables: list[str]
        periods: list[str]
        locality_level: str
        localities: list[str]
        classifications: dict[str, list[str]]
        estimated_cells: int
        request_url: str
        request_params: dict


    class SIDRAFactRow(BaseModel):
        model_config = ConfigDict(extra="forbid")

        table_id: str
        variable_id: str
        period: str

        locality_level: str
        locality_id: str

        classification_tuple: tuple[tuple[str, str], ...]
        category_tuple: tuple[tuple[str, str], ...]

        value_raw: str | None
        value_numeric: float | None
        value_status: Literal[
            "numeric",
            "blank",
            "dash_zero_or_nil",
            "not_available",
            "suppressed_or_unidentified",
            "non_numeric_symbol",
            "header_row",
            "parse_error",
        ]

        unit: str | None
        request_hash: str
        metadata_hash: str
        fetched_at: str


    class DenominatorContract(BaseModel):
        model_config = ConfigDict(extra="forbid")

        mode: Literal[
            "official_sidra_anchor",
            "imported_fixture",
            "independent_population_tensor",
            "sim_informed_population_tensor",
            "synthetic_test_fixture",
            "blocked_missing",
        ]
        source: str
        provenance: list[str]
        state: str
        dashboard_safe: bool | Literal["warning"]
        allowed_for_rates: bool
        warnings: list[str]


    class ProjectionResult(BaseModel):
        model_config = ConfigDict(extra="forbid")

        status: Literal["projected", "blocked", "failed"]
        projection_matrix_id: str | None
        warnings: list[str]
        reason: str | None = None


    class BoundedPushforwardResult(BaseModel):
        model_config = ConfigDict(extra="forbid")

        status: Literal["bounded", "not_required", "blocked", "failed"]
        axes_kept: list[str]
        axes_dropped: list[str]
        warnings: list[str]
        reason: str | None = None
    ''')

    write("src/pegasus/sidra/plan.py", r'''
    from __future__ import annotations

    from dataclasses import replace
    from typing import Any

    from pegasus.core.hashing import content_hash
    from pegasus.sidra.schemas import SIDRAChunk, SIDRAMetadata, SIDRARequest


    DEFAULT_BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"


    def estimate_cells(
        *,
        localities: list[str],
        periods: list[str],
        variables: list[str],
        classifications: dict[str, list[str]],
    ) -> int:
        cells = len(localities) * len(periods) * len(variables)
        for categories in classifications.values():
            cells *= max(1, len(categories))
        return cells


    def _chunk_id(spec: dict[str, Any]) -> str:
        return content_hash(spec)


    def _request_url(base_url: str, table_id: str) -> str:
        return f"{base_url.rstrip('/')}/{table_id}/periodos/{{periods}}/variaveis/{{variables}}"


    def _make_chunk(
        request: SIDRARequest,
        *,
        localities: list[str],
        periods: list[str],
        variables: list[str],
        classifications: dict[str, list[str]],
        base_url: str,
    ) -> SIDRAChunk:
        estimated = estimate_cells(
            localities=localities,
            periods=periods,
            variables=variables,
            classifications=classifications,
        )
        params = {
            "table_id": request.table_id,
            "variables": variables,
            "periods": periods,
            "locality_level": request.locality_level,
            "localities": localities,
            "classifications": classifications,
        }
        return SIDRAChunk(
            chunk_id=_chunk_id(params),
            table_id=request.table_id,
            variables=variables,
            periods=periods,
            locality_level=request.locality_level,
            localities=localities,
            classifications=classifications,
            estimated_cells=estimated,
            request_url=_request_url(base_url, request.table_id),
            request_params=params,
        )


    def _split_list(values: list[str]) -> tuple[list[str], list[str]]:
        if len(values) <= 1:
            raise ValueError("Cannot split singleton list.")
        mid = max(1, len(values) // 2)
        return values[:mid], values[mid:]


    def _largest_classification(classifications: dict[str, list[str]]) -> str | None:
        splittable = {k: v for k, v in classifications.items() if len(v) > 1}
        if not splittable:
            return None
        return sorted(splittable, key=lambda k: (-len(splittable[k]), k))[0]


    def _split_request(request: SIDRARequest) -> list[SIDRARequest]:
        # TDD order: localities, highest-cardinality classification, periods, variables.
        if len(request.localities) > 1:
            left, right = _split_list(request.localities)
            return [
                request.model_copy(update={"localities": left}),
                request.model_copy(update={"localities": right}),
            ]

        cls = _largest_classification(request.classifications)
        if cls is not None:
            left, right = _split_list(request.classifications[cls])
            left_cls = dict(request.classifications)
            right_cls = dict(request.classifications)
            left_cls[cls] = left
            right_cls[cls] = right
            return [
                request.model_copy(update={"classifications": left_cls}),
                request.model_copy(update={"classifications": right_cls}),
            ]

        if len(request.periods) > 1:
            left, right = _split_list(request.periods)
            return [
                request.model_copy(update={"periods": left}),
                request.model_copy(update={"periods": right}),
            ]

        if len(request.variables) > 1:
            left, right = _split_list(request.variables)
            return [
                request.model_copy(update={"variables": left}),
                request.model_copy(update={"variables": right}),
            ]

        raise ValueError("Cannot split SIDRA request below cell ceiling.")


    def validate_request_against_metadata(
        request: SIDRARequest,
        metadata: SIDRAMetadata,
    ) -> list[str]:
        errors: list[str] = []
        table = metadata.tables.get(request.table_id)
        if table is None:
            return [f"Unknown SIDRA table: {request.table_id}"]

        for variable in request.variables:
            if variable not in table.variables:
                errors.append(f"Unknown variable for table {request.table_id}: {variable}")

        for period in request.periods:
            if period not in table.periods:
                errors.append(f"Unknown period for table {request.table_id}: {period}")

        if request.locality_level not in table.locality_levels:
            errors.append(f"Unknown locality level for table {request.table_id}: {request.locality_level}")
        else:
            allowed_locs = set(table.localities_by_level.get(request.locality_level, []))
            for locality in request.localities:
                if locality not in allowed_locs:
                    errors.append(f"Unknown locality {locality} at level {request.locality_level}")

        for cls_id, categories in request.classifications.items():
            if cls_id not in table.classifications:
                errors.append(f"Unknown classification for table {request.table_id}: {cls_id}")
                continue
            allowed = set(table.classifications[cls_id])
            for category in categories:
                if category not in allowed:
                    errors.append(f"Unknown category {category} for classification {cls_id}")

        return errors


    def plan_sidra_chunks(
        request: SIDRARequest,
        metadata: SIDRAMetadata,
        *,
        max_cells_per_request: int = 49900,
        base_url: str = DEFAULT_BASE_URL,
    ) -> list[SIDRAChunk]:
        errors = validate_request_against_metadata(request, metadata)
        if errors:
            raise ValueError("; ".join(errors))

        pending = [request]
        chunks: list[SIDRAChunk] = []

        while pending:
            current = pending.pop(0)
            cells = estimate_cells(
                localities=current.localities,
                periods=current.periods,
                variables=current.variables,
                classifications=current.classifications,
            )

            if cells <= max_cells_per_request:
                chunks.append(
                    _make_chunk(
                        current,
                        localities=current.localities,
                        periods=current.periods,
                        variables=current.variables,
                        classifications=current.classifications,
                        base_url=base_url,
                    )
                )
                continue

            try:
                pending = _split_request(current) + pending
            except ValueError as exc:
                raise ValueError(
                    f"SIDRA request cannot be split below ceiling {max_cells_per_request}; "
                    f"minimum unsplittable cell count={cells}"
                ) from exc

        return chunks
    ''')

    write("src/pegasus/sidra/facts.py", r'''
    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash
    from pegasus.sidra.schemas import SIDRAChunk, SIDRAFactRow


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def classify_sidra_value(raw: Any) -> tuple[float | None, str]:
        if raw is None:
            return None, "blank"

        text = str(raw).strip()
        if text == "":
            return None, "blank"

        if text in {"-", "—", "–"}:
            return None, "dash_zero_or_nil"

        lowered = text.casefold()
        if lowered in {"x", "...", "na", "n/a"}:
            return None, "not_available"

        if lowered in {"..", "…"}:
            return None, "suppressed_or_unidentified"

        normalized = text.replace(".", "").replace(",", ".") if "," in text else text

        try:
            return float(normalized), "numeric"
        except ValueError:
            return None, "non_numeric_symbol"


    def _tuple_from_pairs(value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return tuple()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(tuple(map(str, x)) for x in value)
        if isinstance(value, dict):
            return tuple((str(k), str(v)) for k, v in sorted(value.items()))
        return tuple()


    def normalize_flat_records_to_facts(
        records: list[dict[str, Any]],
        *,
        table_id: str,
        request_hash: str,
        metadata_hash: str,
        unit_by_variable: dict[str, str | None] | None = None,
        fetched_at: str | None = None,
    ) -> list[SIDRAFactRow]:
        unit_by_variable = unit_by_variable or {}
        fetched_at = fetched_at or utc_now()
        facts: list[SIDRAFactRow] = []

        for record in records:
            if record.get("header_row") is True:
                status = "header_row"
                value_numeric = None
            else:
                value_numeric, status = classify_sidra_value(record.get("value"))

            variable_id = str(record.get("variable_id") or record.get("variable") or "")
            if not variable_id:
                continue

            facts.append(
                SIDRAFactRow(
                    table_id=str(record.get("table_id") or table_id),
                    variable_id=variable_id,
                    period=str(record.get("period") or ""),
                    locality_level=str(record.get("locality_level") or ""),
                    locality_id=str(record.get("locality_id") or ""),
                    classification_tuple=_tuple_from_pairs(record.get("classification_tuple")),
                    category_tuple=_tuple_from_pairs(record.get("category_tuple")),
                    value_raw=None if record.get("value") is None else str(record.get("value")),
                    value_numeric=value_numeric,
                    value_status=status,
                    unit=unit_by_variable.get(variable_id),
                    request_hash=request_hash,
                    metadata_hash=metadata_hash,
                    fetched_at=fetched_at,
                )
            )

        return facts


    def facts_to_frame(facts: list[SIDRAFactRow]) -> pl.DataFrame:
        rows = []
        for fact in facts:
            row = fact.model_dump(mode="json")
            row["classification_tuple"] = json.dumps(row["classification_tuple"], ensure_ascii=False)
            row["category_tuple"] = json.dumps(row["category_tuple"], ensure_ascii=False)
            rows.append(row)

        if not rows:
            return pl.DataFrame(
                schema={
                    "table_id": pl.Utf8,
                    "variable_id": pl.Utf8,
                    "period": pl.Utf8,
                    "locality_level": pl.Utf8,
                    "locality_id": pl.Utf8,
                    "classification_tuple": pl.Utf8,
                    "category_tuple": pl.Utf8,
                    "value_raw": pl.Utf8,
                    "value_numeric": pl.Float64,
                    "value_status": pl.Utf8,
                    "unit": pl.Utf8,
                    "request_hash": pl.Utf8,
                    "metadata_hash": pl.Utf8,
                    "fetched_at": pl.Utf8,
                }
            )
        return pl.DataFrame(rows)


    def write_facts_parquet(
        facts: list[SIDRAFactRow],
        *,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        facts_to_frame(facts).write_parquet(output_path)
        return output_path


    def normalize_fixture_json_to_facts(
        *,
        input_path: str | Path,
        output_path: str | Path,
        table_id: str,
        unit_by_variable: dict[str, str | None] | None = None,
    ) -> Path:
        input_path = Path(input_path)
        records = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("SIDRA fixture JSON must contain a list of flat records.")

        request_hash = content_hash({"fixture": str(input_path), "table_id": table_id})
        metadata_hash = content_hash({"fixture_metadata": table_id, "unit_by_variable": unit_by_variable or {}})

        facts = normalize_flat_records_to_facts(
            records,
            table_id=table_id,
            request_hash=request_hash,
            metadata_hash=metadata_hash,
            unit_by_variable=unit_by_variable,
        )
        return write_facts_parquet(facts, output_path=output_path)
    ''')

    write("src/pegasus/sidra/metadata.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.schemas import SIDRAMetadata, SIDRATableMetadata


    def load_table_seed(path: str | Path) -> list[dict]:
        rows: list[dict] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows


    def fixture_sidra_metadata() -> SIDRAMetadata:
        table = SIDRATableMetadata(
            table_id="9606",
            name="Population by municipality, period and classification fixture",
            variables=["93"],
            periods=["2022", "2023"],
            locality_levels=["N6"],
            localities_by_level={
                "N6": ["270030", "270430", "270770"],
            },
            classifications={
                "2": ["0", "4", "5"],
                "58": ["0", "1", "2"],
                "287": ["0", "93070", "93084", "100000"],
            },
            units_by_variable={"93": "persons"},
        )
        return SIDRAMetadata(tables={"9606": table})


    def write_normalized_metadata_tables(
        metadata: SIDRAMetadata,
        *,
        output_dir: str | Path,
    ) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables = []
        variables = []
        classifications = []
        categories = []
        periods = []
        localities = []

        for table in metadata.tables.values():
            tables.append({"table_id": table.table_id, "name": table.name})

            for variable in table.variables:
                variables.append(
                    {
                        "table_id": table.table_id,
                        "variable_id": variable,
                        "unit": table.units_by_variable.get(variable),
                    }
                )

            for period in table.periods:
                periods.append({"table_id": table.table_id, "period": period})

            for level, locs in table.localities_by_level.items():
                for loc in locs:
                    localities.append(
                        {
                            "table_id": table.table_id,
                            "locality_level": level,
                            "locality_id": loc,
                        }
                    )

            for cls_id, cats in table.classifications.items():
                classifications.append(
                    {
                        "table_id": table.table_id,
                        "classification_id": cls_id,
                    }
                )
                for cat in cats:
                    categories.append(
                        {
                            "table_id": table.table_id,
                            "classification_id": cls_id,
                            "category_id": cat,
                        }
                    )

        outputs = {
            "sidra_tables": output_dir / "sidra_tables.parquet",
            "sidra_variables": output_dir / "sidra_variables.parquet",
            "sidra_classifications": output_dir / "sidra_classifications.parquet",
            "sidra_categories": output_dir / "sidra_categories.parquet",
            "sidra_periods": output_dir / "sidra_periods.parquet",
            "sidra_localities": output_dir / "sidra_localities.parquet",
        }

        pl.DataFrame(tables).write_parquet(outputs["sidra_tables"])
        pl.DataFrame(variables).write_parquet(outputs["sidra_variables"])
        pl.DataFrame(classifications).write_parquet(outputs["sidra_classifications"])
        pl.DataFrame(categories).write_parquet(outputs["sidra_categories"])
        pl.DataFrame(periods).write_parquet(outputs["sidra_periods"])
        pl.DataFrame(localities).write_parquet(outputs["sidra_localities"])

        return outputs
    ''')

    write("src/pegasus/sidra/projection.py", r'''
    from __future__ import annotations

    from pegasus.sidra.schemas import ProjectionResult


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
    ''')

    write("src/pegasus/sidra/category_maps.py", r'''
    from __future__ import annotations

    from pegasus.sidra.schemas import BoundedPushforwardResult


    def bounded_pushforward_scaffold(
        *,
        raw_axes: list[str],
        demanded_axes: list[str],
        aggregation: str,
        high_dimensional: bool,
    ) -> BoundedPushforwardResult:
        raw = list(dict.fromkeys(raw_axes))
        demanded = list(dict.fromkeys(demanded_axes))
        drop = [axis for axis in raw if axis not in demanded]

        if not high_dimensional and not drop:
            return BoundedPushforwardResult(
                status="not_required",
                axes_kept=demanded,
                axes_dropped=[],
                warnings=[],
            )

        if aggregation not in {"additive", "compositional"}:
            return BoundedPushforwardResult(
                status="blocked",
                axes_kept=demanded,
                axes_dropped=drop,
                warnings=["bounded_pushforward_illegal_for_aggregation"],
                reason="High-dimensional bounding requires additive/compositional pushforward or denominator recovery.",
            )

        return BoundedPushforwardResult(
            status="bounded",
            axes_kept=demanded,
            axes_dropped=drop,
            warnings=["high_dimensional_bounded_pushforward"],
        )
    ''')

    write("src/pegasus/she/population/schema.py", r'''
    from __future__ import annotations

    from pegasus.sidra.schemas import DenominatorContract


    def official_sidra_anchor_contract(
        *,
        source: str = "SIDRA",
        warnings: list[str] | None = None,
    ) -> DenominatorContract:
        return DenominatorContract(
            mode="official_sidra_anchor",
            source=source,
            provenance=["official"],
            state="fragile",
            dashboard_safe="warning",
            allowed_for_rates=True,
            warnings=warnings or ["fixture_or_unvalidated_sidra_anchor"],
        )


    def blocked_missing_population_contract(
        *,
        reason: str,
    ) -> DenominatorContract:
        return DenominatorContract(
            mode="blocked_missing",
            source="none",
            provenance=[],
            state="illegal_excluded",
            dashboard_safe=False,
            allowed_for_rates=False,
            warnings=[reason],
        )
    ''')

    write("src/pegasus/cli.py", r'''
    from __future__ import annotations

    import importlib.util
    import json
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import (
        build_datasus_manifests,
        load_datasus_config,
        read_request_manifest,
        write_request_manifest,
    )
    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.datasus.profile import profile_table
    from pegasus.datasus.schema_compare import compare_profiles
    from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree
    from pegasus.sidra.facts import normalize_fixture_json_to_facts
    from pegasus.sidra.metadata import fixture_sidra_metadata, write_normalized_metadata_tables
    from pegasus.sidra.plan import plan_sidra_chunks
    from pegasus.sidra.schemas import SIDRARequest
    from pegasus.workflows.construct.build_efg import build_sim_fixture_efg_run

    app = typer.Typer(no_args_is_help=True)
    registries_app = typer.Typer(no_args_is_help=True)
    sidra_app = typer.Typer(no_args_is_help=True)
    datasus_app = typer.Typer(no_args_is_help=True)
    efg_app = typer.Typer(no_args_is_help=True)

    app.add_typer(registries_app, name="registries")
    app.add_typer(sidra_app, name="sidra")
    app.add_typer(datasus_app, name="datasus")
    app.add_typer(efg_app, name="efg")


    def _fail(errors: list[str]) -> None:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)


    @app.command()
    def init() -> None:
        ensure_data_lake(".")
        run_dir = create_empty_output_bundle(Path("data/runs/slice0_empty"))
        print(f"[green]initialized[/green] data lake and scaffold run: {run_dir}")


    @app.command("validate-config")
    def validate_config() -> None:
        errors = validate_config_tree(".")
        if errors:
            _fail(errors)
        print("[green]config valid[/green]")


    @registries_app.command("validate")
    def validate_registries() -> None:
        errors = validate_registry_tree("config/registries")
        if errors:
            _fail(errors)
        print("[green]registries valid[/green]")


    @app.command("validate-run")
    def validate_run(run: Path = typer.Option(..., "--run")) -> None:
        result = validate_output_bundle(run_dir=str(run))
        if not result.ok:
            _fail(result.errors)
        print("[green]run bundle valid[/green]")


    @app.command()
    def doctor() -> None:
        checks: dict[str, str] = {}
        checks["python"] = sys.version.split()[0]

        for mod in ["duckdb", "polars", "pyarrow", "pydantic", "typer", "yaml"]:
            checks[mod] = "ok" if importlib.util.find_spec(mod) else "missing"

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec:
            import torch

            checks["torch"] = getattr(torch, "__version__", "ok")
            checks["torch_cuda_available"] = str(torch.cuda.is_available())
            checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        else:
            checks["torch"] = "missing"
            checks["torch_cuda_available"] = "False"
            checks["cuda_device_name"] = "none"

        checks["Rscript"] = shutil.which("Rscript") or "missing"
        checks["microdatasus"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"
        checks["read.dbc"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"

        checks["data_write"] = "ok"
        try:
            ensure_data_lake(".")
        except Exception as exc:
            checks["data_write"] = f"failed: {exc}"

        reg_errors = validate_registry_tree("config/registries")
        checks["registry_schema"] = "ok" if not reg_errors else f"{len(reg_errors)} errors"

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is still light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


    @datasus_app.command("ingest")
    def datasus_ingest(
        system: str = typer.Option(..., "--system"),
        uf: str = typer.Option(..., "--uf"),
        years: str = typer.Option(..., "--years"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
    ) -> None:
        cfg_payload = load_datasus_config()
        cfg = DatasusConfig.from_mapping(cfg_payload)
        manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

        cache = DatasusCache()
        blocked = False
        failed = False

        for manifest in manifests:
            planned_path = write_request_manifest(manifest)
            print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")

            if dry_run:
                continue

            executed = fetch_datasus_chunk(
                manifest,
                config=cfg,
                cache=cache,
                timeout_seconds=cfg.r_timeout_seconds,
                heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
            )
            executed_path = write_request_manifest(executed)

            if executed.status == "success":
                print(f"[green]success[/green] {executed.system} {executed.uf} {executed.year_start}: {executed_path}")
            elif executed.status == "blocked":
                blocked = True
                print(f"[yellow]blocked[/yellow] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")
            else:
                failed = True
                print(f"[red]{executed.status}[/red] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")

        if failed:
            raise typer.Exit(1)
        if blocked:
            raise typer.Exit(2)


    @datasus_app.command("profile")
    def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
        request = read_request_manifest(manifest)

        raw_path = Path(request.raw_path)
        processed_path = Path(request.processed_path)

        if not raw_path.exists() or not processed_path.exists():
            print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
            raise typer.Exit(2)

        raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
        processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
        compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

        raw_profile = profile_table(raw_path, output_path=raw_profile_path)
        processed_profile = profile_table(processed_path, output_path=processed_profile_path)
        compare_profiles(raw_profile, processed_profile, output_path=compare_path)

        print(f"[green]raw profile[/green] {raw_profile_path}")
        print(f"[green]processed profile[/green] {processed_profile_path}")
        print(f"[green]schema comparison[/green] {compare_path}")


    @datasus_app.command("normalize-sim")
    def datasus_normalize_sim(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(..., "--output"),
        source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
    ) -> None:
        result = normalize_sim_do_events(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )
        print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


    @efg_app.command("build-sim-fixture")
    def efg_build_sim_fixture(
        sim_events: Path = typer.Option(..., "--sim-events"),
        run_dir: Path = typer.Option(..., "--run-dir"),
    ) -> None:
        output = build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
        )
        result = validate_output_bundle(run_dir=str(output))
        if not result.ok:
            _fail(result.errors)
        print(f"[green]sim fixture EFG bundle valid[/green] {output}")


    @sidra_app.command("metadata-fixture")
    def sidra_metadata_fixture(
        output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
    ) -> None:
        metadata = fixture_sidra_metadata()
        outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)
        for name, path in outputs.items():
            print(f"[green]{name}[/green] {path}")


    @sidra_app.command("plan-fixture")
    def sidra_plan_fixture(
        output: Path = typer.Option(Path("data/manifests/sidra/fixture_plan.json"), "--output"),
        max_cells: int = typer.Option(49900, "--max-cells"),
    ) -> None:
        metadata = fixture_sidra_metadata()
        table = metadata.tables["9606"]
        request = SIDRARequest(
            table_id="9606",
            variables=table.variables,
            periods=table.periods,
            locality_level="N6",
            localities=table.localities_by_level["N6"],
            classifications=table.classifications,
        )
        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=max_cells)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[green]planned[/green] chunks={len(chunks)} output={output}")


    @sidra_app.command("normalize-fixture")
    def sidra_normalize_fixture(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(Path("data/processed/sidra/facts/9606/fixture.parquet"), "--output"),
    ) -> None:
        output = normalize_fixture_json_to_facts(
            input_path=input_path,
            output_path=output_path,
            table_id="9606",
            unit_by_variable={"93": "persons"},
        )
        print(f"[green]sidra facts normalized[/green] {output}")


    @sidra_app.command("metadata")
    def sidra_metadata(tables: Path = typer.Option(..., "--tables")) -> None:
        print(f"[yellow]blocked[/yellow] live SIDRA metadata fetch is Slice 2B. Seed received: {tables}")
        raise typer.Exit(2)


    @sidra_app.command("plan")
    def sidra_plan(view: str = typer.Option(..., "--view")) -> None:
        print(f"[yellow]blocked[/yellow] live SIDRA planning is Slice 2B. View received: {view}")
        raise typer.Exit(2)


    @sidra_app.command("extract")
    def sidra_extract(plan: Path = typer.Option(..., "--plan")) -> None:
        print(f"[yellow]blocked[/yellow] live SIDRA extraction is Slice 2B. Plan received: {plan}")
        raise typer.Exit(2)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    ''')

    write("tests/fixtures/sidra/sidra_flat_fixture.json", r'''
    [
      {
        "table_id": "9606",
        "variable_id": "93",
        "period": "2022",
        "locality_level": "N6",
        "locality_id": "270430",
        "classification_tuple": [["2", "Sexo"], ["58", "Cor ou raça"], ["287", "Idade"]],
        "category_tuple": [["2", "0"], ["58", "0"], ["287", "0"]],
        "value": "1025360"
      },
      {
        "table_id": "9606",
        "variable_id": "93",
        "period": "2022",
        "locality_level": "N6",
        "locality_id": "270030",
        "classification_tuple": [["2", "Sexo"], ["58", "Cor ou raça"], ["287", "Idade"]],
        "category_tuple": [["2", "0"], ["58", "0"], ["287", "0"]],
        "value": "234185"
      },
      {
        "table_id": "9606",
        "variable_id": "93",
        "period": "2022",
        "locality_level": "N6",
        "locality_id": "270770",
        "classification_tuple": [["2", "Sexo"], ["58", "Cor ou raça"], ["287", "Idade"]],
        "category_tuple": [["2", "0"], ["58", "0"], ["287", "0"]],
        "value": "-"
      },
      {
        "table_id": "9606",
        "variable_id": "93",
        "period": "2023",
        "locality_level": "N6",
        "locality_id": "270430",
        "classification_tuple": [["2", "Sexo"], ["58", "Cor ou raça"], ["287", "Idade"]],
        "category_tuple": [["2", "0"], ["58", "0"], ["287", "0"]],
        "value": "1.031.597"
      }
    ]
    ''')

    write("tests/unit/test_sidra_planner.py", r'''
    from pegasus.sidra.metadata import fixture_sidra_metadata
    from pegasus.sidra.plan import estimate_cells, plan_sidra_chunks
    from pegasus.sidra.schemas import SIDRARequest


    def test_estimate_cells_multiplies_axes():
        assert estimate_cells(
            localities=["1", "2"],
            periods=["2022"],
            variables=["93"],
            classifications={"2": ["0", "1"], "58": ["0", "4"]},
        ) == 8


    def test_small_request_does_not_require_view_definition():
        metadata = fixture_sidra_metadata()
        request = SIDRARequest(
            table_id="9606",
            variables=["93"],
            periods=["2022"],
            locality_level="N6",
            localities=["270430"],
            classifications={"2": ["0"], "58": ["0"], "287": ["0"]},
        )
        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49900)
        assert len(chunks) == 1
        assert chunks[0].estimated_cells == 1


    def test_large_request_splits_by_localities_first():
        metadata = fixture_sidra_metadata()
        table = metadata.tables["9606"]
        request = SIDRARequest(
            table_id="9606",
            variables=["93"],
            periods=["2022", "2023"],
            locality_level="N6",
            localities=[f"{i:06d}" for i in range(100)],
            classifications={"2": ["0", "1", "2"], "58": ["0", "4", "5"], "287": [str(i) for i in range(100)]},
        )

        # Extend fixture metadata so test is about split behavior, not validation.
        table.localities_by_level["N6"] = request.localities
        table.classifications["287"] = request.classifications["287"]

        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=49900)
        assert len(chunks) > 1
        assert all(c.estimated_cells <= 49900 for c in chunks)
        assert sum(c.estimated_cells for c in chunks) == 100 * 2 * 1 * 3 * 3 * 100
    ''')

    write("tests/unit/test_sidra_facts.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.facts import classify_sidra_value, normalize_fixture_json_to_facts


    def test_sidra_value_status_preservation():
        assert classify_sidra_value("10")[1] == "numeric"
        assert classify_sidra_value("-")[1] == "dash_zero_or_nil"
        assert classify_sidra_value("x")[1] == "not_available"
        assert classify_sidra_value("abc")[1] == "non_numeric_symbol"


    def test_sidra_fixture_normalizes_to_long_form_facts(tmp_path: Path):
        out = tmp_path / "facts.parquet"
        normalize_fixture_json_to_facts(
            input_path="tests/fixtures/sidra/sidra_flat_fixture.json",
            output_path=out,
            table_id="9606",
            unit_by_variable={"93": "persons"},
        )

        df = pl.read_parquet(out)
        assert df.height == 4
        assert {
            "table_id",
            "variable_id",
            "period",
            "locality_level",
            "locality_id",
            "classification_tuple",
            "category_tuple",
            "value_raw",
            "value_numeric",
            "value_status",
            "unit",
            "request_hash",
            "metadata_hash",
            "fetched_at",
        } == set(df.columns)

        statuses = set(df["value_status"].to_list())
        assert "numeric" in statuses
        assert "dash_zero_or_nil" in statuses
    ''')

    write("tests/unit/test_sidra_projection_bounding.py", r'''
    from pegasus.sidra.category_maps import bounded_pushforward_scaffold
    from pegasus.sidra.projection import project_classification_to_axis
    from pegasus.she.population.schema import blocked_missing_population_contract, official_sidra_anchor_contract


    def test_projection_blocks_ratio_without_denominator_recovery():
        result = project_classification_to_axis(
            measure_kind="proportion",
            has_denominator=False,
            projection_matrix_id="race_projection_v1",
        )
        assert result.status == "blocked"
        assert "ratio_projection_requires_denominator_recovery" in result.warnings


    def test_projection_allows_additive_with_matrix():
        result = project_classification_to_axis(
            measure_kind="count",
            has_denominator=False,
            projection_matrix_id="population_axis_projection_v1",
        )
        assert result.status == "projected"


    def test_bounded_pushforward_blocks_nonaggregable_high_dimensional_field():
        result = bounded_pushforward_scaffold(
            raw_axes=["age", "sex", "race"],
            demanded_axes=["sex"],
            aggregation="non_aggregable",
            high_dimensional=True,
        )
        assert result.status == "blocked"


    def test_bounded_pushforward_additive_field():
        result = bounded_pushforward_scaffold(
            raw_axes=["age", "sex", "race"],
            demanded_axes=["sex"],
            aggregation="additive",
            high_dimensional=True,
        )
        assert result.status == "bounded"
        assert result.axes_dropped == ["age", "race"]


    def test_population_denominator_contracts():
        official = official_sidra_anchor_contract()
        assert official.mode == "official_sidra_anchor"
        assert official.allowed_for_rates is True

        blocked = blocked_missing_population_contract(reason="missing_population")
        assert blocked.mode == "blocked_missing"
        assert blocked.allowed_for_rates is False
    ''')

    print("Applied Slice 2A: SIDRA schemas, fixture metadata, chunk planner, long-form fact normalizer, projection/bounding scaffolds, denominator contract.")
    

if __name__ == "__main__":
    main()