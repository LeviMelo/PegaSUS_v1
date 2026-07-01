from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("docs/pegasus_codebase_contract.yaml", r'''
    project_name: PegaSUS

    required_paths:
      - src/pegasus/
      - src/pegasus/core/
      - src/pegasus/registries/
      - src/pegasus/storage/
      - src/pegasus/datasus/
      - src/pegasus/sidra/
      - src/pegasus/geo/
      - src/pegasus/she/
      - src/pegasus/she/population/
      - src/pegasus/she/stdfm/
      - src/pegasus/efg/
      - src/pegasus/pirs/
      - src/pegasus/compute/
      - src/pegasus/output/
      - src/pegasus/dashboard/
      - src/pegasus/workflows/
      - config/registries/
      - config/intents/
      - tests/

    forbidden_paths:
      - problem1/
      - problem2/
      - problem3/
      - notebooks/production/

    forbidden_terms_in_paths:
      - miniPegaSUS
      - problem1
      - problem2
      - problem3

    required_cli_commands:
      - pegasus init
      - pegasus doctor
      - pegasus validate-config
      - pegasus registries validate
      - pegasus sidra metadata
      - pegasus sidra plan
      - pegasus sidra extract
      - pegasus datasus ingest
      - pegasus datasus profile
      - pegasus datasus normalize-sim
      - pegasus efg build-sim-fixture
      - pegasus efg attach-sidra-denominator
      - pegasus compile
      - pegasus validate-run

    required_output_bundle_keys:
      - V_fields
      - E_DAG
      - Q_tensor
      - P_vector
      - UserIntent
      - Warnings
      - ModelAssociations
      - ResidualAssociations
      - Hypotheses
      - Tables
      - Maps
      - VariableDictionary
      - FailedBranches
      - QuarantinedFields
      - ForcedFields
      - RunConfig
      - ReproducibilityManifest

    layer_rules:
      - name: dashboard_read_only
        from: src/pegasus/dashboard/
        forbidden_imports:
          - pegasus.sidra.extract
          - pegasus.datasus.subprocess
          - pegasus.pirs
          - pegasus.she.population.solver
          - pegasus.compute.cuda

      - name: cli_thin_wrapper
        from: src/pegasus/cli.py
        warning_forbidden_imports:
          - pegasus.sidra.extract.extract_chunk_plan
          - pegasus.sidra.extract.read_chunk_plan
          - pegasus.sidra.extract.write_chunk_plan
          - pegasus.sidra.extract.write_extraction_log
          - pegasus.datasus.normalize.normalize_sim_do_events
          - pegasus.output.sidra_denominator_anchor.attach_sidra_population_anchor_to_run

      - name: scripts_do_not_own_production_logic
        from: scripts/
        allowed_behaviors:
          - calls_package_services
        warning_patterns:
          - "def normalize"
          - "def extract"
          - "class .*Client"
          - "pl.read_parquet"
          - "requests.get"

      - name: production_no_pandas
        from: src/
        forbidden_imports:
          - pandas

      - name: sidra_boundary
        from: src/pegasus/sidra/
        expected_modules:
          - api.py
          - cache.py
          - metadata.py
          - registry.py
          - plan.py
          - extract.py
          - normalize.py
          - facts.py
          - category_maps.py
          - projection.py
          - stitching.py
          - regime.py
          - quality.py

    notes:
      - compile may remain blocked until Slice 2D, but the CLI command must exist.
      - output/ may serialize and mutate output bundles, but workflow orchestration should live under workflows/.
      - SIDRA 9606 high-dimensional facts must enter EFG only through bounded legal marginals such as Population(s,t).
      - Any RN field joining SIM and SIDRA must prove time/geography support alignment or fail before field creation.
    ''')

    write("src/pegasus/geo/municipality_crosswalk.py", r'''
    from __future__ import annotations

    import re
    from dataclasses import dataclass


    class MunicipalityCrosswalkError(ValueError):
        """Raised when a municipality code cannot be safely crosswalked."""


    @dataclass(frozen=True)
    class MunicipalityCode:
        source_code: str
        datasus_cod6: str | None
        ibge_cod7: str | None
        state: str
        method: str


    # Minimal explicit AL smoke crosswalk. This is not a national geodata table.
    # It exists to prevent accidental string equality between DATASUS six-digit
    # municipality support and SIDRA/IBGE seven-digit municipality support.
    AL_SMOKE_COD6_TO_COD7: dict[str, str] = {
        "270030": "2700300",  # Arapiraca
        "270430": "2704302",  # Maceió
    }
    AL_SMOKE_COD7_TO_COD6: dict[str, str] = {v: k for k, v in AL_SMOKE_COD6_TO_COD7.items()}


    def _digits(value: object) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\D", "", str(value).strip())
        return text or None


    def normalize_municipality_code(value: object) -> MunicipalityCode:
        digits = _digits(value)
        if digits is None:
            return MunicipalityCode(
                source_code="",
                datasus_cod6=None,
                ibge_cod7=None,
                state="missing",
                method="none",
            )

        if len(digits) == 7:
            return MunicipalityCode(
                source_code=digits,
                datasus_cod6=AL_SMOKE_COD7_TO_COD6.get(digits, digits[:6]),
                ibge_cod7=digits,
                state="mapped" if digits in AL_SMOKE_COD7_TO_COD6 else "assumed_ibge_cod7",
                method="ibge_cod7_identity",
            )

        if len(digits) == 6:
            cod7 = AL_SMOKE_COD6_TO_COD7.get(digits)
            return MunicipalityCode(
                source_code=digits,
                datasus_cod6=digits,
                ibge_cod7=cod7,
                state="mapped" if cod7 else "unmapped_datasus_cod6",
                method="al_smoke_explicit_cod6_to_cod7" if cod7 else "none",
            )

        return MunicipalityCode(
            source_code=digits,
            datasus_cod6=None,
            ibge_cod7=None,
            state="invalid_length",
            method="none",
        )


    def datasus_cod6_to_ibge_cod7(value: object, *, strict: bool = True) -> str | None:
        code = normalize_municipality_code(value)
        if code.ibge_cod7 is not None:
            return code.ibge_cod7
        if strict:
            raise MunicipalityCrosswalkError(
                f"Cannot map DATASUS municipality code {value!r} to IBGE/SIDRA cod7. state={code.state}"
            )
        return None


    def ibge_cod7_to_datasus_cod6(value: object, *, strict: bool = True) -> str | None:
        code = normalize_municipality_code(value)
        if code.datasus_cod6 is not None:
            return code.datasus_cod6
        if strict:
            raise MunicipalityCrosswalkError(
                f"Cannot map IBGE/SIDRA municipality code {value!r} to DATASUS cod6. state={code.state}"
            )
        return None
    ''')

    write("src/pegasus/geo/support.py", r'''
    from __future__ import annotations

    from dataclasses import dataclass, asdict
    from typing import Any

    from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7


    class SupportAlignmentError(ValueError):
        """Raised when numerator/denominator support cannot be safely aligned."""


    @dataclass(frozen=True)
    class SupportAlignmentResult:
        aligned: bool
        reason: str
        numerator_geography: str | None
        denominator_geography: str | None
        numerator_years: list[int | str]
        denominator_years: list[int | str]
        numerator_municipalities_source: list[str]
        denominator_municipalities_source: list[str]
        numerator_municipalities_ibge_cod7: list[str]
        denominator_municipalities_ibge_cod7: list[str]
        common_municipalities_ibge_cod7: list[str]
        missing_from_denominator_ibge_cod7: list[str]
        missing_from_numerator_ibge_cod7: list[str]
        crosswalk: str

        def model(self) -> dict[str, Any]:
            return asdict(self)


    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


    def _years(support: dict[str, Any]) -> list[int | str]:
        out = []
        for value in _as_list(support.get("years")):
            if isinstance(value, int):
                out.append(value)
            else:
                text = str(value)
                out.append(int(text) if text.isdigit() else text)
        return sorted(out, key=lambda x: str(x))


    def _municipalities(support: dict[str, Any]) -> list[str]:
        return sorted(str(x) for x in _as_list(support.get("municipalities")) if x is not None)


    def _municipalities_as_ibge_cod7(
        support: dict[str, Any],
        axes: dict[str, Any],
        *,
        strict: bool,
    ) -> list[str]:
        geography = axes.get("geography")
        codes = _municipalities(support)

        if not codes:
            return []

        if geography == "mun_residence_cod6":
            return sorted(datasus_cod6_to_ibge_cod7(code, strict=strict) for code in codes)

        if geography in {"municipality", "municipality_ibge_cod7", None}:
            mapped = []
            for code in codes:
                if len(code) == 7 and code.isdigit():
                    mapped.append(code)
                else:
                    mapped.append(datasus_cod6_to_ibge_cod7(code, strict=strict))
            return sorted(mapped)

        if strict:
            raise SupportAlignmentError(f"Unsupported geography axis for municipality support alignment: {geography!r}")
        return []


    def align_municipality_year_support(
        *,
        numerator_support: dict[str, Any],
        numerator_axes: dict[str, Any],
        denominator_support: dict[str, Any],
        denominator_axes: dict[str, Any],
        strict_crosswalk: bool = True,
    ) -> SupportAlignmentResult:
        numerator_years = _years(numerator_support)
        denominator_years = _years(denominator_support)
        numerator_source = _municipalities(numerator_support)
        denominator_source = _municipalities(denominator_support)

        try:
            numerator_cod7 = _municipalities_as_ibge_cod7(numerator_support, numerator_axes, strict=strict_crosswalk)
            denominator_cod7 = _municipalities_as_ibge_cod7(denominator_support, denominator_axes, strict=strict_crosswalk)
        except Exception as exc:
            return SupportAlignmentResult(
                aligned=False,
                reason=str(exc),
                numerator_geography=numerator_axes.get("geography"),
                denominator_geography=denominator_axes.get("geography"),
                numerator_years=numerator_years,
                denominator_years=denominator_years,
                numerator_municipalities_source=numerator_source,
                denominator_municipalities_source=denominator_source,
                numerator_municipalities_ibge_cod7=[],
                denominator_municipalities_ibge_cod7=[],
                common_municipalities_ibge_cod7=[],
                missing_from_denominator_ibge_cod7=[],
                missing_from_numerator_ibge_cod7=[],
                crosswalk="datasus_cod6_to_ibge_cod7",
            )

        num_year_set = set(numerator_years)
        den_year_set = set(denominator_years)
        num_set = set(numerator_cod7)
        den_set = set(denominator_cod7)

        missing_from_denominator = sorted(num_set - den_set)
        missing_from_numerator = sorted(den_set - num_set)
        common = sorted(num_set & den_set)

        aligned = (
            num_year_set == den_year_set
            and bool(num_set)
            and num_set == den_set
        )
        if num_year_set != den_year_set:
            reason = "time_support_mismatch"
        elif not num_set:
            reason = "empty_numerator_municipality_support"
        elif num_set != den_set:
            reason = "municipality_support_mismatch"
        else:
            reason = "aligned"

        return SupportAlignmentResult(
            aligned=aligned,
            reason=reason,
            numerator_geography=numerator_axes.get("geography"),
            denominator_geography=denominator_axes.get("geography"),
            numerator_years=numerator_years,
            denominator_years=denominator_years,
            numerator_municipalities_source=numerator_source,
            denominator_municipalities_source=denominator_source,
            numerator_municipalities_ibge_cod7=sorted(num_set),
            denominator_municipalities_ibge_cod7=sorted(den_set),
            common_municipalities_ibge_cod7=common,
            missing_from_denominator_ibge_cod7=missing_from_denominator,
            missing_from_numerator_ibge_cod7=missing_from_numerator,
            crosswalk="datasus_cod6_to_ibge_cod7",
        )


    def assert_municipality_year_support_aligned(
        *,
        numerator_support: dict[str, Any],
        numerator_axes: dict[str, Any],
        denominator_support: dict[str, Any],
        denominator_axes: dict[str, Any],
    ) -> SupportAlignmentResult:
        result = align_municipality_year_support(
            numerator_support=numerator_support,
            numerator_axes=numerator_axes,
            denominator_support=denominator_support,
            denominator_axes=denominator_axes,
            strict_crosswalk=True,
        )
        if not result.aligned:
            raise SupportAlignmentError(
                "Illegal denominator attachment: numerator and denominator municipality-year support are not aligned. "
                f"reason={result.reason}; "
                f"numerator_cod7={result.numerator_municipalities_ibge_cod7}; "
                f"denominator_cod7={result.denominator_municipalities_ibge_cod7}; "
                f"numerator_years={result.numerator_years}; denominator_years={result.denominator_years}"
            )
        return result
    ''')

    write("src/pegasus/workflows/datasus.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

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


    def run_datasus_ingest(
        *,
        system: str,
        uf: str,
        years: str,
        dry_run: bool,
    ) -> dict[str, Any]:
        cfg_payload = load_datasus_config()
        cfg = DatasusConfig.from_mapping(cfg_payload)
        manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

        cache = DatasusCache()
        planned = []
        executed = []
        blocked = False
        failed = False

        for manifest in manifests:
            planned_path = write_request_manifest(manifest)
            planned.append((manifest, planned_path))

            if dry_run:
                continue

            result = fetch_datasus_chunk(
                manifest,
                config=cfg,
                cache=cache,
                timeout_seconds=cfg.r_timeout_seconds,
                heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
            )
            executed_path = write_request_manifest(result)
            executed.append((result, executed_path))

            if result.status == "blocked":
                blocked = True
            elif result.status != "success":
                failed = True

        return {
            "planned": planned,
            "executed": executed,
            "blocked": blocked,
            "failed": failed,
        }


    def run_datasus_profile(*, manifest: str | Path) -> dict[str, Any]:
        request = read_request_manifest(manifest)
        raw_path = Path(request.raw_path)
        processed_path = Path(request.processed_path)

        if not raw_path.exists() or not processed_path.exists():
            return {"status": "blocked", "reason": "raw_or_processed_artifact_missing", "request": request}

        raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
        processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
        compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

        raw_profile = profile_table(raw_path, output_path=raw_profile_path)
        processed_profile = profile_table(processed_path, output_path=processed_profile_path)
        comparison = compare_profiles(raw_profile, processed_profile, output_path=compare_path)

        return {
            "status": "success",
            "request": request,
            "raw_profile_path": raw_profile_path,
            "processed_profile_path": processed_profile_path,
            "compare_path": compare_path,
            "raw_profile": raw_profile,
            "processed_profile": processed_profile,
            "comparison": comparison,
        }


    def run_datasus_normalize_sim(
        *,
        input_path: str | Path,
        output_path: str | Path,
        source_manifest_hash: str,
    ) -> dict[str, Any]:
        return normalize_sim_do_events(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )
    ''')

    write("src/pegasus/workflows/sidra.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.core.config import load_yaml
    from pegasus.sidra.api import SidraClient
    from pegasus.sidra.extract import extract_chunk_plan, read_chunk_plan, write_chunk_plan, write_extraction_log
    from pegasus.sidra.facts import normalize_fixture_json_to_facts
    from pegasus.sidra.metadata import (
        fetch_official_metadata,
        fixture_sidra_metadata,
        metadata_dir_hash,
        read_normalized_metadata_tables,
        table_ids_from_seed,
        write_normalized_metadata_tables,
    )
    from pegasus.sidra.plan import plan_sidra_chunks
    from pegasus.sidra.registry import request_from_view
    from pegasus.sidra.schemas import SIDRARequest


    def sidra_runtime_config() -> dict[str, Any]:
        data = load_yaml("config/sidra.yaml")
        return data.get("sidra", data)


    def run_sidra_metadata_fixture(*, output_dir: str | Path) -> dict[str, Path]:
        metadata = fixture_sidra_metadata()
        return write_normalized_metadata_tables(metadata, output_dir=output_dir)


    def run_sidra_plan_fixture(
        *,
        output: str | Path,
        max_cells: int = 49900,
    ) -> dict[str, Any]:
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
        output_path = write_chunk_plan(chunks, output_path=output)
        return {"chunks": chunks, "output": output_path}


    def run_sidra_normalize_fixture(
        *,
        input_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        return normalize_fixture_json_to_facts(
            input_path=input_path,
            output_path=output_path,
            table_id="9606",
            unit_by_variable={"93": "persons"},
        )


    def run_sidra_metadata(
        *,
        tables: str | Path,
        level: str,
        output_dir: str | Path,
        raw_dir: str | Path,
    ) -> dict[str, Any]:
        table_ids = table_ids_from_seed(tables)
        if not table_ids:
            raise ValueError("No SIDRA table IDs found in seed.")

        client = SidraClient()
        metadata = fetch_official_metadata(
            table_ids=table_ids,
            client=client,
            locality_level=level,
            raw_dir=raw_dir,
        )
        outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)
        return {"table_ids": table_ids, "outputs": outputs}


    def run_sidra_plan(
        *,
        view: str,
        metadata_dir: str | Path,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        sidra_cfg = sidra_runtime_config()
        metadata = read_normalized_metadata_tables(metadata_dir)
        request = request_from_view(view)
        chunks = plan_sidra_chunks(
            request,
            metadata,
            max_cells_per_request=int(sidra_cfg.get("max_cells_per_request", 49900)),
        )
        output_path = Path(output) if output is not None else Path("data/manifests/sidra") / f"{view}.json"
        write_chunk_plan(chunks, output_path=output_path)
        return {"chunks": chunks, "output": output_path}


    def run_sidra_extract(
        *,
        plan: str | Path,
        metadata_dir: str | Path,
        dry_run: bool,
        concurrency: int | None = None,
        log_path: str | Path | None = None,
    ) -> dict[str, Any]:
        sidra_cfg = sidra_runtime_config()
        chunks = read_chunk_plan(plan)
        meta_hash = metadata_dir_hash(metadata_dir)

        if dry_run:
            return {
                "status": "dry_run",
                "chunks": chunks,
                "estimated_cells": sum(c.estimated_cells for c in chunks),
                "metadata_hash": meta_hash,
                "log_path": None,
            }

        client = SidraClient()
        results = extract_chunk_plan(
            chunks,
            client=client,
            concurrency=concurrency or int(sidra_cfg.get("concurrency", 4)),
            metadata_hash=meta_hash,
        )
        resolved_log_path = Path(log_path) if log_path is not None else Path("data/diagnostics/sidra") / f"{Path(plan).stem}.extraction_log.json"
        write_extraction_log(results, output_path=resolved_log_path)
        failures = [r for r in results if r.status != "success"]
        return {
            "status": "success" if not failures else "failed",
            "chunks": chunks,
            "results": results,
            "failures": failures,
            "metadata_hash": meta_hash,
            "log_path": resolved_log_path,
        }
    ''')

    write("src/pegasus/workflows/build_efg.py", r'''
    from __future__ import annotations

    from pathlib import Path

    import polars as pl

    from pegasus.output.sim_efg_bundle import write_sim_fixture_efg_bundle


    def build_sim_fixture_efg_run(
        *,
        sim_events_path: str | Path,
        run_dir: str | Path,
        municipality_cod6: str | None = None,
    ) -> Path:
        run_dir = Path(run_dir)
        source_events_path = Path(sim_events_path)

        if municipality_cod6 is not None:
            filtered_path = run_dir / "Tables" / f"sim_events_mun_{municipality_cod6}.parquet"
            filtered_path.parent.mkdir(parents=True, exist_ok=True)
            df = pl.read_parquet(source_events_path)
            filtered = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
            if filtered.height == 0:
                raise ValueError(f"No SIM fixture events remain after municipality filter {municipality_cod6!r}.")
            filtered.write_parquet(filtered_path)
            source_events_path = filtered_path

        return write_sim_fixture_efg_bundle(
            sim_events_path=source_events_path,
            run_dir=run_dir,
        )
    ''')

    write("src/pegasus/workflows/efg.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.construct.build_efg import build_sim_fixture_efg_run


    def run_build_sim_fixture(
        *,
        sim_events_path: str | Path,
        run_dir: str | Path,
        municipality_cod6: str | None = None,
    ) -> dict[str, Any]:
        output = build_sim_fixture_efg_run(
            sim_events_path=sim_events_path,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
        )
        result = validate_output_bundle(run_dir=str(output))
        return {"run_dir": output, "validation": result}


    def run_attach_sidra_denominator(
        *,
        run_dir: str | Path,
        sidra_facts_path: str | Path,
    ) -> dict[str, Any]:
        output = attach_sidra_population_anchor_to_run(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts_path,
        )
        result = validate_output_bundle(run_dir=str(output))
        return {"run_dir": output, "validation": result}
    ''')

    write("src/pegasus/output/sidra_denominator_anchor.py", r'''
    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash
    from pegasus.geo.support import SupportAlignmentResult, assert_municipality_year_support_aligned
    from pegasus.output.validate import validate_output_bundle
    from pegasus.she.population.sidra_anchor import SidraPopulationAnchor, load_sidra_population_total_anchor


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


    def _load_json_field(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(str(value))


    def _remove_by_values(df: pl.DataFrame, column: str, values: set[str]) -> pl.DataFrame:
        if column not in df.columns or not values:
            return df
        return df.filter(~pl.col(column).is_in(sorted(values)))


    def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str | None = None, remove_values: set[str] | None = None) -> None:
        df = pl.read_parquet(path)

        if remove_column and remove_values:
            df = _remove_by_values(df, remove_column, remove_values)

        if not rows:
            df.write_parquet(path)
            return

        add = pl.DataFrame(rows)

        for col, dtype in df.schema.items():
            if col not in add.columns:
                add = add.with_columns(pl.lit(None).cast(dtype).alias(col))
            else:
                add = add.with_columns(pl.col(col).cast(dtype, strict=False))

        add = add.select(df.columns)
        pl.concat([df, add], how="vertical").write_parquet(path)


    def _row_for_columns(columns: list[str], payload: dict[str, Any]) -> dict[str, Any]:
        return {col: payload.get(col) for col in columns}


    def _get_field_by_name(v: pl.DataFrame, name: str) -> dict[str, Any]:
        rows = v.filter(pl.col("name") == name).to_dicts()
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one field named {name}; found {len(rows)}.")
        return rows[0]


    def _population_v_field(anchor: SidraPopulationAnchor, facts_path: Path) -> dict[str, Any]:
        support = {
            "support": "municipality_year",
            "years": [int(anchor.period) if anchor.period.isdigit() else anchor.period],
            "municipalities": [anchor.locality_id],
            "n_denom": anchor.value,
            "n_eff": anchor.value,
            "cov_S": 1.0,
            "cov_T": 1.0,
            "missingness": 0.0,
            "denom_fragility": 0.0,
            "sidra_table_id": anchor.table_id,
            "sidra_variable_id": anchor.variable_id,
            "total_category_policy": anchor.total_category_policy,
            "source_category_tuple": anchor.category_tuple,
            "source_classification_tuple": anchor.classification_tuple,
            "municipality_code_system": "IBGE/SIDRA cod7",
        }

        axes = {
            "time": "year",
            "geography": "municipality",
            "race_axis_type": None,
            "sidra_source_classifications": {
                "2": "sex",
                "86": "race_color",
                "287": "age",
            },
            "sidra_source_categories": dict(anchor.category_tuple),
            "projection_metadata": {
                "source_classifications": anchor.classification_tuple,
                "source_categories": anchor.category_tuple,
                "target_axes": ["municipality", "year"],
                "projection_matrix_id": "total_category_identity_marginal_9606_v1",
                "total_category_policy": "total_only",
                "fractional_mapping_warnings": [],
            },
            "bounded_pushforward": {
                "operator": "pi_bound_*",
                "axes_kept": ["municipality", "year"],
                "axes_dropped": ["sex", "race_color", "age"],
                "legal": True,
                "reason": "SIDRA 9606 total sex/race/age categories produce Population(s,t).",
            },
        }

        return {
            "field_id": anchor.field_id,
            "name": "SIDRAPopulationTotalAnchor",
            "kind": "extensive_measure",
            "carrier": "Population",
            "unit": "persons",
            "aggregation": "additive",
            "role": _json(["demographic", "exposure_offset"]),
            "source": _json(["SIDRA"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": "Pi_Clsf_to_Axis/pi_bound_*",
            "provenance": _json(["official", "sidra_9606", "bounded_total_category_anchor"]),
            "state": "fragile",
            "dashboard_safe": "warning",
            "warnings": _json(["sidra_total_category_anchor", "population_denominator_contract_fragile_until_crosscheck"]),
            "lineage_hash": anchor.field_id,
            "registry_hash": _json(
                {
                    "sidra_views": "v1.0",
                    "sidra_metadata_hash": anchor.metadata_hash,
                    "sidra_request_hash": anchor.request_hash,
                }
            ),
            "materialization_state": "metadata_only",
            "path": str(facts_path),
        }


    def _rate_v_field(
        *,
        all_deaths: dict[str, Any],
        anchor: SidraPopulationAnchor,
        facts_path: Path,
        support_alignment: SupportAlignmentResult,
    ) -> dict[str, Any]:
        death_support = _load_json_field(all_deaths["support_json"])
        n_events = float(death_support.get("n_events", 0.0))

        field_id = content_hash(
            {
                "kind": "SIMCrudeMortalitySIDRAOfficial",
                "parents": [all_deaths["field_id"], anchor.field_id],
                "operator": "RN",
                "sidra_metadata_hash": anchor.metadata_hash,
                "sidra_request_hash": anchor.request_hash,
                "support_alignment": support_alignment.model(),
            }
        )

        support = {
            "support": "municipality_year",
            "years": [int(anchor.period) if anchor.period.isdigit() else anchor.period],
            "municipalities": support_alignment.numerator_municipalities_ibge_cod7,
            "municipality_code_system": "IBGE/SIDRA cod7",
            "n_events": n_events,
            "n_denom": anchor.value,
            "n_eff": n_events,
            "cov_S": float(len(support_alignment.numerator_municipalities_ibge_cod7)),
            "cov_T": float(len(support_alignment.numerator_years)),
            "missingness": float(death_support.get("missingness", 0.0)),
            "denom_fragility": 0.0,
            "sidra_population_anchor_field_id": anchor.field_id,
            "support_alignment": support_alignment.model(),
            "municipality_crosswalk": "datasus_cod6_to_ibge_cod7",
            "fixture_rate": True,
        }

        axes = {
            "time": "year",
            "geography": "municipality_ibge_cod7",
            "diagnostic_role": "all_deaths",
            "topology": "none",
            "race_axis_type": None,
            "denominator_source": "SIDRA_9606_total_population_anchor",
        }

        return {
            "field_id": field_id,
            "name": "SIMCrudeMortalitySIDRAOfficial",
            "kind": "intensive_density",
            "carrier": "Deaths/Population",
            "unit": "rate",
            "aggregation": "non_aggregable",
            "role": _json(["outcome", "model_only"]),
            "source": _json(["SIM-DO", "SIDRA"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": "RN",
            "provenance": _json(["official", "sidra_denominator_anchor", "sim_fixture_numerator"]),
            "state": "quarantined_descriptive",
            "dashboard_safe": "False",
            "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor", "dashboard_unsafe_fixture_rate", "support_aligned_by_municipality_crosswalk"]),
            "lineage_hash": field_id,
            "registry_hash": _json(
                {
                    "sidra_views": "v1.0",
                    "sidra_metadata_hash": anchor.metadata_hash,
                    "sidra_request_hash": anchor.request_hash,
                }
            ),
            "materialization_state": "metadata_only",
            "path": str(facts_path),
        }


    def _q_rows(*, population_row: dict[str, Any], rate_row: dict[str, Any]) -> list[dict[str, Any]]:
        pop_support = _load_json_field(population_row["support_json"])
        rate_support = _load_json_field(rate_row["support_json"])

        return [
            {
                "field_id": population_row["field_id"],
                "n_events": None,
                "n_denom": float(pop_support["n_denom"]),
                "n_eff": float(pop_support["n_eff"]),
                "cov_S": 1.0,
                "cov_T": 1.0,
                "missingness": 0.0,
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "cv": None,
                "moran_i": None,
                "temporal_roughness": None,
                "spatial_entropy": None,
                "provenance_risk": 0.0,
                "race_axis_source": None,
                "race_axis_target": None,
                "missing_race_share": None,
                "emission_prior_strength": None,
                "race_bridge_cv": None,
                "sensitivity_width": None,
                "bridge_mode": None,
                "state": "fragile",
                "dashboard_safe": "warning",
                "warnings": _json(["sidra_total_category_anchor"]),
                "computed_at": _now(),
                "q_schema_version": "1.0",
            },
            {
                "field_id": rate_row["field_id"],
                "n_events": float(rate_support["n_events"]),
                "n_denom": float(rate_support["n_denom"]),
                "n_eff": float(rate_support["n_eff"]),
                "cov_S": float(rate_support["cov_S"]),
                "cov_T": float(rate_support["cov_T"]),
                "missingness": float(rate_support["missingness"]),
                "zero_inflation": 0.0,
                "denom_fragility": 0.0,
                "cv": None,
                "moran_i": None,
                "temporal_roughness": None,
                "spatial_entropy": None,
                "provenance_risk": 0.2,
                "race_axis_source": None,
                "race_axis_target": None,
                "missing_race_share": None,
                "emission_prior_strength": None,
                "race_bridge_cv": None,
                "sensitivity_width": None,
                "bridge_mode": None,
                "state": "quarantined_descriptive",
                "dashboard_safe": "False",
                "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor", "support_aligned_by_municipality_crosswalk"]),
                "computed_at": _now(),
                "q_schema_version": "1.0",
            },
        ]


    def _vd_rows(*, vd_columns: list[str], population_row: dict[str, Any], rate_row: dict[str, Any]) -> list[dict[str, Any]]:
        pop_support = _load_json_field(population_row["support_json"])
        pop_axes = _load_json_field(population_row["axes_json"])
        rate_support = _load_json_field(rate_row["support_json"])
        rate_axes = _load_json_field(rate_row["axes_json"])

        return [
            _row_for_columns(
                vd_columns,
                {
                    "field_id": population_row["field_id"],
                    "display_name": "SIDRAPopulationTotalAnchor",
                    "technical_name": "SIDRAPopulationTotalAnchor",
                    "definition": "Official SIDRA 9606 total-category resident population anchor bounded to Population(s,t).",
                    "estimand_label": "official_resident_population_total_municipality_year",
                    "source_systems": _json(["SIDRA"]),
                    "carrier": "Population",
                    "unit": "persons",
                    "support_description": _json(pop_support),
                    "axis_description": _json(pop_axes),
                    "provenance_description": _json(["official", "sidra_9606", "bounded_total_category_anchor"]),
                    "state": "fragile",
                    "dashboard_safe": "warning",
                    "interpretation_warning": "Official SIDRA denominator anchor. Total sex/race/age categories are marginalization categories, not modeled axes.",
                    "diagnostic_role": None,
                    "topology": None,
                    "position": None,
                    "icd_group_kind": None,
                    "icd_group_id": None,
                },
            ),
            _row_for_columns(
                vd_columns,
                {
                    "field_id": rate_row["field_id"],
                    "display_name": "SIMCrudeMortalitySIDRAOfficial",
                    "technical_name": "SIMCrudeMortalitySIDRAOfficial",
                    "definition": "SIM fixture all-deaths numerator divided by official SIDRA total resident population denominator after municipality-year support alignment.",
                    "estimand_label": "fixture_crude_mortality_with_official_sidra_denominator",
                    "source_systems": _json(["SIM-DO", "SIDRA"]),
                    "carrier": "Deaths/Population",
                    "unit": "rate",
                    "support_description": _json(rate_support),
                    "axis_description": _json(rate_axes),
                    "provenance_description": _json(["official", "sidra_denominator_anchor", "sim_fixture_numerator"]),
                    "state": "quarantined_descriptive",
                    "dashboard_safe": "False",
                    "interpretation_warning": "Uses official SIDRA denominator and explicit municipality-year support alignment, but SIM numerator remains fixture-derived.",
                    "diagnostic_role": "all_deaths",
                    "topology": "none",
                    "position": None,
                    "icd_group_kind": None,
                    "icd_group_id": None,
                },
            ),
        ]


    def attach_sidra_population_anchor_to_run(
        *,
        run_dir: str | Path,
        sidra_facts_path: str | Path,
    ) -> Path:
        run_dir = Path(run_dir)
        sidra_facts_path = Path(sidra_facts_path)

        anchor = load_sidra_population_total_anchor(sidra_facts_path)

        v_path = run_dir / "V_fields.parquet"
        e_path = run_dir / "E_DAG.parquet"
        q_path = run_dir / "Q_tensor.parquet"
        vd_path = run_dir / "VariableDictionary.parquet"
        warnings_path = run_dir / "Warnings.parquet"
        qf_path = run_dir / "QuarantinedFields.parquet"
        p_path = run_dir / "P_vector.json"

        for path in [v_path, e_path, q_path, vd_path, warnings_path, qf_path, p_path]:
            if not path.exists():
                raise FileNotFoundError(f"Run bundle is missing required file: {path}")

        v = pl.read_parquet(v_path)
        all_deaths = _get_field_by_name(v, "SIMDeathsAll")

        population_row = _population_v_field(anchor, sidra_facts_path)
        support_alignment = assert_municipality_year_support_aligned(
            numerator_support=_load_json_field(all_deaths["support_json"]),
            numerator_axes=_load_json_field(all_deaths["axes_json"]),
            denominator_support=_load_json_field(population_row["support_json"]),
            denominator_axes=_load_json_field(population_row["axes_json"]),
        )
        rate_row = _rate_v_field(
            all_deaths=all_deaths,
            anchor=anchor,
            facts_path=sidra_facts_path,
            support_alignment=support_alignment,
        )

        new_field_ids = {population_row["field_id"], rate_row["field_id"]}
        new_names = {"SIDRAPopulationTotalAnchor", "SIMCrudeMortalitySIDRAOfficial"}

        v_clean = v.filter(~pl.col("name").is_in(sorted(new_names)))
        v_clean.write_parquet(v_path)
        _append_rows(v_path, [population_row, rate_row], remove_column="field_id", remove_values=new_field_ids)

        edge_rows = [
            {
                "edge_id": f"{all_deaths['field_id']}->{rate_row['field_id']}",
                "parent_field_id": all_deaths["field_id"],
                "child_field_id": rate_row["field_id"],
                "operator": "RN",
                "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor", "support_alignment": support_alignment.model()}),
                "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
                "created_at": _now(),
            },
            {
                "edge_id": f"{population_row['field_id']}->{rate_row['field_id']}",
                "parent_field_id": population_row["field_id"],
                "child_field_id": rate_row["field_id"],
                "operator": "RN",
                "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor", "support_alignment": support_alignment.model()}),
                "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
                "created_at": _now(),
            },
        ]
        e = pl.read_parquet(e_path)
        e = e.filter(pl.col("child_field_id") != rate_row["field_id"])
        e.write_parquet(e_path)
        _append_rows(e_path, edge_rows)

        _append_rows(q_path, _q_rows(population_row=population_row, rate_row=rate_row), remove_column="field_id", remove_values=new_field_ids)

        vd = pl.read_parquet(vd_path)
        vd_columns = vd.columns
        vd = vd.filter(~pl.col("field_id").is_in(sorted(new_field_ids)))
        vd.write_parquet(vd_path)
        _append_rows(vd_path, _vd_rows(vd_columns=vd_columns, population_row=population_row, rate_row=rate_row))

        warning_rows = [
            {
                "warning_id": "sidra_total_category_anchor",
                "field_id": population_row["field_id"],
                "source": "pegasus.output.sidra_denominator_anchor",
                "severity": "info",
                "code": "sidra_total_category_anchor",
                "message": "SIDRA 9606 total sex/race/age categories were used to create bounded Population(s,t).",
                "inherited_from": _json([]),
                "created_at": _now(),
            },
            {
                "warning_id": "official_sidra_denominator_anchor",
                "field_id": rate_row["field_id"],
                "source": "pegasus.output.sidra_denominator_anchor",
                "severity": "info",
                "code": "official_sidra_denominator_anchor",
                "message": "SIM fixture crude mortality has an official SIDRA total-population denominator after support alignment.",
                "inherited_from": _json([population_row["field_id"]]),
                "created_at": _now(),
            },
            {
                "warning_id": "support_aligned_by_municipality_crosswalk",
                "field_id": rate_row["field_id"],
                "source": "pegasus.geo.support",
                "severity": "info",
                "code": "support_aligned_by_municipality_crosswalk",
                "message": "DATASUS six-digit municipality support was aligned to SIDRA seven-digit support by explicit crosswalk before RN field creation.",
                "inherited_from": _json([all_deaths["field_id"], population_row["field_id"]]),
                "created_at": _now(),
            },
        ]
        _append_rows(
            warnings_path,
            warning_rows,
            remove_column="warning_id",
            remove_values={row["warning_id"] for row in warning_rows},
        )

        qf_rows = [
            {
                "field_id": population_row["field_id"],
                "state": "fragile",
                "reason": "official_anchor_pending_crosschecks",
                "warnings": _json(["sidra_total_category_anchor"]),
            },
            {
                "field_id": rate_row["field_id"],
                "state": "quarantined_descriptive",
                "reason": "sim_fixture_numerator",
                "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor", "support_aligned_by_municipality_crosswalk"]),
            },
        ]
        _append_rows(qf_path, qf_rows, remove_column="field_id", remove_values=new_field_ids)

        p = json.loads(p_path.read_text(encoding="utf-8"))
        p.setdefault("provenance", {})
        p["provenance"][population_row["field_id"]] = ["official", "sidra_9606", "bounded_total_category_anchor"]
        p["provenance"][rate_row["field_id"]] = ["official", "sidra_denominator_anchor", "sim_fixture_numerator"]
        p_path.write_text(_json(p), encoding="utf-8")

        result = validate_output_bundle(run_dir=str(run_dir))
        if not result.ok:
            raise RuntimeError("Run bundle failed validation after SIDRA denominator anchor attach: " + "; ".join(result.errors))

        return run_dir
    ''')

    write("src/pegasus/cli.py", r'''
    from __future__ import annotations

    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree
    from pegasus.sidra.api import SidraClient, SidraClientConfig
    from pegasus.workflows.acquire.datasus import run_datasus_ingest, run_datasus_normalize_sim, run_datasus_profile
    from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture
    from pegasus.workflows.acquire.sidra import (
        run_sidra_extract,
        run_sidra_metadata,
        run_sidra_metadata_fixture,
        run_sidra_normalize_fixture,
        run_sidra_plan,
        run_sidra_plan_fixture,
        sidra_runtime_config,
    )

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

        try:
            client = SidraClient(
                config=SidraClientConfig.from_mapping(
                    {
                        **sidra_runtime_config(),
                        "timeout_seconds": 5,
                        "max_retries": 0,
                    }
                )
            )
            response = client.ping()
            checks["sidra_network"] = "ok" if response.status_code < 400 else f"HTTP {response.status_code}"
        except Exception as exc:
            checks["sidra_network"] = f"failed: {exc}"

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is light: it checks connectivity but does not run heavy ingestion.[/yellow]")


    @datasus_app.command("ingest")
    def datasus_ingest(
        system: str = typer.Option(..., "--system"),
        uf: str = typer.Option(..., "--uf"),
        years: str = typer.Option(..., "--years"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
    ) -> None:
        result = run_datasus_ingest(system=system, uf=uf, years=years, dry_run=dry_run)
        for manifest, planned_path in result["planned"]:
            print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")
        for manifest, executed_path in result["executed"]:
            if manifest.status == "success":
                print(f"[green]success[/green] {manifest.system} {manifest.uf} {manifest.year_start}: {executed_path}")
            elif manifest.status == "blocked":
                print(f"[yellow]blocked[/yellow] {manifest.system} {manifest.uf} {manifest.year_start}: {manifest.error_message}")
            else:
                print(f"[red]{manifest.status}[/red] {manifest.system} {manifest.uf} {manifest.year_start}: {manifest.error_message}")
        if result["failed"]:
            raise typer.Exit(1)
        if result["blocked"]:
            raise typer.Exit(2)


    @datasus_app.command("profile")
    def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
        result = run_datasus_profile(manifest=manifest)
        if result["status"] == "blocked":
            print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
            raise typer.Exit(2)
        print(f"[green]raw profile[/green] {result['raw_profile_path']}")
        print(f"[green]processed profile[/green] {result['processed_profile_path']}")
        print(f"[green]schema comparison[/green] {result['compare_path']}")


    @datasus_app.command("normalize-sim")
    def datasus_normalize_sim(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(..., "--output"),
        source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
    ) -> None:
        result = run_datasus_normalize_sim(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )
        print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


    @efg_app.command("build-sim-fixture")
    def efg_build_sim_fixture(
        sim_events: Path = typer.Option(..., "--sim-events"),
        run_dir: Path = typer.Option(..., "--run-dir"),
        municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
    ) -> None:
        result = run_build_sim_fixture(
            sim_events_path=sim_events,
            run_dir=run_dir,
            municipality_cod6=municipality_cod6,
        )
        validation = result["validation"]
        if not validation.ok:
            _fail(validation.errors)
        print(f"[green]sim fixture EFG bundle valid[/green] {result['run_dir']}")


    @efg_app.command("attach-sidra-denominator")
    def efg_attach_sidra_denominator(
        run_dir: Path = typer.Option(..., "--run-dir"),
        sidra_facts: Path = typer.Option(..., "--sidra-facts"),
    ) -> None:
        result = run_attach_sidra_denominator(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts,
        )
        validation = result["validation"]
        if not validation.ok:
            _fail(validation.errors)
        print(f"[green]SIDRA denominator anchor attached and run bundle valid[/green] {result['run_dir']}")


    @sidra_app.command("metadata-fixture")
    def sidra_metadata_fixture(
        output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
    ) -> None:
        outputs = run_sidra_metadata_fixture(output_dir=output_dir)
        for name, path in outputs.items():
            print(f"[green]{name}[/green] {path}")


    @sidra_app.command("plan-fixture")
    def sidra_plan_fixture(
        output: Path = typer.Option(Path("data/manifests/sidra/fixture_plan.json"), "--output"),
        max_cells: int = typer.Option(49900, "--max-cells"),
    ) -> None:
        result = run_sidra_plan_fixture(output=output, max_cells=max_cells)
        print(f"[green]planned[/green] chunks={len(result['chunks'])} output={result['output']}")


    @sidra_app.command("normalize-fixture")
    def sidra_normalize_fixture(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(Path("data/processed/sidra/facts/9606/fixture.parquet"), "--output"),
    ) -> None:
        output = run_sidra_normalize_fixture(input_path=input_path, output_path=output_path)
        print(f"[green]sidra facts normalized[/green] {output}")


    @sidra_app.command("metadata")
    def sidra_metadata(
        tables: Path = typer.Option(..., "--tables"),
        level: str = typer.Option("N6", "--level"),
        output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
        raw_dir: Path = typer.Option(Path("data/metadata/sidra/raw"), "--raw-dir"),
    ) -> None:
        try:
            result = run_sidra_metadata(tables=tables, level=level, output_dir=output_dir, raw_dir=raw_dir)
        except ValueError as exc:
            print(f"[red]ERROR[/red] {exc}")
            raise typer.Exit(1) from exc
        print(f"[green]metadata rebuilt[/green] tables={len(result['table_ids'])}")
        for name, path in result["outputs"].items():
            print(f"[green]{name}[/green] {path}")


    @sidra_app.command("plan")
    def sidra_plan(
        view: str = typer.Option(..., "--view"),
        metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        result = run_sidra_plan(view=view, metadata_dir=metadata_dir, output=output)
        print(f"[green]planned[/green] view={view} chunks={len(result['chunks'])} output={result['output']}")


    @sidra_app.command("extract")
    def sidra_extract(
        plan: Path = typer.Option(..., "--plan"),
        metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        concurrency: int | None = typer.Option(None, "--concurrency"),
        log_path: Path | None = typer.Option(None, "--log"),
    ) -> None:
        result = run_sidra_extract(
            plan=plan,
            metadata_dir=metadata_dir,
            dry_run=dry_run,
            concurrency=concurrency,
            log_path=log_path,
        )
        if result["status"] == "dry_run":
            print(
                f"[cyan]dry-run[/cyan] chunks={len(result['chunks'])} "
                f"estimated_cells={result['estimated_cells']} metadata_hash={result['metadata_hash']}"
            )
            return
        failures = result["failures"]
        print(f"[green]extract complete[/green] chunks={len(result['results'])} failures={len(failures)} log={result['log_path']}")
        if failures:
            raise typer.Exit(1)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    ''')

    write("tests/unit/test_geo_support_alignment.py", r'''
    import pytest

    from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7, ibge_cod7_to_datasus_cod6
    from pegasus.geo.support import SupportAlignmentError, assert_municipality_year_support_aligned


    def test_datasus_to_ibge_smoke_crosswalk_is_explicit():
        assert datasus_cod6_to_ibge_cod7("270430") == "2704302"
        assert datasus_cod6_to_ibge_cod7("270030") == "2700300"
        assert ibge_cod7_to_datasus_cod6("2704302") == "270430"


    def test_support_alignment_accepts_cod6_numerator_and_cod7_denominator():
        result = assert_municipality_year_support_aligned(
            numerator_support={"years": [2022], "municipalities": ["270430"]},
            numerator_axes={"geography": "mun_residence_cod6"},
            denominator_support={"years": [2022], "municipalities": ["2704302"]},
            denominator_axes={"geography": "municipality"},
        )
        assert result.aligned is True
        assert result.numerator_municipalities_ibge_cod7 == ["2704302"]
        assert result.denominator_municipalities_ibge_cod7 == ["2704302"]


    def test_support_alignment_rejects_unmatched_municipality_sets():
        with pytest.raises(SupportAlignmentError, match="municipality_support_mismatch"):
            assert_municipality_year_support_aligned(
                numerator_support={"years": [2022], "municipalities": ["270030", "270430"]},
                numerator_axes={"geography": "mun_residence_cod6"},
                denominator_support={"years": [2022], "municipalities": ["2704302"]},
                denominator_axes={"geography": "municipality"},
            )
    ''')

    write("tests/unit/test_sidra_denominator_anchor.py", r'''
    from pathlib import Path
    import json

    import polars as pl
    import pytest

    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.geo.support import SupportAlignmentError
    from pegasus.output.sidra_denominator_anchor import attach_sidra_population_anchor_to_run
    from pegasus.output.validate import validate_output_bundle
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.she.population.sidra_anchor import load_sidra_population_total_anchor
    from pegasus.workflows.construct.build_efg import build_sim_fixture_efg_run


    LIVE_SHAPE_PAYLOAD = [
        {
            "NC": "Nível Territorial (Código)",
            "NN": "Nível Territorial",
            "MC": "Unidade de Medida (Código)",
            "MN": "Unidade de Medida",
            "V": "Valor",
            "D1C": "Município (Código)",
            "D1N": "Município",
            "D2C": "Ano (Código)",
            "D2N": "Ano",
            "D3C": "Variável (Código)",
            "D3N": "Variável",
            "D4C": "Sexo (Código)",
            "D4N": "Sexo",
            "D5C": "Cor ou raça (Código)",
            "D5N": "Cor ou raça",
            "D6C": "Idade (Código)",
            "D6N": "Idade",
        },
        {
            "NC": "6",
            "NN": "Município",
            "MC": "45",
            "MN": "Pessoas",
            "V": "957916",
            "D1C": "2704302",
            "D1N": "Maceió (AL)",
            "D2C": "2022",
            "D2N": "2022",
            "D3C": "93",
            "D3N": "População residente",
            "D4C": "6794",
            "D4N": "Total",
            "D5C": "95251",
            "D5N": "Total",
            "D6C": "100362",
            "D6N": "Total",
        },
    ]


    CHUNK_REQUEST = {
        "table_id": "9606",
        "variables": ["93"],
        "periods": ["2022"],
        "locality_level": "N6",
        "localities": ["2704302"],
        "classifications": {
            "86": ["95251"],
            "2": ["6794"],
            "287": ["100362"],
        },
    }


    def _sidra_facts(tmp_path: Path) -> Path:
        facts = normalize_sidra_payload_to_facts(
            LIVE_SHAPE_PAYLOAD,
            table_id="9606",
            request_hash="request_hash",
            metadata_hash="a" * 64,
            chunk_request=CHUNK_REQUEST,
            unit_by_variable=None,
            fetched_at="2026-06-06T00:00:00+00:00",
        )
        path = tmp_path / "sidra_facts.parquet"
        write_facts_parquet(facts, output_path=path)
        return path


    def _normalized_sim_events(tmp_path: Path) -> Path:
        sim_events = tmp_path / "sim_events.parquet"
        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=sim_events,
            source_manifest_hash="fixture_manifest_hash",
        )
        return sim_events


    def test_total_population_anchor_loader(tmp_path: Path):
        facts_path = _sidra_facts(tmp_path)
        anchor = load_sidra_population_total_anchor(facts_path)

        assert anchor.table_id == "9606"
        assert anchor.variable_id == "93"
        assert anchor.period == "2022"
        assert anchor.locality_level == "N6"
        assert anchor.locality_id == "2704302"
        assert anchor.value == 957916.0
        assert anchor.metadata_hash == "a" * 64


    def test_attach_rejects_unaligned_fixture_denominator(tmp_path: Path):
        sim_events = _normalized_sim_events(tmp_path)
        run_dir = tmp_path / "run_unfiltered"
        sidra_facts = _sidra_facts(tmp_path)

        build_sim_fixture_efg_run(sim_events_path=sim_events, run_dir=run_dir)

        with pytest.raises(SupportAlignmentError, match="municipality_support_mismatch"):
            attach_sidra_population_anchor_to_run(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts,
            )

        result = validate_output_bundle(run_dir=str(run_dir))
        assert result.ok, result.errors
        v = pl.read_parquet(run_dir / "V_fields.parquet")
        assert "SIMCrudeMortalitySIDRAOfficial" not in set(v["name"].to_list())


    def test_attach_sidra_anchor_to_valid_run_bundle_after_support_filter(tmp_path: Path):
        sim_events = _normalized_sim_events(tmp_path)
        run_dir = tmp_path / "run_maceio"
        sidra_facts = _sidra_facts(tmp_path)

        build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
            municipality_cod6="270430",
        )

        attach_sidra_population_anchor_to_run(
            run_dir=run_dir,
            sidra_facts_path=sidra_facts,
        )

        result = validate_output_bundle(run_dir=str(run_dir))
        assert result.ok, result.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        names = set(v["name"].to_list())

        assert "SIDRAPopulationTotalAnchor" in names
        assert "SIMCrudeMortalitySIDRAOfficial" in names

        pop = v.filter(pl.col("name") == "SIDRAPopulationTotalAnchor").row(0, named=True)
        rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)

        assert pop["carrier"] == "Population"
        assert pop["source"] == '["SIDRA"]'
        assert "bounded_total_category_anchor" in pop["provenance"]

        rate_support = json.loads(rate["support_json"])
        assert rate["carrier"] == "Deaths/Population"
        assert "SIDRA" in rate["source"]
        assert rate_support["municipalities"] == ["2704302"]
        assert rate_support["support_alignment"]["aligned"] is True
        assert rate_support["support_alignment"]["numerator_municipalities_source"] == ["270430"]
        assert rate_support["support_alignment"]["denominator_municipalities_source"] == ["2704302"]

        e = pl.read_parquet(run_dir / "E_DAG.parquet")
        assert rate["field_id"] in set(e["child_field_id"].to_list())

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        assert pop["field_id"] in set(q["field_id"].to_list())
        assert rate["field_id"] in set(q["field_id"].to_list())
    ''')

    write("tests/unit/test_cli_workflow_boundaries.py", r'''
    import ast
    from pathlib import Path


    def test_cli_uses_workflow_wrappers_for_mutating_slice_commands():
        tree = ast.parse(Path("src/pegasus/cli.py").read_text(encoding="utf-8"))
        imported_from = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_from.append(node.module)

        forbidden = {
            "pegasus.sidra.extract",
            "pegasus.sidra.metadata",
            "pegasus.sidra.plan",
            "pegasus.sidra.registry",
            "pegasus.sidra.facts",
            "pegasus.datasus.normalize",
            "pegasus.datasus.manifests",
            "pegasus.datasus.subprocess",
            "pegasus.output.sidra_denominator_anchor",
        }
        assert not (set(imported_from) & forbidden)
        assert "pegasus.workflows.acquire.sidra" in imported_from
        assert "pegasus.workflows.acquire.datasus" in imported_from
        assert "pegasus.workflows.efg" in imported_from
    ''')

    write("tests/integration/test_slice2c1_support_integration.py", r'''
    from pathlib import Path
    import json

    import polars as pl

    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.output.validate import validate_output_bundle
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.workflows.construct.build_efg import build_sim_fixture_efg_run
    from pegasus.workflows.efg import run_attach_sidra_denominator


    def _sidra_facts(tmp_path: Path) -> Path:
        payload = [
            {
                "NC": "Nível Territorial (Código)", "NN": "Nível Territorial", "MC": "Unidade de Medida (Código)",
                "MN": "Unidade de Medida", "V": "Valor", "D1C": "Município (Código)", "D1N": "Município",
                "D2C": "Ano (Código)", "D2N": "Ano", "D3C": "Variável (Código)", "D3N": "Variável",
                "D4C": "Sexo (Código)", "D4N": "Sexo", "D5C": "Cor ou raça (Código)", "D5N": "Cor ou raça",
                "D6C": "Idade (Código)", "D6N": "Idade",
            },
            {
                "NC": "6", "NN": "Município", "MC": "45", "MN": "Pessoas", "V": "957916",
                "D1C": "2704302", "D1N": "Maceió (AL)", "D2C": "2022", "D2N": "2022",
                "D3C": "93", "D3N": "População residente", "D4C": "6794", "D4N": "Total",
                "D5C": "95251", "D5N": "Total", "D6C": "100362", "D6N": "Total",
            },
        ]
        chunk_request = {
            "table_id": "9606",
            "variables": ["93"],
            "periods": ["2022"],
            "locality_level": "N6",
            "localities": ["2704302"],
            "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
        }
        facts = normalize_sidra_payload_to_facts(
            payload,
            table_id="9606",
            request_hash="request_hash",
            metadata_hash="b" * 64,
            chunk_request=chunk_request,
            fetched_at="2026-06-06T00:00:00+00:00",
        )
        out = tmp_path / "sidra_facts.parquet"
        write_facts_parquet(facts, output_path=out)
        return out


    def test_maceio_filtered_sim_sidra_denominator_integration_validates(tmp_path: Path):
        sim_events = tmp_path / "sim_events.parquet"
        run_dir = tmp_path / "run"
        sidra_facts = _sidra_facts(tmp_path)

        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=sim_events,
            source_manifest_hash="fixture_manifest_hash",
        )
        build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
            municipality_cod6="270430",
        )
        result = run_attach_sidra_denominator(run_dir=run_dir, sidra_facts_path=sidra_facts)
        assert result["validation"].ok, result["validation"].errors

        bundle_validation = validate_output_bundle(run_dir=str(run_dir))
        assert bundle_validation.ok, bundle_validation.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)
        support = json.loads(rate["support_json"])
        assert support["support_alignment"]["aligned"] is True
        assert support["support_alignment"]["crosswalk"] == "datasus_cod6_to_ibge_cod7"
        assert support["municipalities"] == ["2704302"]

        warnings = pl.read_parquet(run_dir / "Warnings.parquet")
        assert "support_aligned_by_municipality_crosswalk" in set(warnings["warning_id"].to_list())
    ''')

    write("scripts/dev/audits/audit_slice2c1_support_semantics.py", r'''
    from __future__ import annotations

    import argparse
    import json
    from pathlib import Path

    import polars as pl


    def _json(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(str(value))


    def main() -> None:
        parser = argparse.ArgumentParser(description="Audit PegaSUS Slice 2C.1 support semantics.")
        parser.add_argument("--run", required=True, help="Run bundle directory.")
        args = parser.parse_args()

        run_dir = Path(args.run)
        v_path = run_dir / "V_fields.parquet"
        e_path = run_dir / "E_DAG.parquet"
        warnings_path = run_dir / "Warnings.parquet"

        for path in [v_path, e_path, warnings_path]:
            if not path.exists():
                raise SystemExit(f"Missing required run artifact: {path}")

        v = pl.read_parquet(v_path)
        e = pl.read_parquet(e_path)
        warnings = pl.read_parquet(warnings_path)

        failures: list[str] = []
        rows = v.filter(pl.col("operator") == "RN").to_dicts()
        sidra_rates = [row for row in rows if "SIDRA" in str(row.get("source"))]

        for row in sidra_rates:
            support = _json(row.get("support_json"))
            alignment = support.get("support_alignment")
            if not alignment:
                failures.append(f"{row['name']} lacks support_alignment metadata.")
                continue
            if alignment.get("aligned") is not True:
                failures.append(f"{row['name']} support_alignment is not aligned: {alignment}")
            if alignment.get("numerator_municipalities_ibge_cod7") != alignment.get("denominator_municipalities_ibge_cod7"):
                failures.append(f"{row['name']} numerator/denominator cod7 sets differ: {alignment}")
            if support.get("municipality_crosswalk") != "datasus_cod6_to_ibge_cod7":
                failures.append(f"{row['name']} missing explicit municipality_crosswalk marker.")

            parents = e.filter(pl.col("child_field_id") == row["field_id"])
            if parents.height < 2:
                failures.append(f"{row['name']} does not have at least numerator and denominator parent edges.")

        if sidra_rates:
            warning_ids = set(warnings["warning_id"].to_list())
            if "support_aligned_by_municipality_crosswalk" not in warning_ids:
                failures.append("SIDRA rate exists but support_aligned_by_municipality_crosswalk warning is missing.")

        if failures:
            print("AUDIT FAILURES")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)

        print(f"AUDIT PASSED: checked {len(sidra_rates)} SIDRA-linked RN field(s) for support alignment semantics.")


    if __name__ == "__main__":
        main()
    ''')

    print("Applied Slice 2C.1: PegaSUS contract, workflow-thin CLI, explicit municipality support alignment, and integration audits.")


if __name__ == "__main__":
    main()
