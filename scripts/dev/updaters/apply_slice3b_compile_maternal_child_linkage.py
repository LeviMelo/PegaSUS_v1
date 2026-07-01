from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

ROOT = Path.cwd()

CREATE_OR_REPLACE = [
    "src/pegasus/she/maternal_child_linkage.py",
    "src/pegasus/output/maternal_child_compile_attach.py",
    "tests/unit/test_maternal_child_linkage.py",
    "tests/integration/test_slice3b_compile_maternal_child_linkage.py",
    "scripts/dev/audits/audit_slice3b_compile_maternal_child_linkage.py",
]

PATCH = [
    "config/intents/alagoas_smoke.json",
    "src/pegasus/workflows/compile.py",
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
        "config/intents/alagoas_smoke.json",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/output/sidra_denominator_anchor.py",
        "src/pegasus/output/sinasc_efg_bundle.py",
        "src/pegasus/datasus/sinasc_normalize.py",
        "src/pegasus/she/maternal_child.py",
        "src/pegasus/workflows/sinasc.py",
        "tests/fixtures/datasus/sim_do_fixture.csv",
        "tests/fixtures/datasus/sinasc_fixture.csv",
        "scripts/dev/audits/audit_slice2d_compile_smoke.py",
        "scripts/dev/audits/audit_slice3a_sinasc_maternal_child.py",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 3B preflight failed; missing expected files: {missing}")

    compile_py = read("src/pegasus/workflows/compile.py")
    markers = [
        "def run_compile(",
        "run_datasus_normalize_sim(",
        "run_attach_sidra_denominator(",
        "SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD",
    ]
    for marker in markers:
        if marker not in compile_py:
            raise RuntimeError(f"Slice 3B preflight failed; compile marker not found: {marker}")

    if "run_datasus_normalize_sinasc" not in read("src/pegasus/workflows/sinasc.py"):
        raise RuntimeError("Slice 3B preflight failed: Slice 3A SINASC workflow not detected.")

    validator = read("src/pegasus/output/validate.py")
    if "ReproducibilityManifest" not in validator or "Q_tensor" not in validator:
        raise RuntimeError("Slice 3B preflight failed: hardened output validator not detected.")


def patch_intent() -> None:
    path = rel("config/intents/alagoas_smoke.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mandatory = list(payload.get("mandatory_fields") or [])
    additions = [
        "SINASCLiveBirthsAll",
        "SINASCCrudeBirthRateSIDRAOfficial",
        "SINASCLowBirthWeightPrevalence",
        "SINASCPrematurityPrevalence",
        "SINASCCongenitalAnomalyPrevalence",
        "SIMInfantMortalitySINASCBirths",
        "SIMNeonatalMortalitySINASCBirths",
        "SIMPostNeonatalMortalitySINASCBirths",
    ]
    for item in additions:
        if item not in mandatory:
            mandatory.append(item)
    payload["mandatory_fields"] = mandatory
    weights = dict(payload.get("system_weights") or {})
    weights.setdefault("SINASC", 1.0)
    payload["system_weights"] = weights
    context_policy = list(payload.get("context_policy") or [])
    for item in ["sinasc_maternal_child_compile", "live_birth_denominator_linkage"]:
        if item not in context_policy:
            context_policy.append(item)
    payload["context_policy"] = context_policy
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def patch_compile() -> None:
    path = rel("src/pegasus/workflows/compile.py")
    text = path.read_text(encoding="utf-8")

    import_a = "from pegasus.output.maternal_child_compile_attach import attach_maternal_child_compile_fields\n"
    if import_a not in text:
        marker = "from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest\n"
        if marker not in text:
            raise RuntimeError("Cannot patch compile imports: reproducibility import marker not found.")
        text = text.replace(marker, marker + import_a, 1)

    import_b = "from pegasus.workflows.acquire.sinasc import run_datasus_normalize_sinasc\n"
    if import_b not in text:
        marker = "from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture\n"
        if marker not in text:
            raise RuntimeError("Cannot patch compile imports: EFG workflow import marker not found.")
        text = text.replace(marker, marker + import_b, 1)

    if "raw_sinasc_fixture_source = Path(\"tests/fixtures/datasus/sinasc_fixture.csv\")" not in text:
        old = '''        raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
        if not raw_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")
'''
        new = '''        raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
        raw_sinasc_fixture_source = Path("tests/fixtures/datasus/sinasc_fixture.csv")
        if not raw_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")
        if not raw_sinasc_fixture_source.exists():
            raise FileNotFoundError(f"Missing SINASC smoke fixture: {raw_sinasc_fixture_source}")
'''
        if old not in text:
            raise RuntimeError("Cannot patch compile fixture source block; expected SIM fixture block not found.")
        text = text.replace(old, new, 1)

    if "sinasc_events_path = data_root / \"processed\" / \"datasus\" / \"SINASC\" / \"fixture\" / \"sinasc_events.parquet\"" not in text:
        old = '''        raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
        sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_maceio.parquet"
'''
        new = '''        raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
        raw_sinasc_cache_path = data_root / "raw" / "datasus" / "SINASC" / "fixture" / "sinasc_fixture.csv"
        sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        sinasc_events_path = data_root / "processed" / "datasus" / "SINASC" / "fixture" / "sinasc_events.parquet"
        sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_maceio.parquet"
'''
        if old not in text:
            raise RuntimeError("Cannot patch compile path declarations; expected smoke path block not found.")
        text = text.replace(old, new, 1)

    if "source_hashes[\"sinasc_raw_fixture\"]" not in text:
        old = '''        with telemetry.stage("datasus_acquire"):
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_fixture_source, raw_cache_path)
            source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)
'''
        new = '''        with telemetry.stage("datasus_acquire"):
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw_sinasc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_fixture_source, raw_cache_path)
            shutil.copyfile(raw_sinasc_fixture_source, raw_sinasc_cache_path)
            source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)
            source_hashes["sinasc_raw_fixture"] = sha256_file(raw_sinasc_cache_path)
'''
        if old not in text:
            raise RuntimeError("Cannot patch compile datasus_acquire block; expected block not found.")
        text = text.replace(old, new, 1)

    if "source_hashes[\"sinasc_processed_events\"]" not in text:
        old = '''        with telemetry.stage("datasus_decode"):
            run_datasus_normalize_sim(
                input_path=raw_cache_path,
                output_path=sim_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            source_hashes["sim_processed_events"] = sha256_file(sim_events_path)
'''
        new = '''        with telemetry.stage("datasus_decode"):
            run_datasus_normalize_sim(
                input_path=raw_cache_path,
                output_path=sim_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            run_datasus_normalize_sinasc(
                input_path=raw_sinasc_cache_path,
                output_path=sinasc_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            source_hashes["sim_processed_events"] = sha256_file(sim_events_path)
            source_hashes["sinasc_processed_events"] = sha256_file(sinasc_events_path)
'''
        if old not in text:
            raise RuntimeError("Cannot patch compile datasus_decode block; expected block not found.")
        text = text.replace(old, new, 1)

    if "attach_maternal_child_compile_fields(" not in text:
        old = '''        with telemetry.stage("she_build"):
            run_attach_sidra_denominator(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
            )
'''
        new = '''        with telemetry.stage("she_build"):
            run_attach_sidra_denominator(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
            )
            attach_maternal_child_compile_fields(
                run_dir=run_dir,
                sinasc_events_path=sinasc_events_path,
                sim_events_path=sim_events_path,
                municipality_cod6=municipality_cod6,
            )
'''
        if old not in text:
            raise RuntimeError("Cannot patch compile she_build block; expected SIDRA denominator attach block not found.")
        text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    preflight()
    print("Slice 3B updater preflight passed.")
    print("CREATE/REPLACE:")
    for p in CREATE_OR_REPLACE:
        print(f"  {p}")
    print("PATCH:")
    for p in PATCH:
        print(f"  {p}")

    write("src/pegasus/she/maternal_child_linkage.py", r'''
    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import polars as pl


    @dataclass(frozen=True)
    class MaternalChildLinkedSummary:
        years: list[int]
        municipalities_cod6: list[str]
        municipalities_ibge_cod7: list[str]
        births_total: int
        low_birth_weight_births: int
        prematurity_births: int
        congenital_anomaly_births: int
        infant_deaths: int
        neonatal_deaths: int
        postneonatal_deaths: int
        sim_death_records_considered: int
        sinasc_birth_records_considered: int
        denominator_population: float | None = None

        def support_cod6(self) -> dict[str, Any]:
            return {
                "time": {"years": self.years},
                "geography": {"municipality_cod6": self.municipalities_cod6},
                "n_events": self.births_total,
            }

        def support_cod7(self) -> dict[str, Any]:
            return {
                "time": {"years": self.years},
                "geography": {"municipality_ibge_cod7": self.municipalities_ibge_cod7},
                "n_events": self.births_total,
            }

        def rates(self) -> dict[str, float | None]:
            birth_denom = float(self.births_total) if self.births_total else 0.0
            pop_denom = float(self.denominator_population) if self.denominator_population else 0.0
            return {
                "low_birth_weight_prevalence": self.low_birth_weight_births / birth_denom if birth_denom else None,
                "prematurity_prevalence": self.prematurity_births / birth_denom if birth_denom else None,
                "congenital_anomaly_prevalence": self.congenital_anomaly_births / birth_denom if birth_denom else None,
                "infant_mortality": self.infant_deaths / birth_denom if birth_denom else None,
                "neonatal_mortality": self.neonatal_deaths / birth_denom if birth_denom else None,
                "postneonatal_mortality": self.postneonatal_deaths / birth_denom if birth_denom else None,
                "crude_birth_rate": self.births_total / pop_denom if pop_denom else None,
            }


    def _has_column(df: pl.DataFrame, name: str) -> bool:
        return name in df.columns


    def _filter_municipality(df: pl.DataFrame, column: str, municipality_cod6: str | None) -> pl.DataFrame:
        if municipality_cod6 is None or column not in df.columns:
            return df
        return df.filter(pl.col(column) == str(municipality_cod6))


    def _birth_year_column(df: pl.DataFrame) -> str:
        if "birth_year" in df.columns:
            return "birth_year"
        if "year" in df.columns:
            return "year"
        raise ValueError("SINASC normalized events lack birth_year/year column.")


    def _bool_count(df: pl.DataFrame, column: str) -> int:
        if column not in df.columns:
            return 0
        return int(df.filter(pl.col(column) == True).height)  # noqa: E712 - data-state comparison


    def _valid_sinasc(df: pl.DataFrame) -> pl.DataFrame:
        if "record_state" not in df.columns:
            return df
        return df.filter(pl.col("record_state") == "valid")


    def _liveborn_death_filter(df: pl.DataFrame) -> pl.DataFrame:
        if "age_days" not in df.columns:
            return df.slice(0, 0)
        filtered = df.filter(pl.col("age_days").is_not_null() & (pl.col("age_days") >= 0))
        if "death_type" in filtered.columns:
            # SIM TIPOBITO fetal-death coding is source-specific; fixture smoke keeps non-fetal deaths.
            # Do not coerce missing death_type to survived/fetal. Only explicit fetal-like states are excluded.
            filtered = filtered.filter(~pl.col("death_type").cast(pl.Utf8).str.to_lowercase().is_in(["1", "fetal", "obito_fetal", "óbito fetal"]))
        return filtered


    def summarize_maternal_child_linkage(
        *,
        sinasc_events_path: str | Path,
        sim_events_path: str | Path,
        municipality_cod6: str | None,
        municipality_ibge_cod7: str | None,
        denominator_population: float | None = None,
    ) -> MaternalChildLinkedSummary:
        sinasc = pl.read_parquet(sinasc_events_path)
        sim = pl.read_parquet(sim_events_path)

        sinasc = _filter_municipality(sinasc, "mun_residence_cod6", municipality_cod6)
        sim = _filter_municipality(sim, "mun_residence_cod6", municipality_cod6)
        sinasc_valid = _valid_sinasc(sinasc)

        year_col = _birth_year_column(sinasc_valid)
        years = sorted(int(x) for x in sinasc_valid[year_col].drop_nulls().unique().to_list())
        if years and "year" in sim.columns:
            sim = sim.filter(pl.col("year").is_in(years))

        municipalities = (
            sorted(str(x) for x in sinasc_valid["mun_residence_cod6"].drop_nulls().unique().to_list())
            if "mun_residence_cod6" in sinasc_valid.columns
            else ([] if municipality_cod6 is None else [str(municipality_cod6)])
        )
        if municipality_cod6 is not None and str(municipality_cod6) not in municipalities and sinasc_valid.height > 0:
            municipalities.append(str(municipality_cod6))
        municipalities = sorted(set(municipalities))

        cod7s = [str(municipality_ibge_cod7)] if municipality_ibge_cod7 else []

        infant = _liveborn_death_filter(sim)
        infant = infant.filter(pl.col("age_days") < 365.25) if infant.height else infant
        neonatal = infant.filter(pl.col("age_days") < 28) if infant.height else infant
        postneonatal = infant.filter((pl.col("age_days") >= 28) & (pl.col("age_days") < 365.25)) if infant.height else infant

        return MaternalChildLinkedSummary(
            years=years,
            municipalities_cod6=municipalities,
            municipalities_ibge_cod7=cod7s,
            births_total=int(sinasc_valid.height),
            low_birth_weight_births=_bool_count(sinasc_valid, "low_birth_weight_flag"),
            prematurity_births=_bool_count(sinasc_valid, "prematurity_flag"),
            congenital_anomaly_births=_bool_count(sinasc_valid, "congenital_anomaly_flag"),
            infant_deaths=int(infant.height),
            neonatal_deaths=int(neonatal.height),
            postneonatal_deaths=int(postneonatal.height),
            sim_death_records_considered=int(sim.height),
            sinasc_birth_records_considered=int(sinasc_valid.height),
            denominator_population=denominator_population,
        )
    ''')

    write("src/pegasus/output/maternal_child_compile_attach.py", r'''
    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import pyarrow as pa
    import pyarrow.parquet as pq
    import polars as pl

    from pegasus.core.hashing import sha256_file, sha256_text
    from pegasus.geo.municipality_crosswalk import datasus_cod6_to_ibge_cod7
    from pegasus.she.maternal_child_linkage import MaternalChildLinkedSummary, summarize_maternal_child_linkage


    FIELD_IDS = {
        "SINASCLiveBirthsAll",
        "SINASCLowBirthWeightBirths",
        "SINASCPrematurityBirths",
        "SINASCCongenitalAnomalyBirths",
        "SIMInfantDeathsForSINASCBirths",
        "SIMNeonatalDeathsForSINASCBirths",
        "SIMPostNeonatalDeathsForSINASCBirths",
        "SINASCCrudeBirthRateSIDRAOfficial",
        "SINASCLowBirthWeightPrevalence",
        "SINASCPrematurityPrevalence",
        "SINASCCongenitalAnomalyPrevalence",
        "SIMInfantMortalitySINASCBirths",
        "SIMNeonatalMortalitySINASCBirths",
        "SIMPostNeonatalMortalitySINASCBirths",
    }


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


    def _load_json(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        text = str(value)
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return pl.read_parquet(path).to_dicts()


    def _write_rows_like(path: Path, rows: list[dict[str, Any]]) -> None:
        schema = pq.read_table(path).schema
        fixed = [{field.name: row.get(field.name) for field in schema} for row in rows]
        table = pa.Table.from_pylist(fixed, schema=schema) if fixed else pa.Table.from_pylist([], schema=schema)
        pq.write_table(table, path)


    def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str, remove_values: set[str]) -> None:
        existing = _read_rows(path)
        kept = [row for row in existing if str(row.get(remove_column)) not in remove_values]
        _write_rows_like(path, kept + rows)


    def _append_edges(path: Path, rows: list[dict[str, Any]]) -> None:
        existing = _read_rows(path)
        edge_ids = {str(row["edge_id"]) for row in rows}
        child_ids = FIELD_IDS
        kept = [
            row
            for row in existing
            if str(row.get("edge_id")) not in edge_ids and str(row.get("child_field_id")) not in child_ids
        ]
        _write_rows_like(path, kept + rows)


    def _field_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
        for row in rows:
            if row.get("name") == name or row.get("field_id") == name:
                return row
        raise ValueError(f"Field not found in run bundle: {name}")


    def _population_value(population_row: dict[str, Any]) -> float:
        support = _load_json(population_row.get("support_json"))
        for key in ["n_denom", "n_eff", "n_events"]:
            value = support.get(key)
            if value is not None:
                return float(value)
        raise ValueError("SIDRA population anchor support does not expose n_denom/n_eff/n_events.")


    def _support_alignment(summary: MaternalChildLinkedSummary) -> dict[str, Any]:
        return {
            "aligned": True,
            "reason": "municipality_year_exact_after_datasus_cod6_to_ibge_cod7_crosswalk",
            "numerator_municipalities_cod6": summary.municipalities_cod6,
            "numerator_municipalities_ibge_cod7": summary.municipalities_ibge_cod7,
            "denominator_municipalities_ibge_cod7": summary.municipalities_ibge_cod7,
            "numerator_years": summary.years,
            "denominator_years": summary.years,
            "crosswalk": "explicit_smoke_municipality_crosswalk",
        }


    def _lineage(field_id: str, parents: list[str], operator: str, support: dict[str, Any]) -> str:
        return sha256_text(_json({"field_id": field_id, "parents": parents, "operator": operator, "support": support}))


    def _field_row(
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
        operator: str,
        parents: list[str],
        warnings: list[str],
        state: str = "quarantined_descriptive",
        dashboard_safe: str = "False",
    ) -> dict[str, Any]:
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
            "provenance": _json(source + ["slice3b_compile_linkage"]),
            "state": state,
            "dashboard_safe": dashboard_safe,
            "warnings": _json(warnings),
            "lineage_hash": _lineage(field_id, parents, operator, support),
            "registry_hash": "maternal_child_linkage_registry_v1",
            "materialization_state": "materialized",
            "path": "Tables/maternal_child_linkage_summary.parquet",
        }


    def _q_row(field: dict[str, Any], *, n_events: float | int | None, n_denom: float | int | None, warnings: list[str]) -> dict[str, Any]:
        n_eff = n_denom if n_denom is not None else n_events
        return {
            "field_id": field["field_id"],
            "n_events": float(n_events) if n_events is not None else None,
            "n_denom": float(n_denom) if n_denom is not None else None,
            "n_eff": float(n_eff) if n_eff is not None else None,
            "cov_S": 1.0,
            "cov_T": 1.0,
            "missingness": 0.0,
            "zero_inflation": 1.0 if n_events == 0 else 0.0,
            "denom_fragility": 0.0 if n_denom else None,
            "cv": None,
            "moran_i": None,
            "temporal_roughness": None,
            "spatial_entropy": None,
            "provenance_risk": 0.25,
            "race_axis_source": None,
            "race_axis_target": None,
            "missing_race_share": None,
            "emission_prior_strength": None,
            "race_bridge_cv": None,
            "sensitivity_width": None,
            "bridge_mode": None,
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "warnings": _json(warnings),
            "computed_at": _now(),
            "q_schema_version": "1.0",
        }


    def _vd_row(field: dict[str, Any], *, definition: str, estimand: str, interpretation_warning: str) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "display_name": field["name"],
            "technical_name": field["field_id"],
            "definition": definition,
            "estimand_label": estimand,
            "source_systems": field["source"],
            "carrier": field["carrier"],
            "unit": field["unit"],
            "support_description": field["support_json"],
            "axis_description": field["axes_json"],
            "provenance_description": field["provenance"],
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "interpretation_warning": interpretation_warning,
            "diagnostic_role": None,
            "topology": None,
            "position": None,
            "icd_group_kind": None,
            "icd_group_id": None,
        }


    def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": edge_id,
            "parent_field_id": parent,
            "child_field_id": child,
            "operator": operator,
            "operator_params_json": _json(params),
            "registry_versions_json": _json({"maternal_child_linkage": "v1"}),
            "created_at": _now(),
        }


    def _warning_rows(summary: MaternalChildLinkedSummary) -> list[dict[str, Any]]:
        return [
            {
                "warning_id": "slice3b_maternal_child_fixture_rates",
                "field_id": "run",
                "severity": "warning",
                "message": "Slice 3B computes maternal-child smoke rates from fixture SINASC/SIM records over explicit Maceio support; outputs are structurally valid but not inferential estimates.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "slice3b_mortality_birth_denominator_linkage",
                "field_id": "run",
                "severity": "info",
                "message": _json({
                    "births_total": summary.births_total,
                    "infant_deaths": summary.infant_deaths,
                    "neonatal_deaths": summary.neonatal_deaths,
                    "postneonatal_deaths": summary.postneonatal_deaths,
                    "support_alignment": _support_alignment(summary),
                }),
                "created_at": _now(),
                "inherited_from": None,
            },
        ]


    def _build_fields(summary: MaternalChildLinkedSummary, population_row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        alignment = _support_alignment(summary)
        support_births = {
            **summary.support_cod6(),
            "municipality_ibge_cod7": summary.municipalities_ibge_cod7,
            "support_alignment": alignment,
        }
        support_pop = {
            "time": {"years": summary.years},
            "geography": {"municipality_ibge_cod7": summary.municipalities_ibge_cod7},
            "n_events": summary.births_total,
            "n_denom": summary.denominator_population,
            "n_eff": summary.denominator_population,
            "support_alignment": alignment,
        }
        axes_common = {
            "time": "year",
            "geography": "mun_residence_cod6",
            "maternal_race_axis": "RACACORMAE_admin",
            "newborn_race_axis": "RACACOR_admin",
            "race_bridge": None,
        }
        axes_rate = {**axes_common, "support_alignment": alignment}
        warnings = ["fixture_small_n", "slice3b_support_aligned_maternal_child", "race_axes_not_bridged"]

        specs_count = [
            ("SINASCLiveBirthsAll", "SINASC Live Births All", summary.births_total, "SINASC", "LiveBirths", "live_birth_count", "identity_count"),
            ("SINASCLowBirthWeightBirths", "SINASC Low Birth Weight Births", summary.low_birth_weight_births, "SINASC", "LiveBirths", "low_birth_weight_count", "indicator_count"),
            ("SINASCPrematurityBirths", "SINASC Prematurity Births", summary.prematurity_births, "SINASC", "LiveBirths", "prematurity_count", "indicator_count"),
            ("SINASCCongenitalAnomalyBirths", "SINASC Congenital Anomaly Births", summary.congenital_anomaly_births, "SINASC", "LiveBirths", "congenital_anomaly_count", "indicator_count"),
            ("SIMInfantDeathsForSINASCBirths", "SIM Infant Deaths for SINASC Birth Denominator", summary.infant_deaths, "SIM-DO", "Deaths", "infant_death_count", "age_days_indicator_count"),
            ("SIMNeonatalDeathsForSINASCBirths", "SIM Neonatal Deaths for SINASC Birth Denominator", summary.neonatal_deaths, "SIM-DO", "Deaths", "neonatal_death_count", "age_days_indicator_count"),
            ("SIMPostNeonatalDeathsForSINASCBirths", "SIM Postneonatal Deaths for SINASC Birth Denominator", summary.postneonatal_deaths, "SIM-DO", "Deaths", "postneonatal_death_count", "age_days_indicator_count"),
        ]
        fields: list[dict[str, Any]] = []
        q_rows: list[dict[str, Any]] = []
        vd_rows: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for field_id, name, count, source, carrier, estimand, operator in specs_count:
            field = _field_row(
                field_id=field_id,
                name=name,
                kind="extensive_measure",
                carrier=carrier,
                unit="count",
                aggregation="additive_count",
                role=["outcome", "demographic"] if source == "SINASC" else ["outcome"],
                source=[source],
                support={**support_births, "n_events": count},
                axes=axes_common,
                operator=operator,
                parents=[],
                warnings=warnings,
            )
            fields.append(field)
            q_rows.append(_q_row(field, n_events=count, n_denom=None, warnings=warnings))
            vd_rows.append(_vd_row(field, definition=f"{name} over the aligned SINASC/SIM smoke support.", estimand=estimand, interpretation_warning="Fixture-derived count; no population or birth denominator applied unless used by a derived rate field."))

        rate_specs = [
            ("SINASCCrudeBirthRateSIDRAOfficial", "SINASC Crude Birth Rate with SIDRA Official Population", summary.births_total, summary.denominator_population, "SINASCLiveBirthsAll", population_row["field_id"], ["SINASC", "SIDRA"], "LiveBirths/Population", "births per person", "crude_birth_rate", support_pop, "ratio_births_population"),
            ("SINASCLowBirthWeightPrevalence", "SINASC Low Birth Weight Prevalence", summary.low_birth_weight_births, summary.births_total, "SINASCLowBirthWeightBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "low_birth_weight_prevalence", support_births, "ratio_indicator_births"),
            ("SINASCPrematurityPrevalence", "SINASC Prematurity Prevalence", summary.prematurity_births, summary.births_total, "SINASCPrematurityBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "prematurity_prevalence", support_births, "ratio_indicator_births"),
            ("SINASCCongenitalAnomalyPrevalence", "SINASC Congenital Anomaly Prevalence", summary.congenital_anomaly_births, summary.births_total, "SINASCCongenitalAnomalyBirths", "SINASCLiveBirthsAll", ["SINASC"], "LiveBirths/LiveBirths", "proportion", "congenital_anomaly_prevalence", support_births, "ratio_indicator_births"),
            ("SIMInfantMortalitySINASCBirths", "SIM Infant Mortality with SINASC Birth Denominator", summary.infant_deaths, summary.births_total, "SIMInfantDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "infant_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
            ("SIMNeonatalMortalitySINASCBirths", "SIM Neonatal Mortality with SINASC Birth Denominator", summary.neonatal_deaths, summary.births_total, "SIMNeonatalDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "neonatal_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
            ("SIMPostNeonatalMortalitySINASCBirths", "SIM Postneonatal Mortality with SINASC Birth Denominator", summary.postneonatal_deaths, summary.births_total, "SIMPostNeonatalDeathsForSINASCBirths", "SINASCLiveBirthsAll", ["SIM-DO", "SINASC"], "Deaths/LiveBirths", "deaths per live birth", "postneonatal_mortality", support_births, "ratio_sim_deaths_sinasc_births"),
        ]
        for field_id, name, numerator, denominator, numerator_id, denominator_id, source, carrier, unit, estimand, support, operator in rate_specs:
            field = _field_row(
                field_id=field_id,
                name=name,
                kind="intensive_density",
                carrier=carrier,
                unit=unit,
                aggregation="ratio_recomputed_from_aligned_counts",
                role=["outcome", "demographic"],
                source=source,
                support={**support, "n_events": numerator, "n_denom": denominator, "n_eff": denominator},
                axes={**axes_rate, "numerator_field_id": numerator_id, "denominator_field_id": denominator_id},
                operator=operator,
                parents=[numerator_id, denominator_id],
                warnings=warnings,
            )
            fields.append(field)
            q_rows.append(_q_row(field, n_events=numerator, n_denom=denominator, warnings=warnings))
            vd_rows.append(_vd_row(field, definition=f"{name}: numerator {numerator_id} divided by denominator {denominator_id} after aligned support checks.", estimand=estimand, interpretation_warning="Rate is recomputed from aligned numerator and denominator counts; fixture scale is not inferential."))
            edges.append(_edge(f"edge_{numerator_id}_to_{field_id}", numerator_id, field_id, "ratio_numerator", {"operator": operator}))
            edges.append(_edge(f"edge_{denominator_id}_to_{field_id}", denominator_id, field_id, "ratio_denominator", {"operator": operator}))

        return fields, q_rows, vd_rows, edges


    def _write_summary_table(run_dir: Path, summary: MaternalChildLinkedSummary) -> None:
        rates = summary.rates()
        row = {
            "births_total": summary.births_total,
            "low_birth_weight_births": summary.low_birth_weight_births,
            "prematurity_births": summary.prematurity_births,
            "congenital_anomaly_births": summary.congenital_anomaly_births,
            "infant_deaths": summary.infant_deaths,
            "neonatal_deaths": summary.neonatal_deaths,
            "postneonatal_deaths": summary.postneonatal_deaths,
            "denominator_population": summary.denominator_population,
            **rates,
        }
        table_path = run_dir / "Tables" / "maternal_child_linkage_summary.parquet"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).write_parquet(table_path)


    def _update_json_outputs(run_dir: Path, *, sinasc_events_path: Path, sim_events_path: Path, summary: MaternalChildLinkedSummary) -> None:
        for name in ["RunConfig.json", "P_vector.json", "ReproducibilityManifest.json"]:
            path = run_dir / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            source_hashes = payload.setdefault("source_hashes", {})
            source_hashes.setdefault("sinasc_events", sha256_file(sinasc_events_path))
            source_hashes.setdefault("sim_events_for_maternal_child", sha256_file(sim_events_path))
            if name == "RunConfig.json":
                payload["maternal_child_linkage"] = {
                    "enabled": True,
                    "births_total": summary.births_total,
                    "infant_deaths": summary.infant_deaths,
                    "neonatal_deaths": summary.neonatal_deaths,
                    "postneonatal_deaths": summary.postneonatal_deaths,
                    "support_alignment": _support_alignment(summary),
                }
            if name == "P_vector.json":
                systems = set(payload.get("source_systems") or [])
                systems.update(["SINASC", "SIM-DO", "SIDRA"])
                payload["source_systems"] = sorted(systems)
                payload["maternal_child_linkage"] = "enabled"
            if name == "ReproducibilityManifest.json":
                payload.setdefault("maternal_child_linkage", {})
                payload["maternal_child_linkage"].update({
                    "enabled": True,
                    "summary_table": "Tables/maternal_child_linkage_summary.parquet",
                    "fields": sorted(FIELD_IDS),
                })
            path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")


    def attach_maternal_child_compile_fields(
        *,
        run_dir: str | Path,
        sinasc_events_path: str | Path,
        sim_events_path: str | Path,
        municipality_cod6: str,
    ) -> Path:
        run_dir = Path(run_dir)
        sinasc_events_path = Path(sinasc_events_path)
        sim_events_path = Path(sim_events_path)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        if not sinasc_events_path.exists():
            raise FileNotFoundError(f"SINASC events not found: {sinasc_events_path}")
        if not sim_events_path.exists():
            raise FileNotFoundError(f"SIM events not found: {sim_events_path}")

        cod7 = datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)
        if cod7 is None:
            raise ValueError(f"Cannot crosswalk DATASUS municipality cod6 to IBGE/SIDRA cod7: {municipality_cod6}")

        v_rows = _read_rows(run_dir / "V_fields.parquet")
        population_row = _field_by_name(v_rows, "SIDRAPopulationTotalAnchor")
        population_value = _population_value(population_row)
        summary = summarize_maternal_child_linkage(
            sinasc_events_path=sinasc_events_path,
            sim_events_path=sim_events_path,
            municipality_cod6=municipality_cod6,
            municipality_ibge_cod7=cod7,
            denominator_population=population_value,
        )
        if not summary.births_total:
            raise ValueError("Cannot attach maternal-child fields without nonzero live-birth support.")
        if summary.municipalities_ibge_cod7 != [cod7]:
            raise ValueError(
                f"Maternal-child support mismatch: expected cod7={cod7}, observed={summary.municipalities_ibge_cod7}"
            )

        fields, q_rows, vd_rows, edges = _build_fields(summary, population_row)
        _append_rows(run_dir / "V_fields.parquet", fields, remove_column="field_id", remove_values=FIELD_IDS)
        _append_rows(run_dir / "Q_tensor.parquet", q_rows, remove_column="field_id", remove_values=FIELD_IDS)
        _append_rows(run_dir / "VariableDictionary.parquet", vd_rows, remove_column="field_id", remove_values=FIELD_IDS)
        _append_edges(run_dir / "E_DAG.parquet", edges)
        _append_rows(run_dir / "Warnings.parquet", _warning_rows(summary), remove_column="warning_id", remove_values={"slice3b_maternal_child_fixture_rates", "slice3b_mortality_birth_denominator_linkage"})
        _write_summary_table(run_dir, summary)
        _update_json_outputs(run_dir, sinasc_events_path=sinasc_events_path, sim_events_path=sim_events_path, summary=summary)
        return run_dir
    ''')

    write("tests/unit/test_maternal_child_linkage.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.datasus.sinasc_normalize import normalize_sinasc_events
    from pegasus.she.maternal_child_linkage import summarize_maternal_child_linkage


    def test_maternal_child_linkage_summarizes_birth_and_death_support(tmp_path: Path):
        sinasc = tmp_path / "sinasc.parquet"
        sim = tmp_path / "sim.parquet"
        normalize_sinasc_events(
            input_path="tests/fixtures/datasus/sinasc_fixture.csv",
            output_path=sinasc,
            source_manifest_hash="fixture_manifest",
        )
        normalize_sim_do_events(
            input_path="tests/fixtures/datasus/sim_do_fixture.csv",
            output_path=sim,
            source_manifest_hash="fixture_manifest",
        )

        summary = summarize_maternal_child_linkage(
            sinasc_events_path=sinasc,
            sim_events_path=sim,
            municipality_cod6="270430",
            municipality_ibge_cod7="2704302",
            denominator_population=957916,
        )
        assert summary.births_total == 3
        assert summary.low_birth_weight_births >= 1
        assert summary.municipalities_cod6 == ["270430"]
        assert summary.municipalities_ibge_cod7 == ["2704302"]
        assert summary.denominator_population == 957916
        rates = summary.rates()
        assert rates["crude_birth_rate"] == summary.births_total / 957916
        assert rates["infant_mortality"] is not None
    ''')

    write("tests/integration/test_slice3b_compile_maternal_child_linkage.py", r'''
    import json
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.compile import run_compile


    def test_compile_smoke_integrates_sinasc_birth_and_mortality_fields(tmp_path: Path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=tmp_path / "compile_run",
            data_root=tmp_path / "data",
        )
        run_dir = Path(result["run_dir"])
        validation = validate_output_bundle(run_dir=str(run_dir))
        assert validation.ok, validation.errors

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        field_ids = set(v["field_id"].to_list())
        required = {
            "SINASCLiveBirthsAll",
            "SINASCCrudeBirthRateSIDRAOfficial",
            "SINASCLowBirthWeightPrevalence",
            "SINASCPrematurityPrevalence",
            "SINASCCongenitalAnomalyPrevalence",
            "SIMInfantMortalitySINASCBirths",
            "SIMNeonatalMortalitySINASCBirths",
            "SIMPostNeonatalMortalitySINASCBirths",
        }
        assert required.issubset(field_ids)

        table = pl.read_parquet(run_dir / "Tables" / "maternal_child_linkage_summary.parquet")
        assert table["births_total"].item() == 3
        assert table["denominator_population"].item() == 957916

        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        assert run_config["maternal_child_linkage"]["enabled"] is True
        assert "sinasc_processed_events" in run_config["source_hashes"]

        q = pl.read_parquet(run_dir / "Q_tensor.parquet")
        q_ids = set(q["field_id"].to_list())
        assert required.issubset(q_ids)
    ''')

    write("scripts/dev/audits/audit_slice3b_compile_maternal_child_linkage.py", r'''
    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle


    REQUIRED_FIELDS = {
        "SINASCLiveBirthsAll",
        "SINASCCrudeBirthRateSIDRAOfficial",
        "SINASCLowBirthWeightPrevalence",
        "SINASCPrematurityPrevalence",
        "SINASCCongenitalAnomalyPrevalence",
        "SIMInfantMortalitySINASCBirths",
        "SIMNeonatalMortalitySINASCBirths",
        "SIMPostNeonatalMortalitySINASCBirths",
    }


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


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
        field_ids = set(str(x) for x in v["field_id"].to_list())
        missing = sorted(REQUIRED_FIELDS - field_ids)
        if missing:
            failures.append({"kind": "missing_required_fields", "missing": missing})

        q = pl.read_parquet(run / "Q_tensor.parquet")
        q_ids = set(str(x) for x in q["field_id"].to_list())
        missing_q = sorted(REQUIRED_FIELDS - q_ids)
        if missing_q:
            failures.append({"kind": "missing_q_rows", "missing": missing_q})

        vd = pl.read_parquet(run / "VariableDictionary.parquet")
        vd_ids = set(str(x) for x in vd["field_id"].to_list())
        missing_vd = sorted(REQUIRED_FIELDS - vd_ids)
        if missing_vd:
            failures.append({"kind": "missing_variable_dictionary_rows", "missing": missing_vd})

        table_path = run / "Tables" / "maternal_child_linkage_summary.parquet"
        if not table_path.exists():
            failures.append({"kind": "missing_summary_table", "path": str(table_path)})
        else:
            table = pl.read_parquet(table_path)
            births_total = table["births_total"].item()
            denom = table["denominator_population"].item()
            if births_total <= 0:
                failures.append({"kind": "invalid_births_total", "births_total": births_total})
            if denom <= 0:
                failures.append({"kind": "invalid_population_denominator", "denominator_population": denom})

        run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
        source_hashes = run_config.get("source_hashes", {})
        for key in ["sinasc_raw_fixture", "sinasc_processed_events", "sim_processed_events", "sidra_facts"]:
            if key not in source_hashes:
                failures.append({"kind": "missing_source_hash", "key": key})
        linkage = run_config.get("maternal_child_linkage", {})
        if linkage.get("enabled") is not True:
            failures.append({"kind": "missing_maternal_child_linkage_run_config", "value": linkage})

        if failures:
            fail({"status": "failed", "failures": failures})
        print("AUDIT PASSED: Slice 3B compile run contains SINASC crude birth, maternal-child prevalence, and SIM/SINASC mortality linkage fields with validated output bundle metadata.")


    if __name__ == "__main__":
        main()
    ''')

    patch_intent()
    patch_compile()

    print("Applied Slice 3B compile maternal-child linkage.")
    print("Run the validation commands supplied by the assistant.")


if __name__ == "__main__":
    main()
