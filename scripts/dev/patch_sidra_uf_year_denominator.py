from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
BACKUP_ROOT = ROOT / ".codex-tmp" / "uf_year_denominator_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
TEST = ROOT / "tests" / "unit" / "test_sidra_uf_year_denominator_anchor.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = BACKUP_ROOT / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def replace_between(text: str, start: str, end: str, replacement: str) -> str:
    pattern = re.compile(rf"^def {re.escape(start)}\(.*?(?=^def {re.escape(end)}\()", re.S | re.M)
    new_text, count = pattern.subn(replacement.rstrip() + "\n\n", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not replace function {start} before {end}.")
    return new_text


HELPERS = r'''
def _anchor_uf_code(anchor: SidraPopulationAnchor) -> str | None:
    level = str(anchor.locality_level or "").strip().upper()
    locality = str(anchor.locality_id or "").strip()
    if level in {"N3", "UF", "STATE", "FEDERATION_UNIT"} and locality.isdigit() and len(locality) == 2:
        return locality
    return None


def _period_value(value: object) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def _alignment_model(support_alignment: Any) -> dict[str, Any]:
    if isinstance(support_alignment, dict):
        return support_alignment
    model = getattr(support_alignment, "model", None)
    if callable(model):
        payload = model()
        return dict(payload)
    if hasattr(support_alignment, "asdict"):
        return dict(support_alignment.asdict())
    return dict(support_alignment)


def _uf_year_support_alignment(
    *,
    anchor: SidraPopulationAnchor,
    numerator_support: dict[str, Any],
    numerator_axes: dict[str, Any],
    denominator_axes: dict[str, Any],
) -> dict[str, Any] | None:
    uf_code = _anchor_uf_code(anchor)
    if uf_code is None:
        return None

    denominator_year = _period_value(anchor.period)
    numerator_years = [_period_value(x) for x in numerator_support.get("years", [])]
    municipalities = sorted({
        str(x).strip()
        for x in numerator_support.get("municipalities", [])
        if str(x).strip()
    })

    if set(numerator_years) != {denominator_year}:
        raise ValueError(
            "Illegal UF denominator attachment: numerator and denominator year support are not aligned. "
            f"numerator_years={sorted(set(numerator_years))}; denominator_years={[denominator_year]}"
        )

    if not municipalities:
        raise ValueError("Illegal UF denominator attachment: empty numerator municipality support.")

    outside_uf = [
        m for m in municipalities
        if not (m.isdigit() and len(m) == 6 and m.startswith(uf_code))
    ]
    if outside_uf:
        sample = outside_uf[:10]
        raise ValueError(
            "Illegal UF denominator attachment: numerator contains municipalities outside the denominator UF. "
            f"uf={uf_code}; outside_uf_sample={sample}; outside_uf_count={len(outside_uf)}"
        )

    return {
        "aligned": True,
        "reason": "uf_year_aggregate_aligned",
        "support": "uf_year",
        "uf": uf_code,
        "numerator_geography": numerator_axes.get("geography"),
        "denominator_geography": denominator_axes.get("geography"),
        "numerator_years": sorted(set(numerator_years)),
        "denominator_years": [denominator_year],
        "numerator_municipalities_source": municipalities,
        "denominator_municipalities_source": [uf_code],
        "numerator_municipalities_ibge_cod7": [],
        "denominator_municipalities_ibge_cod7": [],
        "common_municipalities_ibge_cod7": [],
        "missing_from_denominator_ibge_cod7": [],
        "missing_from_numerator_ibge_cod7": [],
        "crosswalk": "datasus_cod6_uf_prefix_to_sidra_uf_n3",
        "municipality_count": len(municipalities),
    }


def _build_denominator_support_alignment(
    *,
    anchor: SidraPopulationAnchor,
    numerator_support: dict[str, Any],
    numerator_axes: dict[str, Any],
    denominator_support: dict[str, Any],
    denominator_axes: dict[str, Any],
) -> Any:
    uf_alignment = _uf_year_support_alignment(
        anchor=anchor,
        numerator_support=numerator_support,
        numerator_axes=numerator_axes,
        denominator_axes=denominator_axes,
    )
    if uf_alignment is not None:
        return uf_alignment

    return assert_municipality_year_support_aligned(
        numerator_support=numerator_support,
        numerator_axes=numerator_axes,
        denominator_support=denominator_support,
        denominator_axes=denominator_axes,
    )
'''


NEW_POPULATION = r'''
def _population_v_field(anchor: SidraPopulationAnchor, facts_path: Path) -> dict[str, Any]:
    year = _period_value(anchor.period)
    uf_code = _anchor_uf_code(anchor)

    common = {
        "years": [year],
        "n_denom": anchor.value,
        "n_eff": anchor.value,
        "cov_T": 1.0,
        "missingness": 0.0,
        "denom_fragility": 0.0,
        "sidra_table_id": anchor.table_id,
        "sidra_variable_id": anchor.variable_id,
        "total_category_policy": anchor.total_category_policy,
        "source_category_tuple": anchor.category_tuple,
        "source_classification_tuple": anchor.classification_tuple,
    }

    if uf_code is not None:
        support = {
            "support": "uf_year",
            **common,
            "ufs": [uf_code],
            "state_ibge_code": uf_code,
            "cov_S": 1.0,
            "municipality_code_system": "IBGE/SIDRA UF code",
        }
        axes = {
            "time": "year",
            "geography": "uf",
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
                "target_axes": ["uf", "year"],
                "projection_matrix_id": "total_category_identity_marginal_9606_uf_v1",
                "total_category_policy": "total_only",
                "fractional_mapping_warnings": [],
            },
            "bounded_pushforward": {
                "operator": "pi_bound_*",
                "axes_kept": ["uf", "year"],
                "axes_dropped": ["sex", "race_color", "age"],
                "legal": True,
                "reason": "SIDRA 9606 total sex/race/age categories produce Population(UF,t).",
            },
        }
    else:
        support = {
            "support": "municipality_year",
            **common,
            "municipalities": [anchor.locality_id],
            "cov_S": 1.0,
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
'''


NEW_RATE = r'''
def _rate_v_field(
    *,
    all_deaths: dict[str, Any],
    anchor: SidraPopulationAnchor,
    facts_path: Path,
    support_alignment: Any,
) -> dict[str, Any]:
    death_support = _load_json_field(all_deaths["support_json"])
    n_events = float(death_support.get("n_events", 0.0))
    alignment = _alignment_model(support_alignment)

    field_id = content_hash(
        {
            "kind": "SIMCrudeMortalitySIDRAOfficial",
            "parents": [all_deaths["field_id"], anchor.field_id],
            "operator": "RN",
            "sidra_metadata_hash": anchor.metadata_hash,
            "sidra_request_hash": anchor.request_hash,
            "support_alignment": alignment,
        }
    )

    if alignment.get("support") == "uf_year":
        support = {
            "support": "uf_year",
            "years": alignment["denominator_years"],
            "ufs": [alignment["uf"]],
            "state_ibge_code": alignment["uf"],
            "municipality_count": int(alignment.get("municipality_count") or 0),
            "municipality_code_system": "DATASUS cod6 aggregated by UF prefix",
            "n_events": n_events,
            "n_denom": anchor.value,
            "n_eff": n_events,
            "cov_S": 1.0,
            "cov_T": float(len(alignment["denominator_years"])),
            "missingness": float(death_support.get("missingness", 0.0)),
            "denom_fragility": 0.0,
            "sidra_population_anchor_field_id": anchor.field_id,
            "support_alignment": alignment,
            "municipality_crosswalk": "uf_prefix_cod6_to_sidra_n3",
        }
        axes = {
            "time": "year",
            "geography": "uf",
            "diagnostic_role": "all_deaths",
            "topology": "none",
            "race_axis_type": None,
            "denominator_source": "SIDRA_9606_total_population_anchor",
        }
        provenance = ["official", "sidra_denominator_anchor", "materialized_external_sim_numerator"]
        warnings = ["official_sidra_denominator_anchor", "support_aligned_by_uf_year_aggregate"]
    else:
        support = {
            "support": "municipality_year",
            "years": [int(anchor.period) if str(anchor.period).isdigit() else anchor.period],
            "municipalities": alignment["numerator_municipalities_ibge_cod7"],
            "municipality_code_system": "IBGE/SIDRA cod7",
            "n_events": n_events,
            "n_denom": anchor.value,
            "n_eff": n_events,
            "cov_S": float(len(alignment["numerator_municipalities_ibge_cod7"])),
            "cov_T": float(len(alignment["numerator_years"])),
            "missingness": float(death_support.get("missingness", 0.0)),
            "denom_fragility": 0.0,
            "sidra_population_anchor_field_id": anchor.field_id,
            "support_alignment": alignment,
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
        provenance = ["official", "sidra_denominator_anchor", "sim_fixture_numerator"]
        warnings = ["fixture_small_n", "official_sidra_denominator_anchor", "dashboard_unsafe_fixture_rate", "support_aligned_by_municipality_crosswalk"]

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
        "provenance": _json(provenance),
        "state": "quarantined_descriptive",
        "dashboard_safe": "False",
        "warnings": _json(warnings),
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
'''


def patch_target() -> None:
    text = read(TARGET)

    if "_anchor_uf_code" not in text:
        marker = "def _population_v_field(anchor: SidraPopulationAnchor, facts_path: Path) -> dict[str, Any]:"
        if marker not in text:
            raise RuntimeError("Could not find _population_v_field insertion point.")
        text = text.replace(marker, HELPERS.rstrip() + "\n\n" + marker, 1)

    text = replace_between(text, "_population_v_field", "_rate_v_field", NEW_POPULATION)
    text = replace_between(text, "_rate_v_field", "_q_rows", NEW_RATE)

    old_block = '''    population_row = _population_v_field(anchor, sidra_facts_path)
    support_alignment = assert_municipality_year_support_aligned(
        numerator_support=_load_json_field(all_deaths["support_json"]),
        numerator_axes=_load_json_field(all_deaths["axes_json"]),
        denominator_support=_load_json_field(population_row["support_json"]),
        denominator_axes=_load_json_field(population_row["axes_json"]),
    )
    rate_row = _rate_v_field(
'''
    new_block = '''    population_row = _population_v_field(anchor, sidra_facts_path)
    support_alignment = _build_denominator_support_alignment(
        anchor=anchor,
        numerator_support=_load_json_field(all_deaths["support_json"]),
        numerator_axes=_load_json_field(all_deaths["axes_json"]),
        denominator_support=_load_json_field(population_row["support_json"]),
        denominator_axes=_load_json_field(population_row["axes_json"]),
    )
    rate_row = _rate_v_field(
'''
    if old_block not in text:
        raise RuntimeError("Could not find support_alignment block in attach_sidra_population_anchor_to_run.")
    text = text.replace(old_block, new_block, 1)

    text = text.replace("support_alignment.model()", "_alignment_model(support_alignment)")

    write(TARGET, text)


def write_tests() -> None:
    test = '''from __future__ import annotations

import json

import pytest

from pegasus.output.sidra_denominator_anchor import (
    _population_v_field,
    _uf_year_support_alignment,
)
from pegasus.she.population.sidra_anchor import SidraPopulationAnchor


def _anchor() -> SidraPopulationAnchor:
    return SidraPopulationAnchor(
        field_id="anchor",
        table_id="9606",
        variable_id="93",
        period="2022",
        locality_level="N3",
        locality_id="27",
        value=3_100_000.0,
        unit="persons",
        request_hash="request",
        metadata_hash="metadata",
        classification_tuple=[("2", "Sexo"), ("86", "Cor ou raça"), ("287", "Idade")],
        category_tuple=[("2", "6794"), ("86", "95251"), ("287", "100362")],
    )


def test_sidra_n3_anchor_is_uf_year_not_fake_municipality() -> None:
    row = _population_v_field(_anchor(), facts_path="sidra.parquet")
    support = json.loads(row["support_json"])
    axes = json.loads(row["axes_json"])

    assert support["support"] == "uf_year"
    assert support["ufs"] == ["27"]
    assert "municipalities" not in support
    assert axes["geography"] == "uf"


def test_uf_year_alignment_accepts_alagoas_datasus_cod6_prefix() -> None:
    alignment = _uf_year_support_alignment(
        anchor=_anchor(),
        numerator_support={"years": [2022], "municipalities": ["270030", "270430"]},
        numerator_axes={"geography": "mun_residence_cod6"},
        denominator_axes={"geography": "uf"},
    )

    assert alignment is not None
    assert alignment["aligned"] is True
    assert alignment["support"] == "uf_year"
    assert alignment["uf"] == "27"
    assert alignment["municipality_count"] == 2


def test_uf_year_alignment_rejects_out_of_state_municipality() -> None:
    with pytest.raises(ValueError, match="outside the denominator UF"):
        _uf_year_support_alignment(
            anchor=_anchor(),
            numerator_support={"years": [2022], "municipalities": ["270430", "280030"]},
            numerator_axes={"geography": "mun_residence_cod6"},
            denominator_axes={"geography": "uf"},
        )
'''
    write(TEST, test)


def main() -> None:
    patch_target()
    write_tests()
    print("Patched SIDRA denominator attachment for N3/UF-year aggregate support.")
    print(f"Backups: {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
