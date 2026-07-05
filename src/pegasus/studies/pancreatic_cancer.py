"""National pancreatic cancer (ICD-10 C25) epidemiology study engine.

Malignant neoplasm of the pancreas (C25, WHO chapter II Neoplasms, block C15-C26
digestive organs) is one of the most lethal cancers — case-fatality near 1, so mortality
(SIM underlying cause) closely tracks incidence. This engine turns raw SIM death records
into a rigorous epidemiological picture: temporal trends, age/sex/race gradients, spatial
heterogeneity, and denominator-based rates — the descriptive substrate the LDO then mines
for socioeconomic/context dependencies.

Everything is computed from real records; nothing is fabricated. Cells with no
denominator carry a rate of ``None`` (never an invented number), honoring the prime
directive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

PANCREATIC_ICD = "C25"                 # malignant neoplasm of the pancreas (C25.0-C25.9)
_CAUSE_COL = "underlying_icd_norm"
_YEAR_COL = "year"
_GEO_COL = "mun_residence_cod6"

# Standard demographic bands.
_AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-39", 0, 40), ("40-49", 40, 50), ("50-59", 50, 60),
    ("60-69", 60, 70), ("70-79", 70, 80), ("80+", 80, 200),
)
_RACE_LABELS = {"1": "Branca", "2": "Preta", "3": "Amarela", "4": "Parda", "5": "Indígena"}
# UF (2-digit IBGE) → region, for spatial aggregation of municipality cod6 (first 2 chars).
_UF_REGION = {
    "11": "Norte", "12": "Norte", "13": "Norte", "14": "Norte", "15": "Norte", "16": "Norte", "17": "Norte",
    "21": "Nordeste", "22": "Nordeste", "23": "Nordeste", "24": "Nordeste", "25": "Nordeste",
    "26": "Nordeste", "27": "Nordeste", "28": "Nordeste", "29": "Nordeste",
    "31": "Sudeste", "32": "Sudeste", "33": "Sudeste", "35": "Sudeste",
    "41": "Sul", "42": "Sul", "43": "Sul",
    "50": "Centro-Oeste", "51": "Centro-Oeste", "52": "Centro-Oeste", "53": "Centro-Oeste",
}


def _age_band(age: float | None) -> str | None:
    if age is None:
        return None
    for label, lo, hi in _AGE_BANDS:
        if lo <= age < hi:
            return label
    return None


def extract_pancreatic_deaths(sim: pl.DataFrame, *, cause_col: str = _CAUSE_COL) -> pl.DataFrame:
    """Filter SIM deaths to underlying cause C25 (pancreatic cancer)."""
    return sim.filter(
        pl.col(cause_col).cast(pl.Utf8).str.to_uppercase().str.starts_with(PANCREATIC_ICD)
    )


def _rate_per_100k(count: int | None, pop: float | None) -> float | None:
    if count is None or pop is None or pop <= 0:
        return None
    return round(count / pop * 1e5, 2)


@dataclass
class PancreaticStudy:
    n_deaths: int
    years: tuple[int, ...]
    by_year: list[dict]                 # [{year, deaths, rate_per_100k?}]
    by_sex: list[dict]
    by_age_band: list[dict]
    by_race: list[dict]
    by_region: list[dict]
    by_uf: list[dict]
    male_female_ratio: float | None
    median_age: float | None
    crude_rate_per_100k: float | None
    disease_concepts: list[str]         # the disease-axis concept memberships of C25
    dependencies: list[dict] = field(default_factory=list)   # LDO-discovered context links
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def build_pancreatic_study(
    sim: pl.DataFrame,
    *,
    population_by_year: dict[int, float] | None = None,
    total_population: float | None = None,
    cause_col: str = _CAUSE_COL,
    geo_col: str = _GEO_COL,
    scope: str = "national",
) -> PancreaticStudy:
    """Assemble the pancreatic-cancer epidemiology study from SIM records.

    ``population_by_year`` (year → population) enables per-year rates; ``total_population``
    enables the crude rate. Absent a denominator, counts are reported and rates are ``None``.
    """
    deaths = extract_pancreatic_deaths(sim, cause_col=cause_col)
    n = deaths.height
    warnings: list[str] = []

    has_year = _YEAR_COL in deaths.columns
    years = tuple(sorted(int(y) for y in deaths[_YEAR_COL].drop_nulls().unique().to_list())) if has_year else ()

    def _counts(col: str, relabel: dict | None = None) -> list[dict]:
        if col not in deaths.columns:
            return []
        g = deaths.group_by(col).len().rename({"len": "deaths"}).sort("deaths", descending=True)
        out = []
        for r in g.to_dicts():
            key = r[col]
            if relabel is not None:
                key = relabel.get(str(key), str(key))
            out.append({"group": key, "deaths": r["deaths"]})
        return out

    by_year = []
    if has_year:
        yc = deaths.group_by(_YEAR_COL).len().rename({"len": "deaths"}).sort(_YEAR_COL)
        for r in yc.to_dicts():
            y = int(r[_YEAR_COL])
            pop = (population_by_year or {}).get(y)
            by_year.append({"year": y, "deaths": r["deaths"], "rate_per_100k": _rate_per_100k(r["deaths"], pop)})

    # age band
    by_age = []
    if "age_years" in deaths.columns:
        banded = deaths.with_columns(
            pl.col("age_years").map_elements(_age_band, return_dtype=pl.Utf8).alias("age_band")
        )
        ac = banded.group_by("age_band").len().rename({"len": "deaths"}).drop_nulls("age_band")
        order = [b[0] for b in _AGE_BANDS]
        for r in sorted(ac.to_dicts(), key=lambda x: order.index(x["age_band"]) if x["age_band"] in order else 99):
            by_age.append({"group": r["age_band"], "deaths": r["deaths"]})
        median_age = float(deaths["age_years"].median()) if deaths["age_years"].drop_nulls().len() else None
    else:
        median_age = None

    by_sex = _counts("sex")
    sex_map = {d["group"]: d["deaths"] for d in by_sex}
    m, f = sex_map.get("male", 0), sex_map.get("female", 0)
    mf_ratio = round(m / f, 2) if f else None

    by_race = _counts("race_color_admin", relabel=_RACE_LABELS)

    # spatial: municipality cod6 → UF (first 2) → region
    by_uf, by_region = [], []
    if geo_col in deaths.columns:
        geo = deaths.with_columns([
            pl.col(geo_col).cast(pl.Utf8).str.slice(0, 2).alias("uf"),
        ]).with_columns(
            pl.col("uf").map_elements(lambda u: _UF_REGION.get(u, "?"), return_dtype=pl.Utf8).alias("region")
        )
        uc = geo.group_by("uf").len().rename({"len": "deaths"}).sort("deaths", descending=True)
        by_uf = [{"group": r["uf"], "deaths": r["deaths"]} for r in uc.to_dicts()]
        rc = geo.group_by("region").len().rename({"len": "deaths"}).sort("deaths", descending=True)
        by_region = [{"group": r["region"], "deaths": r["deaths"]} for r in rc.to_dicts()]

    # disease-axis concept memberships of C25 (uses the built disease semantic axis)
    concepts: list[str] = []
    try:
        from pegasus.disease import concept_registry
        concepts = sorted({
            a.concept_id for a in concept_registry.assertions_for_code("C25")
        })
    except Exception as exc:  # disease extra optional at study time
        warnings.append(f"disease_axis_unavailable:{type(exc).__name__}")

    return PancreaticStudy(
        n_deaths=n,
        years=years,
        by_year=by_year,
        by_sex=by_sex,
        by_age_band=by_age,
        by_race=by_race,
        by_region=by_region,
        by_uf=by_uf,
        male_female_ratio=mf_ratio,
        median_age=median_age,
        crude_rate_per_100k=_rate_per_100k(n, total_population),
        disease_concepts=concepts,
        provenance={"scope": scope, "source": "SIM-DO underlying cause", "icd": PANCREATIC_ICD,
                    "n_source_deaths": sim.height},
        warnings=warnings,
    )


__all__ = ["PANCREATIC_ICD", "extract_pancreatic_deaths", "PancreaticStudy", "build_pancreatic_study"]
