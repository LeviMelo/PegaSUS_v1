from __future__ import annotations
from pegasus.storage import read_table

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.geo.state_panel import clean_datasus_municipalities


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




def _congenital_anomaly_count(df: pl.DataFrame) -> int:
    if "congenital_anomaly_flag" not in df.columns:
        return 0
    return int(df.filter(pl.col("congenital_anomaly_flag") == True).height)  # noqa: E712


def _assert_plausible_anomaly_rate(*, births_total: int, congenital_anomaly_births: int) -> None:
    if births_total <= 0:
        return
    # Tiny fixtures can be intentionally high-prevalence to exercise decoder states.
    # The plausibility gate is intended for real production-sized SINASC materializations.
    if births_total < 1000:
        return
    rate = congenital_anomaly_births / float(births_total)
    if rate > 0.20:
        raise ValueError(
            f"SINASC congenital anomaly count is implausibly high: "
            f"{congenital_anomaly_births}/{births_total} ({rate:.3f})."
        )


def summarize_maternal_child_linkage(
    *,
    sinasc_events_path: str | Path,
    sim_events_path: str | Path,
    municipality_cod6: str | None,
    municipality_ibge_cod7: str | None,
    datasus_uf_prefix: str,
    denominator_population: float | None = None,
) -> MaternalChildLinkedSummary:
    sinasc = pl.from_arrow(read_table(sinasc_events_path))
    sim = pl.from_arrow(read_table(sim_events_path))

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
    municipalities, invalid_municipalities = clean_datasus_municipalities(municipalities, uf_prefix=datasus_uf_prefix)

    cod7s = [str(municipality_ibge_cod7)] if municipality_ibge_cod7 else []

    infant = _liveborn_death_filter(sim)
    infant = infant.filter(pl.col("age_days") < 365.25) if infant.height else infant
    neonatal = infant.filter(pl.col("age_days") < 28) if infant.height else infant
    postneonatal = infant.filter((pl.col("age_days") >= 28) & (pl.col("age_days") < 365.25)) if infant.height else infant
    congenital_anomaly_births = _congenital_anomaly_count(sinasc_valid)
    _assert_plausible_anomaly_rate(births_total=int(sinasc_valid.height), congenital_anomaly_births=congenital_anomaly_births)

    return MaternalChildLinkedSummary(
        years=years,
        municipalities_cod6=municipalities,
        municipalities_ibge_cod7=cod7s,
        births_total=int(sinasc_valid.height),
        low_birth_weight_births=_bool_count(sinasc_valid, "low_birth_weight_flag"),
        prematurity_births=_bool_count(sinasc_valid, "prematurity_flag"),
        congenital_anomaly_births=congenital_anomaly_births,
        infant_deaths=int(infant.height),
        neonatal_deaths=int(neonatal.height),
        postneonatal_deaths=int(postneonatal.height),
        sim_death_records_considered=int(sim.height),
        sinasc_birth_records_considered=int(sinasc_valid.height),
        denominator_population=denominator_population,
    )
