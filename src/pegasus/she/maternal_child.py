from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.geo.state_panel import clean_datasus_municipalities


@dataclass(frozen=True)
class MaternalChildSummary:
    years: list[int]
    municipalities_cod6: list[str]
    births_total: int
    low_birth_weight_births: int
    prematurity_births: int
    cesarean_births: int
    congenital_anomaly_births: int
    low_apgar5_births: int
    adolescent_mother_births: int
    advanced_maternal_age_births: int
    insufficient_prenatal_births: int
    birth_weight_missing_or_invalid: int
    gestational_age_missing_or_invalid: int
    apgar_missing_or_invalid: int
    race_missing_or_ignored: int

    def rates(self) -> dict[str, float | None]:
        denom = float(self.births_total) if self.births_total else 0.0
        names = {
            "low_birth_weight_prevalence": self.low_birth_weight_births,
            "prematurity_prevalence": self.prematurity_births,
            "cesarean_prevalence": self.cesarean_births,
            "congenital_anomaly_prevalence": self.congenital_anomaly_births,
            "low_apgar5_prevalence": self.low_apgar5_births,
            "adolescent_mother_share": self.adolescent_mother_births,
            "advanced_maternal_age_share": self.advanced_maternal_age_births,
            "insufficient_prenatal_share": self.insufficient_prenatal_births,
        }
        if denom <= 0:
            return {key: None for key in names}
        return {key: value / denom for key, value in names.items()}

    def support(self) -> dict[str, Any]:
        return {
            "time": {"years": self.years},
            "geography": {"municipality_cod6": self.municipalities_cod6},
            "n_events": self.births_total,
        }


def _bool_count(df: pl.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df.filter(pl.col(column) == True).height)  # noqa: E712 - explicit data-state comparison




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


def _state_not_valid_count(df: pl.DataFrame, column: str, valid_state: str = "valid") -> int:
    if column not in df.columns:
        return 0
    return int(df.filter(pl.col(column) != valid_state).height)


def summarize_maternal_child_events(
    events_path: str | Path,
    *,
    municipality_cod6: str | None = None,
    datasus_uf_prefix: str,
) -> MaternalChildSummary:
    df = pl.read_parquet(events_path)
    if municipality_cod6 is not None:
        df = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
    valid = df.filter(pl.col("record_state") == "valid") if "record_state" in df.columns else df

    years = sorted(int(x) for x in valid["birth_year"].drop_nulls().unique().to_list()) if "birth_year" in valid.columns else []
    municipalities = (
        sorted(str(x) for x in valid["mun_residence_cod6"].drop_nulls().unique().to_list())
        if "mun_residence_cod6" in valid.columns
        else []
    )
    municipalities, invalid_municipalities = clean_datasus_municipalities(municipalities, uf_prefix=datasus_uf_prefix)
    congenital_anomaly_births = _congenital_anomaly_count(valid)
    _assert_plausible_anomaly_rate(births_total=int(valid.height), congenital_anomaly_births=congenital_anomaly_births)

    race_missing = 0
    for col in ["mother_race_state", "newborn_race_state"]:
        if col in valid.columns:
            race_missing += int(valid.filter(pl.col(col) != "valid_admin_race").height)

    return MaternalChildSummary(
        years=years,
        municipalities_cod6=municipalities,
        births_total=int(valid.height),
        low_birth_weight_births=_bool_count(valid, "low_birth_weight_flag"),
        prematurity_births=_bool_count(valid, "prematurity_flag"),
        cesarean_births=_bool_count(valid, "cesarean_flag"),
        congenital_anomaly_births=congenital_anomaly_births,
        low_apgar5_births=_bool_count(valid, "low_apgar5_flag"),
        adolescent_mother_births=_bool_count(valid, "adolescent_mother_flag"),
        advanced_maternal_age_births=_bool_count(valid, "advanced_maternal_age_flag"),
        insufficient_prenatal_births=_bool_count(valid, "insufficient_prenatal_flag"),
        birth_weight_missing_or_invalid=_state_not_valid_count(valid, "birth_weight_state"),
        gestational_age_missing_or_invalid=_state_not_valid_count(valid, "gestational_age_state"),
        apgar_missing_or_invalid=_state_not_valid_count(valid, "apgar5_state"),
        race_missing_or_ignored=race_missing,
    )
