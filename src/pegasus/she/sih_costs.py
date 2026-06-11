from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.registries.sih_cost import COST_COMPONENTS


@dataclass(frozen=True)
class SIHCostSummary:
    admissions_total: int
    inpatient_deaths: int
    years: list[int]
    municipalities_cod6: list[str]
    municipalities_cod7: list[str]
    stay_days_total: int
    stay_days_observed: int
    cost_sums: dict[str, float]
    cost_observed: dict[str, int]
    principal_diagnosis_states: dict[str, int]

    @property
    def inpatient_fatality(self) -> float | None:
        return None if self.admissions_total == 0 else self.inpatient_deaths / self.admissions_total

    @property
    def mean_los(self) -> float | None:
        return None if self.stay_days_observed == 0 else self.stay_days_total / self.stay_days_observed

    def mean_costs(self) -> dict[str, float | None]:
        return {key: (None if self.cost_observed.get(key, 0) == 0 else self.cost_sums[key] / self.cost_observed[key]) for key in COST_COMPONENTS}

    def support(self) -> dict[str, Any]:
        return {"years": self.years, "municipalities": self.municipalities_cod6, "municipalities_ibge_cod7": self.municipalities_cod7, "n_events": self.admissions_total}


def summarize_sih_costs(events_path: str | Path, *, municipality_cod6: str | None = None) -> SIHCostSummary:
    df = pl.read_parquet(events_path)
    if municipality_cod6 is not None and "mun_residence_cod6" in df.columns:
        df = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
    valid = df.filter(pl.col("record_state") == "valid") if "record_state" in df.columns else df
    cost_columns = {"VAL_SH": "hospital_service_cost_real", "VAL_SP": "professional_service_cost_real", "VAL_UTI": "icu_cost_real", "VAL_TOT": "total_admission_cost_real"}
    cost_sums: dict[str, float] = {}
    cost_obs: dict[str, int] = {}
    for raw, col in cost_columns.items():
        cost_sums[raw] = float(valid[col].drop_nulls().sum()) if col in valid.columns else 0.0
        cost_obs[raw] = int(valid[col].drop_nulls().len()) if col in valid.columns else 0
    diag_counts = {}
    if "principal_icd_parse_state" in valid.columns:
        for row in valid.group_by("principal_icd_parse_state").len().to_dicts():
            diag_counts[str(row["principal_icd_parse_state"])] = int(row["len"])
    years = sorted(int(x) for x in valid["admission_year"].drop_nulls().unique().to_list()) if "admission_year" in valid.columns else []
    cod6 = sorted(str(x) for x in valid["mun_residence_cod6"].drop_nulls().unique().to_list()) if "mun_residence_cod6" in valid.columns else []
    cod7 = sorted(str(x) for x in valid["mun_residence_cod7"].drop_nulls().unique().to_list()) if "mun_residence_cod7" in valid.columns else []
    stay_observed = int(valid["stay_length_days"].drop_nulls().len()) if "stay_length_days" in valid.columns else 0
    stay_total = int(valid["stay_length_days"].drop_nulls().sum()) if "stay_length_days" in valid.columns and stay_observed else 0
    return SIHCostSummary(int(valid.height), int((valid["death_flag"] == True).sum()) if "death_flag" in valid.columns else 0, years, cod6, cod7, stay_total, stay_observed, cost_sums, cost_obs, diag_counts)
