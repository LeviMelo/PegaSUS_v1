from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.registries.cnes_capacity import CAPACITY_COMPONENTS


@dataclass(frozen=True)
class CNESCapacitySummary:
    facilities_total: int
    years: list[int]
    municipalities_cod6: list[str]
    municipalities_cod7: list[str]
    capacity_sums: dict[str, int]
    invalid_flag_count: int
    flag_count: int
    zero_facility_cnpj_count: int
    nullified_maintainer_cnpj_count: int

    @property
    def invalid_flag_share(self) -> float | None:
        return None if self.flag_count == 0 else self.invalid_flag_count / self.flag_count

    @property
    def zero_facility_cnpj_share(self) -> float | None:
        return None if self.facilities_total == 0 else self.zero_facility_cnpj_count / self.facilities_total

    def support(self) -> dict[str, Any]:
        return {"years": self.years, "municipalities": self.municipalities_cod6, "municipalities_ibge_cod7": self.municipalities_cod7, "n_events": self.facilities_total}


def summarize_cnes_capacity(events_path: str | Path, *, municipality_cod6: str | None = None) -> CNESCapacitySummary:
    df = pl.read_parquet(events_path)
    if municipality_cod6 is not None and "mun_facility_cod6" in df.columns:
        df = df.filter(pl.col("mun_facility_cod6") == str(municipality_cod6))
    valid = df.filter(pl.col("record_state") == "valid") if "record_state" in df.columns else df
    capacity_sums = {key: 0 for key in CAPACITY_COMPONENTS}
    for row in valid.to_dicts():
        payload = json.loads(row.get("capacity_vector_json") or "{}")
        for key in capacity_sums:
            value = payload.get(key)
            if isinstance(value, int):
                capacity_sums[key] += value
    years = sorted(int(x) for x in valid["year"].drop_nulls().unique().to_list()) if "year" in valid.columns else []
    cod6 = sorted(str(x) for x in valid["mun_facility_cod6"].drop_nulls().unique().to_list()) if "mun_facility_cod6" in valid.columns else []
    cod7 = sorted(str(x) for x in valid["mun_facility_cod7"].drop_nulls().unique().to_list()) if "mun_facility_cod7" in valid.columns else []
    return CNESCapacitySummary(
        facilities_total=int(valid.height),
        years=years,
        municipalities_cod6=cod6,
        municipalities_cod7=cod7,
        capacity_sums=capacity_sums,
        invalid_flag_count=int(valid["invalid_flag_count"].sum()) if "invalid_flag_count" in valid.columns else 0,
        flag_count=int(valid["flag_count"].sum()) if "flag_count" in valid.columns else 0,
        zero_facility_cnpj_count=int((valid["facility_cnpj_state"] == "NullifiedZeroCNPJ").sum()) if "facility_cnpj_state" in valid.columns else 0,
        nullified_maintainer_cnpj_count=int((valid["maintainer_cnpj_state"] == "NullifiedZeroCNPJ").sum()) if "maintainer_cnpj_state" in valid.columns else 0,
    )
