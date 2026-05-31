from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class AdapterColumnContract:
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]


@dataclass(frozen=True)
class NormalizationResult:
    source_system: str
    normalized: pl.DataFrame
    warnings: list[str]


class DatasusSourceAdapter(Protocol):
    source_system: str
    normalized_schema_name: str

    def column_contract(self) -> AdapterColumnContract:
        ...

    def normalize(
        self,
        df: pl.DataFrame,
        *,
        source_manifest_hash: str = "",
    ) -> NormalizationResult:
        ...