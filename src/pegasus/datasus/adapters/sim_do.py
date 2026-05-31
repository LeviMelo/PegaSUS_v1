from __future__ import annotations

import polars as pl

from pegasus.datasus.adapters.base import AdapterColumnContract, NormalizationResult
from pegasus.datasus.normalize_sim import normalize_sim_do


class SimDoAdapter:
    source_system = "SIM-DO"
    normalized_schema_name = "SIM_DO_NORMALIZED_SCHEMA"

    def column_contract(self) -> AdapterColumnContract:
        return AdapterColumnContract(
            required_columns=(
                "DTOBITO",
                "IDADE",
                "SEXO",
                "RACACOR",
                "CODMUNRES",
                "CAUSABAS",
            ),
            optional_columns=(
                "DTNASC",
                "CODMUNOCOR",
                "CODESTAB",
                "DIFDATA",
                "TPPOS",
                "ALTCAUSA",
                "ASSISTMED",
                "GRAVIDEZ",
                "PUERPERIO",
                "TIPOBITO",
                "SEMAGESTAC",
                "LINHAA",
                "LINHAB",
                "LINHAC",
                "LINHAD",
                "LINHAII",
                "CAUSABAS_O",
            ),
        )

    def normalize(
        self,
        df: pl.DataFrame,
        *,
        source_manifest_hash: str = "",
    ) -> NormalizationResult:
        warnings = self._validate_input_columns(df)
        normalized = normalize_sim_do(df, source_manifest_hash=source_manifest_hash)

        return NormalizationResult(
            source_system=self.source_system,
            normalized=normalized,
            warnings=warnings,
        )

    def _validate_input_columns(self, df: pl.DataFrame) -> list[str]:
        observed = {column.upper() for column in df.columns}
        contract = self.column_contract()

        warnings: list[str] = []

        missing_required = [
            column for column in contract.required_columns if column.upper() not in observed
        ]

        if missing_required:
            warnings.append(
                "missing_required_columns:"
                + ",".join(sorted(missing_required))
            )

        missing_optional = [
            column for column in contract.optional_columns if column.upper() not in observed
        ]

        if missing_optional:
            warnings.append(
                "missing_optional_columns:"
                + ",".join(sorted(missing_optional))
            )

        return warnings