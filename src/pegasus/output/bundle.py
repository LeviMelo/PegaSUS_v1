from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.core.constants import OUTPUT_BUNDLE_KEYS
from pegasus.core.exceptions import DataContractError
from pegasus.core.manifests import (
    ReproducibilityManifest,
    build_environment_manifest,
    utc_now_iso,
    write_json,
)
from pegasus.output.schemas import PARQUET_FILENAMES, PARQUET_SCHEMAS, empty_df


class OutputBundle:
    """Filesystem representation of the locked 17-key PegaSUS output bundle.

    Some keys are represented as Parquet files, some as JSON files, and some as
    directories. The logical keys remain exactly the 17 keys specified by the
    production contract.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "Tables").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "Maps").mkdir(parents=True, exist_ok=True)
        self._write_empty_schema_tables()

    def _write_empty_schema_tables(self) -> None:
        for key, schema in PARQUET_SCHEMAS.items():
            filename = PARQUET_FILENAMES[key]
            path = self.run_dir / filename
            if not path.exists():
                empty_df(schema).write_parquet(path)

    def write_user_intent(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "UserIntent.json", payload)

    def write_run_config(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "RunConfig.json", payload)

    def write_p_vector(self, payload: dict[str, Any] | None = None) -> None:
        write_json(self.run_dir / "P_vector.json", payload or {})

    def write_reproducibility_manifest(
        self,
        run_id: str,
        registry_hashes: dict[str, str] | None = None,
        source_hashes: dict[str, str] | None = None,
    ) -> None:
        manifest = ReproducibilityManifest(
            run_id=run_id,
            created_at=utc_now_iso(),
            environment=build_environment_manifest(),
            registry_hashes=registry_hashes or {},
            source_hashes=source_hashes or {},
        )
        write_json(self.run_dir / "ReproducibilityManifest.json", manifest)

    def validate_minimal(self) -> None:
        missing: list[str] = []

        expected_paths = {
            "V_fields": self.run_dir / "V_fields.parquet",
            "E_DAG": self.run_dir / "E_DAG.parquet",
            "Q_tensor": self.run_dir / "Q_tensor.parquet",
            "P_vector": self.run_dir / "P_vector.json",
            "UserIntent": self.run_dir / "UserIntent.json",
            "Warnings": self.run_dir / "Warnings.parquet",
            "ModelAssociations": self.run_dir / "ModelAssociations.parquet",
            "ResidualAssociations": self.run_dir / "ResidualAssociations.parquet",
            "Hypotheses": self.run_dir / "Hypotheses.parquet",
            "Tables": self.run_dir / "Tables",
            "Maps": self.run_dir / "Maps",
            "VariableDictionary": self.run_dir / "VariableDictionary.parquet",
            "FailedBranches": self.run_dir / "FailedBranches.parquet",
            "QuarantinedFields": self.run_dir / "QuarantinedFields.parquet",
            "ForcedFields": self.run_dir / "ForcedFields.parquet",
            "RunConfig": self.run_dir / "RunConfig.json",
            "ReproducibilityManifest": self.run_dir / "ReproducibilityManifest.json",
        }

        for key in OUTPUT_BUNDLE_KEYS:
            if key not in expected_paths:
                missing.append(f"{key}: no expected path mapping")

        for key, path in expected_paths.items():
            if not path.exists():
                missing.append(f"{key}: {path}")

        if missing:
            raise DataContractError(
                "Output bundle is missing required keys:\n" + "\n".join(missing)
            )

        self.validate_parquet_schemas()

    def validate_parquet_schemas(self) -> None:
        errors: list[str] = []

        for key, expected_schema in PARQUET_SCHEMAS.items():
            filename = PARQUET_FILENAMES[key]
            path = self.run_dir / filename

            if not path.exists():
                errors.append(f"{key}: missing {path}")
                continue

            actual = pl.read_parquet(path).schema

            for column, expected_dtype in expected_schema.items():
                if column not in actual:
                    errors.append(f"{key}: missing column {column}")
                    continue

                actual_dtype = actual[column]
                if actual_dtype != expected_dtype:
                    errors.append(
                        f"{key}: column {column} has dtype {actual_dtype}, "
                        f"expected {expected_dtype}"
                    )

        if errors:
            raise DataContractError(
                "Output bundle Parquet schema validation failed:\n" + "\n".join(errors)
            )

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.run_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)