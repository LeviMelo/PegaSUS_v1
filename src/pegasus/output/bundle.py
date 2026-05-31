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


class OutputBundle:
    """Filesystem representation of the locked 17-key PegaSUS output bundle."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        for name in OUTPUT_BUNDLE_KEYS:
            path = self.run_dir / name
            if name in {"V_fields", "Tables", "Maps"}:
                path.mkdir(parents=True, exist_ok=True)

        self._write_empty_tables()

    def _write_empty_tables(self) -> None:
        empty = pl.DataFrame()

        table_files = {
            "E_DAG": "E_DAG.parquet",
            "Q_tensor": "Q_tensor.parquet",
            "Warnings": "Warnings.parquet",
            "ModelAssociations": "ModelAssociations.parquet",
            "ResidualAssociations": "ResidualAssociations.parquet",
            "Hypotheses": "Hypotheses.parquet",
            "VariableDictionary": "VariableDictionary.parquet",
            "FailedBranches": "FailedBranches.parquet",
            "QuarantinedFields": "QuarantinedFields.parquet",
            "ForcedFields": "ForcedFields.parquet",
        }

        for key, filename in table_files.items():
            path = self.run_dir / filename
            if not path.exists():
                empty.write_parquet(path)

    def write_user_intent(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "UserIntent.json", payload)

    def write_run_config(self, payload: dict[str, Any]) -> None:
        write_json(self.run_dir / "RunConfig.json", payload)

    def write_p_vector(self, payload: dict[str, Any] | None = None) -> None:
        write_json(self.run_dir / "P_vector.json", payload or {})

    def write_reproducibility_manifest(self, run_id: str) -> None:
        manifest = ReproducibilityManifest(
            run_id=run_id,
            created_at=utc_now_iso(),
            environment=build_environment_manifest(),
        )
        write_json(self.run_dir / "ReproducibilityManifest.json", manifest)

    def validate_minimal(self) -> None:
        missing: list[str] = []

        expected_paths = {
            "V_fields": self.run_dir / "V_fields",
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

        for key, path in expected_paths.items():
            if not path.exists():
                missing.append(f"{key}: {path}")

        if missing:
            raise DataContractError(
                "Output bundle is missing required keys:\n" + "\n".join(missing)
            )

    def read_json(self, name: str) -> dict[str, Any]:
        path = self.run_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)