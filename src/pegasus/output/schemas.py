from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


OUTPUT_BUNDLE_FILES = {
    "V_fields": "V_fields.parquet",
    "E_DAG": "E_DAG.parquet",
    "Q_tensor": "Q_tensor.parquet",
    "P_vector": "P_vector.json",
    "UserIntent": "UserIntent.json",
    "Warnings": "Warnings.parquet",
    "ModelAssociations": "ModelAssociations.parquet",
    "ResidualAssociations": "ResidualAssociations.parquet",
    "Hypotheses": "Hypotheses.parquet",
    "Tables": "Tables",
    "Maps": "Maps",
    "VariableDictionary": "VariableDictionary.parquet",
    "FailedBranches": "FailedBranches.parquet",
    "QuarantinedFields": "QuarantinedFields.parquet",
    "ForcedFields": "ForcedFields.parquet",
    "RunConfig": "RunConfig.json",
    "ReproducibilityManifest": "ReproducibilityManifest.json",
}

OUTPUT_KEYS = tuple(OUTPUT_BUNDLE_FILES.keys())


class OutputSchemaRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    required_keys: tuple[str, ...] = OUTPUT_KEYS


class OutputValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
