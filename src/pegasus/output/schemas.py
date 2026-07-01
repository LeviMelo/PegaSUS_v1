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


# --- Output profile contract (single source of truth; MSD-II §II.5 / MII-SCOPE-01) ---
# Which first-class keys must be non-empty per DataScope run_profile. Consumed by
# BOTH the output validator and the bundle manager's declared-empty emission -- kept
# here so the two never drift.
PROFILE_NONEMPTY: dict[str, set[str]] = {
    "core_vital": {
        "V_fields", "E_DAG", "Q_tensor", "P_vector", "UserIntent",
        "VariableDictionary", "RunConfig", "ReproducibilityManifest",
    },
    "contextual": {
        "V_fields", "E_DAG", "Q_tensor", "P_vector", "UserIntent", "Warnings",
        "ModelAssociations", "Hypotheses", "Tables", "VariableDictionary",
        "RunConfig", "ReproducibilityManifest",
    },
    "full": set(OUTPUT_BUNDLE_FILES),
}

# The LDO's typed LinkRecord (Hypotheses) is the authoritative inference output
# (MSD-II §II.8 / MII-OUT-01). ModelAssociations & ResidualAssociations are the
# legacy pairwise tables it subsumes -- never required, always allowed empty.
LEGACY_INFERENCE_KEYS: frozenset[str] = frozenset({"ModelAssociations", "ResidualAssociations"})
# All inference outputs are produced only at ExecutionStage=investigate (the LDO).
INFERENCE_KEYS: frozenset[str] = frozenset({"ModelAssociations", "ResidualAssociations", "Hypotheses"})


def required_nonempty_keys(run_profile: str, execution_stage: str) -> set[str]:
    """First-class keys that MUST be non-empty for (run_profile, execution_stage).

    Legacy inference tables are dropped always (LinkRecord-authoritative); the
    remaining inference outputs (Hypotheses) are required only at ``investigate``.
    """
    required = set(PROFILE_NONEMPTY.get(run_profile, PROFILE_NONEMPTY["core_vital"]))
    required -= set(LEGACY_INFERENCE_KEYS)
    if execution_stage != "investigate":
        required -= set(INFERENCE_KEYS)
    return required


class OutputSchemaRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    required_keys: tuple[str, ...] = OUTPUT_KEYS


class OutputValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
