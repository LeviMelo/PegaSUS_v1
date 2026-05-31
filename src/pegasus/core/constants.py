from __future__ import annotations

OUTPUT_BUNDLE_KEYS: tuple[str, ...] = (
    "V_fields",
    "E_DAG",
    "Q_tensor",
    "P_vector",
    "UserIntent",
    "Warnings",
    "ModelAssociations",
    "ResidualAssociations",
    "Hypotheses",
    "Tables",
    "Maps",
    "VariableDictionary",
    "FailedBranches",
    "QuarantinedFields",
    "ForcedFields",
    "RunConfig",
    "ReproducibilityManifest",
)

SIDRA_MAX_CELLS_PER_REQUEST = 49_900

PROJECT_NAME = "PegaSUS"
PACKAGE_NAME = "pegasus"