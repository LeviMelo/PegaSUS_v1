from enum import Enum


class SourceSystem(str, Enum):
    SIM_DO = "SIM-DO"
    SIH_RD = "SIH-RD"
    SINASC = "SINASC"
    CNES_ST = "CNES-ST"
    SIDRA = "SIDRA"


class Budget(str, Enum):
    fast = "fast"
    standard = "standard"
    deep = "deep"


class GeoMode(str, Enum):
    native = "native"
    AMC = "AMC"
    geneallocated = "geneallocated"
    hybrid = "hybrid"


class FieldState(str, Enum):
    verified = "verified"
    fragile = "fragile"
    forced_fragile = "forced_fragile"
    quarantined_descriptive = "quarantined_descriptive"
    quarantined_nochildren = "quarantined_nochildren"
    illegal_excluded = "illegal_excluded"


class MaterializationState(str, Enum):
    unmaterialized = "unmaterialized"
    metadata_only = "metadata_only"
    planned = "planned"
    materialized = "materialized"
    cached = "cached"
    blocked = "blocked"
    failed = "failed"
    quarantined = "quarantined"
