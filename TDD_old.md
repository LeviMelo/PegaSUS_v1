# PegaSUS — Technical Design Document

Production Codebase Contract, Monolithic Scaffolding Specification, and Implementation Hardening Rules

## Status of This Document

This Technical Design Document defines how PegaSUS MUST be physically implemented. It is subordinate to the Master System Document in mathematical ontology and superior to ad-hoc implementation choices in software architecture. The Master System Document defines what PegaSUS is. This document defines how the codebase MUST enforce it.

PegaSUS is not an ETL project. PegaSUS is a measurement-process-aware epidemiological compiler. The codebase MUST preserve this identity through directory structure, schemas, registries, interfaces, runtime gates, output contracts, and abort behavior.

The historical `miniPegaSUS` name is retired. The package namespace is `pegasus`. Legacy terminology and directory names such as `problem1`, `problem2`, and `problem3` MUST NOT appear in production modules. The operative architecture is:

$$
\mathcal{D}
\longrightarrow
SHE
\longrightarrow
EFG
\longrightarrow
PIRS
\longrightarrow
\mathcal{O}_{run}
$$

where:

- `she/` implements the Substrate Harmonization Engine.
- `efg/` implements the Epidemiological Field Graph.
- `pirs/` implements the Parametric Inference & Residual Scanner.
- `output/` emits the immutable 17-key run bundle.

# 1. Non-Negotiable Implementation Doctrine

## 1.1 Anti-ETL Doctrine

The codebase MUST NOT degenerate into scripts that fetch tables, join columns, compute rates, and export CSV files. Every production transformation MUST operate through typed objects with explicit support, axes, carrier, unit, aggregation law, provenance, state, warnings, and lineage.

The following are forbidden:

- Silent fallback to simpler computations.
- Silent replacement of missingness with zero.
- Silent replacement of administrative race/color axes with IBGE self-declared axes.
- Silent replacement of SIM-DO or SINASC event streams with SIDRA Civil Registry aggregates.
- Generic “beds” variables without CNES capacity-vector index.
- Generic SIH cost variables when the estimand requires `VAL_SH`, `VAL_SP`, `VAL_UTI`, or `VAL_TOT`.
- Facility joins through unsanitized CNPJ-like identifiers.
- High-dimensional SIDRA exposure into EFG without legal bounded pushforward.
- Empirical correlation compression before candidate pruning.
- Dashboard-triggered computation.
- Production logic hidden in notebooks or one-off scripts.
- Production use of Pandas as the substrate engine for large data operations.

Pandas MAY be used in tests, tiny fixtures, and validation notebooks. It MUST NOT become the production substrate engine.

## 1.2 Non-Simplification Doctrine

Blueprint-locked modules MUST exist as directories, schemas, interfaces, validators, and blocked-state implementations before domain logic is written. A module may be inactive because its solver is not calibrated, but it MUST be visible, typed, and callable through a controlled interface.

The following modules MUST NOT be omitted:

- ST-DFM registry, input schema, output schema, certification table, and blocked solver state.
- Population denominator tensor interface and solver registry.
- `Bridge_R` / `Bridge_{\mathcal{R}}` race-axis bridge interface and prior validator.
- Seven-part structural legality predicate plus declaration-process gate.
- Source-field registry.
- Carrier registry.
- Unit registry.
- Aggregation registry.
- Provenance registry.
- Quality/state permissions.
- ICD six-state parser.
- Diagnostic topology registry.
- CNES capacity-vector registry.
- SIH cost-component registry.
- SIDRA normalized fact store.
- SIDRA classification projection and high-dimensional bounding.
- `Q(v)` state tensor.
- PIRS model/residual/HSIC interfaces.
- 17-key output bundle validator.

If an inactive module is invoked, it MUST emit a structured blocked state or raise a typed exception. It MUST NOT fall back to an unrelated simpler path.

## 1.3 Compute Boundary Doctrine

The production compute boundary is fixed.

CPU/out-of-core substrate:

- DuckDB.
- Polars.
- Apache Arrow.
- Parquet.
- GeoParquet.
- JSON/YAML registries.
- Disk-backed cache.

GPU numerical kernels:

- Native PyTorch CUDA only.

CUDA-backed modules:

- PIRS HSIC exact/Nyström/RFF kernels.
- ST-DFM production solver.
- Race bridge posterior simulation and tensor-embedded modes.
- Selected population tensor kernels when computationally justified.

Deferred and banned for v1:

- WSL2 requirement.
- RAPIDS.
- cuDF.
- Polars GPU.
- GPU-backed data warehouse.
- Docker-first implementation.
- Runtime CPU reference branches inside production numerical modules.

CPU reference checks belong in `tests/`, `notebooks/validation/`, or `diagnostics/`. They MUST NOT be blended into production code paths.

## 1.4 Monolithic Scaffolding Approach

PegaSUS MUST be scaffolded as a complete monolith before domain logic is implemented.

Slice 0 MUST create:

- Full repository tree.
- Package namespace.
- CLI skeleton.
- Registry schemas.
- Pydantic contracts.
- Exception hierarchy.
- Data lake layout.
- Output bundle skeleton.
- Run validator.
- Subsystem interfaces.
- Stubbed but typed module boundaries for SHE, EFG, PIRS, ST-DFM, population tensor, race bridge, SIDRA, DATASUS, CNES, geography, and dashboard.

No ingestion logic, rate computation, model fitting, HSIC scan, or visualization MAY be written before Slice 0 passes validation.

The purpose of Slice 0 is to physically constrain all future code. Domain logic is then inserted into a pre-existing compiler skeleton rather than grown as ad-hoc ETL.

# 2. Repository Topology

## 2.1 Root Layout

The repository root MUST be:

```text
PegaSUS/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  .env.example
  environment.yml
  uv.lock
  renv.lock

  config/
    project.yaml
    paths.yaml
    compute.yaml
    datasus.yaml
    sidra.yaml
    output.yaml

    registries/
      registry_manifest.yaml
      source_fields.yaml
      composite_decoders.yaml
      carrier_registry.yaml
      unit_registry.yaml
      aggregation_registry.yaml
      provenance_registry.yaml
      quality_permissions.yaml
      race_axis_registry.yaml
      race_bridge_priors.yaml
      icd_catalog.yaml
      icd_quality_groups.yaml
      diagnostic_topology.yaml
      clinical_event_definitions.yaml
      cnes_capacity_registry.yaml
      sih_cost_registry.yaml
      sidra_table_seed.jsonl
      sidra_views.yaml
      sidra_category_maps.yaml
      sidra_stitching.yaml
      sidra_regime_registry.yaml
      municipality_crosswalk_sources.yaml
      join_affordances.yaml
      bridge_grammars.yaml
      stdfm_registry.yaml
      population_solver_registry.yaml
      model_registry.yaml
      residual_registry.yaml
      hsic_registry.yaml
      null_registry.yaml
      output_schema.yaml

    intents/
      alagoas_smoke.json
      alagoas_sim_only.json
      alagoas_maternal_child.json
      northeast_region.json
      national_blocked.json

  src/
    pegasus/
      __init__.py
      __main__.py
      cli.py

      core/
      registries/
      storage/
      datasus/
      sidra/
      geo/
      she/
      efg/
      pirs/
      compute/
      output/
      dashboard/
      workflows/

  scripts/
    doctor.py
    check_cuda.py
    check_r_microdatasus.R
    validate_registry_hashes.py

  tests/
    unit/
    integration/
    smoke/
    golden/
    fixtures/

  notebooks/
    validation/
    exploration/

  docs/
    architecture.md
    data_lake.md
    registries.md
    datasus_subsystem.md
    sidra_subsystem.md
    she.md
    efg.md
    pirs.md
    race_bridge.md
    output_bundle.md
    compute_backend.md
    development_slices.md
    validation.md

  data/
    raw/
    processed/
    metadata/
    cache/
    manifests/
    intermediate/
    runs/
    diagnostics/
```

Production modules MUST NOT import from `scripts/`, `tests/`, or `notebooks/`.

## 2.2 Package Layout

The package tree MUST be:

```text
src/pegasus/
  __init__.py
  __main__.py
  cli.py

  core/
    __init__.py
    config.py
    paths.py
    logging.py
    hashing.py
    manifests.py
    exceptions.py
    constants.py
    enums.py
    schemas.py
    types.py
    validation.py
    time.py
    ids.py

  registries/
    __init__.py
    loader.py
    validators.py
    manifest.py
    source_fields.py
    composite_decoders.py
    carrier.py
    unit.py
    aggregation.py
    provenance.py
    quality.py
    race_axis.py
    race_bridge.py
    icd.py
    diagnostic_topology.py
    events.py
    cnes_capacity.py
    sih_cost.py
    sidra.py
    bridge.py
    population.py
    models.py
    residuals.py
    hsic.py
    nulls.py
    output.py

  storage/
    __init__.py
    duckdb.py
    polars.py
    arrow.py
    parquet.py
    geopaquet.py
    cache.py
    dataset.py
    materialization.py
    locks.py

  datasus/
    __init__.py
    client_microdatasus.py
    subprocess.py
    cache.py
    manifests.py
    profile.py
    schema_compare.py
    normalize.py
    decoders.py
    icd_parser.py
    joins.py
    quality.py
    raw_processed.py

    r_scripts/
      fetch_process_microdatasus.R

  sidra/
    __init__.py
    api.py
    cache.py
    metadata.py
    registry.py
    plan.py
    extract.py
    normalize.py
    facts.py
    category_maps.py
    stitching.py
    projection.py
    regime.py
    quality.py

  geo/
    __init__.py
    municipality_crosswalk.py
    support.py
    amc.py
    geneallocation.py
    adjacency.py
    geodata.py
    spatial_index.py

  she/
    __init__.py
    substrate.py
    source_registry.py
    composite_decode.py
    clinical_events.py
    cnes_capacity.py
    sih_costs.py
    civil_health.py
    zero_variance.py
    high_dimensional.py

    population/
      __init__.py
      schema.py
      loss.py
      solvers.py
      sparse_admm.py
      block_coordinate.py
      state_space.py
      race_modes.py
      torch_kernels.py
      diagnostics.py

    stdfm/
      __init__.py
      schema.py
      regime.py
      objective.py
      torch_solver.py
      certification.py
      blocked.py

  efg/
    __init__.py
    node.py
    lineage.py
    field_tensor.py
    operators.py
    align.py
    legality.py
    declaration.py
    dag.py
    q_tensor.py
    materialize.py
    equivalence.py
    core_seed.py
    bridges.py
    race_bridge.py
    warning.py
    failed_branch.py

  pirs/
    __init__.py
    field_selection.py
    design.py
    models.py
    families.py
    residuals.py
    crossfit.py
    hsic.py
    nystrom.py
    rff.py
    nulls.py
    fdr.py
    diagnostics.py

  compute/
    __init__.py
    devices.py
    torch_backend.py
    memory.py
    kernels.py
    random.py

  output/
    __init__.py
    bundle.py
    schemas.py
    variable_dictionary.py
    warnings.py
    tables.py
    maps.py
    validate.py
    reproducibility.py
    serializers.py

  dashboard/
    __init__.py
    read_only.py
    contracts.py

  workflows/
    __init__.py
    compile.py
    ingest_datasus.py
    ingest_sidra.py
    build_substrate.py
    build_efg.py
    run_pirs.py
    serialize.py
    validate_run.py
```

The directories `she/`, `efg/`, and `pirs/` are mandatory. The legacy directory names `problem1/`, `problem2/`, and `problem3/` are forbidden.

# 3. Runtime Configuration

## 3.1 Project Configuration

`config/project.yaml` MUST contain:

```yaml
project:
  name: "PegaSUS"
  package: "pegasus"
  version: "0.1.0"
  registry_version: "v1.0"
  timezone: "America/Maceio"
  default_locale: "pt-BR"
```

## 3.2 Paths Configuration

`config/paths.yaml` MUST contain:

```yaml
paths:
  data_root: "data"
  raw: "data/raw"
  processed: "data/processed"
  metadata: "data/metadata"
  cache: "data/cache"
  manifests: "data/manifests"
  intermediate: "data/intermediate"
  runs: "data/runs"
  diagnostics: "data/diagnostics"
```

## 3.3 Compute Configuration

`config/compute.yaml` MUST contain:

```yaml
compute:
  data_backend: "duckdb_polars_arrow"
  production_pandas_allowed: false
  numerical_backend: "pytorch"
  rapids_enabled: false
  wsl_required: false

  cuda:
    enabled_for:
      - "pirs_hsic_exact"
      - "pirs_hsic_nystrom"
      - "pirs_hsic_rff"
      - "race_bridge_posterior"
      - "race_bridge_tensor_embedded"
      - "stdfm_solver"
      - "population_tensor_kernels"
    require_cuda_for_region_hsic: true
    dtype: "float32"
    max_vram_fraction: 0.80
```

## 3.4 DATASUS Configuration

`config/datasus.yaml` MUST contain:

```yaml
datasus:
  backend: "microdatasus"
  rscript_path: "Rscript"
  preserve_raw: true
  preserve_processed: true
  r_timeout_seconds: 7200
  heartbeat_timeout_seconds: 900

  systems:
    - "SIM-DO"
    - "SINASC"
    - "SIH-RD"
    - "CNES-ST"

  chunking:
    SIM-DO: "uf_year"
    SINASC: "uf_year"
    SIH-RD: "uf_year_or_month_by_size"
    CNES-ST: "uf_month_or_year_by_source"

  optional_ftp_discovery: false
```

## 3.5 SIDRA Configuration

`config/sidra.yaml` MUST contain:

```yaml
sidra:
  base_url: "https://servicodados.ibge.gov.br/api/v3/agregados"
  max_cells_per_request: 49900
  view_mode: "flat"
  concurrency: 16
  timeout_seconds: 60
  retry_status_codes: [429, 500, 502, 503, 504]
  max_retries: 5
  backoff_initial_seconds: 0.25
  backoff_max_seconds: 10.0
  persist_raw_chunks: true
  persist_normalized_facts: true
  persist_full_tables_by_default: false
```

SIDRA view definitions are selective controls for large or redundant tables. They MUST NOT be required for every SIDRA table.

# 4. Data Lake Contract

## 4.1 Directory Layout

The data lake MUST be:

```text
data/
  raw/
    datasus/
      SIM-DO/
      SIH-RD/
      SINASC/
      CNES-ST/
    sidra/
      chunks/
    geo/
    external/

  processed/
    datasus/
      SIM-DO/
      SIH-RD/
      SINASC/
      CNES-ST/
    sidra/
      facts/
    geo/

  metadata/
    sidra/
      raw/
      normalized/
    datasus/
      profiles/
      schema_compare/
      variable_catalog/
    registries/
    geo/

  cache/
    sidra/
      http/
      values/
    datasus/
      microdatasus/

  manifests/
    datasus/
    sidra/
    runs/

  intermediate/
    she/
    efg/
    pirs/

  runs/
    <run_id>/

  diagnostics/
```

## 4.2 Immutability Rules

Raw artifacts MUST be immutable. Processed artifacts MUST be immutable. Repeated requests MUST either hit a content-addressed cache or create a new hash-specific directory.

A raw DATASUS path MUST follow:

```text
data/raw/datasus/{system}/uf={UF}/period={start}_{end}/{request_hash}/raw.{ext}
```

A processed DATASUS path MUST follow:

```text
data/processed/datasus/{system}/uf={UF}/period={start}_{end}/{request_hash}/processed.parquet
```

A SIDRA raw chunk path MUST follow:

```text
data/raw/sidra/chunks/{chunk_id}.json
```

A SIDRA normalized fact path MUST follow:

```text
data/processed/sidra/facts/{table_or_view_id}/{chunk_id}.parquet
```

# 5. Registry System

## 5.1 Registry Lifecycle

Registries are code-governing artifacts. They MUST be versioned, hashed, validated, and frozen into every run.

Every registry file MUST have:

```yaml
schema_version: "1.0"
registry_version: "..."
created_at: "YYYY-MM-DD"
updated_at: "YYYY-MM-DD"
provenance: "..."
entries: []
```

Every entry MUST have:

```yaml
id: "..."
status: "stable | calibrated | mutable_solver | experimental | deferred"
description: "..."
warnings: []
```

`config/registries/registry_manifest.yaml` MUST contain hashes for all registry files:

```yaml
registry_set:
  name: "PegaSUS_core"
  version: "v1.0"
  created_at: "YYYY-MM-DD"
  owner: "..."
  git_commit: "..."
  dirty_allowed: false

registries:
  source_fields:
    path: "source_fields.yaml"
    schema_version: "1.0"
    sha256: "..."
  carrier_registry:
    path: "carrier_registry.yaml"
    schema_version: "1.0"
    sha256: "..."
  unit_registry:
    path: "unit_registry.yaml"
    schema_version: "1.0"
    sha256: "..."
```

A registry change MUST require:

1. Version bump.
2. Changelog entry.
3. SHA-256 update.
4. Schema validation.
5. Compatibility declaration.
6. Run-manifest preservation of the old hash.

## 5.2 Source Field Registry

`source_fields.yaml` MUST map every raw source field to a canonical typed field before any data enter SHE.

Entry schema:

```yaml
id: "SIM-DO.CAUSABAS"
source_system: "SIM-DO"
raw_field: "CAUSABAS"
canonical_field: "underlying_icd"
type: "icd10"
carrier: "Deaths"
axis_or_mark: "H"
required_for:
  - "mortality_icd"
missingness_policy: "six_state_icd_parser"
parser: "icd10_six_state"
declaration_process: null
state_if_missing: "missing"
warnings: []
```

Every raw column MUST route through exactly one of:

- `Decode`
- `Parse`
- `PreserveMark`
- `Exclude`

Unmapped source fields MUST be excluded with audit metadata. They MUST NOT silently enter the substrate.

## 5.3 Composite Decoder Registry

`composite_decoders.yaml` MUST define:

- `Decode_SIM_IDADE`
- `Decode_SIH_AGE`
- `Decode_PESO`
- `Decode_count2`
- `Clamp_bool`
- `Filter_CNPJ`
- `Decode_cat`

The implementation MUST treat numeric-looking administrative fields as structural codes until decoded.

## 5.4 Carrier Registry

`carrier_registry.yaml` MUST define legal numerator-denominator carrier relations.

Example:

```yaml
- id: "deaths_over_population"
  numerator_carrier: "Deaths"
  denominator_carrier: "Population"
  role: "mortality_rate"
  output_kind: "intensive_density"
  default_unit: "rate"
  legal: true
  warnings: []
```

The registry MUST include:

- `Deaths / Population`
- `HospitalAdmissions / Population`
- `LiveBirths / Population`
- `BirthsToWomen / Women15_49`
- `InfantDeaths / LiveBirths`
- `NeonatalDeaths / LiveBirths`
- `PostNeonatalDeaths / LiveBirths`
- `MaternalDeaths / LiveBirths`
- `LowBirthWeightBirths / LiveBirths`
- `PretermBirths / LiveBirths`
- `CesareanBirths / LiveBirths`
- `CongenitalAnomalies / LiveBirths`
- `HospitalDeaths / HospitalAdmissions`
- `ICUDays / HospitalAdmissions`
- `HospitalDays / HospitalAdmissions`
- `HospitalCosts_SH / HospitalAdmissions`
- `HospitalCosts_SP / HospitalAdmissions`
- `HospitalCosts_UTI / HospitalAdmissions`
- `HospitalCosts_TOT / HospitalAdmissions`
- `FacilityCapacityVector_k / Population`
- `FacilityCapacityVector_k / Facilities`
- `Facilities / Population`
- `Physicians_k / Population`
- `DomicilesWithSanitation / Domiciles`

The generic carrier `Beds` MUST be illegal unless rewritten as `FacilityCapacityVector_k`.

## 5.5 Unit Registry

`unit_registry.yaml` MUST enforce unit legality.

Entry schema:

```yaml
- id: "counts_over_person_years"
  numerator_unit: "counts"
  denominator_unit: "person_years"
  output_unit: "rate"
  legal_roles:
    - "mortality_rate"
    - "hospitalization_rate"
```

SIH economic units MUST be separated:

- `reais_hospital_services` for `VAL_SH`
- `reais_professional_services` for `VAL_SP`
- `reais_icu_services` for `VAL_UTI`
- `reais_total_billing` for `VAL_TOT`

These units MUST NOT be pooled unless a registered `CostComponentSum` operator declares the composite estimand.

## 5.6 Quality Permissions Registry

`quality_permissions.yaml` MUST define field-state permissions:

```yaml
verified:
  model_outcome: true
  model_covariate: true
  generate_rn: true
  icd_descend: true
  dashboard_safe: true

fragile:
  model_outcome: true
  model_covariate: true
  generate_rn: false
  icd_descend: true
  dashboard_safe: "warning"

forced_fragile:
  model_outcome: true
  model_covariate: true
  generate_rn: false
  icd_descend: false
  dashboard_safe: false

quarantined_descriptive:
  model_outcome: false
  model_covariate: true
  generate_rn: false
  icd_descend: false
  dashboard_safe: false

quarantined_nochildren:
  model_outcome: false
  model_covariate: false
  generate_rn: false
  icd_descend: false
  dashboard_safe: false

illegal_excluded:
  model_outcome: false
  model_covariate: false
  generate_rn: false
  icd_descend: false
  dashboard_safe: false
```

## 5.7 Race Axis Registry

`race_axis_registry.yaml` MUST define measurement-process-indexed race axes:

```yaml
IBGE:
  race_axis: "self_declared"

SIM-DO:
  race_axis: "administrative_death_declaration"

SIH-RD:
  race_axis: "billing_record"

SINASC:
  race_axis: "administrative_mixed"

CNES-ST:
  race_axis: "none"

SIDRA:
  race_axis: "table_specific"
```

The EFG MUST block direct race-specific division when numerator and denominator race axes are declaration-incommensurable.

## 5.8 Race Bridge Prior Registry

`race_bridge_priors.yaml` MUST define source-specific emission matrices.

Validator rules:

- Matrix $C$ MUST exist.
- Shape MUST be `5 x 6`.
- Entries MUST be finite.
- Entries MUST be nonnegative.
- Rows MUST sum to 1 within tolerance.
- Self categories MUST match registry.
- Observed categories MUST match registry.
- Uncertainty object MUST exist.
- Provenance MUST be non-empty.
- Placeholder priors MUST block analytic posterior output.

Failure state:

```text
Bridge_R_status = blocked_prior_pending
```

When priors are invalid, the compiler MAY emit raw administrative race counts and race missingness observer fields. It MUST NOT emit denominator-aligned posterior race rates.

# 6. Pydantic Data Contracts

All core contracts MUST be implemented as Pydantic v2 models. Runtime objects MUST validate at module boundaries. Serialization MUST use explicit schema versions.

## 6.1 Core Enumerations

```python
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
```

## 6.2 UserIntent

```python
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class GeographySelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    level: Literal["municipality", "AMC", "UF", "region", "country"]
    codes: list[str]
    uf: list[str] = Field(default_factory=list)

class TimeWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_year: int
    end_year: int
    start_month: int | None = None
    end_month: int | None = None

class UserIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    geography: GeographySelector
    time: TimeWindow
    health_seeds: list[str] = Field(default_factory=list)
    mandatory_fields: list[str] = Field(default_factory=list)
    system_weights: dict[str, float] = Field(default_factory=dict)
    context_policy: list[str] = Field(default_factory=list)
    budget: Literal["fast", "standard", "deep"]
    geo_mode: Literal["native", "AMC", "geneallocated", "hybrid"]
    force_selectors: list[str] = Field(default_factory=list)
    exclude_systems: list[str] = Field(default_factory=list)

    execution_scale: Literal[
        "smoke",
        "state",
        "region",
        "national_blocked"
    ]

    race_tensor_mode: Literal[
        "decoupled",
        "downstream_bridge",
        "embedded_fixedC",
        "embedded_posteriorC",
        "embedded_sensitivity"
    ] = "decoupled"

    population_mode: Literal[
        "official_sidra_anchor",
        "imported_fixture",
        "independent_population_tensor",
        "sim_informed_population_tensor",
        "synthetic_test_fixture",
        "blocked_missing"
    ] = "official_sidra_anchor"
```

## 6.3 Lineage

```python
class Lineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent_ids: list[str]
    operator_type: str
    operator_params: dict
    registry_versions: dict[str, str]
    source_manifest_hashes: list[str] = Field(default_factory=list)
    code_version: str
```

Field identity MUST be:

$$
id(v)=SHA256(Lineage(v))
$$

## 6.4 FieldNode

```python
class FieldNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    kind: Literal[
        "extensive_measure",
        "intensive_density",
        "marked_functional",
        "context_gradient",
        "bridge_divergence",
        "bridge_module",
        "observer_proxy",
        "latent_context",
        "model_residual"
    ]

    carrier: str
    unit: str
    support: dict
    axes: dict
    aggregation: Literal[
        "additive",
        "weighted_mean",
        "statistical_functional",
        "compositional",
        "non_aggregable"
    ]

    role: list[str]
    source: list[str]
    operator: str | None
    provenance: list[str]
    state: FieldState
    warnings: list[str]
    lineage: Lineage
    materialization_state: MaterializationState
    path: str | None = None
    dashboard_safe: bool | Literal["warning"] = False
```

## 6.5 DATASUSRequestManifest

```python
class DATASUSRequestManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: Literal["SIM-DO", "SIH-RD", "SINASC", "CNES-ST"]
    uf: str

    year_start: int
    month_start: int | None
    year_end: int
    month_end: int | None

    information_system: str
    fetch_function: str
    process_function: str

    raw_path: str
    processed_path: str
    raw_sha256: str
    processed_sha256: str

    row_counts: dict[str, int]
    column_lists: dict[str, list[str]]

    started_at: str
    ended_at: str
    duration_seconds: float

    rscript_path: str
    r_version: str | None
    microdatasus_version: str | None
    read_dbc_version: str | None

    stdout_path: str
    stderr_path: str
    heartbeat_path: str
    exit_code: int
    status: Literal["success", "failed", "cached", "timeout", "blocked"]
    error_message: str | None
    request_hash: str
```

## 6.6 SIDRAFact

```python
class SIDRAFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    variable_id: str
    period: str

    locality_level: str
    locality_id: str

    classification_tuple: tuple[tuple[str, str], ...]
    category_tuple: tuple[tuple[str, str], ...]

    value_raw: str | None
    value_numeric: float | None
    value_status: Literal[
        "numeric",
        "blank",
        "dash_zero_or_nil",
        "not_available",
        "suppressed_or_unidentified",
        "non_numeric_symbol",
        "header_row",
        "parse_error"
    ]

    unit: str | None
    request_hash: str
    metadata_hash: str
    fetched_at: str
```

SIDRA MUST remain a normalized fact store. Wide panels are derived artifacts, not base substrate.

## 6.7 QState

```python
class QState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_id: str

    n_events: float | None
    n_denom: float | None
    n_eff: float | None
    cov_S: float | None
    cov_T: float | None
    missingness: float | None
    zero_inflation: float | None
    denom_fragility: float | None
    cv: float | None
    moran_i: float | None
    temporal_roughness: float | None
    spatial_entropy: float | None
    provenance_risk: float

    race_axis_source: str | None = None
    race_axis_target: str | None = None
    missing_race_share: float | None = None
    emission_prior_strength: float | None = None
    race_bridge_cv: float | None = None
    sensitivity_width: float | None = None
    bridge_mode: str | None = None

    state: FieldState
    dashboard_safe: bool | Literal["warning"]
    warnings: list[str]
    computed_at: str
    q_schema_version: str = "1.0"
```

## 6.8 WarningRecord

```python
class WarningRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning_id: str
    field_id: str | None
    source: str
    severity: Literal["info", "warning", "downgrade", "abort"]
    code: str
    message: str
    inherited_from: list[str] = Field(default_factory=list)
    created_at: str
```

## 6.9 FailedBranch

```python
class FailedBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failed_branch_id: str
    attempted_operator: str
    parent_field_ids: list[str]
    failure_stage: Literal[
        "alignment",
        "support",
        "axes",
        "carrier",
        "unit",
        "aggregation",
        "provenance",
        "quality",
        "declaration",
        "materialization",
        "solver",
        "output_validation"
    ]
    failed_terms: list[str]
    reason: str
    warnings: list[str]
    created_at: str
```

## 6.10 AlignmentResult, DeltaResult, and OperatorResult

```python
class AlignmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    aligned_left_id: str | None
    aligned_right_id: str | None
    operations_applied: list[str]
    support_after_alignment: dict | None
    warnings: list[str]
    failure_reason: str | None

class DeltaResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legal: bool
    delta_support: int
    delta_axes: int
    delta_carrier: int
    delta_unit: int
    delta_aggregation: int
    delta_provenance: int
    delta_quality: int
    delta_declaration: int
    failed_terms: list[str]
    warnings: list[str]
    failed_branch_id: str | None

class OperatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "blocked", "failed"]
    output_field_id: str | None
    materialization_state: MaterializationState
    provenance: list[str]
    warnings: list[str]
    failed_branch_id: str | None
```

# 7. DATASUS Interprocess Boundary

## 7.1 Boundary Rule

Python MUST orchestrate DATASUS acquisition. R/microdatasus MUST execute DATASUS fetching and first-pass processing. Python MUST NOT reimplement microdatasus internals.

The boundary is:

$$
Python\ orchestrator
\to
Rscript\ subprocess
\to
microdatasus
\to
raw/processed\ artifacts
\to
Python\ profiling/normalization
$$

`rpy2` is not the default. The production boundary is subprocess-based.

## 7.2 R Invocation Contract

Every R invocation MUST write:

```text
raw.*
processed.*
manifest.json
stdout.log
stderr.log
heartbeat.json
```

A zero exit code is not sufficient for success. Success is:

$$
exit\_code=0
\land manifest.json\ exists
\land raw\ artifact\ exists
\land processed\ artifact\ exists
\land hashes\ validate
$$

## 7.3 Exit Codes

The R bridge MUST use:

| Code | Meaning |
|---:|---|
| 0 | Success |
| 10 | Invalid input arguments |
| 11 | Unsupported DATASUS system |
| 12 | Invalid UF/year/month range |
| 20 | Network/download failure |
| 21 | DATASUS source unavailable |
| 22 | Retry exhausted |
| 30 | microdatasus fetch failure |
| 31 | microdatasus process failure |
| 32 | Raw output write failure |
| 33 | Processed output write failure |
| 40 | Memory allocation failure or suspected out-of-memory |
| 41 | Timeout preemption |
| 50 | Manifest write failure |
| 60 | Unknown R exception |

Unknown nonzero exit codes MUST be recorded as `r_unknown_failure`.

## 7.4 Heartbeat Contract

`heartbeat.json` MUST have:

```json
{
  "stage": "fetching | processing | writing_raw | writing_processed | done",
  "timestamp": "...",
  "rows_so_far": null,
  "message": "..."
}
```

Python MUST terminate the process tree when the heartbeat is stale beyond `heartbeat_timeout_seconds`.

On Windows, termination MUST clean child R processes. Zombie R processes are forbidden.

## 7.5 Chunking Contract

DATASUS requests MUST be chunked by:

$$
system,\ UF,\ year,\ month\ range
$$

Default chunking:

| Source | Chunking |
|---|---|
| SIM-DO | UF × year |
| SINASC | UF × year |
| SIH-RD | UF × year or UF × month if large |
| CNES-ST | UF × month or UF × year by source behavior |

No R request may span an unbounded multi-year/multi-UF region.

## 7.6 Python Orchestrator Signature

```python
def fetch_datasus_chunk(
    request: DATASUSRequestManifest,
    *,
    config: DatasusConfig,
    cache: DatasusCache,
    timeout_seconds: int,
    heartbeat_timeout_seconds: int,
) -> DATASUSRequestManifest:
    """Execute one microdatasus-backed R subprocess and return a validated manifest."""
```

# 8. DATASUS Normalization Contract

## 8.1 Profiling

Every raw and processed DATASUS artifact MUST be profiled before normalization.

Minimum profile per column:

- Column name.
- Observed dtype.
- Row count.
- Missing count.
- Missing rate.
- Nonblank count.
- Unique count.
- Unique rate.
- Top values.
- Full category list if low-cardinality.
- Numeric parse rate.
- Numeric minimum and maximum.
- Date parse rate.
- Date minimum and maximum.
- Value length distribution.
- Digit length distribution.
- Shape flags.
- Name-based role.
- Observed kind.

Shape flags are evidence, not semantic truth.

## 8.2 Raw/Processed Comparison

The schema comparison MUST emit:

- Raw-only columns.
- Processed-only columns.
- Common columns with changed missingness.
- Common columns with changed cardinality.
- Common columns with changed observed kind.
- Common columns with changed top-value distribution.
- Binary-like columns with outliers.
- Columns that lost nonblank values after processing.
- Columns that gained unique values after processing.

This is diagnostic evidence. It MUST NOT automatically override registry semantics.

## 8.3 Date Parsing Rule

Date parsing MUST run only on candidate date fields identified by registry and source profile. The system MUST NOT apply generic date parsing to every eight-digit field.

Invalid placeholders:

```text
00000000
0000-00-00
00/00/0000
99999999
9999-99-99
0
blank
```

MUST become explicit invalid or missing states. Numeric identifiers MUST NOT be parsed as dates because they have eight digits.

## 8.4 ICD Parsing Rule

ICD parsing MUST implement six states:

- `valid`
- `ill-defined`
- `blank`
- `invalid`
- `unparseable`
- `missing`

Blank, invalid, unparseable, and missing codes MUST NOT be coerced into R99.

Asterisks and source-specific line markers MUST be stripped only by field-specific parser rules. The raw string MUST remain in lineage.

## 8.5 SIM-DO Normalized Schema

`sim_events.parquet` MUST include:

```text
event_id
source_system
year
death_date
death_hour
birth_date
age_source
age_days
age_years
age_unit
raw_age_code
sex
race_color_admin
race_axis_type
race_missingness_state
mun_residence_cod6
mun_residence_cod7
mun_occurrence_cod6
mun_occurrence_cod7
place_of_death
facility_code
facility_code_state
underlying_icd_raw
underlying_icd_norm
underlying_icd_parse_state
cause_chain_raw
cause_chain_norm
cause_chain_parse_states
associated_conditions_raw
associated_conditions_norm
associated_conditions_parse_states
death_type
fetal_or_liveborn_status_source
maternal_age_years
maternal_education_legacy
maternal_education_2010
maternal_occupation_cbo
maternal_living_children_count
maternal_deceased_children_count
pregnancy_type
gestational_weeks_death
gestational_age_group_death
delivery_type_death_context
death_timing_relative_to_delivery
birth_weight_death_context_grams
death_during_pregnancy
death_during_puerperium
medical_assistance
exam_performed
surgery_performed
autopsy_performed
svo_iml_municipality
certificate_date
reporting_delay
investigation_status
investigation_date
cause_altered
raw_record_hash
processed_record_hash
source_manifest_hash
```

## 8.6 SIH-RD Normalized Schema

`sih_events.parquet` MUST include:

```text
event_id
source_system
year
admit_date
discharge_date
age_source
age_days
age_years
age_unit
age_unit_state
sex
race_color_billing
race_axis_type
race_missingness_state
mun_residence_cod6
mun_residence_cod7
mun_movement_cod6
mun_movement_cod7
principal_icd_raw
principal_icd_norm
principal_icd_parse_state
secondary_icd_raw
secondary_icd_norm
secondary_icd_parse_state
all_diagnosis_codes_norm
all_diagnosis_parse_states
procedure_requested_raw
procedure_requested_norm
procedure_performed_raw
procedure_performed_norm
procedure_code_parse_state
stay_length_days
stay_length_valid_state
icu_type_mark
icu_mark_state
icu_days_month_total
icu_days_hospitalization_total
hospital_service_cost_real
professional_service_cost_real
icu_cost_real
total_admission_cost_real
federal_hospital_service_cost_real
federal_professional_service_cost_real
intermediate_care_cost_real
cost_valid_state
hospital_death
hospital_death_state
high_risk_pregnancy_mark
hospital_cnpj
hospital_cnpj_state
maintainer_cnpj
maintainer_cnpj_state
facility_match_status
facility_match_type
facility_match_confidence
facility_join_warning
fallback_support
raw_record_hash
processed_record_hash
source_manifest_hash
```

SIH economic components MUST remain separate.

## 8.7 SINASC Normalized Schema

`sinasc_events.parquet` MUST include:

```text
event_id
source_system
year
birth_date
birth_hour
facility_code
facility_code_state
mun_birth_occurrence_cod6
mun_birth_occurrence_cod7
mun_residence_cod6
mun_residence_cod7
birth_location
maternal_age_years
maternal_age_state
maternal_birth_date
maternal_marital_status
maternal_education_legacy
maternal_education_2010
maternal_school_grade
maternal_occupation_cbo
maternal_living_children_count
maternal_deceased_children_count
prior_pregnancy_count
prior_vaginal_delivery_count
prior_cesarean_delivery_count
maternal_state_of_birth
maternal_municipality_of_birth
maternal_birth_state_code
newborn_sex
newborn_race_admin
newborn_race_axis_type
newborn_race_missingness_state
maternal_race_admin
maternal_race_axis_type
gestational_age_group
gestational_weeks
gestational_weeks_state
pregnancy_type
delivery_type
prenatal_visit_group
apgar_1min
apgar1_state
apgar_5min
apgar5_state
birth_weight_grams
birth_weight_state
anomaly_flag
anomaly_flag_state
anomaly_icd_raw
anomaly_icd_norm
anomaly_icd_parse_state
birth_record_entry_date
birth_record_receipt_date
birth_reporting_delay
raw_record_hash
processed_record_hash
source_manifest_hash
```

## 8.8 CNES-ST Normalized Schema

`cnes_facilities.parquet` MUST include:

```text
facility_id
facility_id_state
source_system
year
month
facility_municipality_cod6
facility_municipality_cod7
facility_cnpj
facility_cnpj_state
maintainer_cnpj
maintainer_cnpj_state
sus_linkage
sus_linkage_state
facility_room_capacity_vector_json
facility_bed_capacity_vector_json
workforce_capacity_vector_json
programmatic_flag_vector_json
care_complexity_flag_vector_json
care_modality_flag_vector_json
urgency_emergency_flag
urgency_emergency_flag_state
surgical_obstetric_center_flag_vector_json
support_service_flag_vector_json
hospital_bed_presence_flag
hospital_bed_presence_flag_state
invalid_flag_share
zero_cnpj_share
raw_record_hash
processed_record_hash
source_manifest_hash
```

CNES boolean-like fields MUST pass through `Clamp_bool`. Values greater than 1 MUST become `InvalidFlagState`, not `true`.

CNPJ-like identifiers MUST pass through `Filter_CNPJ`. `0` and `00000000000000` MUST be nullified before linkage.

# 9. SIDRA Subsystem Contract

## 9.1 Normalized Fact Rule

SIDRA MUST be stored as long-form facts:

$$
(table,\ variable,\ period,\ locality,\ classification\text{-}category\ tuple)\to value
$$

SIDRA MUST NOT become a wide-table substrate.

## 9.2 Metadata Pipeline

The metadata pipeline MUST execute:

1. Load selected table seed.
2. Fetch `/metadados`.
3. Fetch `/periodos`.
4. Fetch `/localidades/{level}`.
5. Normalize metadata.
6. Validate table, variable, period, locality, classification, and category references.
7. Build extraction plan.

Normalized metadata outputs:

```text
sidra_tables.parquet
sidra_variables.parquet
sidra_classifications.parquet
sidra_categories.parquet
sidra_periods.parquet
sidra_localities.parquet
```

Manual JSONL table seeds are selection seeds, not permanent truth.

## 9.3 SIDRA Chunk Planner

Cell estimate:

$$
cells
=
n_{localities}
\times
n_{periods}
\times
n_{variables}
\times
\prod_c n_{categories,c}
$$

The hard ceiling is:

$$
SIDRA\_MAX\_CELLS\_PER\_REQUEST=49{,}900
$$

If a planned request exceeds the ceiling, the planner MUST split deterministically by:

1. Localities.
2. Highest-cardinality classification.
3. Periods.
4. Variables.

If splitting cannot produce legal chunks, compilation MUST abort.

## 9.4 SIDRA Chunk Schema

```python
class SIDRAChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    table_id: str
    variables: list[str]
    periods: list[str]
    locality_level: str
    localities: list[str]
    classifications: dict[str, list[str]]
    estimated_cells: int
    request_url: str
    request_params: dict
```

`chunk_id` MUST be `SHA256(chunk_spec)`.

## 9.5 SIDRA Planner Signature

```python
def plan_sidra_chunks(
    request: SIDRARequest,
    metadata: SIDRAMetadata,
    *,
    max_cells_per_request: int = 49900,
) -> list[SIDRAChunk]:
    """Plan bounded SIDRA requests. Abort if legal chunking cannot satisfy the cell ceiling."""
```

## 9.6 SIDRA Classification Projection

SIDRA classification tuples MUST be projected through registered matrices before EFG materialization.

For additive measures:

$$
Y^{axis}_{s,t,\alpha,q}
=
\sum_{\chi}
\Pi^{(q)}_{\alpha,\chi}Y^{raw}_{s,t,\chi,q}
$$

For rates and proportions, projection MUST recover numerator and denominator first. Direct projection of proportions is illegal unless the registry declares a valid weighted operation.

## 9.7 High-Dimensional Bounding

The SHE-to-EFG boundary MUST execute:

$$
Y^{raw}
\to
Y^\dagger
\to
\Pi_{Clsf\to Axis}
\to
\pi^{bound}_*
\to
\Delta
\to
V_{fields}
$$

A high-dimensional SIDRA field MUST NOT enter EFG unbounded when legal bounded pushforward exists.

# 10. SHE Implementation Contract

## 10.1 SHE Inputs and Outputs

SHE input:

- Normalized DATASUS event/facility tables.
- Normalized SIDRA facts.
- Geospatial support.
- Registries.
- User intent.
- Compute configuration.

SHE output:

```python
class SubstrateBundle(BaseModel):
    B_official: list[FieldNode]
    B_harmonized: list[FieldNode]
    B_deflated: list[FieldNode]
    B_reconstructed: list[FieldNode]
    B_latent: list[FieldNode]
    B_cross_sectional: list[FieldNode]
    B_excluded: list[FieldNode]
    warnings: list[WarningRecord]
    failed_branches: list[FailedBranch]
```

## 10.2 SHE Build Signature

```python
def build_substrate(
    *,
    intent: UserIntent,
    datasus_artifacts: list[DATASUSRequestManifest],
    sidra_artifacts: list[SIDRAFact],
    registries: RegistryBundle,
    geo_support: GeoSupport,
    compute: ComputeConfig,
) -> SubstrateBundle:
    """Compile raw source artifacts into admissible substrate fields."""
```

## 10.3 Composite Decoders

Mandatory decoder signatures:

```python
def decode_sim_idade(raw: str | int | None) -> DecodedAge:
    """Decode SIM-DO IDADE structural code into canonical age units."""

def decode_sih_age(cod_idade: str | int | None, idade: str | int | None) -> DecodedAge:
    """Decode SIH age using COD_IDADE as the unit source."""

def decode_physical_scalar(
    raw: str | int | float | None,
    *,
    unit: str,
    lower: float,
    upper: float,
    sentinels: set[str],
) -> DecodedScalar:
    """Decode a physical scalar without treating sentinels as valid values."""

def decode_count2(raw: str | int | None, *, sentinels: set[str]) -> DecodedCount:
    """Decode two-character count fields while preserving leading-zero semantics."""

def clamp_bool(raw: object) -> DecodedBoolean:
    """Decode administrative boolean-like fields without clamping outliers to true."""

def filter_cnpj(raw: object) -> FilteredCNPJ:
    """Sanitize corporate identifiers and nullify all-zero identifiers."""
```

## 10.4 Zero-Variance Drop

SHE MUST drop all 100% missing or 100% constant source columns from analytic field space. Dropped fields MUST enter audit metadata only.

If a user forces a zero-variance field, it MUST become:

```text
State = quarantined_descriptive
ModelCovariate = false
ModelOutcome = false
HSICCandidate = false
```

## 10.5 Population Denominator Modes

The denominator mode MUST be explicit:

```python
class DenominatorContract(BaseModel):
    mode: Literal[
        "official_sidra_anchor",
        "imported_fixture",
        "independent_population_tensor",
        "sim_informed_population_tensor",
        "synthetic_test_fixture",
        "blocked_missing"
    ]
    source: str
    provenance: list[str]
    state: FieldState
    dashboard_safe: bool | Literal["warning"]
    allowed_for_rates: bool
    warnings: list[str]
```

Rules:

- `official_sidra_anchor` is allowed for rates.
- `independent_population_tensor` is allowed for rates with reconstructed provenance.
- `sim_informed_population_tensor` is allowed for SIM rates only with feedback-risk warning and dashboard downgrade.
- `synthetic_test_fixture` is forbidden outside tests.
- `blocked_missing` forbids rate generation.

## 10.6 Population Solver Gate

Dense Brazil-scale optimization is forbidden.

If:

$$
|\Omega_{epi}|>10^7
$$

then `projected_gradient_small` MUST abort.

Allowed production solver registry:

- `projected_gradient_small`
- `regional_block_coordinate`
- `sparse_ADMM`
- `primal_dual_sparse`
- `state_space_smoother`

The solver MAY be mutable. Its input/output contract is not mutable.

## 10.7 ST-DFM Contract

ST-DFM MUST expose typed objects before solver completion.

```python
class STDFMInput(BaseModel):
    field_ids: list[str]
    y_matrix_path: str
    y_matrix_shape: tuple[int, int]
    support_index_path: str
    observed_mask_path: str
    link_function_by_field: dict[str, str]
    denominator_by_field: dict[str, str] | None
    covariates_z_path: str | None
    spatial_laplacian_path: str | None
    registry_version: str

class STDFMOutput(BaseModel):
    status: Literal["blocked_solver_pending", "fitted", "failed", "uncertified"]
    latent_factor_path: str | None
    loading_matrix_path: str | None
    reconstructed_fields_path: str | None
    certification_table_path: str | None
    uncertainty_path: str | None
    warnings: list[str]
```

If a field is routed to `bounded_interpolate` and the solver is unavailable, the state MUST be `blocked_solver_pending`. The system MUST NOT silently route it to direct or cross-sectional output.

# 11. EFG Implementation Contract

## 11.1 EFG Build Signature

```python
def build_efg(
    *,
    substrate: SubstrateBundle,
    intent: UserIntent,
    registries: RegistryBundle,
    compute: ComputeConfig,
) -> EFGResult:
    """Build the typed epidemiological field graph."""
```

## 11.2 Field Graph Rule

The EFG is a field graph, not a cell graph. A node represents:

$$
v:L_v\to\mathbb{R}
$$

Cells live in field tensors. Nodes live in `V_fields`.

Production graph storage MUST be table-backed:

```text
V_fields.parquet
E_DAG.parquet
```

NetworkX MAY be used only for smoke tests and developer diagnostics. It MUST NOT be the production graph engine.

## 11.3 Delta Predicate Interface

The production legality predicate is:

$$
\Delta
=
\Delta_{support}
\Delta_{axes}
\Delta_{carrier}
\Delta_{unit}
\Delta_{aggregation}
\Delta_{provenance}
\Delta_{quality}
\Delta_{declaration}
$$

Signature:

```python
def evaluate_delta(
    *,
    parents: list[FieldNode],
    operator: OperatorSpec,
    intent: UserIntent,
    registries: RegistryBundle,
    alignment: AlignmentResult | None = None,
) -> DeltaResult:
    """Evaluate full EFG legality, including declaration-process compatibility."""
```

A `DeltaResult` with `legal=False` MUST create a `FailedBranch`.

## 11.4 Declaration Gate

Race-specific direct division MUST fail when:

$$
RaceAxis(\nu)\neq RaceAxis(\mu)
$$

unless the numerator has passed through `Bridge_R`.

Signature:

```python
def evaluate_declaration_compatibility(
    *,
    numerator: FieldNode,
    denominator: FieldNode | None,
    operator: OperatorSpec,
    registries: RegistryBundle,
) -> DeclarationResult:
    """Reject declaration-process-incommensurable operations."""
```

## 11.5 Alignment Interface

```python
def align_fields(
    *,
    left: FieldNode,
    right: FieldNode,
    operator: OperatorSpec,
    intent: UserIntent,
    geo_support: GeoSupport,
    registries: RegistryBundle,
) -> AlignmentResult:
    """Align support and axes before legality evaluation."""
```

Alignment MUST handle:

- Spatial support.
- Temporal support.
- Age support.
- Sex support.
- Race declaration process.
- Diagnostic topology.
- Facility support.
- SIDRA context classification.

## 11.6 Operator Registry Interface

```python
def apply_operator(
    *,
    operator: OperatorSpec,
    parents: list[FieldNode],
    delta: DeltaResult,
    materializer: FieldMaterializer,
    registries: RegistryBundle,
) -> OperatorResult:
    """Apply a legal operator and return a structured result."""
```

Operators MUST include:

- `sigma_C`
- `pi_*`
- `pi^*`
- `Pi_Clsf_to_Axis`
- `pi_bound_*`
- `RN`
- `Psi_phi`
- `Std_W`
- `Comp_epsilon`
- `Lag_tau`
- `Bridge`
- `Contrast`
- `Shrink`
- `CostComponentSum`
- `ModelResidual`

## 11.7 Race Bridge Interface

```python
def execute_bridge_r(
    *,
    administrative_counts: FieldTensorRef,
    self_declared_denominator: FieldTensorRef,
    source_axis: str,
    target_axis: str,
    prior: RaceEmissionPrior,
    mode: Literal["fixedC_dynamicW", "posteriorC", "sensitivity"],
    support: dict,
    seed: int,
    compute: ComputeConfig,
) -> RaceBridgeResult:
    """Emit raw counts, missingness observer field, posterior self-aligned estimate, and sensitivity intervals."""
```

The race bridge MUST NOT overwrite raw administrative counts.

Race bridge output MUST include:

- Raw administrative count.
- Missing/unknown race observer field.
- Self-aligned Bayesian race estimate when priors validate.
- Sensitivity interval estimate when requested.
- Bridge metadata.
- Warnings.
- Q-state extensions.

## 11.8 Q Tensor Interface

```python
def compute_q_state(
    *,
    field: FieldNode,
    tensor: FieldTensorRef | None,
    denominator: FieldTensorRef | None,
    provenance: list[str],
    warnings: list[WarningRecord],
    registries: RegistryBundle,
) -> QState:
    """Compute Q(v) and assign state permissions."""
```

`Q(v)` MUST control:

- Model outcome permission.
- Model covariate permission.
- RN generation permission.
- ICD descent permission.
- Dashboard safety.
- HSIC eligibility.

# 12. PIRS Implementation Contract

## 12.1 PIRS Input and Output

PIRS receives legal EFG fields only. It MUST NOT generate raw epidemiological variables except model-derived residual fields.

```python
class ModelInput(BaseModel):
    outcome_field_id: str
    covariate_field_ids: list[str]
    offset_field_id: str | None
    family: str
    support_index_path: str
    design_matrix_path: str
    outcome_vector_path: str
    q_state_filter: dict
    registry_versions: dict[str, str]

class ModelOutput(BaseModel):
    model_id: str
    status: Literal["fitted", "failed", "skipped"]
    family: str
    coefficients_path: str | None
    diagnostics: dict
    fitted_values_path: str | None
    residual_field_id: str | None
    warnings: list[str]
```

## 12.2 PIRS Run Signature

```python
def run_pirs(
    *,
    efg: EFGResult,
    intent: UserIntent,
    registries: RegistryBundle,
    compute: ComputeConfig,
) -> PIRSResult:
    """Run parametric models, residual extraction, and nonlinear residual scanner."""
```

## 12.3 Residual Contract

```python
class ResidualField(BaseModel):
    field_id: str
    parent_model_id: str
    residual_type: Literal[
        "deviance",
        "pearson",
        "randomized_quantile",
        "standardized",
        "ilr"
    ]
    support: dict
    provenance: list[str]
    state: FieldState
```

Residual fields MUST have:

```text
provenance = ["model_derived"]
```

## 12.4 HSIC Input and Output

```python
class HSICInput(BaseModel):
    outcome_residual_field_id: str
    covariate_field_id: str
    support_intersection: dict
    mode: Literal["exact", "nystrom", "rff"]
    kernel: Literal["rbf", "linear", "matern"]
    bandwidth_policy: Literal["median_subsample", "fixed", "registry"]
    landmark_policy: Literal["uniform_seeded", "kmeans_seeded"] | None
    n_landmarks: int | None
    null_strategy: str
    permutations: int
    seed: int

class HSICOutput(BaseModel):
    hypothesis_id: str
    outcome_field_id: str
    covariate_field_id: str
    residual_field_id: str
    hsic_mode: str
    statistic: float
    p_value: float | None
    q_value: float | None
    null_strategy: str
    fdr_method: str
    n_eff: float
    approximation_diagnostics: dict
    warnings: list[str]
```

## 12.5 Residual Modes

PIRS MUST implement:

- `in_sample`
- `cross_fitted`
- `parametric_bootstrap`
- `posterior_predictive`

Defaults:

| Budget | Residual mode |
|---|---|
| fast | in-sample |
| standard | cross-fitted |
| deep | cross-fitted + parametric bootstrap |

Standard and deep HSIC MUST NOT use in-sample residuals.

## 12.6 Null Regimes

Null strategies MUST include:

| Panel type | Null strategy |
|---|---|
| Annual municipal panel | spatial block + cyclic time shift |
| Monthly seasonal panel | season-preserving moving-block circular shift |
| Cross-sectional census | geo-adjacency shuffle |
| Facility stock | restricted intra-UF swap |
| Sparse stratified | bootstrap within strata |

Within-season arbitrary random shuffling is forbidden.

# 13. Output Bundle Contract

## 13.1 Immutable 17-Key Bundle

Every run MUST emit exactly 17 first-class keys:

```json
{
  "V_fields": "...",
  "E_DAG": "...",
  "Q_tensor": "...",
  "P_vector": "...",
  "UserIntent": "...",
  "Warnings": "...",
  "ModelAssociations": "...",
  "ResidualAssociations": "...",
  "Hypotheses": "...",
  "Tables": "...",
  "Maps": "...",
  "VariableDictionary": "...",
  "FailedBranches": "...",
  "QuarantinedFields": "...",
  "ForcedFields": "...",
  "RunConfig": "...",
  "ReproducibilityManifest": "..."
}
```

No production run is valid unless all 17 keys exist.

## 13.2 Output Directory

A run directory MUST be:

```text
data/runs/<run_id>/
  V_fields.parquet
  E_DAG.parquet
  Q_tensor.parquet
  P_vector.json
  UserIntent.json
  Warnings.parquet
  ModelAssociations.parquet
  ResidualAssociations.parquet
  Hypotheses.parquet
  Tables/
  Maps/
  VariableDictionary.parquet
  FailedBranches.parquet
  QuarantinedFields.parquet
  ForcedFields.parquet
  RunConfig.json
  ReproducibilityManifest.json
```

## 13.3 V_fields Schema

`V_fields.parquet` MUST include:

```text
field_id
name
kind
carrier
unit
aggregation
role
source
support_json
axes_json
operator
provenance
state
dashboard_safe
warnings
lineage_hash
registry_hash
materialization_state
path
```

## 13.4 E_DAG Schema

`E_DAG.parquet` MUST include:

```text
edge_id
parent_field_id
child_field_id
operator
operator_params_json
registry_versions_json
created_at
```

## 13.5 Q_tensor Schema

`Q_tensor.parquet` MUST include all `QState` fields plus:

```text
computed_at
q_schema_version
```

## 13.6 VariableDictionary Schema

`VariableDictionary.parquet` MUST include:

```text
field_id
display_name
technical_name
definition
estimand_label
source_systems
carrier
unit
support_description
axis_description
provenance_description
state
dashboard_safe
interpretation_warning
```

Race-related fields MUST use only allowed estimand labels.

Facility-capacity fields MUST declare:

```text
capacity_index
```

Cost fields MUST declare:

```text
cost_component
```

Diagnostic fields MUST declare:

```text
diagnostic_role
topology
position
```

SIDRA fields MUST declare:

```text
projection_matrix_id
stitch_metadata
```

when applicable.

## 13.7 Hypotheses Schema

`Hypotheses.parquet` MUST include:

```text
hypothesis_id
outcome_field_id
covariate_field_id
residual_field_id
statistic
p_value
q_value
hsic_mode
residual_mode
fold_scheme
bootstrap_count
residual_uncertainty
null_strategy
fdr_method
n_eff
state
warnings
approximation_diagnostics_json
```

## 13.8 ReproducibilityManifest Schema and Global Telemetry

`ReproducibilityManifest.json` MUST be the first-class home for global run telemetry. Telemetry is an observed execution artifact, not user configuration. It MUST NOT be stored only inside `RunConfig.json`.

`RunConfig.json` records intended configuration. `ReproducibilityManifest.json` records what actually happened during execution.

`ReproducibilityManifest.json` MUST include:

```json
{
  "run_id": "...",
  "created_at": "...",
  "completed_at": "...",
  "status": "success | failed | aborted | partial",
  "code_version": {
    "package_version": "...",
    "git_commit": "...",
    "git_dirty": false
  },
  "environment": {
    "python_version": "...",
    "os": "...",
    "duckdb_version": "...",
    "polars_version": "...",
    "pyarrow_version": "...",
    "torch_version": "...",
    "torch_cuda_available": true,
    "cuda_device_name": "...",
    "r_version": "...",
    "microdatasus_version": "...",
    "read_dbc_version": "..."
  },
  "registry_hashes": {},
  "source_manifest_hashes": [],
  "random_seeds": {},
  "telemetry": {
    "total_wall_seconds": 0.0,
    "stage_wall_seconds": {
      "config_load_seconds": 0.0,
      "registry_validation_seconds": 0.0,
      "datasus_acquire_seconds": 0.0,
      "datasus_profile_seconds": 0.0,
      "datasus_normalize_seconds": 0.0,
      "sidra_metadata_seconds": 0.0,
      "sidra_plan_seconds": 0.0,
      "sidra_fetch_seconds": 0.0,
      "sidra_normalize_seconds": 0.0,
      "geo_support_seconds": 0.0,
      "she_build_seconds": 0.0,
      "population_solver_seconds": 0.0,
      "stdfm_seconds": 0.0,
      "efg_build_seconds": 0.0,
      "q_tensor_seconds": 0.0,
      "pirs_model_seconds": 0.0,
      "pirs_hsic_seconds": 0.0,
      "output_serialization_seconds": 0.0,
      "output_validation_seconds": 0.0
    },
    "stage_status": {
      "config_load": "success | skipped | blocked | failed",
      "registry_validation": "success | skipped | blocked | failed",
      "datasus_acquire": "success | skipped | blocked | failed",
      "datasus_profile": "success | skipped | blocked | failed",
      "datasus_normalize": "success | skipped | blocked | failed",
      "sidra_metadata": "success | skipped | blocked | failed",
      "sidra_plan": "success | skipped | blocked | failed",
      "sidra_fetch": "success | skipped | blocked | failed",
      "sidra_normalize": "success | skipped | blocked | failed",
      "geo_support": "success | skipped | blocked | failed",
      "she_build": "success | skipped | blocked | failed",
      "population_solver": "success | skipped | blocked | failed",
      "stdfm": "success | skipped | blocked | failed",
      "efg_build": "success | skipped | blocked | failed",
      "q_tensor": "success | skipped | blocked | failed",
      "pirs_model": "success | skipped | blocked | failed",
      "pirs_hsic": "success | skipped | blocked | failed",
      "output_serialization": "success | skipped | blocked | failed",
      "output_validation": "success | skipped | blocked | failed"
    },
    "resource_summary": {
      "peak_rss_mb": null,
      "peak_vram_mb": null,
      "duckdb_temp_bytes": null,
      "rows_read": {},
      "rows_written": {},
      "parquet_bytes_written": 0
    }
  }
}
```

Stage-level telemetry MUST be emitted even when a run aborts. If a stage is not reached, its status MUST be `skipped`. If a stage is structurally unavailable because a solver or module is scaffolded but inactive, its status MUST be `blocked`. If a stage starts and fails, its elapsed time MUST be recorded with status `failed`.

The compiler MUST write telemetry incrementally during execution to prevent silent freezing. At minimum, each major workflow boundary MUST flush the current telemetry state before starting the next stage.

## 13.9 Output Validator Signature

```python
def validate_output_bundle(
    *,
    run_dir: str,
    schema_registry: OutputSchemaRegistry,
) -> OutputValidationResult:
    """Validate exact 17-key output bundle and all mandatory cross-references."""
```

Validator MUST check:

- All 17 keys exist.
- No extra first-class keys exist.
- User intent is frozen.
- Run config is frozen.
- Registry versions are recorded.
- Source hashes are recorded.
- `Q_tensor` is nonempty.
- `VariableDictionary` covers all `V_fields`.
- `Warnings` link to valid field IDs or run scope.
- `E_DAG` IDs match `V_fields`.
- `illegal_excluded` fields do not enter model outputs.
- Dashboard-safety flags exist.
- Race bridge metadata exists for race-bridged fields.
- CNES capacity metadata exists for capacity fields.
- SIH cost-component metadata exists for cost fields.
- `ReproducibilityManifest.json` contains global telemetry.
- `telemetry.total_wall_seconds` exists and is nonnegative.
- Every executed stage has a nonnegative wall-clock duration.
- Every stage has a terminal status.
- Aborted runs still contain partial telemetry.
- DATASUS per-chunk timings in `DATASUSRequestManifest` are consistent with aggregate `datasus_acquire_seconds`.
- SIDRA per-chunk timings, when present, are consistent with aggregate `sidra_fetch_seconds`.
- CUDA-required stages record CUDA availability and device metadata.
- No run with missing global telemetry can validate as a completed production run.

# 14. CLI Contract

The CLI MUST use Typer or an equivalent typed command framework.

Required commands:

```text
pegasus init
pegasus doctor
pegasus validate-config
pegasus registries validate
pegasus sidra metadata --tables config/registries/sidra_table_seed.jsonl
pegasus sidra plan --view <view_id>
pegasus sidra extract --plan <plan_id>
pegasus datasus ingest --system SIM-DO --uf AL --years 2022
pegasus datasus profile --manifest <manifest_path>
pegasus compile --intent config/intents/alagoas_smoke.json
pegasus validate-run --run data/runs/<run_id>
```

Scripts MUST call package services. Scripts MUST NOT own production logic.

# 15. Doctor Command

`pegasus doctor` MUST check:

- Python version.
- Package imports.
- DuckDB availability.
- Polars availability.
- PyArrow availability.
- PyTorch availability.
- CUDA availability and device name.
- Rscript availability.
- microdatasus availability.
- read.dbc availability.
- Write permissions in data directories.
- Registry schema validity.
- SIDRA network connectivity.

`pegasus doctor` MUST NOT run heavy ingestion.

# 16. Development by Topological Dependency

No chronological estimates are allowed. Progress is measured by dependency slices.

## 16.1 Slice 0 — Monolithic Scaffold

Implement:

- Repository tree.
- `src/pegasus/` namespace.
- CLI skeleton.
- Config loader.
- Path manager.
- Logging system.
- Hashing utilities.
- Manifest models.
- Exception hierarchy.
- Registry loader.
- Registry validators.
- Pydantic models.
- Data lake creation.
- Output bundle skeleton.
- Output validator.
- Dashboard read-only placeholder.
- Typed blocked states for ST-DFM, population tensor, race bridge, and PIRS.

Acceptance:

- `pegasus validate-config` passes.
- `pegasus registries validate` passes.
- Empty 17-key bundle scaffold validates.
- No domain ingestion code exists outside the scaffold.

## 16.2 Slice 1 — SIM-DO Ingestion and Basic EFG

Implement:

- microdatasus subprocess boundary.
- DATASUS cache.
- DATASUS profiler.
- Raw/processed schema comparison.
- SIM-DO normalization.
- SIM structural age decoder.
- SIM six-state ICD parser.
- Municipality crosswalk.
- FieldNode and Lineage hashing.
- Carrier/unit/provenance/quality registries.
- Delta predicate evaluator.
- Declaration gate.
- Basic EFG serialization.
- Q tensor core metrics.
- Crude mortality.
- One ICD chapter mortality.
- One ICD block mortality.
- Administrative race counts.
- Failed branch for direct race-specific SIM numerator over IBGE denominator.

Acceptance:

- SIM raw and processed artifacts are saved.
- SIM normalized events are produced.
- ICD parser emits six states.
- Direct race-specific mortality fails without `Bridge_R`.
- A valid 17-key run bundle is emitted.

## 16.3 Slice 2 — SIDRA Metadata, Fact Store, and Denominator Anchor

Implement:

- SIDRA metadata fetch.
- SIDRA normalized metadata tables.
- SIDRA chunk planner with 49,900-cell ceiling.
- SIDRA extractor.
- SIDRA flat-response normalizer.
- SIDRA long-form fact store.
- Official population denominator anchor.
- SIDRA category-map scaffold.
- SIDRA classification projection scaffold.
- Bounded pushforward scaffold.

Acceptance:

- SIDRA request over 49,900 cells is split or aborted.
- SIDRA facts preserve `value_raw`, `value_numeric`, and `value_status`.
- Small bounded tables do not require view definitions.
- Large redundant tables use view definitions.
- Population denominator contract validates.

## 16.4 Slice 3 — SINASC and Maternal-Child Fields

Implement:

- SINASC ingestion and normalization.
- Newborn and maternal race axes.
- Birth weight decoder.
- Count-with-leading-zero decoders.
- Gestational age state handling.
- APGAR state handling.
- Anomaly ICD parser.
- Clinical event definitions for infant, neonatal, postneonatal, low birth weight, prematurity, cesarean, and congenital anomaly fields.
- Birth and maternal-child EFG fields.

Acceptance:

- Crude birth rate validates.
- Infant mortality validates.
- Neonatal mortality validates.
- Low birth weight validates.
- Prematurity validates.
- Anomaly prevalence validates.
- Race-axis declarations remain separate.

## 16.5 Slice 4 — Downstream Race Bridge

Implement:

- Race bridge prior validator.
- `Bridge_R` fixedC dynamic-weight mode.
- Raw administrative race output.
- Missing race observer fields.
- Self-aligned posterior output when priors validate.
- Sensitivity interval output.
- Race-adjusted Q-state fields.
- Race bridge metadata in output bundle.

Acceptance:

- Empty or invalid C matrix blocks posterior output.
- Missing race is preserved as observer field.
- Raw administrative race counts are not overwritten.
- Race-bridged fields carry bridge warnings and sensitivity metadata.

## 16.6 Slice 5 — CNES-ST and SIH-RD

Implement:

- CNES ingestion and normalization.
- Boolean sentinel decoder.
- CNPJ filter.
- CNES capacity-vector registry.
- Capacity density fields by vector index.
- Invalid flag share observer fields.
- Zero CNPJ share observer fields.
- SIH ingestion and normalization.
- SIH age decoder.
- SIH diagnostic topology.
- SIH cost-component separation.
- Facility-linkage affordance registry.
- Hospitalization fields.
- Inpatient fatality.
- LOS fields.
- Cost fields by component.

Acceptance:

- Generic beds fail without capacity index.
- Boolean outliers become invalid states.
- All-zero CNPJ is nullified.
- Unsanitized facility-flow linkage fails.
- Generic SIH cost fails when component-specific estimand is required.
- `VAL_SH`, `VAL_SP`, `VAL_UTI`, and `VAL_TOT` remain separate.

## 16.7 Slice 6 — Population Tensor

Implement:

- Population tensor schema.
- Independent denominator mode.
- Solver registry.
- Sparse/block solver interface.
- ADMM scaffold.
- Population tensor diagnostics.
- SIM-informed mode warning propagation.
- Dense national solver abort gate.

Acceptance:

- Independent denominator mode works.
- SIM-informed mode emits feedback-risk warning.
- Dense national projected-gradient path aborts above threshold.
- Population tensor output metadata appears in the run bundle.

## 16.8 Slice 7 — SIDRA Stitching, Projection, and ST-DFM

Implement:

- SIDRA longitudinal stitching.
- Classification projection matrices.
- High-dimensional bounding.
- ST-DFM input/output schema.
- ST-DFM blocked state.
- ST-DFM certification table.
- ST-DFM PyTorch solver when calibration is available.

Acceptance:

- Table identity is not treated as concept identity.
- GDP and labor/CEMPRE-style stitched fields preserve segment provenance.
- Fractional classification projection emits warnings.
- ST-DFM solver pending emits `blocked_solver_pending`, not direct fallback.
- ST-DFM verified promotion requires certification.

## 16.9 Slice 8 — PIRS Parametric Models

Implement:

- Field selection by $U_{\mathcal{I}}(v)$ and $Q(v)$.
- Design matrix builder.
- Count models with exposure offsets.
- Binomial/proportion models.
- SIH economic model routing.
- Residual extraction.
- Cross-fitting.

Acceptance:

- Quarantined fields cannot become outcomes.
- Zero-variance fields cannot enter design matrices.
- Exposure offsets come from carrier registry.
- Residual fields inherit `model_derived` provenance.

## 16.10 Slice 9 — HSIC Residual Scanner

Implement:

- Exact HSIC for small validation support.
- Nyström HSIC.
- RFF HSIC.
- PyTorch CUDA backend.
- Null strategies.
- FDR correction.
- Approximation diagnostics.

Acceptance:

- Standard/deep HSIC uses cross-fitted residuals.
- Monthly nulls preserve season.
- Approximation diagnostics are emitted.
- CUDA-required region runs abort if CUDA is unavailable.

## 16.11 Slice 10 — Read-Only Dashboard

Implement:

- Read-only run directory loader.
- Read-only variable dictionary viewer.
- Read-only table/map/hypothesis viewer.
- No computation triggers.

Acceptance:

- Dashboard cannot fetch DATASUS.
- Dashboard cannot fetch SIDRA.
- Dashboard cannot run SHE.
- Dashboard cannot run EFG.
- Dashboard cannot run PIRS.
- Dashboard cannot run ST-DFM.
- Dashboard cannot run `Bridge_R`.
- Dashboard cannot run population solver.

# 17. Hard Abort Conditions

The compiler MUST abort when any of the following occur:

- Invalid user intent.
- Missing required registry.
- Registry hash mismatch.
- Unknown source system.
- Source ingestion failure without cached valid artifact.
- Missing required source field for a mandatory core variable.
- Required composite decoder missing for a structural field.
- SIDRA request exceeding 49,900 cells after planning.
- Unsupported SIDRA table, variable, period, category, or locality.
- Dense national population optimization requested above scale threshold.
- Direct administrative race numerator divided by IBGE self-declared denominator.
- Race bridge requested without valid emission-prior object.
- Direct allocation of an intensive geneallocated field.
- ST-DFM proportion field promoted to verified without denominator or survey uncertainty.
- Standard/deep residual HSIC using in-sample residuals.
- Monthly seasonal null using arbitrary within-season shuffle.
- SIDRA Civil Registry used to replace SIM/SINASC event streams.
- Unsanitized all-zero CNPJ used in facility-flow linkage.
- Generic CNES `Beds` carrier requested without capacity-vector index.
- Generic SIH cost field requested when the estimand requires `VAL_SH`, `VAL_SP`, or `VAL_UTI`.
- High-dimensional SIDRA field exposed to EFG without legal bounded pushforward.
- Diagnostic topology erased without explicit projection.
- Output bundle fails the 17-key schema contract.
- Dashboard attempts to trigger computation.

# 18. Downgrade and Quarantine Conditions

The compiler MUST downgrade rather than abort when an object is legal but unstable.

Downgrade triggers:

- Sparse support.
- High missingness.
- Fragile denominator.
- SIM-informed denominator feedback risk.
- Uncertified latent reconstruction.
- High ST-DFM factor instability.
- System measurement divergence.
- Race bridge high coefficient of variation.
- Wide race sensitivity interval.
- Missing local calibration for race bridge.
- Facility linkage uncertainty.
- Nullified corporate identifiers.
- Invalid CNES boolean flag shares.
- Geneallocated support used for inference.
- SIDRA stitching without overlap but with declared methodological continuity.
- Fractional classification projection.
- Insufficient HSIC blocks.

Downgraded fields MUST remain auditable. They MUST NOT silently disappear.

# 19. Acceptance Test Suite

## 19.1 Non-Simplification Tests

The suite MUST verify:

- ST-DFM registry exists.
- ST-DFM invoked before solver emits `blocked_solver_pending`.
- `Bridge_R` registry exists.
- Declaration gate blocks direct SIM administrative race numerator over IBGE self-declared denominator.
- SIDRA facts are long-form.
- Missingness observer fields are emitted.
- `Q_tensor` controls dashboard safety.
- Output bundle has exactly 17 keys.
- No legacy `problem1`, `problem2`, or `problem3` package exists.

## 19.2 DATASUS Tests

The suite MUST verify:

- Raw and processed artifacts are both saved.
- Raw/processed schema comparison is emitted.
- Generic date parsing is not applied to all columns.
- ICD parser distinguishes all six states.
- SIM pregnancy and puerperium fields are preserved when present.
- Facility ID missingness does not drop records.
- CNPJ zero-fillers are filtered before CNPJ joins.
- CNES boolean outliers are not clamped to true.
- SIH cost components remain separate.

## 19.3 SIDRA Tests

The suite MUST verify:

- Metadata is rebuilt from API or validated fixture.
- Cell estimator includes variables, periods, localities, and classification categories.
- Requests above 49,900 cells are split or aborted.
- Flat header rows are dropped.
- `value_raw`, `value_numeric`, and `value_status` are preserved.
- Total categories are not mixed into modeling categories unless policy allows.
- Small bounded table can be fetched without SIDRA view definition.
- Large redundant table uses SIDRA view definition.
- High-dimensional facts are bounded before EFG exposure.

## 19.4 Race Bridge Tests

The suite MUST verify:

- Empty C matrix blocks posterior output.
- Rows of C sum to 1.
- Missing category is preserved as observer field.
- Raw administrative counts are preserved.
- Posterior field carries bridge provenance.
- Sensitivity interval width contributes to Q-state downgrade.

## 19.5 EFG Tests

The suite MUST verify:

- Carrier legality rejects generic beds.
- Unit legality rejects pooled SIH cost components.
- Aggregation legality rejects direct rate pushforward.
- Provenance legality blocks `illegal_excluded`.
- Quality legality enforces state permissions.
- Diagnostic topology distinguishes SIM underlying cause, terminal chain, associated conditions, SIH principal diagnosis, SIH secondary diagnoses, and SINASC anomalies.
- Zero-variance fields cannot enter PIRS.

## 19.6 PIRS Tests

The suite MUST verify:

- Design matrices exclude zero-variance fields.
- Exposure offsets come from carrier registry.
- Standard/deep residuals are cross-fitted.
- HSIC nulls preserve season for monthly panels.
- CUDA-required HSIC aborts if CUDA is unavailable.
- Approximation diagnostics are emitted.

## 19.7 Output Tests

The suite MUST verify:

- All 17 keys exist.
- Cross-references are valid.
- `Warnings` records link to field or run scope.
- `VariableDictionary` covers all fields.
- `FailedBranches` records illegal operations.
- `QuarantinedFields` records state-limited fields.
- `ForcedFields` records user-forced fields.
- `ReproducibilityManifest` records source hashes, registry hashes, package versions, R versions, and random seeds.

# 20. Formula-to-Code Appendix Requirement

Before implementing any numerical module, the mathematical formula MUST be rewritten into typed pseudocode with:

- Inputs.
- Outputs.
- Shape notation.
- Dtype.
- Support assumptions.
- Missingness handling.
- Epsilon stabilization.
- Warnings emitted.
- State downgrade rules.
- Failure modes.

This requirement applies to:

- Population tensor loss.
- Embedded `Bridge_R` inside population solver.
- ST-DFM objective.
- Q tensor metrics.
- PIRS likelihoods.
- Residual extraction.
- Exact HSIC.
- Nyström HSIC.
- RFF HSIC.
- Null permutation regimes.

No visually corrupted or ambiguous formula block from the MSD may be copied directly into production without this disambiguation.

# 21. Dashboard Contract

The dashboard is not part of the first implementation milestone. The first valid dashboard is strictly read-only over completed run directories.

Allowed before dashboard implementation:

- Package namespace placeholder.
- Read-only contract.
- Documentation.

Forbidden:

- Interactive computation.
- Data fetch.
- SIDRA extraction.
- DATASUS ingestion.
- SHE execution.
- EFG execution.
- PIRS execution.
- HSIC execution.
- ST-DFM execution.
- Population solver execution.
- Race bridge execution.

# 22. Final Implementation Rule

PegaSUS MUST be built as a compiler whose source artifacts, field objects, graph edges, warnings, state tensor, and output bundle make every transformation auditable.

The codebase may be staged. Solvers may be mutable inside declared contracts. Domain coverage may expand by vertical dependency slice. But the architecture MUST NOT simplify itself into ETL.

The first accepted implementation is the one that can run:

```text
pegasus compile --intent config/intents/alagoas_smoke.json
```

and emit a valid 17-key bundle in which:

- SIM-DO raw and processed artifacts are both cached.
- SIM events are normalized.
- ICD parser emits six states.
- Municipality crosswalk is applied.
- Typed crude mortality is generated.
- At least one ICD-restricted mortality field is generated.
- Administrative race counts are generated.
- Direct administrative-race numerator over IBGE self-declared denominator fails without `Bridge_R`.
- `Q(v)` is emitted and controls state.
- `E_DAG` lineage is emitted.
- `VariableDictionary` explains every field.
- `Warnings` contains meaningful warnings.
- `FailedBranches` records rejected illegal operations.
- Output validation passes.
- ST-DFM exists as scaffold or solver.
- Population tensor exists as scaffold or solver.
- `Bridge_R` exists as scaffold or executor.
- PIRS exists as scaffold or executor.
- No dashboard computation is involved.
- No WSL2 or RAPIDS dependency exists.
- PyTorch CUDA is verified only when CUDA-required modules are invoked.


