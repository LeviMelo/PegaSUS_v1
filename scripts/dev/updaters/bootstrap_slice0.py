from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap
from pathlib import Path
from datetime import datetime, timezone
import yaml

ROOT = Path.cwd()

REGISTRY_FILES = [
    "datasus/source_fields.yaml",
    "datasus/composite_decoders.yaml",
    "carrier_registry.yaml",
    "unit_registry.yaml",
    "aggregation_registry.yaml",
    "provenance_registry.yaml",
    "ontology/quality_permissions.yaml",
    "demographic/race_axis_registry.yaml",
    "demographic/race_bridge_priors.yaml",
    "health/icd_catalog.yaml",
    "health/icd_quality_groups.yaml",
    "health/diagnostic_topology.yaml",
    "health/clinical_event_definitions.yaml",
    "health/cnes_capacity_registry.yaml",
    "health/sih_cost_registry.yaml",
    "sidra_views.yaml",
    "sidra_category_maps.yaml",
    "sidra_stitching.yaml",
    "sidra_regime_registry.yaml",
    "spatial/municipality_crosswalk_sources.yaml",
    "fields/join_affordances.yaml",
    "fields/bridge_grammars.yaml",
    "inference/stdfm_registry.yaml",
    "demographic/population_solver_registry.yaml",
    "inference/model_registry.yaml",
    "inference/residual_registry.yaml",
    "inference/hsic_registry.yaml",
    "ontology/null_registry.yaml",
    "output_schema.yaml",
]


PACKAGE_FILES = {
    "core": [
        "__init__.py", "config.py", "paths.py", "logging.py", "hashing.py",
        "manifests.py", "exceptions.py", "constants.py", "enums.py",
        "schemas.py", "types.py", "validation.py", "time.py", "ids.py",
    ],
    "registries": [
        "__init__.py", "loader.py", "validators.py", "manifest.py",
        "source_fields.py", "composite_decoders.py", "carrier.py", "unit.py",
        "aggregation.py", "provenance.py", "quality.py", "race_axis.py",
        "race_bridge.py", "icd.py", "diagnostic_topology.py", "events.py",
        "cnes_capacity.py", "sih_cost.py", "sidra.py", "bridge.py",
        "population.py", "models.py", "residuals.py", "hsic.py", "nulls.py",
        "output.py",
    ],
    "storage": [
        "__init__.py", "duckdb.py", "polars.py", "arrow.py", "parquet.py",
        "geopaquet.py", "cache.py", "dataset.py", "materialization.py", "locks.py",
    ],
    "datasus": [
        "__init__.py", "client_microdatasus.py", "subprocess.py", "cache.py",
        "manifests.py", "profile.py", "schema_compare.py", "normalize.py",
        "decoders.py", "icd_parser.py", "joins.py", "quality.py", "raw_processed.py",
    ],
    "sidra": [
        "__init__.py", "api.py", "cache.py", "metadata.py", "registry.py",
        "plan.py", "extract.py", "normalize.py", "facts.py", "category_maps.py",
        "stitching.py", "projection.py", "regime.py", "quality.py",
    ],
    "geo": [
        "__init__.py", "municipality_crosswalk.py", "support.py", "amc.py",
        "geneallocation.py", "adjacency.py", "geodata.py", "spatial_index.py",
    ],
    "she": [
        "__init__.py", "substrate.py", "source_registry.py", "composite_decode.py",
        "clinical_events.py", "cnes_capacity.py", "sih_costs.py", "civil_health.py",
        "zero_variance.py", "high_dimensional.py",
    ],
    "efg": [
        "__init__.py", "node.py", "lineage.py", "field_tensor.py", "operators.py",
        "align.py", "legality.py", "declaration.py", "dag.py", "q_tensor.py",
        "materialize.py", "equivalence.py", "core_seed.py", "bridges.py",
        "race_bridge.py", "warning.py", "failed_branch.py",
    ],
    "pirs": [
        "__init__.py", "field_selection.py", "design.py", "models.py", "families.py",
        "residuals.py", "crossfit.py", "hsic.py", "nystrom.py", "rff.py",
        "nulls.py", "fdr.py", "diagnostics.py",
    ],
    "compute": [
        "__init__.py", "devices.py", "torch_backend.py", "memory.py",
        "kernels.py", "random.py",
    ],
    "output": [
        "__init__.py", "bundle.py", "schemas.py", "variable_dictionary.py",
        "warnings.py", "tables.py", "maps.py", "validate.py",
        "reproducibility.py", "serializers.py",
    ],
    "dashboard": ["__init__.py", "read_only.py", "contracts.py"],
    "workflows": [
        "__init__.py", "compile.py", "ingest_datasus.py", "ingest_sidra.py",
        "build_substrate.py", "build_efg.py", "run_pirs.py", "serialize.py",
        "validate_run.py",
    ],
}


def write(path: str | Path, content: str, overwrite: bool = True) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def touch(path: str | Path) -> None:
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def yaml_registry(name: str, entries: list[dict] | None = None) -> str:

    entries = entries or []
    payload = {
        "schema_version": "1.0",
        "registry_version": "v1.0",
        "created_at": "2026-06-06",
        "updated_at": "2026-06-06",
        "provenance": f"Slice 0 scaffold registry for {name}; not a calibrated analytic registry.",
        "entries": entries,
    }
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


def make_root_files() -> None:
    write("pyproject.toml", """
    [build-system]
    requires = ["setuptools>=69", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "pegasus"
    version = "0.1.0"
    description = "PegaSUS epidemiological compiler monolithic scaffold"
    readme = "README.md"
    requires-python = ">=3.11"
    dependencies = [
      "pydantic>=2.7",
      "typer>=0.12",
      "PyYAML>=6.0",
      "rich>=13.7",
      "duckdb>=1.0",
      "polars>=1.0",
      "pyarrow>=16.0"
    ]

    [project.optional-dependencies]
    dev = ["pytest>=8.0", "ruff>=0.5", "mypy>=1.10"]
    gpu = ["torch>=2.4"]

    [project.scripts]
    pegasus = "pegasus.cli:app"

    [tool.setuptools.packages.find]
    where = ["src"]

    [tool.ruff]
    line-length = 100
    target-version = "py311"

    [tool.pytest.ini_options]
    testpaths = ["tests"]
    """)

    write("README.md", """
    # PegaSUS

    PegaSUS is a measurement-process-aware epidemiological compiler.

    This repository starts with Slice 0: monolithic scaffold only. Domain ingestion, rate
    computation, model fitting, HSIC scanning, and visualization are intentionally blocked
    until the scaffold validates.
    """)

    write("LICENSE", """
    Copyright (c) 2026.

    Local personal research scaffold. Replace this file with the final license before public release.
    """)

    write(".env.example", """
    PEGASUS_ENV=local
    PEGASUS_DATA_ROOT=data
    """)

    write("environment.yml", """
    name: pegasus
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python=3.11
      - pip
      - duckdb
      - polars
      - pyarrow
      - pydantic
      - typer
      - pyyaml
      - rich
      - pytest
      - pip:
          - -e .[dev]
    """)

    touch("uv.lock")
    touch("renv.lock")

    write(".gitignore", """
    # Python
    __pycache__/
    *.py[cod]
    *.pyo
    *.pyd
    .pytest_cache/
    .ruff_cache/
    .mypy_cache/
    .coverage
    htmlcov/
    build/
    dist/
    *.egg-info/

    # Virtual environments
    .venv/
    venv/
    env/
    .env

    # VS Code / OS
    .vscode/*.log
    .DS_Store
    Thumbs.db

    # Jupyter
    .ipynb_checkpoints/

    # PegaSUS data lake: keep directory placeholders, ignore payloads
    data/raw/**
    data/processed/**
    data/metadata/**
    data/cache/**
    data/manifests/**
    data/intermediate/**
    data/runs/**
    data/diagnostics/**

    !data/raw/.gitkeep
    !data/processed/.gitkeep
    !data/metadata/.gitkeep
    !data/cache/.gitkeep
    !data/manifests/.gitkeep
    !data/intermediate/.gitkeep
    !data/runs/.gitkeep
    !data/diagnostics/.gitkeep

    # Local logs and temp files
    *.log
    *.tmp
    *.bak
    """)


def make_configs() -> None:
    write("config/project.yaml", """
    project:
      name: "PegaSUS"
      package: "pegasus"
      version: "0.1.0"
      registry_version: "v1.0"
      timezone: "America/Maceio"
      default_locale: "pt-BR"
    """)

    write("config/paths.yaml", """
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
    """)

    write("config/compute.yaml", """
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
    """)

    write("config/datasus.yaml", """
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
    """)

    write("config/sidra.yaml", """
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
    """)

    write("config/output.yaml", """
    output:
      exact_first_class_keys: 17
      require_global_telemetry: true
      write_partial_telemetry_on_abort: true
    """)

    intents = {
        "alagoas_smoke.json": {
            "geography": {"level": "UF", "codes": ["27"], "uf": ["AL"]},
            "time": {"start_year": 2022, "end_year": 2022},
            "health_seeds": [],
            "mandatory_fields": [],
            "system_weights": {},
            "context_policy": [],
            "budget": "fast",
            "geo_mode": "native",
            "force_selectors": [],
            "exclude_systems": [],
            "execution_scale": "smoke",
            "race_tensor_mode": "decoupled",
            "population_mode": "official_sidra_anchor",
        },
        "alagoas_sim_only.json": {
            "geography": {"level": "UF", "codes": ["27"], "uf": ["AL"]},
            "time": {"start_year": 2022, "end_year": 2022},
            "health_seeds": [],
            "mandatory_fields": ["V_M01"],
            "system_weights": {"SIM-DO": 1.0},
            "context_policy": [],
            "budget": "fast",
            "geo_mode": "native",
            "force_selectors": [],
            "exclude_systems": ["SIH-RD", "SINASC", "CNES-ST", "SIDRA"],
            "execution_scale": "smoke",
            "race_tensor_mode": "decoupled",
            "population_mode": "blocked_missing",
        },
        "alagoas_maternal_child.json": {
            "geography": {"level": "UF", "codes": ["27"], "uf": ["AL"]},
            "time": {"start_year": 2022, "end_year": 2022},
            "health_seeds": [],
            "mandatory_fields": ["V_B01", "V_C01", "V_C02"],
            "system_weights": {"SIM-DO": 1.0, "SINASC": 1.0, "SIDRA": 1.0},
            "context_policy": ["maternal_child"],
            "budget": "standard",
            "geo_mode": "native",
            "force_selectors": [],
            "exclude_systems": [],
            "execution_scale": "state",
            "race_tensor_mode": "decoupled",
            "population_mode": "official_sidra_anchor",
        },
        "northeast_region.json": {
            "geography": {"level": "region", "codes": ["2"], "uf": []},
            "time": {"start_year": 2022, "end_year": 2022},
            "health_seeds": [],
            "mandatory_fields": [],
            "system_weights": {},
            "context_policy": [],
            "budget": "standard",
            "geo_mode": "AMC",
            "force_selectors": [],
            "exclude_systems": [],
            "execution_scale": "region",
            "race_tensor_mode": "downstream_bridge",
            "population_mode": "official_sidra_anchor",
        },
        "national_blocked.json": {
            "geography": {"level": "country", "codes": ["BR"], "uf": []},
            "time": {"start_year": 2022, "end_year": 2022},
            "health_seeds": [],
            "mandatory_fields": [],
            "system_weights": {},
            "context_policy": [],
            "budget": "deep",
            "geo_mode": "AMC",
            "force_selectors": [],
            "exclude_systems": [],
            "execution_scale": "national_blocked",
            "race_tensor_mode": "embedded_posteriorC",
            "population_mode": "blocked_missing",
        },
    }

    for name, payload in intents.items():
        write(f"config/intents/{name}", json.dumps(payload, indent=2, ensure_ascii=False))


def make_registries() -> None:
    reg_dir = ROOT / "config" / "registries"
    reg_dir.mkdir(parents=True, exist_ok=True)

    special_entries = {
        "datasus/composite_decoders.yaml": [
            {"id": "Decode_SIM_IDADE", "status": "stable", "description": "SIM structural age decoder.", "warnings": []},
            {"id": "Decode_SIH_AGE", "status": "stable", "description": "SIH age decoder using COD_IDADE.", "warnings": []},
            {"id": "Decode_PESO", "status": "stable", "description": "Physical scalar decoder for birth weight.", "warnings": []},
            {"id": "Decode_count2", "status": "stable", "description": "Two-character count decoder.", "warnings": []},
            {"id": "Clamp_bool", "status": "stable", "description": "Sentinel-safe boolean decoder.", "warnings": []},
            {"id": "Filter_CNPJ", "status": "stable", "description": "CNPJ sanitizer and all-zero nullifier.", "warnings": []},
            {"id": "Decode_cat", "status": "stable", "description": "Categorical socioeconomic decoder.", "warnings": []},
        ],
        "ontology/quality_permissions.yaml": [
            {"id": "verified", "status": "stable", "description": "Full analytic permissions.", "warnings": []},
            {"id": "fragile", "status": "stable", "description": "Analytic with warning.", "warnings": []},
            {"id": "forced_fragile", "status": "stable", "description": "Forced unstable field.", "warnings": []},
            {"id": "quarantined_descriptive", "status": "stable", "description": "Descriptive only.", "warnings": []},
            {"id": "quarantined_nochildren", "status": "stable", "description": "No downstream children.", "warnings": []},
            {"id": "illegal_excluded", "status": "stable", "description": "Excluded from analytic graph.", "warnings": []},
        ],
        "demographic/race_axis_registry.yaml": [
            {"id": "IBGE.self_declared", "status": "stable", "description": "IBGE self-declared race axis.", "warnings": []},
            {"id": "SIM-DO.administrative_death_declaration", "status": "stable", "description": "SIM death declaration race axis.", "warnings": []},
            {"id": "SIH-RD.billing_record", "status": "stable", "description": "SIH billing race axis.", "warnings": []},
            {"id": "SINASC.administrative_mixed", "status": "stable", "description": "SINASC administrative mixed race axis.", "warnings": []},
        ],
        "output_schema.yaml": [
            {"id": "immutable_17_key_bundle", "status": "stable", "description": "Exact run bundle key contract.", "warnings": []}
        ],
    }

    for file_name in REGISTRY_FILES:
        entries = special_entries.get(
            file_name,
            [{"id": file_name.replace(".yaml", ""), "status": "deferred", "description": "Slice 0 placeholder registry.", "warnings": ["scaffold_only"]}],
        )
        write(reg_dir / file_name, yaml_registry(file_name, entries))

    write("config/registries/sidra_table_seed.jsonl", """
    {"table_id":"9606","purpose":"population_denominator_anchor","status":"stable","notes":"Slice 0 metadata seed only; no fetch performed."}
    {"table_id":"5938","purpose":"gdp_context_candidate","status":"deferred","notes":"Slice 0 metadata seed only; no fetch performed."}
    """)

    manifest_entries = {}
    for file_name in REGISTRY_FILES:
        path = reg_dir / file_name
        manifest_entries[file_name.replace(".yaml", "")] = {
            "path": file_name,
            "schema_version": "1.0",
            "sha256": sha256_file(path),
        }

    manifest_entries["sidra_table_seed"] = {
        "path": "sidra_table_seed.jsonl",
        "schema_version": "1.0",
        "sha256": sha256_file(reg_dir / "sidra_table_seed.jsonl"),
    }

    manifest_yaml = {
        "registry_set": {
            "name": "PegaSUS_core",
            "version": "v1.0",
            "created_at": "2026-06-06",
            "owner": "local",
            "git_commit": "uninitialized",
            "dirty_allowed": False,
        },
        "registries": manifest_entries,
    }

    try:
        import yaml
        content = yaml.safe_dump(manifest_yaml, sort_keys=False, allow_unicode=True)
    except Exception:
        content = json.dumps(manifest_yaml, indent=2, ensure_ascii=False)

    write("config/registries/registry_manifest.yaml", content)


def make_dirs() -> None:
    dirs = [
        "config/registries",
        "config/intents",
        "src/pegasus",
        "scripts",
        "tests/unit",
        "tests/integration",
        "tests/smoke",
        "tests/golden",
        "tests/fixtures",
        "notebooks/validation",
        "notebooks/exploration",
        "docs",
        "data/raw/datasus/SIM-DO",
        "data/raw/datasus/SIH-RD",
        "data/raw/datasus/SINASC",
        "data/raw/datasus/CNES-ST",
        "data/raw/sidra/chunks",
        "data/raw/geo",
        "data/raw/external",
        "data/processed/datasus/SIM-DO",
        "data/processed/datasus/SIH-RD",
        "data/processed/datasus/SINASC",
        "data/processed/datasus/CNES-ST",
        "data/processed/sidra/facts",
        "data/processed/geo",
        "data/metadata/sidra/raw",
        "data/metadata/sidra/normalized",
        "data/metadata/datasus/profiles",
        "data/metadata/datasus/schema_compare",
        "data/metadata/datasus/variable_catalog",
        "data/metadata/registries",
        "data/metadata/geo",
        "data/cache/sidra/http",
        "data/cache/sidra/values",
        "data/cache/datasus/microdatasus",
        "data/manifests/datasus",
        "data/manifests/sidra",
        "data/manifests/runs",
        "data/intermediate/she",
        "data/intermediate/efg",
        "data/intermediate/pirs",
        "data/runs",
        "data/diagnostics",
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        if d.startswith("data/"):
            touch(Path(d) / ".gitkeep")

    for doc in [
        "architecture.md", "data_lake.md", "registries.md", "datasus_subsystem.md",
        "sidra_subsystem.md", "she.md", "efg.md", "pirs.md", "race_bridge.md",
        "output_bundle.md", "compute_backend.md", "development_slices.md", "validation.md",
    ]:
        write(f"docs/{doc}", f"# {doc.replace('_', ' ').replace('.md', '').title()}\n\nSlice 0 scaffold placeholder.\n")


def default_stub(module_path: str) -> str:
    return f'''
    """
    Slice 0 scaffold module: {module_path}

    This module intentionally contains no domain logic. Future implementation slices
    must replace blocked stubs through typed contracts.
    """

    from pegasus.core.exceptions import BlockedModuleError


    def blocked(*, module: str = "{module_path}", reason: str = "slice0_scaffold_only") -> None:
        raise BlockedModuleError(module=module, reason=reason)
    '''


def make_package() -> None:
    write("src/pegasus/__init__.py", '__version__ = "0.1.0"\n')
    write("src/pegasus/__main__.py", """
    from pegasus.cli import app

    if __name__ == "__main__":
        app()
    """)

    for package, files in PACKAGE_FILES.items():
        pkg_dir = ROOT / "src" / "pegasus" / package
        pkg_dir.mkdir(parents=True, exist_ok=True)
        for file_name in files:
            rel = f"{package}/{file_name}"
            if file_name == "__init__.py":
                write(pkg_dir / file_name, f'"""PegaSUS {package} package."""\n')
            else:
                write(pkg_dir / file_name, default_stub(rel), overwrite=False)

    for subdir, files in {
        "she/population": [
            "__init__.py", "schema.py", "loss.py", "solvers.py", "sparse_admm.py",
            "block_coordinate.py", "state_space.py", "race_modes.py",
            "torch_kernels.py", "diagnostics.py",
        ],
        "she/stdfm": [
            "__init__.py", "schema.py", "regime.py", "objective.py",
            "torch_solver.py", "certification.py", "blocked.py",
        ],
        "datasus/r_scripts": ["fetch_process_microdatasus.R"],
    }.items():
        d = ROOT / "src" / "pegasus" / subdir
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            if f.endswith(".R"):
                write(d / f, """
                # Slice 0 placeholder only.
                # The microdatasus R bridge is implemented in Slice 1.
                quit(status = 41)
                """)
            elif f == "__init__.py":
                write(d / f, f'"""PegaSUS {subdir} package."""\n')
            else:
                write(d / f, default_stub(f"{subdir}/{f}"), overwrite=False)


def make_core_modules() -> None:
    write("src/pegasus/core/exceptions.py", """
    class PegasusError(Exception):
        \"\"\"Base PegaSUS exception.\"\"\"


    class ConfigError(PegasusError):
        \"\"\"Configuration failure.\"\"\"


    class RegistryValidationError(PegasusError):
        \"\"\"Registry validation failure.\"\"\"


    class OutputValidationError(PegasusError):
        \"\"\"Output bundle validation failure.\"\"\"


    class BlockedModuleError(PegasusError):
        \"\"\"Raised when an architecturally visible but inactive module is invoked.\"\"\"

        def __init__(self, *, module: str, reason: str = "blocked_state"):
            self.module = module
            self.reason = reason
            super().__init__(f"{module} is blocked: {reason}")
    """)

    write("src/pegasus/core/enums.py", """
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
    """)

    write("src/pegasus/core/hashing.py", """
    from __future__ import annotations

    import hashlib
    import json
    from pathlib import Path
    from typing import Any


    def stable_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


    def sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


    def sha256_file(path: str | Path) -> str:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()


    def content_hash(value: Any) -> str:
        return sha256_text(stable_json(value))
    """)

    write("src/pegasus/core/config.py", """
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    import yaml

    from pegasus.core.exceptions import ConfigError


    REQUIRED_CONFIGS = [
        "project.yaml",
        "paths.yaml",
        "compute.yaml",
        "datasus.yaml",
        "sidra.yaml",
        "output.yaml",
    ]


    def load_yaml(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Missing config file: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Config file is not a mapping: {path}")
        return data


    def validate_config_tree(root: str | Path = ".") -> list[str]:
        root = Path(root)
        errors: list[str] = []
        for name in REQUIRED_CONFIGS:
            path = root / "config" / name
            if not path.exists():
                errors.append(f"missing config/{name}")
                continue
            try:
                load_yaml(path)
            except Exception as exc:
                errors.append(f"invalid config/{name}: {exc}")
        return errors


    def load_all_configs(root: str | Path = ".") -> dict[str, dict[str, Any]]:
        root = Path(root)
        return {name: load_yaml(root / "config" / name) for name in REQUIRED_CONFIGS}
    """)

    write("src/pegasus/core/paths.py", """
    from __future__ import annotations

    from pathlib import Path

    from pegasus.core.config import load_yaml


    DATA_LAKE_DIRS = [
        "data/raw/datasus/SIM-DO",
        "data/raw/datasus/SIH-RD",
        "data/raw/datasus/SINASC",
        "data/raw/datasus/CNES-ST",
        "data/raw/sidra/chunks",
        "data/raw/geo",
        "data/raw/external",
        "data/processed/datasus/SIM-DO",
        "data/processed/datasus/SIH-RD",
        "data/processed/datasus/SINASC",
        "data/processed/datasus/CNES-ST",
        "data/processed/sidra/facts",
        "data/processed/geo",
        "data/metadata/sidra/raw",
        "data/metadata/sidra/normalized",
        "data/metadata/datasus/profiles",
        "data/metadata/datasus/schema_compare",
        "data/metadata/datasus/variable_catalog",
        "data/metadata/registries",
        "data/metadata/geo",
        "data/cache/sidra/http",
        "data/cache/sidra/values",
        "data/cache/datasus/microdatasus",
        "data/manifests/datasus",
        "data/manifests/sidra",
        "data/manifests/runs",
        "data/intermediate/she",
        "data/intermediate/efg",
        "data/intermediate/pirs",
        "data/runs",
        "data/diagnostics",
    ]


    def ensure_data_lake(root: str | Path = ".") -> list[Path]:
        root = Path(root)
        created = []
        for rel in DATA_LAKE_DIRS:
            path = root / rel
            path.mkdir(parents=True, exist_ok=True)
            keep = path / ".gitkeep"
            keep.touch(exist_ok=True)
            created.append(path)
        return created


    def configured_paths(root: str | Path = ".") -> dict:
        return load_yaml(Path(root) / "config" / "paths.yaml")["paths"]
    """)

    write("src/pegasus/core/schemas.py", """
    from __future__ import annotations

    from typing import Literal

    from pydantic import BaseModel, ConfigDict, Field

    from pegasus.core.enums import FieldState, MaterializationState


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

        execution_scale: Literal["smoke", "state", "region", "national_blocked"]

        race_tensor_mode: Literal[
            "decoupled",
            "downstream_bridge",
            "embedded_fixedC",
            "embedded_posteriorC",
            "embedded_sensitivity",
        ] = "decoupled"

        population_mode: Literal[
            "official_sidra_anchor",
            "imported_fixture",
            "independent_population_tensor",
            "sim_informed_population_tensor",
            "synthetic_test_fixture",
            "blocked_missing",
        ] = "official_sidra_anchor"


    class Lineage(BaseModel):
        model_config = ConfigDict(extra="forbid")

        parent_ids: list[str]
        operator_type: str
        operator_params: dict
        registry_versions: dict[str, str]
        source_manifest_hashes: list[str] = Field(default_factory=list)
        code_version: str


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
            "model_residual",
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
            "non_aggregable",
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
            "parse_error",
        ]

        unit: str | None
        request_hash: str
        metadata_hash: str
        fetched_at: str


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
            "output_validation",
        ]
        failed_terms: list[str]
        reason: str
        warnings: list[str]
        created_at: str


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


    class DenominatorContract(BaseModel):
        model_config = ConfigDict(extra="forbid")

        mode: Literal[
            "official_sidra_anchor",
            "imported_fixture",
            "independent_population_tensor",
            "sim_informed_population_tensor",
            "synthetic_test_fixture",
            "blocked_missing",
        ]
        source: str
        provenance: list[str]
        state: FieldState
        dashboard_safe: bool | Literal["warning"]
        allowed_for_rates: bool
        warnings: list[str]
    """)


def make_registry_modules() -> None:
    write("src/pegasus/registries/loader.py", """
    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    import yaml

    from pegasus.core.hashing import sha256_file


    @dataclass(frozen=True)
    class RegistryBundle:
        root: Path
        registries: dict[str, dict[str, Any]]
        hashes: dict[str, str]


    def load_registry_file(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Registry is not a mapping: {path}")
        return data


    def load_registries(root: str | Path = "config/registries") -> RegistryBundle:
        root = Path(root)
        registries: dict[str, dict[str, Any]] = {}
        hashes: dict[str, str] = {}
        for path in sorted(root.glob("*.yaml")):
            registries[path.stem] = load_registry_file(path)
            hashes[path.stem] = sha256_file(path)
        seed = root / "sidra_table_seed.jsonl"
        if seed.exists():
            hashes["sidra_table_seed"] = sha256_file(seed)
        return RegistryBundle(root=root, registries=registries, hashes=hashes)
    """)

    write("src/pegasus/registries/validators.py", """
    from __future__ import annotations

    import json
    from pathlib import Path

    import yaml


    REQUIRED_REGISTRY_KEYS = {
        "schema_version",
        "registry_version",
        "created_at",
        "updated_at",
        "provenance",
        "entries",
    }


    def validate_registry_file(path: str | Path) -> list[str]:
        path = Path(path)
        errors: list[str] = []

        if path.name == "registry_manifest.yaml":
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "registry_set" not in data:
                errors.append("registry_manifest missing registry_set")
            if "registries" not in data:
                errors.append("registry_manifest missing registries")
            return errors

        if path.suffix == ".jsonl":
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path.name}:{i}: invalid jsonl: {exc}")
            return errors

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            return [f"{path.name}: registry is not a mapping"]

        missing = REQUIRED_REGISTRY_KEYS - set(data)
        for key in sorted(missing):
            errors.append(f"{path.name}: missing {key}")

        if "entries" in data and not isinstance(data["entries"], list):
            errors.append(f"{path.name}: entries is not a list")

        for idx, entry in enumerate(data.get("entries", [])):
            if not isinstance(entry, dict):
                errors.append(f"{path.name}: entry {idx} is not a mapping")
                continue
            for key in ["id", "status", "description", "warnings"]:
                if key not in entry:
                    errors.append(f"{path.name}: entry {idx} missing {key}")

        return errors


    def validate_registry_tree(root: str | Path = "config/registries") -> list[str]:
        root = Path(root)
        errors: list[str] = []
        required = [
            "registry_manifest.yaml",
            "datasus/source_fields.yaml",
            "datasus/composite_decoders.yaml",
            "carrier_registry.yaml",
            "unit_registry.yaml",
            "aggregation_registry.yaml",
            "provenance_registry.yaml",
            "ontology/quality_permissions.yaml",
            "demographic/race_axis_registry.yaml",
            "demographic/race_bridge_priors.yaml",
            "health/icd_catalog.yaml",
            "health/icd_quality_groups.yaml",
            "health/diagnostic_topology.yaml",
            "health/clinical_event_definitions.yaml",
            "health/cnes_capacity_registry.yaml",
            "health/sih_cost_registry.yaml",
            "sidra_table_seed.jsonl",
            "sidra_views.yaml",
            "sidra_category_maps.yaml",
            "sidra_stitching.yaml",
            "sidra_regime_registry.yaml",
            "spatial/municipality_crosswalk_sources.yaml",
            "fields/join_affordances.yaml",
            "fields/bridge_grammars.yaml",
            "inference/stdfm_registry.yaml",
            "demographic/population_solver_registry.yaml",
            "inference/model_registry.yaml",
            "inference/residual_registry.yaml",
            "inference/hsic_registry.yaml",
            "ontology/null_registry.yaml",
            "output_schema.yaml",
        ]
        for name in required:
            path = root / name
            if not path.exists():
                errors.append(f"missing registry: {name}")
            else:
                errors.extend(validate_registry_file(path))
        return errors
    """)


def make_output_modules() -> None:
    write("src/pegasus/output/schemas.py", """
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
    """)

    write("src/pegasus/output/bundle.py", """
    from __future__ import annotations

    import json
    import platform
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    from pegasus.output.schemas import OUTPUT_BUNDLE_FILES


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _write_table(path: Path, rows: list[dict], schema: pa.Schema) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, path)


    def create_empty_output_bundle(run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "Tables").mkdir(exist_ok=True)
        (run_dir / "Maps").mkdir(exist_ok=True)

        field_id = "slice0_scaffold_field"

        v_schema = pa.schema([
            ("field_id", pa.string()),
            ("name", pa.string()),
            ("kind", pa.string()),
            ("carrier", pa.string()),
            ("unit", pa.string()),
            ("aggregation", pa.string()),
            ("role", pa.string()),
            ("source", pa.string()),
            ("support_json", pa.string()),
            ("axes_json", pa.string()),
            ("operator", pa.string()),
            ("provenance", pa.string()),
            ("state", pa.string()),
            ("dashboard_safe", pa.string()),
            ("warnings", pa.string()),
            ("lineage_hash", pa.string()),
            ("registry_hash", pa.string()),
            ("materialization_state", pa.string()),
            ("path", pa.string()),
        ])
        _write_table(run_dir / "V_fields.parquet", [{
            "field_id": field_id,
            "name": "Slice 0 Scaffold Field",
            "kind": "observer_proxy",
            "carrier": "none",
            "unit": "none",
            "aggregation": "non_aggregable",
            "role": json.dumps(["model_only"]),
            "source": json.dumps(["scaffold"]),
            "support_json": json.dumps({}),
            "axes_json": json.dumps({}),
            "operator": None,
            "provenance": json.dumps(["synthetic"]),
            "state": "quarantined_descriptive",
            "dashboard_safe": "False",
            "warnings": json.dumps(["slice0_scaffold_only"]),
            "lineage_hash": "slice0",
            "registry_hash": "uncomputed",
            "materialization_state": "metadata_only",
            "path": None,
        }], v_schema)

        edge_schema = pa.schema([
            ("edge_id", pa.string()),
            ("parent_field_id", pa.string()),
            ("child_field_id", pa.string()),
            ("operator", pa.string()),
            ("operator_params_json", pa.string()),
            ("registry_versions_json", pa.string()),
            ("created_at", pa.string()),
        ])
        _write_table(run_dir / "E_DAG.parquet", [], edge_schema)

        q_schema = pa.schema([
            ("field_id", pa.string()),
            ("n_events", pa.float64()),
            ("n_denom", pa.float64()),
            ("n_eff", pa.float64()),
            ("cov_S", pa.float64()),
            ("cov_T", pa.float64()),
            ("missingness", pa.float64()),
            ("zero_inflation", pa.float64()),
            ("denom_fragility", pa.float64()),
            ("cv", pa.float64()),
            ("moran_i", pa.float64()),
            ("temporal_roughness", pa.float64()),
            ("spatial_entropy", pa.float64()),
            ("provenance_risk", pa.float64()),
            ("race_axis_source", pa.string()),
            ("race_axis_target", pa.string()),
            ("missing_race_share", pa.float64()),
            ("emission_prior_strength", pa.float64()),
            ("race_bridge_cv", pa.float64()),
            ("sensitivity_width", pa.float64()),
            ("bridge_mode", pa.string()),
            ("state", pa.string()),
            ("dashboard_safe", pa.string()),
            ("warnings", pa.string()),
            ("computed_at", pa.string()),
            ("q_schema_version", pa.string()),
        ])
        _write_table(run_dir / "Q_tensor.parquet", [{
            "field_id": field_id,
            "n_events": 0.0,
            "n_denom": 0.0,
            "n_eff": 0.0,
            "cov_S": 0.0,
            "cov_T": 0.0,
            "missingness": 1.0,
            "zero_inflation": 0.0,
            "denom_fragility": 1.0,
            "cv": None,
            "moran_i": None,
            "temporal_roughness": None,
            "spatial_entropy": None,
            "provenance_risk": 1.0,
            "race_axis_source": None,
            "race_axis_target": None,
            "missing_race_share": None,
            "emission_prior_strength": None,
            "race_bridge_cv": None,
            "sensitivity_width": None,
            "bridge_mode": None,
            "state": "quarantined_descriptive",
            "dashboard_safe": "False",
            "warnings": json.dumps(["slice0_scaffold_only"]),
            "computed_at": _now(),
            "q_schema_version": "1.0",
        }], q_schema)

        warn_schema = pa.schema([
            ("warning_id", pa.string()),
            ("field_id", pa.string()),
            ("source", pa.string()),
            ("severity", pa.string()),
            ("code", pa.string()),
            ("message", pa.string()),
            ("inherited_from", pa.string()),
            ("created_at", pa.string()),
        ])
        _write_table(run_dir / "Warnings.parquet", [{
            "warning_id": "slice0_scaffold_only",
            "field_id": field_id,
            "source": "pegasus.output.bundle",
            "severity": "info",
            "code": "slice0_scaffold_only",
            "message": "Slice 0 scaffold bundle; no domain computation has run.",
            "inherited_from": json.dumps([]),
            "created_at": _now(),
        }], warn_schema)

        empty_assoc_schema = pa.schema([
            ("id", pa.string()),
            ("status", pa.string()),
            ("warnings", pa.string()),
        ])
        for name in ["ModelAssociations", "ResidualAssociations"]:
            _write_table(run_dir / f"{name}.parquet", [], empty_assoc_schema)

        hyp_schema = pa.schema([
            ("hypothesis_id", pa.string()),
            ("outcome_field_id", pa.string()),
            ("covariate_field_id", pa.string()),
            ("residual_field_id", pa.string()),
            ("statistic", pa.float64()),
            ("p_value", pa.float64()),
            ("q_value", pa.float64()),
            ("hsic_mode", pa.string()),
            ("residual_mode", pa.string()),
            ("fold_scheme", pa.string()),
            ("bootstrap_count", pa.int64()),
            ("residual_uncertainty", pa.string()),
            ("null_strategy", pa.string()),
            ("fdr_method", pa.string()),
            ("n_eff", pa.float64()),
            ("state", pa.string()),
            ("warnings", pa.string()),
            ("approximation_diagnostics_json", pa.string()),
        ])
        _write_table(run_dir / "Hypotheses.parquet", [], hyp_schema)

        vd_schema = pa.schema([
            ("field_id", pa.string()),
            ("display_name", pa.string()),
            ("technical_name", pa.string()),
            ("definition", pa.string()),
            ("estimand_label", pa.string()),
            ("source_systems", pa.string()),
            ("carrier", pa.string()),
            ("unit", pa.string()),
            ("support_description", pa.string()),
            ("axis_description", pa.string()),
            ("provenance_description", pa.string()),
            ("state", pa.string()),
            ("dashboard_safe", pa.string()),
            ("interpretation_warning", pa.string()),
        ])
        _write_table(run_dir / "VariableDictionary.parquet", [{
            "field_id": field_id,
            "display_name": "Slice 0 Scaffold Field",
            "technical_name": "slice0_scaffold_field",
            "definition": "Non-analytic placeholder used to validate the output contract.",
            "estimand_label": "scaffold_non_estimand",
            "source_systems": json.dumps(["scaffold"]),
            "carrier": "none",
            "unit": "none",
            "support_description": "No epidemiological support.",
            "axis_description": "No epidemiological axes.",
            "provenance_description": "Synthetic scaffold artifact.",
            "state": "quarantined_descriptive",
            "dashboard_safe": "False",
            "interpretation_warning": "Not an epidemiological field.",
        }], vd_schema)

        fb_schema = pa.schema([
            ("failed_branch_id", pa.string()),
            ("attempted_operator", pa.string()),
            ("parent_field_ids", pa.string()),
            ("failure_stage", pa.string()),
            ("failed_terms", pa.string()),
            ("reason", pa.string()),
            ("warnings", pa.string()),
            ("created_at", pa.string()),
        ])
        _write_table(run_dir / "FailedBranches.parquet", [], fb_schema)

        qf_schema = pa.schema([
            ("field_id", pa.string()),
            ("state", pa.string()),
            ("reason", pa.string()),
            ("warnings", pa.string()),
        ])
        _write_table(run_dir / "QuarantinedFields.parquet", [{
            "field_id": field_id,
            "state": "quarantined_descriptive",
            "reason": "slice0_scaffold_only",
            "warnings": json.dumps(["synthetic_placeholder"]),
        }], qf_schema)
        _write_table(run_dir / "ForcedFields.parquet", [], qf_schema)

        (run_dir / "P_vector.json").write_text(json.dumps({
            "schema_version": "1.0",
            "provenance": {"slice0_scaffold_field": ["synthetic"]},
        }, indent=2), encoding="utf-8")

        (run_dir / "UserIntent.json").write_text(json.dumps({
            "frozen": True,
            "intent_source": "slice0_scaffold",
        }, indent=2), encoding="utf-8")

        (run_dir / "RunConfig.json").write_text(json.dumps({
            "frozen": True,
            "budget": "fast",
            "geo_mode": "native",
            "slice": "0",
        }, indent=2), encoding="utf-8")

        stages = [
            "config_load", "registry_validation", "datasus_acquire", "datasus_profile",
            "datasus_normalize", "sidra_metadata", "sidra_plan", "sidra_fetch",
            "sidra_normalize", "geo_support", "she_build", "population_solver",
            "stdfm", "efg_build", "q_tensor", "pirs_model", "pirs_hsic",
            "output_serialization", "output_validation",
        ]
        manifest = {
            "run_id": run_dir.name,
            "created_at": _now(),
            "completed_at": _now(),
            "status": "success",
            "code_version": {
                "package_version": "0.1.0",
                "git_commit": "uninitialized",
                "git_dirty": False,
            },
            "environment": {
                "python_version": sys.version,
                "os": platform.platform(),
                "duckdb_version": None,
                "polars_version": None,
                "pyarrow_version": pa.__version__,
                "torch_version": None,
                "torch_cuda_available": False,
                "cuda_device_name": None,
                "r_version": None,
                "microdatasus_version": None,
                "read_dbc_version": None,
            },
            "registry_hashes": {},
            "source_manifest_hashes": [],
            "random_seeds": {},
            "telemetry": {
                "total_wall_seconds": 0.0,
                "stage_wall_seconds": {f"{s}_seconds": 0.0 for s in stages},
                "stage_status": {
                    s: ("success" if s in {"config_load", "registry_validation", "output_serialization", "output_validation"} else "skipped")
                    for s in stages
                },
                "resource_summary": {
                    "peak_rss_mb": None,
                    "peak_vram_mb": None,
                    "duckdb_temp_bytes": None,
                    "rows_read": {},
                    "rows_written": {},
                    "parquet_bytes_written": 0,
                },
            },
        }
        (run_dir / "ReproducibilityManifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        expected = set(OUTPUT_BUNDLE_FILES.values())
        for child in run_dir.iterdir():
            if child.name not in expected:
                raise RuntimeError(f"Unexpected first-class output artifact: {child}")

        return run_dir
    """)

    write("src/pegasus/output/validate.py", """
    from __future__ import annotations

    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    from pegasus.output.schemas import OUTPUT_BUNDLE_FILES, OutputSchemaRegistry, OutputValidationResult


    def _read(path: Path):
        return pq.read_table(path)


    def validate_output_bundle(
        *,
        run_dir: str,
        schema_registry: OutputSchemaRegistry | None = None,
    ) -> OutputValidationResult:
        schema_registry = schema_registry or OutputSchemaRegistry()
        root = Path(run_dir)
        errors: list[str] = []
        warnings: list[str] = []

        if not root.exists():
            return OutputValidationResult(ok=False, errors=[f"run_dir does not exist: {root}"], warnings=[])

        expected_names = {OUTPUT_BUNDLE_FILES[key] for key in schema_registry.required_keys}
        found_names = {p.name for p in root.iterdir()}

        missing = expected_names - found_names
        extra = found_names - expected_names

        for name in sorted(missing):
            errors.append(f"missing first-class artifact: {name}")
        for name in sorted(extra):
            errors.append(f"extra first-class artifact: {name}")

        if errors:
            return OutputValidationResult(ok=False, errors=errors, warnings=warnings)

        try:
            v = _read(root / "V_fields.parquet")
            q = _read(root / "Q_tensor.parquet")
            vd = _read(root / "VariableDictionary.parquet")
            edges = _read(root / "E_DAG.parquet")
            warnings_table = _read(root / "Warnings.parquet")
        except Exception as exc:
            return OutputValidationResult(ok=False, errors=[f"parquet read failure: {exc}"], warnings=warnings)

        if q.num_rows == 0:
            errors.append("Q_tensor is empty")

        v_ids = set(v.column("field_id").to_pylist()) if "field_id" in v.column_names else set()
        vd_ids = set(vd.column("field_id").to_pylist()) if "field_id" in vd.column_names else set()

        if not v_ids.issubset(vd_ids):
            errors.append("VariableDictionary does not cover all V_fields")

        if edges.num_rows:
            for col in ["parent_field_id", "child_field_id"]:
                if col in edges.column_names:
                    bad = set(edges.column(col).to_pylist()) - v_ids
                    if bad:
                        errors.append(f"E_DAG {col} contains IDs absent from V_fields: {sorted(bad)}")

        if warnings_table.num_rows and "field_id" in warnings_table.column_names:
            bad_warnings = {
                x for x in warnings_table.column("field_id").to_pylist()
                if x is not None and x not in v_ids
            }
            if bad_warnings:
                errors.append(f"Warnings link to invalid field IDs: {sorted(bad_warnings)}")

        try:
            manifest = json.loads((root / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
            telemetry = manifest["telemetry"]
            if telemetry.get("total_wall_seconds") is None or telemetry["total_wall_seconds"] < 0:
                errors.append("telemetry.total_wall_seconds missing or negative")
            for stage, status in telemetry.get("stage_status", {}).items():
                if status not in {"success", "skipped", "blocked", "failed"}:
                    errors.append(f"invalid telemetry stage status: {stage}={status}")
            for stage, duration in telemetry.get("stage_wall_seconds", {}).items():
                if duration is None or duration < 0:
                    errors.append(f"invalid telemetry stage duration: {stage}={duration}")
        except Exception as exc:
            errors.append(f"invalid ReproducibilityManifest.json telemetry: {exc}")

        return OutputValidationResult(ok=not errors, errors=errors, warnings=warnings)
    """)


def make_cli() -> None:
    write("src/pegasus/cli.py", """
    from __future__ import annotations

    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree

    app = typer.Typer(no_args_is_help=True)
    registries_app = typer.Typer(no_args_is_help=True)
    sidra_app = typer.Typer(no_args_is_help=True)
    datasus_app = typer.Typer(no_args_is_help=True)

    app.add_typer(registries_app, name="registries")
    app.add_typer(sidra_app, name="sidra")
    app.add_typer(datasus_app, name="datasus")


    def _fail(errors: list[str]) -> None:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)


    @app.command()
    def init() -> None:
        ensure_data_lake(".")
        run_dir = create_empty_output_bundle(Path("data/runs/slice0_empty"))
        print(f"[green]initialized[/green] data lake and scaffold run: {run_dir}")


    @app.command("validate-config")
    def validate_config() -> None:
        errors = validate_config_tree(".")
        if errors:
            _fail(errors)
        print("[green]config valid[/green]")


    @registries_app.command("validate")
    def validate_registries() -> None:
        errors = validate_registry_tree("config/registries")
        if errors:
            _fail(errors)
        print("[green]registries valid[/green]")


    @app.command("validate-run")
    def validate_run(run: Path = typer.Option(..., "--run")) -> None:
        result = validate_output_bundle(run_dir=str(run))
        if not result.ok:
            _fail(result.errors)
        print("[green]run bundle valid[/green]")


    @app.command()
    def doctor() -> None:
        checks: dict[str, str] = {}
        checks["python"] = sys.version.split()[0]

        for mod in ["duckdb", "polars", "pyarrow", "pydantic", "typer", "yaml"]:
            checks[mod] = "ok" if importlib.util.find_spec(mod) else "missing"

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec:
            import torch
            checks["torch"] = getattr(torch, "__version__", "ok")
            checks["torch_cuda_available"] = str(torch.cuda.is_available())
            checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        else:
            checks["torch"] = "missing"
            checks["torch_cuda_available"] = "False"
            checks["cuda_device_name"] = "none"

        checks["Rscript"] = shutil.which("Rscript") or "missing"
        checks["data_write"] = "ok"
        try:
            ensure_data_lake(".")
        except Exception as exc:
            checks["data_write"] = f"failed: {exc}"

        reg_errors = validate_registry_tree("config/registries")
        checks["registry_schema"] = "ok" if not reg_errors else f"{len(reg_errors)} errors"

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is Slice 0-light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


    @sidra_app.command("metadata")
    def sidra_metadata(tables: Path = typer.Option(..., "--tables")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA metadata is Slice 2. Seed received: {tables}")
        raise typer.Exit(2)


    @sidra_app.command("plan")
    def sidra_plan(view: str = typer.Option(..., "--view")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA planning is Slice 2. View received: {view}")
        raise typer.Exit(2)


    @sidra_app.command("extract")
    def sidra_extract(plan: Path = typer.Option(..., "--plan")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA extraction is Slice 2. Plan received: {plan}")
        raise typer.Exit(2)


    @datasus_app.command("ingest")
    def datasus_ingest(
        system: str = typer.Option(..., "--system"),
        uf: str = typer.Option(..., "--uf"),
        years: str = typer.Option(..., "--years"),
    ) -> None:
        print(f"[yellow]blocked[/yellow] DATASUS ingest is Slice 1+. Request: {system} {uf} {years}")
        raise typer.Exit(2)


    @datasus_app.command("profile")
    def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
        print(f"[yellow]blocked[/yellow] DATASUS profiling is Slice 1+. Manifest: {manifest}")
        raise typer.Exit(2)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    """)


def make_scripts_and_tests() -> None:
    write("scripts/doctor.py", """
    from pegasus.cli import doctor

    if __name__ == "__main__":
        doctor()
    """)

    write("scripts/check_cuda.py", """
    import importlib.util

    if not importlib.util.find_spec("torch"):
        print("torch missing")
    else:
        import torch
        print({"torch": torch.__version__, "cuda": torch.cuda.is_available()})
    """)

    write("scripts/check_r_microdatasus.R", """
    cat("Slice 0 R diagnostic placeholder. microdatasus check is formalized in later slices.\\n")
    """)

    write("scripts/validate_registry_hashes.py", """
    from pegasus.registries.validators import validate_registry_tree

    errors = validate_registry_tree("config/registries")
    if errors:
        for e in errors:
            print(e)
        raise SystemExit(1)
    print("registries valid")
    """)

    write("tests/smoke/test_slice0_contract.py", """
    from pathlib import Path

    from pegasus.core.config import validate_config_tree
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree


    def test_slice0_contract(tmp_path: Path):
        assert not validate_config_tree(".")
        assert not validate_registry_tree("config/registries")
        run_dir = create_empty_output_bundle(tmp_path / "run")
        result = validate_output_bundle(run_dir=str(run_dir))
        assert result.ok, result.errors
    """)


def main() -> None:
    make_dirs()
    make_root_files()
    make_configs()
    make_registries()
    make_package()
    make_core_modules()
    make_registry_modules()
    make_output_modules()
    make_cli()
    make_scripts_and_tests()

    print(f"Slice 0 scaffold written to: {ROOT}")
    print("Next: install package and validate.")


if __name__ == "__main__":
    main()