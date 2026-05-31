from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from pegasus.core.exceptions import ConfigurationError
from pegasus.core.paths import ProjectPaths


class ProjectBlock(BaseModel):
    name: str
    package_name: str
    version: str
    registry_version: str
    timezone: str = "America/Maceio"
    default_locale: str = "pt-BR"


class ProjectConfig(BaseModel):
    project: ProjectBlock


class PathsBlock(BaseModel):
    root: str = "."
    data_root: str = "data"
    raw: str = "data/raw"
    processed: str = "data/processed"
    metadata: str = "data/metadata"
    cache: str = "data/cache"
    manifests: str = "data/manifests"
    intermediate: str = "data/intermediate"
    runs: str = "data/runs"
    diagnostics: str = "data/diagnostics"
    config: str = "config"
    registries: str = "config/registries"
    intents: str = "config/intents"


class PathsConfig(BaseModel):
    paths: PathsBlock


class CudaBlock(BaseModel):
    dtype: Literal["float32", "float64"] = "float32"
    max_vram_fraction: float = Field(default=0.80, ge=0.1, le=1.0)
    required_for_region_modules: list[str] = Field(default_factory=list)
    preferred_for_modules: list[str] = Field(default_factory=list)


class ComputeBlock(BaseModel):
    data_backend: str = "duckdb_polars_arrow"
    numerical_backend: str = "pytorch"
    rapids_enabled: bool = False
    cudf_enabled: bool = False
    polars_gpu_enabled: bool = False
    wsl_required: bool = False
    cuda: CudaBlock = Field(default_factory=CudaBlock)


class ComputeConfig(BaseModel):
    compute: ComputeBlock


class SidraBlock(BaseModel):
    base_url: str
    max_cells_per_request: int = 49_900
    view_mode: str = "flat"
    concurrency: int = 16
    timeout_seconds: int = 60
    retry_status_codes: list[int] = Field(default_factory=lambda: [429, 500, 502, 503, 504])
    max_retries: int = 5
    backoff_initial_seconds: float = 0.25
    backoff_max_seconds: float = 10.0
    persist_raw_chunks: bool = True
    persist_normalized_facts: bool = True
    persist_full_tables_by_default: bool = False


class SidraConfig(BaseModel):
    sidra: SidraBlock


class DatasusBlock(BaseModel):
    backend: str = "microdatasus"
    rscript_path: str = "Rscript"
    preserve_raw: bool = True
    preserve_processed: bool = True
    systems: list[str] = Field(default_factory=list)
    optional_ftp_discovery: bool = False
    r_timeout_seconds: int = 7200
    heartbeat_timeout_seconds: int = 900


class DatasusConfig(BaseModel):
    datasus: DatasusBlock


class PegasusConfig(BaseModel):
    project: ProjectBlock
    paths: PathsBlock
    compute: ComputeBlock
    sidra: SidraBlock
    datasus: DatasusBlock


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Missing configuration file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping: {path}")

    return data


def load_config(root: str | Path = ".") -> PegasusConfig:
    root = Path(root).resolve()
    config_dir = root / "config"

    project = ProjectConfig.model_validate(read_yaml(config_dir / "project.yaml"))
    paths = PathsConfig.model_validate(read_yaml(config_dir / "paths.yaml"))
    compute = ComputeConfig.model_validate(read_yaml(config_dir / "compute.yaml"))
    sidra = SidraConfig.model_validate(read_yaml(config_dir / "sidra.yaml"))
    datasus = DatasusConfig.model_validate(read_yaml(config_dir / "datasus.yaml"))

    return PegasusConfig(
        project=project.project,
        paths=paths.paths,
        compute=compute.compute,
        sidra=sidra.sidra,
        datasus=datasus.datasus,
    )


def resolve_project_paths(root: str | Path = ".") -> ProjectPaths:
    return ProjectPaths.from_root(root)