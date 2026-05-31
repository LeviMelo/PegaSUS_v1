from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SIDRAQuerySpec(BaseModel):
    table_id: str
    periods: list[str]
    variables: list[str]
    localities: str
    classifications: list[str] = Field(default_factory=list)
    view: Literal["flat", "normal"] = "flat"

    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class SIDRAFetchManifest(BaseModel):
    table_id: str
    periods: list[str]
    variables: list[str]
    localities: str
    classifications: list[str]
    view: str
    url: str
    cache_key: str
    raw_json_path: str
    facts_path: str
    n_raw_top_level_items: int | None = None
    n_facts: int
    empty_result: bool
    warnings: list[str] = Field(default_factory=list)


class SIDRANormalizationResult(BaseModel):
    n_facts: int
    warnings: list[str] = Field(default_factory=list)


def query_spec_from_mapping(data: dict[str, Any]) -> SIDRAQuerySpec:
    return SIDRAQuerySpec.model_validate(data)