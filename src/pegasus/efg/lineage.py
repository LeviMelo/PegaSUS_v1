from __future__ import annotations

from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import Lineage


def make_lineage(
    *,
    parent_ids: list[str],
    operator_type: str,
    operator_params: dict[str, Any],
    registry_versions: dict[str, str] | None = None,
    source_manifest_hashes: list[str] | None = None,
    code_version: str = "0.1.0",
) -> Lineage:
    return Lineage(
        parent_ids=parent_ids,
        operator_type=operator_type,
        operator_params=operator_params,
        registry_versions=registry_versions or {"registry_set": "v1.0"},
        source_manifest_hashes=source_manifest_hashes or [],
        code_version=code_version,
    )


def lineage_hash(lineage: Lineage) -> str:
    return content_hash(lineage.model_dump(mode="json"))


def field_id_from_lineage(lineage: Lineage) -> str:
    return lineage_hash(lineage)
