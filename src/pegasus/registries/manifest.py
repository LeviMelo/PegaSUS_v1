"""Thin accessors for the canonical registry manifest."""

from __future__ import annotations

from pathlib import Path

from pegasus.registries.generic import load_registry_payload, registry_manifest

REGISTRY_FILES = ("registry_manifest.yaml",)


def load_registry_manifest_payload(*, root: str | Path = "config/registries", required: bool = True) -> dict:
    return load_registry_payload(REGISTRY_FILES, root=root, required=required)


def canonical_registry_manifest(*, root: str | Path = "config/registries", required: bool = True) -> dict:
    return registry_manifest(REGISTRY_FILES, root=root, required=required)


__all__ = ["REGISTRY_FILES", "canonical_registry_manifest", "load_registry_manifest_payload"]
