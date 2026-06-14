"""Thin registry wrapper for SIDRA registry entries."""

from __future__ import annotations

from pathlib import Path

from pegasus.registries.generic import RegistryEntry, active_entries, get_entry, load_entries, registry_manifest

REGISTRY_FILES = ('sidra_registry.yaml', 'sidra.yaml', 'sidra_tables.yaml')


def load_sidra_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return load_entries(REGISTRY_FILES, root=root, required=required)


def active_sidra_entries(*, root: str | Path = "config/registries", required: bool = False) -> tuple[RegistryEntry, ...]:
    return active_entries(REGISTRY_FILES, root=root, required=required)


def get_sidra_entry(entry_id: str, *, root: str | Path = "config/registries", required: bool = True) -> RegistryEntry:
    return get_entry(REGISTRY_FILES, entry_id, root=root, required=required)


def sidra_registry_manifest(*, root: str | Path = "config/registries", required: bool = False) -> dict:
    return registry_manifest(REGISTRY_FILES, root=root, required=required)


__all__ = [
    "REGISTRY_FILES",
    "load_sidra_entries",
    "active_sidra_entries",
    "get_sidra_entry",
    "sidra_registry_manifest",
]
