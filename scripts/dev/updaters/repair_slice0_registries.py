from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path.cwd()
REG_DIR = ROOT / "config" / "registries"

REGISTRY_FILES = [
    "source_fields.yaml",
    "composite_decoders.yaml",
    "carrier_registry.yaml",
    "unit_registry.yaml",
    "aggregation_registry.yaml",
    "provenance_registry.yaml",
    "quality_permissions.yaml",
    "race_axis_registry.yaml",
    "race_bridge_priors.yaml",
    "icd_catalog.yaml",
    "icd_quality_groups.yaml",
    "diagnostic_topology.yaml",
    "clinical_event_definitions.yaml",
    "cnes_capacity_registry.yaml",
    "sih_cost_registry.yaml",
    "sidra_views.yaml",
    "sidra_category_maps.yaml",
    "sidra_stitching.yaml",
    "sidra_regime_registry.yaml",
    "municipality_crosswalk_sources.yaml",
    "join_affordances.yaml",
    "bridge_grammars.yaml",
    "stdfm_registry.yaml",
    "population_solver_registry.yaml",
    "model_registry.yaml",
    "residual_registry.yaml",
    "hsic_registry.yaml",
    "null_registry.yaml",
    "output_schema.yaml",
]

SPECIAL_ENTRIES = {
    "composite_decoders.yaml": [
        {"id": "Decode_SIM_IDADE", "status": "stable", "description": "SIM structural age decoder.", "warnings": []},
        {"id": "Decode_SIH_AGE", "status": "stable", "description": "SIH age decoder using COD_IDADE.", "warnings": []},
        {"id": "Decode_PESO", "status": "stable", "description": "Physical scalar decoder for birth weight.", "warnings": []},
        {"id": "Decode_count2", "status": "stable", "description": "Two-character count decoder.", "warnings": []},
        {"id": "Clamp_bool", "status": "stable", "description": "Sentinel-safe boolean decoder.", "warnings": []},
        {"id": "Filter_CNPJ", "status": "stable", "description": "CNPJ sanitizer and all-zero nullifier.", "warnings": []},
        {"id": "Decode_cat", "status": "stable", "description": "Categorical socioeconomic decoder.", "warnings": []},
    ],
    "quality_permissions.yaml": [
        {"id": "verified", "status": "stable", "description": "Full analytic permissions.", "warnings": []},
        {"id": "fragile", "status": "stable", "description": "Analytic with warning.", "warnings": []},
        {"id": "forced_fragile", "status": "stable", "description": "Forced unstable field.", "warnings": []},
        {"id": "quarantined_descriptive", "status": "stable", "description": "Descriptive only.", "warnings": []},
        {"id": "quarantined_nochildren", "status": "stable", "description": "No downstream children.", "warnings": []},
        {"id": "illegal_excluded", "status": "stable", "description": "Excluded from analytic graph.", "warnings": []},
    ],
    "race_axis_registry.yaml": [
        {"id": "IBGE.self_declared", "status": "stable", "description": "IBGE self-declared race axis.", "warnings": []},
        {"id": "SIM-DO.administrative_death_declaration", "status": "stable", "description": "SIM death declaration race axis.", "warnings": []},
        {"id": "SIH-RD.billing_record", "status": "stable", "description": "SIH billing race axis.", "warnings": []},
        {"id": "SINASC.administrative_mixed", "status": "stable", "description": "SINASC administrative mixed race axis.", "warnings": []},
    ],
    "output_schema.yaml": [
        {"id": "immutable_17_key_bundle", "status": "stable", "description": "Exact run bundle key contract.", "warnings": []}
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_payload(file_name: str) -> dict:
    entries = SPECIAL_ENTRIES.get(
        file_name,
        [
            {
                "id": file_name.replace(".yaml", ""),
                "status": "deferred",
                "description": "Slice 0 placeholder registry.",
                "warnings": ["scaffold_only"],
            }
        ],
    )
    return {
        "schema_version": "1.0",
        "registry_version": "v1.0",
        "created_at": "2026-06-06",
        "updated_at": "2026-06-06",
        "provenance": f"Slice 0 scaffold registry for {file_name}; not a calibrated analytic registry.",
        "entries": entries,
    }


def main() -> None:
    REG_DIR.mkdir(parents=True, exist_ok=True)

    for file_name in REGISTRY_FILES:
        path = REG_DIR / file_name
        with path.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(
                registry_payload(file_name),
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    seed_path = REG_DIR / "sidra_table_seed.jsonl"
    with seed_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({
            "table_id": "9606",
            "purpose": "population_denominator_anchor",
            "status": "stable",
            "notes": "Slice 0 metadata seed only; no fetch performed.",
        }, ensure_ascii=False) + "\n")
        f.write(json.dumps({
            "table_id": "5938",
            "purpose": "gdp_context_candidate",
            "status": "deferred",
            "notes": "Slice 0 metadata seed only; no fetch performed.",
        }, ensure_ascii=False) + "\n")

    manifest_entries = {}
    for file_name in REGISTRY_FILES:
        manifest_entries[file_name.replace(".yaml", "")] = {
            "path": file_name,
            "schema_version": "1.0",
            "sha256": sha256_file(REG_DIR / file_name),
        }

    manifest_entries["sidra_table_seed"] = {
        "path": "sidra_table_seed.jsonl",
        "schema_version": "1.0",
        "sha256": sha256_file(seed_path),
    }

    manifest = {
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

    with (REG_DIR / "registry_manifest.yaml").open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(
            manifest,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )

    print("Rewrote Slice 0 registry YAML files and refreshed registry_manifest.yaml hashes.")


if __name__ == "__main__":
    main()