from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.registries.source_fields import resolve_source_field_entry, source_field_registry_summary
from pegasus.she.source_registry import resolve_source_field


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Slice 13B registry-backed source-field semantics.")
    parser.add_argument("--registry-root", default="config/registries")
    args = parser.parse_args()
    root = Path(args.registry_root)
    summary = source_field_registry_summary(registry_root=root)
    if summary["entry_count"] < 40:
        raise SystemExit(f"source-field registry too small: {summary['entry_count']}")
    required = [
        ("SIM-DO", "underlying_icd_norm", "Deaths", "ICD10"),
        ("SIM-DO", "race_color_admin", "Deaths", "counts"),
        ("SINASC", "low_birth_weight_flag", "LiveBirths", "counts"),
        ("SIH-RD", "hospital_service_cost_real", "HospitalAdmissions", "BRL"),
        ("CNES-ST", "QTLEIT05", "Facilities", "beds"),
        ("SIDRA", "value_numeric", "ContextCells", "raw_sidra_value"),
    ]
    for system, column, carrier, unit in required:
        entry = resolve_source_field_entry(source_system=system, column_name=column, registry_root=root)
        if entry.carrier != carrier or entry.unit != unit:
            raise SystemExit(f"unexpected registry entry for {system}.{column}: {entry.as_manifest()}")
        resolution = resolve_source_field(source_system=system, column_name=column, registry_root=root)
        if not resolution.registry_backed:
            raise SystemExit(f"SHE resolution is not registry-backed for {system}.{column}")
        if not resolution.spec.registry_hash:
            raise SystemExit(f"SHE resolution missing registry hash for {system}.{column}")
    unknown = resolve_source_field(source_system="SIM-DO", column_name="UNDECLARED_COLUMN", registry_root=root)
    if unknown.known or unknown.spec.admissible:
        raise SystemExit("unknown source field was not quarantined as audit-only")
    print(
        "AUDIT PASSED: Slice 13B registry-backed source-field semantics validated "
        + json.dumps(
            {
                "entry_count": summary["entry_count"],
                "pattern_count": summary["pattern_count"],
                "registry_hash": summary["registry_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
