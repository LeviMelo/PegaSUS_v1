from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()

CNES_CAPACITY_REGISTRY = """from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class CNESCapacityRegistryError(ValueError):
    \"\"\"Raised when CNES capacity requests erase vector-indexed semantics.\"\"\"


@dataclass(frozen=True)
class CNESCapacityComponent:
    raw_field: str
    component_id: str
    carrier: str
    unit: str
    family: Literal["bed", "room"]
    label: str


CAPACITY_COMPONENTS: dict[str, CNESCapacityComponent] = {
    "QTLEITP1": CNESCapacityComponent("QTLEITP1", "clinical_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP1", "bed", "Clinical beds"),
    "QTLEITP2": CNESCapacityComponent("QTLEITP2", "surgical_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP2", "bed", "Surgical beds"),
    "QTLEITP3": CNESCapacityComponent("QTLEITP3", "obstetric_bed_capacity", "FacilityCapacityVector", "facility_capacity_units_QTLEITP3", "bed", "Obstetric beds"),
    "QTINST01": CNESCapacityComponent("QTINST01", "consulting_room_capacity", "FacilityCapacityVector", "facility_capacity_units_QTINST01", "room", "Consulting rooms / infrastructure component 01"),
    "QTINST34": CNESCapacityComponent("QTINST34", "room_infrastructure_capacity_34", "FacilityCapacityVector", "facility_capacity_units_QTINST34", "room", "Infrastructure/room capacity component 34"),
}


def get_capacity_component(raw_field: str) -> CNESCapacityComponent:
    key = raw_field.upper()
    if key not in CAPACITY_COMPONENTS:
        raise CNESCapacityRegistryError(f"Unknown CNES capacity vector component: {raw_field}")
    return CAPACITY_COMPONENTS[key]


def require_vector_index(raw_field: str | None) -> CNESCapacityComponent:
    if raw_field is None or raw_field.strip().lower() in {"beds", "bed", "rooms", "room", "capacity", "generic_beds"}:
        raise CNESCapacityRegistryError("Generic CNES beds/capacity request is illegal without a capacity-vector index.")
    return get_capacity_component(raw_field)


def registry_manifest() -> dict[str, object]:
    return {
        "schema_version": "1.1",
        "registry": "cnes_capacity_vector",
        "components": {k: v.__dict__ for k, v in CAPACITY_COMPONENTS.items()},
    }
"""

CNES_FIXTURE = """CNES,CODMUN,COMPETEN,CPF_CNPJ,CNPJ_MAN,QTLEITP1,QTLEITP2,QTLEITP3,QTINST01,QTINST34,GESPRG1,NIVATE_A,ATENDAMB,URGEMERG,LEITHOSP
0000001,270430,202201,00000000000000,12345678000190,10,4,2,3,7,1,0,1,2,1
0000002,270430,202201,11222333000181,00000000000000,20,3,1,5,4,0,1,1,1,1
0000003,270030,202201,22333444000191,33444555000192,5,2,6,2,1,1,1,0,0,9
"""


def path(rel: str) -> Path:
    return ROOT / rel


def read_text(rel: str) -> str:
    return path(rel).read_text(encoding="utf-8")


def write_text(rel: str, text: str) -> None:
    target = path(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8")


def preflight() -> None:
    required = [
        "src/pegasus/registries/cnes_capacity.py",
        "tests/fixtures/datasus/cnes_st_fixture.csv",
        "src/pegasus/output/cnes_sih_efg_bundle.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/cnes_sih_compile_attach.py",
        "tests/integration/test_slice5b_compile_cnes_sih_integration.py",
    ]
    missing = [rel for rel in required if not path(rel).exists()]
    if missing:
        raise RuntimeError(f"Slice 5B repair preflight failed; missing files: {missing}")

    bundle_text = read_text("src/pegasus/output/cnes_sih_efg_bundle.py")
    if "build_cnes_sih_rows" not in bundle_text:
        raise RuntimeError("Slice 5B repair preflight failed: cnes_sih_efg_bundle.py is not the v2 modular row-builder version.")

    test_text = read_text("tests/integration/test_slice5b_compile_cnes_sih_integration.py")
    if "cnes_capacity_qtinst34" not in test_text:
        raise RuntimeError("Slice 5B repair preflight failed: Slice 5B integration test does not target QTINST34.")

    validate_text = read_text("src/pegasus/output/validate.py")
    if "_validate_cnes_sih_contract" not in validate_text:
        raise RuntimeError("Slice 5B repair preflight failed: validator does not contain the Slice 5B CNES/SIH contract.")


def patch_cnes_capacity_registry() -> None:
    write_text("src/pegasus/registries/cnes_capacity.py", CNES_CAPACITY_REGISTRY)


def patch_cnes_fixture() -> None:
    # Full fixture rewrite is deliberate: it makes the empirical CNES-ST capacity
    # contract exercise both the older QTINST01 fixture component and the stricter
    # QTINST34 component required by the Slice 5B audit/test.
    write_text("tests/fixtures/datasus/cnes_st_fixture.csv", CNES_FIXTURE)


def patch_warning_contract() -> None:
    rel = "src/pegasus/output/cnes_sih_efg_bundle.py"
    text = read_text(rel)
    changed = text.replace("all-zero-cnpj-nullified", "all_zero_cnpj_nullified")
    if "all_zero_cnpj_nullified" not in changed:
        raise RuntimeError("Could not ensure all_zero_cnpj_nullified warning code in cnes_sih_efg_bundle.py")
    write_text(rel, changed)


def patch_audit_context_if_needed() -> None:
    # Guarded compatibility no-op unless a local audit drifted to QTINST01.
    rel = "scripts/dev/audits/audit_slice5b_compile_cnes_sih.py"
    if not path(rel).exists():
        return
    text = read_text(rel)
    if "cnes_capacity_qtinst34" not in text and "cnes_capacity_qtinst01" in text:
        text = text.replace("cnes_capacity_qtinst01", "cnes_capacity_qtinst34")
        text = text.replace("QTINST01", "QTINST34")
        write_text(rel, text)


def summarize() -> None:
    print("Slice 5B repair applied.")
    print("CHANGES:")
    print("  src/pegasus/registries/cnes_capacity.py")
    print("    - Adds QTINST34 as a first-class CNES capacity-vector component.")
    print("    - Keeps QTINST01 and QTLEITP1/2/3; generic beds still fail.")
    print("  tests/fixtures/datasus/cnes_st_fixture.csv")
    print("    - Adds empirical QTINST34 fixture values so normalization/compile exercise it.")
    print("  src/pegasus/output/cnes_sih_efg_bundle.py")
    print("    - Restores committed 5A warning-code spelling: all_zero_cnpj_nullified.")
    print("  scripts/dev/audits/audit_slice5b_compile_cnes_sih.py")
    print("    - Guarded compatibility no-op unless a local audit drifted to QTINST01.")


def main() -> None:
    preflight()
    patch_cnes_capacity_registry()
    patch_cnes_fixture()
    patch_warning_contract()
    patch_audit_context_if_needed()
    summarize()


if __name__ == "__main__":
    main()
