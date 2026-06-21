from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
BACKUP_ROOT = ROOT / ".codex-tmp" / "maternal_child_state_support_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

TARGET = ROOT / "src" / "pegasus" / "output" / "maternal_child_compile_attach.py"
TEST = ROOT / "tests" / "unit" / "test_maternal_child_state_attach_contract.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = BACKUP_ROOT / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def patch_attach_function() -> None:
    text = read(TARGET)

    text = text.replace(
        "    municipality_cod6: str,\n) -> Path:",
        "    municipality_cod6: str | None,\n) -> Path:",
        1,
    )

    old = '''    cod7 = datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)
    if cod7 is None:
        raise ValueError(f"Cannot crosswalk DATASUS municipality cod6 to IBGE/SIDRA cod7: {municipality_cod6}")

    v_rows = _read_rows(run_dir / "V_fields.parquet")
    population_row = _field_by_name(v_rows, "SIDRAPopulationTotalAnchor")
    population_value = _population_value(population_row)
    summary = summarize_maternal_child_linkage(
        sinasc_events_path=sinasc_events_path,
        sim_events_path=sim_events_path,
        municipality_cod6=municipality_cod6,
        municipality_ibge_cod7=cod7,
        denominator_population=population_value,
    )
    if not summary.births_total:
        raise ValueError("Cannot attach maternal-child fields without nonzero live-birth support.")
    if summary.municipalities_ibge_cod7 != [cod7]:
        raise ValueError(
            f"Maternal-child support mismatch: expected cod7={cod7}, observed={summary.municipalities_ibge_cod7}"
        )
'''

    new = '''    cod7 = None
    if municipality_cod6 is not None:
        cod7 = datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)
        if cod7 is None:
            raise ValueError(f"Cannot crosswalk DATASUS municipality cod6 to IBGE/SIDRA cod7: {municipality_cod6}")

    v_rows = _read_rows(run_dir / "V_fields.parquet")
    population_row = _field_by_name(v_rows, "SIDRAPopulationTotalAnchor")
    population_value = _population_value(population_row)
    summary = summarize_maternal_child_linkage(
        sinasc_events_path=sinasc_events_path,
        sim_events_path=sim_events_path,
        municipality_cod6=municipality_cod6,
        municipality_ibge_cod7=cod7,
        denominator_population=population_value,
    )
    if not summary.births_total:
        raise ValueError("Cannot attach maternal-child fields without nonzero live-birth support.")

    if municipality_cod6 is not None:
        if summary.municipalities_ibge_cod7 != [cod7]:
            raise ValueError(
                f"Maternal-child support mismatch: expected cod7={cod7}, observed={summary.municipalities_ibge_cod7}"
            )
    else:
        if len(summary.municipalities_cod6) <= 1:
            raise ValueError(
                "State-level maternal-child attachment requires multi-municipality DATASUS cod6 support; "
                f"observed={summary.municipalities_cod6}"
            )
'''

    if old not in text:
        raise RuntimeError(
            "Could not find the old single-municipality cod7 block in maternal_child_compile_attach.py. "
            "The file may have diverged; inspect attach_maternal_child_compile_fields manually."
        )

    text = text.replace(old, new, 1)
    write(TARGET, text)


def write_tests() -> None:
    test = '''from __future__ import annotations

import inspect

from pegasus.output import maternal_child_compile_attach as mc


def test_maternal_child_attach_accepts_optional_municipality_cod6() -> None:
    signature = inspect.signature(mc.attach_maternal_child_compile_fields)
    annotation = signature.parameters["municipality_cod6"].annotation
    assert annotation in {"str | None", str | None}


def test_maternal_child_attach_has_state_level_branch() -> None:
    source = inspect.getsource(mc.attach_maternal_child_compile_fields)

    assert "if municipality_cod6 is not None:" in source
    assert "else:" in source
    assert "State-level maternal-child attachment requires multi-municipality" in source
    assert "datasus_cod6_to_ibge_cod7(municipality_cod6, strict=True)" in source
'''
    write(TEST, test)


def main() -> None:
    patch_attach_function()
    write_tests()
    print("Patched maternal-child attachment for state-level municipality_cod6=None support.")
    print(f"Backups: {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
