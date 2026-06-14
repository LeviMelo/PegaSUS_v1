from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from textwrap import dedent

ROOT = Path.cwd()
SIDRA_ANCHOR = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
AUDIT = ROOT / "scripts" / "dev" / "audits" / "audit_slice28za_sidra_anchor_storage_boundary.py"
TEST_UNIT = ROOT / "tests" / "unit" / "test_slice28za_sidra_anchor_storage_boundary.py"
TEST_INT = ROOT / "tests" / "integration" / "test_slice28za_sidra_anchor_storage_boundary_audit.py"
DOC = ROOT / "docs" / "production_boundaries.md"


def _backup(path: Path, suffix: str) -> None:
    backup = path.with_name(path.name + suffix)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _replace_top_level_function(text: str, name: str, replacement: str) -> str:
    """Replace an entire top-level function using AST line numbers.

    This deliberately avoids exact multi-line block matching inside function
    bodies. The previous 28ZA updater failed because it expected a fragile
    three-line Polars block to exist verbatim.
    """
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno - 1
            end = getattr(node, "end_lineno", None)
            if end is None:
                raise RuntimeError(f"Python AST did not expose end_lineno for {name!r}")
            repl_lines = replacement.strip("\n").splitlines()
            return "\n".join(lines[:start] + repl_lines + lines[end:]) + "\n"
    raise RuntimeError(f"Could not find top-level function {name!r}")


def _ensure_import(text: str, import_line: str, *, after: str) -> str:
    if import_line in text:
        return text
    if after not in text:
        raise RuntimeError(f"Could not locate import anchor: {after!r}")
    return text.replace(after, after + import_line)


REMOVE_BY_VALUES = r'''
def _remove_by_values(rows: list[dict[str, Any]], column: str, values: set[str]) -> list[dict[str, Any]]:
    """Return rows excluding entries whose column value is in values.

    Row-level filtering keeps this production attacher behind output.table_io and
    pegasus.storage instead of calling Polars/PyArrow parquet APIs directly.
    Both raw and stringified comparisons are accepted because bundle parquet
    readers can preserve ids as Python strings while some test fixtures use
    simple scalar values.
    """
    if not values:
        return [dict(row) for row in rows]
    raw_values = set(values)
    text_values = {str(value) for value in values}
    return [
        dict(row)
        for row in rows
        if row.get(column) not in raw_values and str(row.get(column)) not in text_values
    ]
'''


APPEND_ROWS = r'''
def _append_rows(path: Path, rows: list[dict[str, Any]], *, remove_column: str | None = None, remove_values: set[str] | None = None) -> None:
    """Append rows to an existing bundle table through output.table_io.

    Existing rows can be removed by a single key before append. Schema
    preservation is delegated to write_rows_like(), which delegates to the
    canonical pegasus.storage boundary.
    """
    existing = read_rows(path)
    if remove_column and remove_values:
        existing = _remove_by_values(existing, remove_column, remove_values)
    write_rows_like(path, existing + [dict(row) for row in rows])
'''


GET_FIELD = r'''
def _get_field_by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one V_fields row named {name!r}; found {len(matches)}.")
    return matches[0]
'''


ATTACH_FUNCTION = r'''
def attach_sidra_population_anchor_to_run(
    *,
    run_dir: str | Path,
    sidra_facts_path: str | Path,
) -> Path:
    run_dir = Path(run_dir)
    sidra_facts_path = Path(sidra_facts_path)

    anchor = load_sidra_population_total_anchor(sidra_facts_path)

    v_path = run_dir / "V_fields.parquet"
    e_path = run_dir / "E_DAG.parquet"
    q_path = run_dir / "Q_tensor.parquet"
    vd_path = run_dir / "VariableDictionary.parquet"
    warnings_path = run_dir / "Warnings.parquet"
    qf_path = run_dir / "QuarantinedFields.parquet"
    p_path = run_dir / "P_vector.json"

    for path in [v_path, e_path, q_path, vd_path, warnings_path, qf_path, p_path]:
        if not path.exists():
            raise FileNotFoundError(f"Run bundle is missing required file: {path}")

    v_rows = read_rows(v_path)
    all_deaths = _get_field_by_name(v_rows, "SIMDeathsAll")

    population_row = _population_v_field(anchor, sidra_facts_path)
    support_alignment = assert_municipality_year_support_aligned(
        numerator_support=_load_json_field(all_deaths["support_json"]),
        numerator_axes=_load_json_field(all_deaths["axes_json"]),
        denominator_support=_load_json_field(population_row["support_json"]),
        denominator_axes=_load_json_field(population_row["axes_json"]),
    )
    rate_row = _rate_v_field(
        all_deaths=all_deaths,
        anchor=anchor,
        facts_path=sidra_facts_path,
        support_alignment=support_alignment,
    )

    new_field_ids = {population_row["field_id"], rate_row["field_id"]}
    new_names = {"SIDRAPopulationTotalAnchor", "SIMCrudeMortalitySIDRAOfficial"}

    # Preserve the original behavior: remove prior rows with these semantic
    # names first, then remove by ids during append so repeated attaches are
    # idempotent even if a field id changes because source metadata changed.
    v_rows = _remove_by_values(v_rows, "name", new_names)
    write_rows_like(v_path, v_rows)
    _append_rows(v_path, [population_row, rate_row], remove_column="field_id", remove_values=new_field_ids)

    edge_rows = [
        {
            "edge_id": f"{all_deaths['field_id']}->{rate_row['field_id']}",
            "parent_field_id": all_deaths["field_id"],
            "child_field_id": rate_row["field_id"],
            "operator": "RN",
            "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor", "support_alignment": support_alignment.model()}),
            "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
            "created_at": _now(),
        },
        {
            "edge_id": f"{population_row['field_id']}->{rate_row['field_id']}",
            "parent_field_id": population_row["field_id"],
            "child_field_id": rate_row["field_id"],
            "operator": "RN",
            "operator_params_json": _json({"role": "mortality_rate", "denominator": "SIDRA_9606_total_population_anchor", "support_alignment": support_alignment.model()}),
            "registry_versions_json": _json({"sidra_metadata_hash": anchor.metadata_hash}),
            "created_at": _now(),
        },
    ]
    _append_rows(e_path, edge_rows, remove_column="child_field_id", remove_values={rate_row["field_id"]})

    _append_rows(q_path, _q_rows(population_row=population_row, rate_row=rate_row), remove_column="field_id", remove_values=new_field_ids)

    vd_columns = list(table_schema(vd_path).names)
    _append_rows(
        vd_path,
        _vd_rows(vd_columns=vd_columns, population_row=population_row, rate_row=rate_row),
        remove_column="field_id",
        remove_values=new_field_ids,
    )

    warning_rows = [
        {
            "warning_id": "sidra_total_category_anchor",
            "field_id": population_row["field_id"],
            "source": "pegasus.output.sidra_denominator_anchor",
            "severity": "info",
            "code": "sidra_total_category_anchor",
            "message": "SIDRA 9606 total sex/race/age categories were used to create bounded Population(s,t).",
            "inherited_from": _json([]),
            "created_at": _now(),
        },
        {
            "warning_id": "official_sidra_denominator_anchor",
            "field_id": rate_row["field_id"],
            "source": "pegasus.output.sidra_denominator_anchor",
            "severity": "info",
            "code": "official_sidra_denominator_anchor",
            "message": "SIM fixture crude mortality has an official SIDRA total-population denominator after support alignment.",
            "inherited_from": _json([population_row["field_id"]]),
            "created_at": _now(),
        },
        {
            "warning_id": "support_aligned_by_municipality_crosswalk",
            "field_id": rate_row["field_id"],
            "source": "pegasus.geo.support",
            "severity": "info",
            "code": "support_aligned_by_municipality_crosswalk",
            "message": "DATASUS six-digit municipality support was aligned to SIDRA seven-digit support by explicit crosswalk before RN field creation.",
            "inherited_from": _json([all_deaths["field_id"], population_row["field_id"]]),
            "created_at": _now(),
        },
    ]
    _append_rows(
        warnings_path,
        warning_rows,
        remove_column="warning_id",
        remove_values={row["warning_id"] for row in warning_rows},
    )

    qf_rows = [
        {
            "field_id": population_row["field_id"],
            "state": "fragile",
            "reason": "official_anchor_pending_crosschecks",
            "warnings": _json(["sidra_total_category_anchor"]),
        },
        {
            "field_id": rate_row["field_id"],
            "state": "quarantined_descriptive",
            "reason": "sim_fixture_numerator",
            "warnings": _json(["fixture_small_n", "official_sidra_denominator_anchor", "support_aligned_by_municipality_crosswalk"]),
        },
    ]
    _append_rows(qf_path, qf_rows, remove_column="field_id", remove_values=new_field_ids)

    p = json.loads(p_path.read_text(encoding="utf-8"))
    p.setdefault("provenance", {})
    p["provenance"][population_row["field_id"]] = ["official", "sidra_9606", "bounded_total_category_anchor"]
    p["provenance"][rate_row["field_id"]] = ["official", "sidra_denominator_anchor", "sim_fixture_numerator"]
    p_path.write_text(_json(p), encoding="utf-8")

    result = validate_output_bundle(run_dir=str(run_dir))
    if not result.ok:
        raise RuntimeError("Run bundle failed validation after SIDRA denominator anchor attach: " + "; ".join(result.errors))

    return run_dir
'''


AUDIT_SOURCE = r'''
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src" / "pegasus" / "output" / "sidra_denominator_anchor.py"
BANNED = (
    r"\bpl\.read_parquet\b",
    r"\.write_parquet\b",
    r"\bpyarrow\.parquet\b",
    r"\bpq\.read_table\b",
    r"\bpq\.write_table\b",
)


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    errors: list[str] = []
    for pattern in BANNED:
        if re.search(pattern, text):
            errors.append(f"storage bypass remains in {TARGET.relative_to(ROOT)}: {pattern}")
    if "from pegasus.output.table_io import" not in text:
        errors.append("sidra_denominator_anchor.py does not import output.table_io")
    if "import polars as pl" in text:
        errors.append("sidra_denominator_anchor.py still imports Polars directly")
    payload = {
        "audit": "slice28za_sidra_anchor_storage_boundary",
        "target": str(TARGET.relative_to(ROOT)),
        "errors": errors,
        "status": "failed" if errors else "passed",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


TEST_UNIT_SOURCE = r'''
from __future__ import annotations

from pegasus.output.sidra_denominator_anchor import _append_rows, _get_field_by_name, _remove_by_values
from pegasus.output.table_io import read_rows, write_rows


def test_slice28za_row_helpers_preserve_append_replace_semantics(tmp_path):
    path = tmp_path / "table.parquet"
    write_rows(
        path,
        [
            {"id": "a", "name": "old", "value": 1},
            {"id": "b", "name": "keep", "value": 2},
        ],
    )

    _append_rows(path, [{"id": "a", "name": "new", "value": 3}], remove_column="id", remove_values={"a"})

    rows = read_rows(path)
    assert [row["id"] for row in rows] == ["b", "a"]
    assert _get_field_by_name(rows, "new")["value"] == 3
    assert _remove_by_values(rows, "id", {"b"}) == [{"id": "a", "name": "new", "value": 3}]


def test_slice28za_sidra_anchor_source_uses_storage_boundary():
    import pegasus.output.sidra_denominator_anchor as module

    text = module.__loader__.get_source(module.__name__)
    assert "pl.read_parquet" not in text
    assert ".write_parquet" not in text
    assert "pyarrow.parquet" not in text
    assert "from pegasus.output.table_io import" in text
'''


TEST_INT_SOURCE = r'''
from __future__ import annotations

import json
import subprocess
import sys


def test_slice28za_sidra_anchor_storage_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/dev/audits/audit_slice28za_sidra_anchor_storage_boundary.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert payload["errors"] == []
'''


def _patch_sidra_anchor() -> None:
    if not SIDRA_ANCHOR.exists():
        raise FileNotFoundError(SIDRA_ANCHOR)
    _backup(SIDRA_ANCHOR, ".slice28za_sidra_storage_boundary_v2.bak")
    text = SIDRA_ANCHOR.read_text(encoding="utf-8")

    # Remove direct Polars import. Row filtering is now performed in Python over
    # rows read/written through output.table_io.
    text = re.sub(r"\nimport polars as pl\n", "\n", text)
    text = _ensure_import(
        text,
        "from pegasus.output.table_io import read_rows, table_schema, write_rows_like\n",
        after="from pegasus.output.validate import validate_output_bundle\n",
    )

    # Replace whole functions, never fragile internal multi-line snippets.
    text = _replace_top_level_function(text, "_remove_by_values", REMOVE_BY_VALUES)
    text = _replace_top_level_function(text, "_append_rows", APPEND_ROWS)
    text = _replace_top_level_function(text, "_get_field_by_name", GET_FIELD)
    text = _replace_top_level_function(text, "attach_sidra_population_anchor_to_run", ATTACH_FUNCTION)

    banned = ["pl.read_parquet", ".write_parquet", "pyarrow.parquet", "pq.read_table", "pq.write_table"]
    remaining = [token for token in banned if token in text]
    if remaining:
        raise RuntimeError(f"Storage bypass tokens remain in sidra_denominator_anchor.py: {remaining}")

    SIDRA_ANCHOR.write_text(text, encoding="utf-8")


def _write_auxiliary_files() -> None:
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    TEST_UNIT.parent.mkdir(parents=True, exist_ok=True)
    TEST_INT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(AUDIT_SOURCE.lstrip(), encoding="utf-8")
    TEST_UNIT.write_text(TEST_UNIT_SOURCE.lstrip(), encoding="utf-8")
    TEST_INT.write_text(TEST_INT_SOURCE.lstrip(), encoding="utf-8")
    if DOC.exists():
        doc = DOC.read_text(encoding="utf-8")
        note = "\n\nSlice 28ZA extends the storage-boundary adoption to the SIDRA denominator anchor. The module may still use row-level Python transformations, but Parquet reads/writes are routed through output.table_io and pegasus.storage.\n"
        if "Slice 28ZA extends the storage-boundary adoption" not in doc:
            DOC.write_text(doc.rstrip() + note, encoding="utf-8")


def main() -> None:
    if not (ROOT / "src" / "pegasus").exists():
        raise RuntimeError("Run this updater from the PegaSUS repository root")
    _patch_sidra_anchor()
    _write_auxiliary_files()
    print("Slice 28ZA SIDRA denominator anchor storage-boundary adoption v2 applied.")


if __name__ == "__main__":
    main()
