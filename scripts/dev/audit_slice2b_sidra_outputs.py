from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    log_path = Path("data/diagnostics/sidra/population_9606_total_2022_alagoas_smoke.extraction_log.json")
    plan_path = Path("data/manifests/sidra/population_9606_total_2022_alagoas_smoke.json")

    if not log_path.exists():
        raise SystemExit(f"Missing extraction log: {log_path}")
    if not plan_path.exists():
        raise SystemExit(f"Missing plan: {plan_path}")

    plan = read_json(plan_path)
    log = read_json(log_path)

    print("\nPLAN")
    print(json.dumps(plan, indent=2, ensure_ascii=False)[:4000])

    print("\nLOG")
    print(json.dumps(log, indent=2, ensure_ascii=False)[:4000])

    if len(log) != 1:
        raise AssertionError(f"Expected exactly one smoke chunk result; got {len(log)}")

    result = log[0]
    if result["status"] != "success":
        raise AssertionError(f"Smoke extraction was not successful: {result}")

    raw_path = Path(result["raw_path"])
    facts_path = Path(result["facts_path"])

    if not raw_path.exists():
        raise AssertionError(f"Missing raw chunk file: {raw_path}")
    if not facts_path.exists():
        raise AssertionError(f"Missing facts parquet: {facts_path}")

    raw = read_json(raw_path)
    payload = raw.get("payload")

    print("\nRAW PAYLOAD FIRST 2 ROWS")
    if isinstance(payload, list):
        for row in payload[:2]:
            print(json.dumps(row, indent=2, ensure_ascii=False)[:4000])
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False)[:4000])

    df = pl.read_parquet(facts_path)

    print("\nFACTS SCHEMA")
    print(df.schema)

    print("\nFACTS ROWS")
    print(df)

    if df.height != 1:
        raise AssertionError(f"Expected exactly one normalized fact row; got {df.height}")

    row = df.row(0, named=True)

    expected = {
        "table_id": "9606",
        "variable_id": "93",
        "period": "2022",
        "locality_level": "N6",
        "locality_id": "2704302",
        "value_status": "numeric",
        "unit": "Pessoas",
    }

    hard_failures = []
    for key, value in expected.items():
        if row.get(key) != value:
            hard_failures.append((key, value, row.get(key)))

    if row.get("value_numeric") is None:
        hard_failures.append(("value_numeric", "non-null population value", None))

    cat_tuple = row.get("category_tuple")
    cls_tuple = row.get("classification_tuple")

    if not cat_tuple or cat_tuple in {"[]", "null"}:
        hard_failures.append(("category_tuple", "nonempty total race/sex/age categories", cat_tuple))

    if not cls_tuple or cls_tuple in {"[]", "null"}:
        hard_failures.append(("classification_tuple", "nonempty classification metadata", cls_tuple))

    for expected_fragment in ['"2","6794"', '"86","95251"', '"287","100362"']:
        compact = str(cat_tuple).replace(" ", "")
        if expected_fragment not in compact:
            hard_failures.append(("category_tuple", f"contains {expected_fragment}", cat_tuple))

    metadata_hash = row.get("metadata_hash")
    if metadata_hash in {None, "", "metadata_unset"}:
        hard_failures.append(("metadata_hash", "content hash of normalized SIDRA metadata", metadata_hash))
    elif len(str(metadata_hash)) != 64:
        hard_failures.append(("metadata_hash", "64-character sha256", metadata_hash))

    if hard_failures:
        print("\nAUDIT FAILURES")
        for key, expected_value, actual_value in hard_failures:
            print(f"- {key}: expected {expected_value!r}; got {actual_value!r}")
        raise SystemExit(1)

    print("\nAUDIT PASSED: live SIDRA smoke fact has correct long-form semantics.")


if __name__ == "__main__":
    main()
