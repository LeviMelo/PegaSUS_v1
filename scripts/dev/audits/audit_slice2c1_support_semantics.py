from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _json(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return json.loads(str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PegaSUS Slice 2C.1 support semantics.")
    parser.add_argument("--run", required=True, help="Run bundle directory.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    v_path = run_dir / "V_fields.parquet"
    e_path = run_dir / "E_DAG.parquet"
    warnings_path = run_dir / "Warnings.parquet"

    for path in [v_path, e_path, warnings_path]:
        if not path.exists():
            raise SystemExit(f"Missing required run artifact: {path}")

    v = pl.read_parquet(v_path)
    e = pl.read_parquet(e_path)
    warnings = pl.read_parquet(warnings_path)

    failures: list[str] = []
    rows = v.filter(pl.col("operator") == "RN").to_dicts()
    sidra_rates = [row for row in rows if "SIDRA" in str(row.get("source"))]

    for row in sidra_rates:
        support = _json(row.get("support_json"))
        alignment = support.get("support_alignment")
        if not alignment:
            failures.append(f"{row['name']} lacks support_alignment metadata.")
            continue
        if alignment.get("aligned") is not True:
            failures.append(f"{row['name']} support_alignment is not aligned: {alignment}")
        if alignment.get("numerator_municipalities_ibge_cod7") != alignment.get("denominator_municipalities_ibge_cod7"):
            failures.append(f"{row['name']} numerator/denominator cod7 sets differ: {alignment}")
        if support.get("municipality_crosswalk") != "datasus_cod6_to_ibge_cod7":
            failures.append(f"{row['name']} missing explicit municipality_crosswalk marker.")

        parents = e.filter(pl.col("child_field_id") == row["field_id"])
        if parents.height < 2:
            failures.append(f"{row['name']} does not have at least numerator and denominator parent edges.")

    if sidra_rates:
        warning_ids = set(warnings["warning_id"].to_list())
        if "support_aligned_by_municipality_crosswalk" not in warning_ids:
            failures.append("SIDRA rate exists but support_aligned_by_municipality_crosswalk warning is missing.")

    if failures:
        print("AUDIT FAILURES")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"AUDIT PASSED: checked {len(sidra_rates)} SIDRA-linked RN field(s) for support alignment semantics.")


if __name__ == "__main__":
    main()
