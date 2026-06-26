"""Build the real national municipality crosswalk from SIDRA/IBGE N6 localities.

Replaces the prior 2-entry ``fixture_smoke_v1`` crosswalk (which only contained
two Alagoas municipalities and forced every other code through a cod7[:6] slice
fallback) with the full set of Brazilian municipalities. The DATASUS cod6 is the
first six digits of the IBGE cod7 (the 7th digit is the check digit).

Source: data/metadata/sidra/normalized/sidra_localities.parquet (N6 localities,
fetched from the IBGE SIDRA API — national coverage, not a fixture).

Usage:  python scripts/build_municipality_crosswalk.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
LOCALITIES = ROOT / "data" / "metadata" / "sidra" / "normalized" / "sidra_localities.parquet"
OUTPUT = ROOT / "config" / "registries" / "municipality_crosswalk_codes.yaml"


def build() -> tuple[int, list[tuple[str, str]]]:
    if not LOCALITIES.exists():
        raise SystemExit(
            f"SIDRA localities not found at {LOCALITIES}. Run `pegasus sidra metadata` first."
        )
    df = pl.read_parquet(LOCALITIES)
    cod7s = (
        df.filter(pl.col("locality_level") == "N6")
        .select(pl.col("locality_id").cast(pl.Utf8).str.strip_chars())
        .unique()
        .to_series()
        .to_list()
    )
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cod7 in sorted(c for c in cod7s if c and len(c) == 7 and c.isdigit()):
        cod6 = cod7[:6]
        if cod6 in seen:
            continue
        seen.add(cod6)
        pairs.append((cod6, cod7))
    return len(pairs), pairs


def main() -> None:
    count, pairs = build()
    lines = [
        'schema_version: "1.0"',
        'registry_version: "ibge_sidra_n6_national_v1"',
        'provenance: "Derived from SIDRA/IBGE N6 municipal localities (national coverage). cod6 = cod7[:6]."',
        "entries:",
    ]
    for cod6, cod7 in pairs:
        lines.append(f'  - datasus_cod6: "{cod6}"')
        lines.append(f'    ibge_cod7: "{cod7}"')
        lines.append('    provenance: "ibge_sidra_n6_municipal_locality"')
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {count} municipalities -> {OUTPUT}")
    ufs = sorted({cod6[:2] for cod6, _ in pairs})
    print(f"  UF prefixes covered: {len(ufs)} ({ufs[:5]}...{ufs[-3:]})")


if __name__ == "__main__":
    main()
