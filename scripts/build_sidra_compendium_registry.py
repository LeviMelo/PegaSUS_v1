"""Parse SIDRA_COMPENDIUM.md into a machine-readable registry.

The compendium is a curated catalog of 96 epidemiologically-relevant SIDRA/IBGE
tables with national (5,570-municipality) coverage. This script converts it,
reproducibly, into ``config/registries/sidra/sidra_compendium.json`` so ingestion and
EFG context-field construction can be driven from data rather than hand-typed
view entries. Re-run whenever the compendium changes.

Usage:  python scripts/build_sidra_compendium_registry.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SIDRA_COMPENDIUM.md"
OUTPUT = ROOT / "config" / "registries" / "sidra" / "sidra_compendium.json"

_GROUP_RE = re.compile(r"^###\s+GROUP\s+\d+:\s*(.+?)\s*$")
_TAB_RE = re.compile(r"^####\s+Tab\s+(\d+)\s*\|\s*(.+?)\s*$")
_META_RE = re.compile(r"\*\*(\w+)\*\*:\s*([^|]+?)(?=\s*\|\s*\*\*|\s*$)")
# Variable, with optional leading description (some tables write `93` (U: Px, ...)).
_VAR_RE = re.compile(r"`(\d+)`\s*\((?:([^;()]+?);\s*)?U:\s*([^,]+),\s*Type:\s*(\w+),\s*DK:\s*(\w+)\)")
# Classification, with optional inline category list after Cats: N.
_CLSF_RE = re.compile(r"`Clsf\s+(\d+)`\s*\(([^;]+);\s*Axis:\s*([^,]+),\s*Cats:\s*(\d+)(?:;\s*(.+?))?\)")
_CAT_RE = re.compile(r"`([\w]+)`\s*([^,)]+)")
_SAME_AS_RE = re.compile(r"[Ss]ame as Tab\s+(\d+)")


def _blocks(text: str) -> list[tuple[str, str, str]]:
    """Yield (group, header, body) per table block."""
    out: list[tuple[str, str, str]] = []
    group = ""
    current: list[str] | None = None
    header = ""
    for line in text.splitlines():
        gm = _GROUP_RE.match(line)
        if gm:
            group = gm.group(1)
            continue
        tm = _TAB_RE.match(line)
        if tm:
            if current is not None:
                out.append((current_group, header, "\n".join(current)))
            header = line
            current = []
            current_group = group
            continue
        if current is not None:
            current.append(line)
    if current is not None:
        out.append((current_group, header, "\n".join(current)))
    return out


def _parse_meta(body: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for line in body.splitlines():
        if "**Tier**" in line or "**Res**" in line or "**Per**" in line:
            for key, value in _META_RE.findall(line):
                meta[key.lower()] = value.strip()
    return meta


def _parse_vars(body: str) -> list[dict[str, object]]:
    return [
        {
            "variable_id": vid,
            "description": (desc or "").strip(),
            "unit": unit.strip(),
            "type": vtype.strip(),
            "default_keep": dk.strip().upper() == "Y",
        }
        for vid, desc, unit, vtype, dk in _VAR_RE.findall(body)
    ]


def _parse_classifications(body: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cid, name, axis, ncats, cats_blob in _CLSF_RE.findall(body):
        categories = [
            {"category_id": cat_id, "label": label.strip()}
            for cat_id, label in _CAT_RE.findall(cats_blob or "")
        ]
        out.append(
            {
                "classification_id": cid,
                "name": name.strip(),
                "axis": axis.strip(),
                "category_count": int(ncats),
                "categories": categories,
            }
        )
    return out


def parse_compendium(text: str) -> dict[str, object]:
    tables: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for group, header, body in _blocks(text):
        tm = _TAB_RE.match(header)
        if not tm:
            continue
        table_id, name = tm.group(1), tm.group(2).strip()
        meta = _parse_meta(body)
        variables = _parse_vars(body)
        same_as = _SAME_AS_RE.search(body)
        tables[table_id] = {
            "table_id": table_id,
            "name": name,
            "group": group,
            "tier": meta.get("tier", ""),
            "research": meta.get("res", ""),
            "subject": meta.get("subj", ""),
            "periods": meta.get("per", ""),
            "localities": meta.get("locs", ""),
            "variables": variables,
            "variables_same_as": same_as.group(1) if same_as else None,
            "classifications": _parse_classifications(body),
        }
        order.append(table_id)
    # Resolve "Same as Tab X" variable references.
    for tid in order:
        ref = tables[tid]["variables_same_as"]
        if ref and not tables[tid]["variables"] and ref in tables:
            tables[tid]["variables"] = list(tables[ref]["variables"])
    return {"schema_version": "1.0", "source": "SIDRA_COMPENDIUM.md", "table_order": order, "tables": tables}


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    registry = parse_compendium(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    tables = registry["tables"]
    by_tier: dict[str, int] = {}
    with_vars = sum(1 for t in tables.values() if t["variables"])
    with_clsf = sum(1 for t in tables.values() if t["classifications"])
    for t in tables.values():
        by_tier[t["tier"]] = by_tier.get(t["tier"], 0) + 1
    print(f"parsed {len(tables)} tables -> {OUTPUT}")
    print(f"  with variables: {with_vars}  with classifications: {with_clsf}")
    print(f"  by tier: {dict(sorted(by_tier.items()))}")


if __name__ == "__main__":
    main()
