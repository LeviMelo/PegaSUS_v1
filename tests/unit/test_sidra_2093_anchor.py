"""POP-01 — SIDRA table 2093 registered as the 2000/2010 census anchor (MSD-III §II.4)."""

from __future__ import annotations

import json
from pathlib import Path


def test_table_2093_registered_as_census_composition_anchor() -> None:
    comp = json.loads(Path("config/registries/sidra/sidra_compendium.json").read_text(encoding="utf-8"))
    assert "2093" in comp["tables"] and "2093" in comp["table_order"]

    t = comp["tables"]["2093"]
    axes = {c["axis"] for c in t["classifications"]}
    # the census race × sex × situation × age composition (§II.4)
    assert {"race_color", "sex", "household_situation", "age_group"} <= axes
    assert "2000" in t["periods"] and "2010" in t["periods"]     # the third-census window 9606 lacks
    assert any(v["variable_id"] == "93" for v in t["variables"]) # resident population (Pessoas)

    seed = Path("config/registries/sidra/sidra_table_seed.jsonl").read_text(encoding="utf-8")
    assert '"2093"' in seed and "anchor" in seed.lower()
