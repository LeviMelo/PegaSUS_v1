from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_hsic_dashboard_cards(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_hsic_dashboard_cards(path: str | Path) -> dict[str, Any]:
    payload = load_hsic_dashboard_cards(path)
    cards = payload.get("cards") or []
    return {
        "schema_version": payload.get("schema_version"),
        "slice": payload.get("slice"),
        "read_only": bool(payload.get("read_only")),
        "card_count": len(cards),
        "top_hypothesis_id": cards[0].get("hypothesis_id") if cards else None,
        "source_manifest_path": payload.get("source_manifest_path"),
        "ranking_path": payload.get("ranking_path"),
    }


__all__ = ["load_hsic_dashboard_cards", "inspect_hsic_dashboard_cards"]
