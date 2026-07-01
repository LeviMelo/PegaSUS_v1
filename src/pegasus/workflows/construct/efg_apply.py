
from __future__ import annotations

from pathlib import Path
from typing import Any

from pegasus.efg.promotion_apply import apply_efg_promotion_plan_to_run


def run_apply_efg_promotion_plan(
    *,
    run_dir: str | Path,
    promotion_plan: str | Path | None = None,
    validate: bool = True,
    bundle=None,
) -> dict[str, Any]:
    result = apply_efg_promotion_plan_to_run(
        run_dir=run_dir,
        promotion_plan=promotion_plan,
        validate=validate,
        bundle=bundle,
    )
    return result.as_manifest()
