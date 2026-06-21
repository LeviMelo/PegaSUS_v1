from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.efg.dag import EFGResult, build_efg
from pegasus.efg.materialization_manifest import load_substrate_bundle_for_efg
from pegasus.output.sim_efg_bundle import write_sim_compiler_bundle
from pegasus.she.substrate import SubstrateBundle


def build_sim_compiler_run(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
    source_mode: str = "fixture_only",
) -> Path:
    run_dir = Path(run_dir)
    source_events_path = Path(sim_events_path)

    if municipality_cod6 is not None:
        filtered_path = run_dir / "Tables" / f"sim_events_mun_{municipality_cod6}.parquet"
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.read_parquet(source_events_path)
        filtered = df.filter(pl.col("mun_residence_cod6") == str(municipality_cod6))
        if filtered.height == 0:
            raise ValueError(f"No SIM events remain after municipality filter {municipality_cod6!r}.")
        filtered.write_parquet(filtered_path)
        source_events_path = filtered_path

    return write_sim_compiler_bundle(
        sim_events_path=source_events_path,
        run_dir=run_dir,
        source_mode=source_mode,
    )


def build_sim_fixture_efg_run(
    *,
    sim_events_path: str | Path,
    run_dir: str | Path,
    municipality_cod6: str | None = None,
) -> Path:
    """Compatibility fixture workflow; canonical compile calls build_sim_compiler_run."""

    return build_sim_compiler_run(
        sim_events_path=sim_events_path,
        run_dir=run_dir,
        municipality_cod6=municipality_cod6,
    )


def build_autonomous_efg(
    *,
    substrate: SubstrateBundle,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
    intent_constraints: dict[str, Any] | None = None,
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> EFGResult:
    """Workflow boundary for the Macro-Slice 19A autonomous EFG core."""

    return build_efg(
        substrate=substrate,
        registry_root=registry_root,
        intent=intent,
        intent_constraints=intent_constraints,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )


def build_autonomous_efg_from_manifest(
    *,
    substrate_manifest: str | Path,
    output_path: str | Path | None = None,
    registry_root: str | Path = "config/registries",
    intent: Any = None,
    intent_constraints: dict[str, Any] | None = None,
    operator_budget: int = 256,
    operator_mode: str = "standard",
) -> EFGResult:
    substrate = load_substrate_bundle_for_efg(substrate_manifest)
    result = build_autonomous_efg(
        substrate=substrate,
        registry_root=registry_root,
        intent=intent,
        intent_constraints=intent_constraints,
        operator_budget=operator_budget,
        operator_mode=operator_mode,
    )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result.as_manifest(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result
