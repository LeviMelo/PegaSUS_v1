import json
from pathlib import Path

import pytest

from pegasus.core.schemas import UserIntent
from pegasus.registries.race_bridge import RaceBridgeRegistryError, load_race_bridge_registry, resolve_race_bridge_plan


def test_race_bridge_registry_loads_and_validates_smoke_prior():
    entries = load_race_bridge_registry("config/registries/demographic/race_bridge_priors.yaml")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.id == "fixedC_sim_admin_to_ibge_selfdeclared_smoke_v1"
    assert entry.source_axis == "SIM_ADMIN_RACACOR"
    assert entry.target_axis == "IBGE_SELF_DECLARED_RACE"
    assert entry.prior_path.exists()
    prior = entry.load_prior()
    assert prior.prior_hash == entry.prior_hash


def test_decoupled_intent_does_not_request_bridge():
    payload = json.loads(Path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))
    assert "race_bridge_policy" not in payload
    intent = UserIntent.model_validate(payload)
    plan = resolve_race_bridge_plan(intent=intent, municipality_cod6="270430")
    assert plan.status == "not_requested"
    assert plan.requires_attach is False


def test_downstream_bridge_intent_resolves_registry_prior():
    payload = json.loads(Path("config/intents/alagoas_smoke_race_bridge.json").read_text(encoding="utf-8"))
    assert "race_bridge_policy" not in payload
    intent = UserIntent.model_validate(payload)
    assert intent.race_tensor_mode == "downstream_bridge"
    plan = resolve_race_bridge_plan(intent=intent, municipality_cod6="270430")
    assert plan.status == "planned"
    assert plan.requires_attach is True
    assert plan.prior_path is not None
    assert plan.prior_path.exists()
    assert plan.prior_hash is not None


def test_embedded_mode_resolves_the_same_registry_prior_for_the_population_tensor():
    """MSD §2.8.5/§2.8.6: "embedded_*" feeds the same Bridge_R prior directly into
    the population tensor's birth/death race stratification (pegasus.sidra.
    population_cube.build) instead of attaching a standalone EFG field."""
    payload = json.loads(Path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))
    payload["race_tensor_mode"] = "embedded_fixedC"
    intent = UserIntent.model_validate(payload)
    plan = resolve_race_bridge_plan(intent=intent, municipality_cod6="270430")
    assert plan.status == "embedded"
    assert plan.requires_attach is False
    assert plan.prior_path is not None
    assert plan.prior_path.exists()
    assert plan.prior_hash is not None
