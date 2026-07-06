"""REG-07 registry status-admission contract + decode-wiring integrity gate (MSD-II §II.1).

Closes a latent coherence bug the architectural review surfaced: `registries.generic.active_entries`
used a strict `status == "active"` test while `registries.semantic.active_entries` used the graded-
active union, so a naive loader unification to the strict form would silently drop graded-active
entries — concretely `config/registries/fields/bridge_grammars.yaml`'s `active_artifact_required`
and `active_warning` EFG bridge-grammar operators. The two filters agreed on today's generic-fed
data (all bare `active`), so NO test caught the divergence. Both stacks now share a single
`is_active` rule; these tests pin it as an executable contract and prove the decode wiring resolves
(the thing a green suite otherwise says nothing about).
"""

from __future__ import annotations

from pegasus.registries import generic, semantic, validators
from pegasus.registries.generic import is_active
from pegasus.registries.semantic import active_entries as semantic_active


def test_status_admission_rule_is_the_graded_active_union() -> None:
    for status in (
        "active", "active_warning", "active_reduced", "active_observer",
        "active_artifact_required", "active_approximation", "active_sparse_warning",
        "active_small_scale", "active_gpu", "stable", "planned_contract", "experimental",
    ):
        assert is_active(status), f"{status!r} must count as active"
    for status in ("deferred", "legacy_identity", "baseline", "deprecated", "scaffold", "", None):
        assert not is_active(status), f"{status!r} must NOT count as active"


def test_both_accessor_stacks_share_one_admission_rule() -> None:
    # The fix: generic (Stack A, RegistryEntry) and semantic (Stack B, dict) resolve the SAME
    # is_active object — re-inlining a status test in either is the divergence that caused the bug.
    assert semantic.is_active is generic.is_active


def test_bridge_grammars_graded_active_operators_survive_admission() -> None:
    # The concrete latent-bug case: these two operators are admitted only because is_active accepts
    # the graded-active variants; a strict `== "active"` filter drops them.
    entries = semantic_active("fields/bridge_grammars.yaml")
    statuses = {str(e.get("status")) for e in entries}
    assert "active_artifact_required" in statuses
    assert "active_warning" in statuses
    strict_only = [e for e in entries if str(e.get("status")) == "active"]
    assert len(strict_only) < len(entries), "the strict filter would have dropped graded-active operators"


def test_registry_decode_wiring_resolves_end_to_end() -> None:
    # Executable-authority check (§II.1): every decoder/parser/composite_decoder a registry declares
    # resolves through the single resolver, vocabularies agree, routes name their callable. This is
    # the decode-integrity gate the REG-07 loader work must keep green.
    errors = validators.validate_registry_authority("config/registries")
    assert errors == [], f"registry decode authority broken: {errors}"
