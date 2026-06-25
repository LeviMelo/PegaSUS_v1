from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from pegasus.core.hashing import sha256_file
from pegasus.core.schemas import UserIntent
from pegasus.efg.race_bridge import RaceBridgePrior, load_race_bridge_prior
from pegasus.geo.uf import uf_from_datasus_cod6


class RaceBridgeRegistryError(ValueError):
    """Raised when a race bridge registry cannot produce a legal plan."""


@dataclass(frozen=True)
class RaceBridgeRegistryEntry:
    id: str
    status: str
    description: str
    warnings: list[str]
    source_system: str
    source_axis: str
    target_axis: str
    mode: str
    prior_path: Path
    enabled_for_compile: bool
    region_scope: list[str]
    period_start: int | None
    period_end: int | None
    dashboard_policy: str
    prior_hash: str
    registry_path: Path
    registry_hash: str

    @property
    def bridge_id(self) -> str:
        return self.id

    def load_prior(self) -> RaceBridgePrior:
        prior = load_race_bridge_prior(self.prior_path)
        if prior.bridge_id != self.id:
            raise RaceBridgeRegistryError(
                f"Race bridge prior ID mismatch: registry={self.id!r}; prior={prior.bridge_id!r}"
            )
        if prior.mode != self.mode:
            raise RaceBridgeRegistryError(
                f"Race bridge prior mode mismatch: registry={self.mode!r}; prior={prior.mode!r}"
            )
        if prior.source_axis != self.source_axis or prior.target_axis != self.target_axis:
            raise RaceBridgeRegistryError(
                "Race bridge prior axis mismatch: "
                f"registry={self.source_axis}->{self.target_axis}; "
                f"prior={prior.source_axis}->{prior.target_axis}"
            )
        return prior

    def as_manifest(self) -> dict[str, Any]:
        return {
            "bridge_id": self.id,
            "status": self.status,
            "source_system": self.source_system,
            "source_axis": self.source_axis,
            "target_axis": self.target_axis,
            "mode": self.mode,
            "prior_path": str(self.prior_path),
            "prior_hash": self.prior_hash,
            "registry_path": str(self.registry_path),
            "registry_hash": self.registry_hash,
            "enabled_for_compile": self.enabled_for_compile,
            "region_scope": self.region_scope,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "dashboard_policy": self.dashboard_policy,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class RaceBridgePlan:
    status: Literal["not_requested", "planned", "blocked"]
    reason: str | None
    intent_mode: str
    source_system: str
    source_axis: str
    target_axis: str
    municipality_cod6: str | None
    bridge_id: str | None = None
    prior_path: Path | None = None
    prior_hash: str | None = None
    registry_path: Path | None = None
    registry_hash: str | None = None
    dashboard_policy: str | None = None
    warnings: list[str] | None = None

    @property
    def requires_attach(self) -> bool:
        return self.status == "planned" and self.prior_path is not None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "intent_mode": self.intent_mode,
            "source_system": self.source_system,
            "source_axis": self.source_axis,
            "target_axis": self.target_axis,
            "municipality_cod6": self.municipality_cod6,
            "bridge_id": self.bridge_id,
            "prior_path": str(self.prior_path) if self.prior_path is not None else None,
            "prior_hash": self.prior_hash,
            "registry_path": str(self.registry_path) if self.registry_path is not None else None,
            "registry_hash": self.registry_hash,
            "dashboard_policy": self.dashboard_policy,
            "warnings": self.warnings or [],
        }


def _repo_path(path: str | Path, *, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (base / value).resolve()


def _load_registry_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise RaceBridgeRegistryError(f"Race bridge registry is not a mapping: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise RaceBridgeRegistryError(f"Race bridge registry entries is not a list: {path}")
    return payload


def _uf_from_datasus_cod6(municipality_cod6: str | None) -> str | None:
    return uf_from_datasus_cod6(municipality_cod6)


def _entry_from_payload(raw: dict[str, Any], *, registry_path: Path, repo_root: Path) -> RaceBridgeRegistryEntry:
    required = ["id", "status", "description", "warnings", "source_system", "source_axis", "target_axis", "mode", "prior_path"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise RaceBridgeRegistryError(f"Race bridge registry entry missing keys: id={raw.get('id')!r} missing={missing}")
    prior_path = _repo_path(str(raw["prior_path"]), base=repo_root)
    if not prior_path.exists():
        raise RaceBridgeRegistryError(f"Race bridge prior file does not exist: {prior_path}")
    warnings = [str(x) for x in (raw.get("warnings") or [])]
    region_scope = [str(x) for x in (raw.get("region_scope") or ["*"])]
    period_start = raw.get("period_start")
    period_end = raw.get("period_end")
    return RaceBridgeRegistryEntry(
        id=str(raw["id"]),
        status=str(raw["status"]),
        description=str(raw["description"]),
        warnings=warnings,
        source_system=str(raw["source_system"]),
        source_axis=str(raw["source_axis"]),
        target_axis=str(raw["target_axis"]),
        mode=str(raw["mode"]),
        prior_path=prior_path,
        enabled_for_compile=bool(raw.get("enabled_for_compile", False)),
        region_scope=region_scope,
        period_start=int(period_start) if period_start is not None else None,
        period_end=int(period_end) if period_end is not None else None,
        dashboard_policy=str(raw.get("dashboard_policy") or "downgrade_when_sensitive"),
        prior_hash=sha256_file(prior_path),
        registry_path=registry_path,
        registry_hash=sha256_file(registry_path),
    )


def load_race_bridge_registry(
    registry_path: str | Path = "config/registries/race_bridge_priors.yaml",
    *,
    repo_root: str | Path = ".",
) -> list[RaceBridgeRegistryEntry]:
    repo_root = Path(repo_root).resolve()
    registry_path = _repo_path(registry_path, base=repo_root)
    payload = _load_registry_payload(registry_path)
    entries = [_entry_from_payload(raw, registry_path=registry_path, repo_root=repo_root) for raw in payload["entries"]]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            duplicates.add(entry.id)
        seen.add(entry.id)
        entry.load_prior()
    if duplicates:
        raise RaceBridgeRegistryError(f"Duplicate race bridge prior IDs: {sorted(duplicates)}")
    return entries


def select_compile_race_bridge_prior(
    *,
    municipality_cod6: str,
    source_system: str = "SIM-DO",
    source_axis: str = "SIM_ADMIN_RACACOR",
    target_axis: str = "IBGE_SELF_DECLARED_RACE",
    registry_path: str | Path = "config/registries/race_bridge_priors.yaml",
    repo_root: str | Path = ".",
) -> RaceBridgeRegistryEntry:
    entries = load_race_bridge_registry(registry_path, repo_root=repo_root)
    uf = _uf_from_datasus_cod6(municipality_cod6)
    candidates: list[RaceBridgeRegistryEntry] = []
    for entry in entries:
        if not entry.enabled_for_compile:
            continue
        if entry.source_system != source_system:
            continue
        if entry.source_axis != source_axis or entry.target_axis != target_axis:
            continue
        if entry.mode != "fixedC_dynamic_weight":
            continue
        if "*" not in entry.region_scope and uf not in entry.region_scope:
            continue
        candidates.append(entry)
    if not candidates:
        raise RaceBridgeRegistryError(
            "No enabled race bridge prior matched compile request: "
            f"source={source_system} axis={source_axis}->{target_axis} municipality_cod6={municipality_cod6}"
        )
    if len(candidates) > 1:
        ids = [entry.id for entry in candidates]
        raise RaceBridgeRegistryError(f"Ambiguous race bridge prior selection: {ids}")
    return candidates[0]


def resolve_race_bridge_plan(
    *,
    intent: UserIntent,
    municipality_cod6: str,
    registry_path: str | Path = "config/registries/race_bridge_priors.yaml",
    repo_root: str | Path = ".",
) -> RaceBridgePlan:
    mode = str(intent.race_tensor_mode)
    if mode == "decoupled":
        return RaceBridgePlan(
            status="not_requested",
            reason="intent race_tensor_mode is decoupled",
            intent_mode=mode,
            source_system="SIM-DO",
            source_axis="SIM_ADMIN_RACACOR",
            target_axis="IBGE_SELF_DECLARED_RACE",
            municipality_cod6=municipality_cod6,
            warnings=[],
        )
    if mode != "downstream_bridge":
        return RaceBridgePlan(
            status="blocked",
            reason=f"race_tensor_mode={mode!r} requires embedded/population-tensor bridge support not implemented in Slice 4B",
            intent_mode=mode,
            source_system="SIM-DO",
            source_axis="SIM_ADMIN_RACACOR",
            target_axis="IBGE_SELF_DECLARED_RACE",
            municipality_cod6=municipality_cod6,
            warnings=["embedded_race_bridge_mode_blocked"],
        )
    entry = select_compile_race_bridge_prior(
        municipality_cod6=municipality_cod6,
        registry_path=registry_path,
        repo_root=repo_root,
    )
    prior = entry.load_prior()
    return RaceBridgePlan(
        status="planned",
        reason=None,
        intent_mode=mode,
        source_system=entry.source_system,
        source_axis=entry.source_axis,
        target_axis=entry.target_axis,
        municipality_cod6=municipality_cod6,
        bridge_id=entry.id,
        prior_path=entry.prior_path,
        prior_hash=prior.prior_hash,
        registry_path=entry.registry_path,
        registry_hash=entry.registry_hash,
        dashboard_policy=entry.dashboard_policy,
        warnings=entry.warnings,
    )
