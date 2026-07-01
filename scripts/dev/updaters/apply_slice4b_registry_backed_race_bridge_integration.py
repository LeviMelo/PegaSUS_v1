from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path.cwd()

CREATE_OR_REPLACE = [
    "config/registries/demographic/race_bridge_priors.yaml",
    "config/intents/alagoas_smoke_race_bridge.json",
    "src/pegasus/registries/race_bridge.py",
    "src/pegasus/workflows/race_bridge.py",
    "src/pegasus/output/race_bridge_attach.py",
    "src/pegasus/output/validate.py",
    "tests/unit/test_race_bridge_registry.py",
    "tests/unit/test_race_bridge_attach_idempotency.py",
    "tests/integration/test_slice4b_compile_race_bridge_registry.py",
    "scripts/dev/audits/audit_slice4b_compile_race_bridge.py",
]

PATCH = [
    "src/pegasus/workflows/compile.py",
    "src/pegasus/cli.py",
]

FORBIDDEN_FAILED_4B = [
    "src/pegasus/she/race_bridge_policy.py",
    "src/pegasus/registries/race_bridge_registry.py",
]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = rel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def preflight() -> None:
    required = [
        "pyproject.toml",
        "config/intents/alagoas_smoke.json",
        "config/registries/registry_manifest.yaml",
        "config/registries/demographic/race_bridge_priors.yaml",
        "src/pegasus/core/schemas.py",
        "src/pegasus/cli.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/workflows/race_bridge.py",
        "src/pegasus/output/race_bridge_attach.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/output/maternal_child_compile_attach.py",
        "src/pegasus/efg/race_bridge.py",
        "tests/fixtures/race_bridge/fixedC_valid.json",
        "tests/fixtures/race_bridge/fixedC_invalid_empty.json",
        "tests/fixtures/race_bridge/fixedC_invalid_rowsum.json",
        "tests/fixtures/datasus/sim_do_fixture.csv",
        "tests/fixtures/datasus/sinasc_fixture.csv",
        "scripts/dev/audits/audit_slice4a_race_bridge.py",
        "scripts/dev/audits/audit_slice3b_compile_maternal_child_linkage.py",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 4B-redo preflight failed; missing expected files: {missing}")

    stale = [p for p in FORBIDDEN_FAILED_4B if rel(p).exists()]
    if stale:
        raise RuntimeError(
            "Slice 4B-redo preflight found stale failed-4B files. Remove them before proceeding: "
            + repr(stale)
        )

    schemas = read("src/pegasus/core/schemas.py")
    if "race_bridge_policy" in schemas:
        raise RuntimeError("Slice 4B-redo preflight failed: UserIntent appears polluted with race_bridge_policy.")
    for marker in ["extra=\"forbid\"", "race_tensor_mode", "downstream_bridge"]:
        if marker not in schemas:
            raise RuntimeError(f"Slice 4B-redo preflight failed; strict UserIntent marker missing: {marker}")

    baseline_intent = json.loads(read("config/intents/alagoas_smoke.json"))
    if "race_bridge_policy" in baseline_intent:
        raise RuntimeError("Slice 4B-redo preflight failed: baseline alagoas_smoke.json contains forbidden race_bridge_policy.")

    compile_py = read("src/pegasus/workflows/compile.py")
    for marker in [
        "def run_compile",
        "attach_maternal_child_compile_fields",
        "run_datasus_normalize_sinasc",
        "race_tensor_mode",
        "telemetry.stage(\"she_build\")",
    ]:
        if marker not in compile_py:
            raise RuntimeError(f"Slice 4B-redo preflight failed; Slice 3B compile marker not found: {marker}")

    race_workflow = read("src/pegasus/workflows/race_bridge.py")
    if "run_attach_race_bridge" not in race_workflow:
        raise RuntimeError("Slice 4B-redo preflight failed: Slice 4A race_bridge workflow missing.")

    validator = read("src/pegasus/output/validate.py")
    if "Q_tensor does not cover all V_fields" not in validator or "COMPILE_TELEMETRY_STAGES" not in validator:
        raise RuntimeError("Slice 4B-redo preflight failed: Slice 2E hardened validator not detected.")


def write_race_bridge_registry() -> None:
    write("src/pegasus/registries/race_bridge.py", r'''
    from __future__ import annotations

    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Literal

    import yaml

    from pegasus.core.hashing import sha256_file
    from pegasus.core.schemas import UserIntent
    from pegasus.efg.race_bridge import RaceBridgePrior, load_race_bridge_prior


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
        if not municipality_cod6:
            return None
        prefix = str(municipality_cod6)[:2]
        return {"27": "AL"}.get(prefix)


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
        registry_path: str | Path = "config/registries/demographic/race_bridge_priors.yaml",
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
        registry_path: str | Path = "config/registries/demographic/race_bridge_priors.yaml",
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
        registry_path: str | Path = "config/registries/demographic/race_bridge_priors.yaml",
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
    ''')


def write_race_bridge_workflow() -> None:
    write("src/pegasus/workflows/race_bridge.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    from pegasus.core.schemas import UserIntent
    from pegasus.efg.race_bridge import (
        fixedc_dynamic_weight_bridge,
        load_race_bridge_prior,
        summarize_sim_admin_race_counts,
    )
    from pegasus.output.race_bridge_attach import attach_race_bridge_to_run
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.race_bridge import resolve_race_bridge_plan


    def _load_intent(path: str | Path) -> UserIntent:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return UserIntent.model_validate(payload)


    def run_validate_race_bridge_prior(*, bridge_prior_path: str | Path) -> dict[str, Any]:
        prior = load_race_bridge_prior(bridge_prior_path)
        return {
            "bridge_id": prior.bridge_id,
            "mode": prior.mode,
            "prior_hash": prior.prior_hash,
            "source_axis": prior.source_axis,
            "target_axis": prior.target_axis,
            "source_categories": prior.source_categories,
            "target_categories": prior.target_categories,
            "sensitivity_width": prior.sensitivity_width,
        }


    def run_plan_race_bridge(
        *,
        sim_events_path: str | Path,
        bridge_prior_path: str | Path | None = None,
        intent_path: str | Path | None = None,
        registry_path: str | Path = "config/registries/demographic/race_bridge_priors.yaml",
        municipality_cod6: str | None = None,
        year: int | None = None,
    ) -> dict[str, Any]:
        if bridge_prior_path is None:
            if intent_path is None or municipality_cod6 is None:
                raise ValueError("Bridge prior path is required unless intent_path and municipality_cod6 are supplied for registry planning.")
            intent = _load_intent(intent_path)
            plan = resolve_race_bridge_plan(intent=intent, municipality_cod6=municipality_cod6, registry_path=registry_path)
            if plan.status != "planned" or plan.prior_path is None:
                raise ValueError(f"Race bridge plan is not attachable: {plan.as_manifest()}")
            bridge_prior_path = plan.prior_path
            plan_payload = plan.as_manifest()
        else:
            plan_payload = None
        prior = load_race_bridge_prior(bridge_prior_path)
        counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6, year=year)
        posterior = fixedc_dynamic_weight_bridge(counts, prior)
        summary = {
            "bridge_id": prior.bridge_id,
            "prior_hash": prior.prior_hash,
            "support": counts.support,
            "raw_admin_counts": counts.raw_admin_counts,
            "missing_count": counts.missing_count,
            "missing_share": counts.missing_share,
            "posterior_counts": posterior.posterior_counts,
            "lower_counts": posterior.lower_counts,
            "upper_counts": posterior.upper_counts,
            "sensitivity_width": posterior.sensitivity_width,
            "race_bridge_cv": posterior.race_bridge_cv,
            "will_downgrade_dashboard_safety": posterior.sensitivity_width > 0.05 or posterior.missing_share > 0,
            "plan": plan_payload,
        }
        return {"summary": summary, "summary_json": json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2)}


    def run_attach_race_bridge(
        *,
        run_dir: str | Path,
        sim_events_path: str | Path,
        bridge_prior_path: str | Path,
        municipality_cod6: str | None = None,
    ) -> dict[str, Any]:
        output = attach_race_bridge_to_run(
            run_dir=run_dir,
            sim_events_path=sim_events_path,
            bridge_prior_path=bridge_prior_path,
            municipality_cod6=municipality_cod6,
        )
        validation = validate_output_bundle(run_dir=str(output))
        run_config = {}
        config_path = Path(output) / "RunConfig.json"
        if config_path.exists():
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        return {"run_dir": output, "validation": validation, "race_bridge": run_config.get("race_bridge")}
    ''')


def write_race_bridge_attach() -> None:
    write("src/pegasus/output/race_bridge_attach.py", r'''
    from __future__ import annotations

    import json
    import platform
    import sys
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import sha256_file, sha256_text
    from pegasus.efg.race_bridge import (
        ADMIN_RACE_LABELS,
        RaceBridgePosterior,
        fixedc_dynamic_weight_bridge,
        load_race_bridge_prior,
        summarize_sim_admin_race_counts,
    )
    from pegasus.output.validate import validate_output_bundle

    RACE_BRIDGE_FIELD_IDS = {
        *(f"SIMRaceAdminRawCount_{code}" for code in ADMIN_RACE_LABELS),
        "SIMRaceBridgeMissingRaceObserver",
        *(f"SIMRaceBridgePosteriorCount_{target}" for target in ["branca", "preta", "amarela", "parda", "indigena"]),
    }
    RACE_BRIDGE_WARNING_IDS = {
        "race_bridge_admin_axis_preserved",
        "race_bridge_missing_preserved",
        "race_bridge_posterior_sensitivity",
    }


    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


    def _append_rows(path: Path, rows: list[dict[str, Any]], *, id_column: str | None = None, remove_values: set[str] | None = None) -> None:
        existing = pl.read_parquet(path)
        if id_column and id_column in existing.columns:
            ids = set(remove_values or set())
            ids.update(str(row[id_column]) for row in rows if row.get(id_column) is not None)
            if ids:
                existing = existing.filter(~pl.col(id_column).cast(pl.Utf8).is_in(sorted(ids)))
        if not rows:
            existing.write_parquet(path)
            return
        new = pl.DataFrame(rows)
        for column, dtype in existing.schema.items():
            if column not in new.columns:
                new = new.with_columns(pl.lit(None).cast(dtype).alias(column))
            else:
                new = new.with_columns(pl.col(column).cast(dtype, strict=False))
        for column in new.columns:
            if column not in existing.columns:
                new = new.drop(column)
        new = new.select(existing.columns)
        pl.concat([existing, new], how="vertical").write_parquet(path)


    def _field_row(
        *,
        field_id: str,
        name: str,
        kind: str,
        carrier: str,
        unit: str,
        aggregation: str,
        support: dict[str, Any],
        axes: dict[str, Any],
        operator: str,
        provenance: list[str],
        warnings: list[str],
        state: str,
        dashboard_safe: bool,
        path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "field_id": field_id,
            "name": name,
            "kind": kind,
            "carrier": carrier,
            "unit": unit,
            "aggregation": aggregation,
            "role": _json(["observer", "demographic", "race_bridge"]),
            "source": _json(["SIM-DO", "Bridge_R"]),
            "support_json": _json(support),
            "axes_json": _json(axes),
            "operator": operator,
            "provenance": _json(provenance),
            "state": state,
            "dashboard_safe": "True" if dashboard_safe else "False",
            "warnings": _json(warnings),
            "lineage_hash": sha256_text(_json({"field_id": field_id, "support": support, "axes": axes, "operator": operator})),
            "registry_hash": axes.get("prior_hash") or "race_bridge_registry_unset",
            "materialization_state": "materialized",
            "path": path,
        }


    def _q_row(
        *,
        field: dict[str, Any],
        n_events: float | int | None,
        warnings: list[str],
        missingness: float,
        denom_fragility: float,
        race_axis_source: str | None = None,
        race_axis_target: str | None = None,
        missing_race_share: float | None = None,
        race_bridge_cv: float | None = None,
        sensitivity_width: float | None = None,
        bridge_mode: str | None = None,
    ) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "n_events": float(n_events) if n_events is not None else None,
            "n_denom": None,
            "n_eff": float(n_events) if n_events is not None else None,
            "cov_S": None,
            "cov_T": None,
            "missingness": float(missingness),
            "zero_inflation": 0.0,
            "denom_fragility": float(denom_fragility),
            "cv": race_bridge_cv,
            "moran_i": None,
            "temporal_roughness": None,
            "spatial_entropy": None,
            "provenance_risk": 0.65 if field["state"] != "verified" else 0.25,
            "race_axis_source": race_axis_source,
            "race_axis_target": race_axis_target,
            "missing_race_share": missing_race_share,
            "emission_prior_strength": None,
            "race_bridge_cv": race_bridge_cv,
            "sensitivity_width": sensitivity_width,
            "bridge_mode": bridge_mode,
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "warnings": _json(warnings),
            "computed_at": _now(),
            "q_schema_version": "1.0",
        }


    def _vd_row(*, field: dict[str, Any], definition: str, estimand: str, warning: str) -> dict[str, Any]:
        return {
            "field_id": field["field_id"],
            "display_name": field["name"],
            "technical_name": field["field_id"],
            "definition": definition,
            "estimand_label": estimand,
            "source_systems": _json(["SIM-DO", "Bridge_R"]),
            "carrier": field["carrier"],
            "unit": field["unit"],
            "support_description": "SIM death-event support filtered by municipality/year; raw administrative race preserved separately from bridge posterior target axis.",
            "axis_description": "Bridge_R maps SIM administrative race/color observer counts to a target self-declared race axis under a validated fixed-C emission prior.",
            "provenance_description": "Derived from normalized SIM events and race bridge prior; posterior counts are sensitivity-bounded observer fields.",
            "state": field["state"],
            "dashboard_safe": field["dashboard_safe"],
            "interpretation_warning": warning,
        }


    def _edge(edge_id: str, parent: str, child: str, operator: str, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": edge_id,
            "parent_field_id": parent,
            "child_field_id": child,
            "operator": operator,
            "operator_params_json": _json(params),
            "registry_versions_json": _json({"race_bridge": params.get("emission_matrix_registry_version")}),
            "created_at": _now(),
        }


    def _warning_rows(posterior: RaceBridgePosterior) -> list[dict[str, Any]]:
        return [
            {
                "warning_id": "race_bridge_admin_axis_preserved",
                "field_id": "run",
                "severity": "warning",
                "message": "SIM administrative race/color is preserved as a raw declaration-process axis and is not overwritten by IBGE self-declared race.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_missing_preserved",
                "field_id": "SIMRaceBridgeMissingRaceObserver",
                "severity": "warning",
                "message": f"Missing/invalid administrative race is preserved as an observer count: n={posterior.missing_count}, share={posterior.missing_share:.6f}.",
                "created_at": _now(),
                "inherited_from": None,
            },
            {
                "warning_id": "race_bridge_posterior_sensitivity",
                "field_id": "run",
                "severity": "downgrade",
                "message": f"Bridge_R posterior fields carry sensitivity interval width={posterior.sensitivity_width:.6f} and remain dashboard-downgraded unless calibrated.",
                "created_at": _now(),
                "inherited_from": None,
            },
        ]


    def _build_rows(posterior: RaceBridgePosterior, *, sim_events_path: Path, bridge_prior_path: Path) -> dict[str, list[dict[str, Any]]]:
        support = {
            **posterior.support,
            "n_missing_race": posterior.missing_count,
            "missing_race_share": posterior.missing_share,
        }
        metadata = posterior.metadata()
        common_warnings = [
            "race_bridge_admin_axis_preserved",
            "race_bridge_bayesian_ecological_warning",
            "race_bridge_sensitivity_metadata",
        ]
        fields: list[dict[str, Any]] = []
        q_rows: list[dict[str, Any]] = []
        vd_rows: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        table_path = "Tables/race_bridge_summary.parquet"

        for code, label in ADMIN_RACE_LABELS.items():
            field = _field_row(
                field_id=f"SIMRaceAdminRawCount_{code}",
                name=f"SIM Raw Administrative Race Count {label}",
                kind="extensive_measure",
                carrier="death_event",
                unit="deaths",
                aggregation="additive_count",
                support={**support, "race_color_admin": code, "n_events": posterior.raw_admin_counts.get(code, 0)},
                axes={"race_axis": posterior.prior.source_axis, "race_admin_code": code, "race_admin_label": label, "raw_admin_preserved": True},
                operator="raw_admin_race_count",
                provenance=["SIM-DO", str(sim_events_path)],
                warnings=["race_bridge_admin_axis_preserved"],
                state="verified",
                dashboard_safe=True,
                path=table_path,
            )
            fields.append(field)
            q_rows.append(_q_row(
                field=field,
                n_events=posterior.raw_admin_counts.get(code, 0),
                warnings=["race_bridge_admin_axis_preserved"],
                missingness=0.0,
                denom_fragility=0.0,
                race_axis_source=posterior.prior.source_axis,
            ))
            vd_rows.append(_vd_row(field=field, definition="Raw SIM administrative race/color death count preserved before Bridge_R.", estimand="raw_administrative_race_count", warning="Raw administrative race axis; not equivalent to IBGE self-declared race."))

        missing_field = _field_row(
            field_id="SIMRaceBridgeMissingRaceObserver",
            name="SIM Missing Administrative Race Observer",
            kind="observer_proxy",
            carrier="death_event",
            unit="deaths",
            aggregation="missingness_count",
            support={**support, "n_events": posterior.missing_count},
            axes={"race_axis": posterior.prior.source_axis, "missing_category_preserved": True, **metadata},
            operator="missing_race_observer",
            provenance=["SIM-DO", str(sim_events_path)],
            warnings=common_warnings,
            state="fragile" if posterior.missing_count else "verified",
            dashboard_safe=False if posterior.missing_count else True,
            path=table_path,
        )
        fields.append(missing_field)
        q_rows.append(_q_row(
            field=missing_field,
            n_events=posterior.missing_count,
            warnings=common_warnings,
            missingness=posterior.missing_share,
            denom_fragility=posterior.sensitivity_width,
            race_axis_source=posterior.prior.source_axis,
            missing_race_share=posterior.missing_share,
            race_bridge_cv=posterior.race_bridge_cv,
            sensitivity_width=posterior.sensitivity_width,
            bridge_mode=posterior.prior.mode,
        ))
        vd_rows.append(_vd_row(field=missing_field, definition="Observer field for SIM records with missing, ignored, sentinel, or invalid administrative race/color.", estimand="missing_race_observer", warning="Missing race is preserved and is not imputed into posterior target categories."))

        for target, value in posterior.posterior_counts.items():
            field_id = f"SIMRaceBridgePosteriorCount_{target}"
            axes = {
                "race_axis": posterior.prior.target_axis,
                "race_target_category": target,
                "raw_admin_fields": [f"SIMRaceAdminRawCount_{code}" for code in posterior.raw_admin_counts],
                **metadata,
                "lower_count": posterior.lower_counts[target],
                "upper_count": posterior.upper_counts[target],
            }
            state = "fragile" if posterior.sensitivity_width > 0.05 or posterior.missing_share > 0 else "verified"
            field = _field_row(
                field_id=field_id,
                name=f"SIM Race Bridge Posterior Count {target}",
                kind="bridge_module",
                carrier="death_event",
                unit="deaths_posterior",
                aggregation="posterior_count_from_fixedC",
                support={**support, "n_events": value},
                axes=axes,
                operator="Bridge_R_fixedC_dynamic_weight",
                provenance=["SIM-DO", "Bridge_R", str(sim_events_path), str(bridge_prior_path)],
                warnings=common_warnings,
                state=state,
                dashboard_safe=False if state != "verified" else True,
                path=table_path,
            )
            fields.append(field)
            q_rows.append(_q_row(
                field=field,
                n_events=value,
                warnings=common_warnings,
                missingness=posterior.missing_share,
                denom_fragility=posterior.sensitivity_width,
                race_axis_source=posterior.prior.source_axis,
                race_axis_target=posterior.prior.target_axis,
                missing_race_share=posterior.missing_share,
                race_bridge_cv=posterior.race_bridge_cv,
                sensitivity_width=posterior.sensitivity_width,
                bridge_mode=posterior.prior.mode,
            ))
            vd_rows.append(_vd_row(field=field, definition="Bridge_R posterior target-axis death count from raw SIM administrative race counts and a validated fixed-C emission prior.", estimand="race_bridge_posterior_count", warning="Bridge-derived observer field with sensitivity interval; not a direct measurement."))
            for source_code in posterior.raw_admin_counts:
                edges.append(_edge(
                    f"edge_SIMRaceAdminRawCount_{source_code}_to_{field_id}",
                    f"SIMRaceAdminRawCount_{source_code}",
                    field_id,
                    "Bridge_R_fixedC_dynamic_weight",
                    metadata,
                ))
        return {"fields": fields, "q_rows": q_rows, "vd_rows": vd_rows, "edges": edges, "warnings": _warning_rows(posterior)}


    def attach_race_bridge_to_run(
        *,
        run_dir: str | Path,
        sim_events_path: str | Path,
        bridge_prior_path: str | Path,
        municipality_cod6: str | None = None,
    ) -> Path:
        run_dir = Path(run_dir)
        sim_events_path = Path(sim_events_path)
        bridge_prior_path = Path(bridge_prior_path)
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
        if not sim_events_path.exists():
            raise FileNotFoundError(f"SIM events not found: {sim_events_path}")
        if not bridge_prior_path.exists():
            raise FileNotFoundError(f"Bridge prior not found: {bridge_prior_path}")

        prior = load_race_bridge_prior(bridge_prior_path)
        counts = summarize_sim_admin_race_counts(sim_events_path, municipality_cod6=municipality_cod6)
        posterior = fixedc_dynamic_weight_bridge(counts, prior)
        rows = _build_rows(posterior, sim_events_path=sim_events_path, bridge_prior_path=bridge_prior_path)

        summary_rows = []
        for target in prior.target_categories:
            summary_rows.append({
                "race_target_category": target,
                "posterior_count": posterior.posterior_counts[target],
                "lower_count": posterior.lower_counts[target],
                "upper_count": posterior.upper_counts[target],
                "sensitivity_width": posterior.sensitivity_width,
                "race_bridge_cv": posterior.race_bridge_cv,
                "missing_race_share": posterior.missing_share,
                "prior_hash": prior.prior_hash,
                "bridge_id": prior.bridge_id,
            })
        summary_path = run_dir / "Tables" / "race_bridge_summary.parquet"
        pl.DataFrame(summary_rows).write_parquet(summary_path)

        _append_rows(run_dir / "V_fields.parquet", rows["fields"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
        _append_rows(run_dir / "Q_tensor.parquet", rows["q_rows"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
        _append_rows(run_dir / "VariableDictionary.parquet", rows["vd_rows"], id_column="field_id", remove_values=RACE_BRIDGE_FIELD_IDS)
        _append_rows(run_dir / "E_DAG.parquet", rows["edges"], id_column="edge_id")
        _append_rows(run_dir / "Warnings.parquet", rows["warnings"], id_column="warning_id", remove_values=RACE_BRIDGE_WARNING_IDS)

        bridge_manifest = {
            "bridge_id": prior.bridge_id,
            "mode": prior.mode,
            "prior_hash": prior.prior_hash,
            "prior_path": str(bridge_prior_path),
            "source_axis": prior.source_axis,
            "target_axis": prior.target_axis,
            "missing_race_share": posterior.missing_share,
            "sensitivity_width": posterior.sensitivity_width,
            "race_bridge_cv": posterior.race_bridge_cv,
            "raw_admin_counts_preserved": True,
            "missing_category_preserved": True,
            "attach_stage": "race_bridge",
            "posterior_field_ids": [f"SIMRaceBridgePosteriorCount_{target}" for target in prior.target_categories],
            "raw_admin_field_ids": [f"SIMRaceAdminRawCount_{code}" for code in ADMIN_RACE_LABELS],
        }
        source_updates = {
            "race_bridge_prior": sha256_file(bridge_prior_path),
            "race_bridge_summary": sha256_file(summary_path),
        }

        run_config_path = run_dir / "RunConfig.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
        run_config["race_bridge"] = bridge_manifest
        run_config.setdefault("source_hashes", {}).update(source_updates)
        run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        manifest_path = run_dir / "ReproducibilityManifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.setdefault("source_hashes", {}).update(source_updates)
            manifest["race_bridge"] = bridge_manifest
            manifest.setdefault("environment", {}).update({"python": sys.version.split()[0], "platform": platform.platform()})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        p_path = run_dir / "P_vector.json"
        if p_path.exists():
            p = json.loads(p_path.read_text(encoding="utf-8"))
            if isinstance(p, dict):
                p["race_bridge"] = bridge_manifest
                p.setdefault("source_hashes", {}).update(source_updates)
                p_path.write_text(json.dumps(p, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

        validation = validate_output_bundle(run_dir=str(run_dir))
        if not validation.ok:
            raise RuntimeError("Race bridge attachment produced invalid output bundle: " + "; ".join(validation.errors))
        return run_dir
    ''')


def write_validator() -> None:
    # Replace with the Slice 2E hardened validator plus a narrow, bridge-activated metadata contract.
    write("src/pegasus/output/validate.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import pyarrow.parquet as pq

    from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES, TERMINAL_STAGE_STATUSES
    from pegasus.output.schemas import OUTPUT_BUNDLE_FILES, OutputSchemaRegistry, OutputValidationResult

    REQUIRED_V_FIELDS_COLUMNS = {"field_id", "name", "kind", "carrier", "unit", "aggregation", "role", "source", "support_json", "axes_json", "operator", "provenance", "state", "dashboard_safe", "warnings", "lineage_hash", "registry_hash", "materialization_state", "path"}
    REQUIRED_E_DAG_COLUMNS = {"edge_id", "parent_field_id", "child_field_id", "operator", "operator_params_json", "registry_versions_json", "created_at"}
    REQUIRED_Q_TENSOR_COLUMNS = {"field_id", "n_events", "n_denom", "n_eff", "cov_S", "cov_T", "missingness", "zero_inflation", "denom_fragility", "provenance_risk", "state", "dashboard_safe", "warnings", "computed_at", "q_schema_version"}
    OPTIONAL_RACE_Q_COLUMNS = {"race_axis_source", "race_axis_target", "missing_race_share", "race_bridge_cv", "sensitivity_width", "bridge_mode"}
    REQUIRED_VARIABLE_DICTIONARY_COLUMNS = {"field_id", "display_name", "technical_name", "definition", "estimand_label", "source_systems", "carrier", "unit", "support_description", "axis_description", "provenance_description", "state", "dashboard_safe", "interpretation_warning"}
    FIELD_REFERENCE_COLUMNS = {"field_id", "parent_field_id", "child_field_id", "outcome_field_id", "covariate_field_id", "residual_field_id"}
    RACE_BRIDGE_POSTERIOR_KEYS = {"numerator_axis_source", "denominator_axis_target", "bridge_operator", "emission_matrix_registry_version", "bridge_mode", "missing_race_share", "race_bridge_cv", "sensitivity_width", "race_axis_warning", "bayesian_ecological_bridge_warning", "prior_hash", "lower_count", "upper_count"}
    RUN_CONFIG_RACE_BRIDGE_KEYS = {"bridge_id", "mode", "prior_hash", "source_axis", "target_axis", "missing_race_share", "sensitivity_width", "race_bridge_cv", "raw_admin_counts_preserved", "missing_category_preserved", "attach_stage"}


    def _read(path: Path):
        return pq.read_table(path)


    def _column_values(table, column: str) -> list[Any]:
        if column not in table.column_names:
            return []
        return table.column(column).to_pylist()


    def _nonnull(values: list[Any]) -> set[Any]:
        return {value for value in values if value is not None and value != ""}


    def _load_json_file(path: Path, *, errors: list[str], name: str) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid {name}: {exc}")
            return None


    def _load_json_cell(value: Any, *, errors: list[str], context: str) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except Exception as exc:
            errors.append(f"invalid JSON cell at {context}: {exc}")
            return None


    def _require_columns(*, table_name: str, actual: set[str], required: set[str], errors: list[str]) -> None:
        missing = sorted(required - actual)
        if missing:
            errors.append(f"{table_name} missing required columns: {missing}")


    def _validate_first_class_keys(root: Path, schema_registry: OutputSchemaRegistry, errors: list[str]) -> None:
        expected_names = {OUTPUT_BUNDLE_FILES[key] for key in schema_registry.required_keys}
        found_names = {p.name for p in root.iterdir()}
        for name in sorted(expected_names - found_names):
            errors.append(f"missing first-class artifact: {name}")
        for name in sorted(found_names - expected_names):
            errors.append(f"extra first-class artifact: {name}")
        for key, name in OUTPUT_BUNDLE_FILES.items():
            if key not in schema_registry.required_keys:
                continue
            path = root / name
            if not path.exists():
                continue
            if key in {"Tables", "Maps"}:
                if not path.is_dir():
                    errors.append(f"first-class artifact is not a directory: {name}")
            elif not path.is_file():
                errors.append(f"first-class artifact is not a file: {name}")


    def _validate_manifest_and_config(*, root: Path, errors: list[str], warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        user_intent = _load_json_file(root / "UserIntent.json", errors=errors, name="UserIntent.json")
        run_config = _load_json_file(root / "RunConfig.json", errors=errors, name="RunConfig.json")
        manifest = _load_json_file(root / "ReproducibilityManifest.json", errors=errors, name="ReproducibilityManifest.json")
        p_vector = _load_json_file(root / "P_vector.json", errors=errors, name="P_vector.json")
        if not isinstance(user_intent, dict):
            errors.append("UserIntent.json is not a frozen JSON object")
            user_intent = {}
        if not isinstance(run_config, dict):
            errors.append("RunConfig.json is not a frozen JSON object")
            run_config = {}
        if not isinstance(manifest, dict):
            errors.append("ReproducibilityManifest.json is not a JSON object")
            manifest = {}
        if not isinstance(p_vector, (dict, list)):
            errors.append("P_vector.json must be a JSON object or list")
        compile_mode = run_config.get("compile_mode") or manifest.get("compile_mode")
        is_compile_run = bool(compile_mode)
        source_hashes = manifest.get("source_hashes")
        registry_hashes = manifest.get("registry_hashes")
        if is_compile_run:
            if not isinstance(source_hashes, dict) or not source_hashes:
                errors.append("ReproducibilityManifest.json missing nonempty source_hashes for compile run")
            if not isinstance(registry_hashes, dict) or not registry_hashes:
                errors.append("ReproducibilityManifest.json missing nonempty registry_hashes for compile run")
            if not isinstance(run_config.get("source_hashes"), dict) or not run_config.get("source_hashes"):
                errors.append("RunConfig.json missing nonempty source_hashes for compile run")
            if not isinstance(run_config.get("registry_hashes"), dict) or not run_config.get("registry_hashes"):
                errors.append("RunConfig.json missing nonempty registry_hashes for compile run")
        else:
            if not isinstance(source_hashes, dict) or not source_hashes:
                warnings.append("ReproducibilityManifest.json has empty or missing source_hashes on non-compile run")
            if not isinstance(registry_hashes, dict) or not registry_hashes:
                warnings.append("ReproducibilityManifest.json has empty or missing registry_hashes on non-compile run")
        _validate_telemetry(manifest=manifest, is_compile_run=is_compile_run, errors=errors)
        return user_intent, run_config, manifest


    def _validate_telemetry(*, manifest: dict[str, Any], is_compile_run: bool, errors: list[str]) -> None:
        telemetry = manifest.get("telemetry")
        if not isinstance(telemetry, dict):
            errors.append("ReproducibilityManifest.json missing global telemetry object")
            return
        total_wall_seconds = telemetry.get("total_wall_seconds")
        if not isinstance(total_wall_seconds, (int, float)) or total_wall_seconds < 0:
            errors.append("telemetry.total_wall_seconds missing or negative")
        stage_status = telemetry.get("stage_status")
        stage_wall_seconds = telemetry.get("stage_wall_seconds")
        if not isinstance(stage_status, dict):
            errors.append("telemetry.stage_status missing or not a mapping")
            stage_status = {}
        if not isinstance(stage_wall_seconds, dict):
            errors.append("telemetry.stage_wall_seconds missing or not a mapping")
            stage_wall_seconds = {}
        if is_compile_run:
            missing_status = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_status))
            missing_duration = sorted(set(COMPILE_TELEMETRY_STAGES) - set(stage_wall_seconds))
            if missing_status:
                errors.append(f"telemetry.stage_status missing compile stages: {missing_status}")
            if missing_duration:
                errors.append(f"telemetry.stage_wall_seconds missing compile stages: {missing_duration}")
        for stage, status in stage_status.items():
            if status not in TERMINAL_STAGE_STATUSES:
                errors.append(f"invalid telemetry stage status: {stage}={status}")
        for stage, duration in stage_wall_seconds.items():
            if not isinstance(duration, (int, float)) or duration < 0:
                errors.append(f"invalid telemetry stage duration: {stage}={duration}")


    def _validate_race_bridge_contract(*, root: Path, v, q, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
        rows = v.to_pylist()
        q_rows = {str(row.get("field_id")): row for row in q.to_pylist() if row.get("field_id") is not None}
        bridge_rows = [row for row in rows if str(row.get("field_id", "")).startswith("SIMRaceBridge") or str(row.get("field_id", "")).startswith("SIMRaceAdminRawCount_")]
        if not bridge_rows:
            return
        run_bridge = run_config.get("race_bridge")
        manifest_bridge = manifest.get("race_bridge")
        if not isinstance(run_bridge, dict):
            errors.append("RunConfig.json missing race_bridge metadata while race bridge fields exist")
            run_bridge = {}
        if not isinstance(manifest_bridge, dict):
            errors.append("ReproducibilityManifest.json missing race_bridge metadata while race bridge fields exist")
            manifest_bridge = {}
        missing_run_keys = sorted(RUN_CONFIG_RACE_BRIDGE_KEYS - set(run_bridge))
        if missing_run_keys:
            errors.append(f"RunConfig.race_bridge missing keys: {missing_run_keys}")
        if run_bridge.get("raw_admin_counts_preserved") is not True:
            errors.append("RunConfig.race_bridge.raw_admin_counts_preserved must be true")
        if run_bridge.get("missing_category_preserved") is not True:
            errors.append("RunConfig.race_bridge.missing_category_preserved must be true")
        if manifest_bridge.get("prior_hash") != run_bridge.get("prior_hash"):
            errors.append("ReproducibilityManifest.race_bridge.prior_hash must match RunConfig.race_bridge.prior_hash")
        if not (root / "Tables" / "race_bridge_summary.parquet").exists():
            errors.append("race bridge fields exist but Tables/race_bridge_summary.parquet is missing")
        for row in bridge_rows:
            fid = str(row.get("field_id"))
            q_row = q_rows.get(fid)
            if q_row is None:
                errors.append(f"race bridge field missing Q_tensor row: {fid}")
                continue
            if fid.startswith("SIMRaceBridgePosteriorCount_"):
                axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
                if not isinstance(axes, dict):
                    errors.append(f"race bridge posterior axes_json is not an object: {fid}")
                    continue
                absent = sorted(RACE_BRIDGE_POSTERIOR_KEYS - set(axes))
                if absent:
                    errors.append(f"race bridge posterior field missing metadata keys: {fid} {absent}")
                if axes.get("bridge_operator") != "Bridge_R_fixedC_dynamic_weight":
                    errors.append(f"race bridge posterior field has wrong bridge_operator: {fid} {axes.get('bridge_operator')}")
                sensitivity = float(axes.get("sensitivity_width") or 0.0)
                if row.get("dashboard_safe") == "True" and sensitivity > 0.05:
                    errors.append(f"race bridge posterior dashboard safety not downgraded despite sensitivity width: {fid}")
                if q_row.get("dashboard_safe") == "True" and sensitivity > 0.05:
                    errors.append(f"race bridge posterior Q dashboard safety not downgraded despite sensitivity width: {fid}")
                for column in OPTIONAL_RACE_Q_COLUMNS & set(q.column_names):
                    if q_row.get(column) is None:
                        errors.append(f"race bridge posterior Q_tensor missing {column}: {fid}")
            if fid == "SIMRaceBridgeMissingRaceObserver":
                axes = _load_json_cell(row.get("axes_json"), errors=errors, context=f"V_fields.axes_json[{fid}]")
                if isinstance(axes, dict) and axes.get("missing_category_preserved") is not True:
                    errors.append("SIMRaceBridgeMissingRaceObserver must declare missing_category_preserved=true")


    def _validate_parquet_contracts(*, root: Path, run_config: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
        try:
            v = _read(root / "V_fields.parquet")
            q = _read(root / "Q_tensor.parquet")
            vd = _read(root / "VariableDictionary.parquet")
            edges = _read(root / "E_DAG.parquet")
            warnings_table = _read(root / "Warnings.parquet")
            failed_branches = _read(root / "FailedBranches.parquet")
            quarantined = _read(root / "QuarantinedFields.parquet")
            forced = _read(root / "ForcedFields.parquet")
            model_assoc = _read(root / "ModelAssociations.parquet")
            residual_assoc = _read(root / "ResidualAssociations.parquet")
            hypotheses = _read(root / "Hypotheses.parquet")
        except Exception as exc:
            errors.append(f"parquet read failure: {exc}")
            return
        _require_columns(table_name="V_fields", actual=set(v.column_names), required=REQUIRED_V_FIELDS_COLUMNS, errors=errors)
        _require_columns(table_name="E_DAG", actual=set(edges.column_names), required=REQUIRED_E_DAG_COLUMNS, errors=errors)
        _require_columns(table_name="Q_tensor", actual=set(q.column_names), required=REQUIRED_Q_TENSOR_COLUMNS, errors=errors)
        _require_columns(table_name="VariableDictionary", actual=set(vd.column_names), required=REQUIRED_VARIABLE_DICTIONARY_COLUMNS, errors=errors)
        if q.num_rows == 0:
            errors.append("Q_tensor is empty")
        v_ids = _nonnull(_column_values(v, "field_id"))
        q_ids = _nonnull(_column_values(q, "field_id"))
        vd_ids = _nonnull(_column_values(vd, "field_id"))
        if not v_ids:
            errors.append("V_fields has no field_id values")
        if not v_ids.issubset(vd_ids):
            errors.append(f"VariableDictionary does not cover all V_fields: {sorted(v_ids - vd_ids)}")
        if not v_ids.issubset(q_ids):
            errors.append(f"Q_tensor does not cover all V_fields: {sorted(v_ids - q_ids)}")
        if edges.num_rows:
            for col in ["parent_field_id", "child_field_id"]:
                bad = _nonnull(_column_values(edges, col)) - v_ids
                if bad:
                    errors.append(f"E_DAG {col} contains IDs absent from V_fields: {sorted(bad)}")
        if warnings_table.num_rows and "field_id" in warnings_table.column_names:
            bad_warnings = {x for x in warnings_table.column("field_id").to_pylist() if x is not None and x not in {"", "run"} and x not in v_ids}
            if bad_warnings:
                errors.append(f"Warnings link to invalid field IDs: {sorted(bad_warnings)}")
        if failed_branches.num_rows and "parent_field_ids" in failed_branches.column_names:
            for idx, raw in enumerate(failed_branches.column("parent_field_ids").to_pylist()):
                parents = _load_json_cell(raw, errors=errors, context=f"FailedBranches.parent_field_ids[{idx}]")
                if parents is None:
                    continue
                if not isinstance(parents, list):
                    errors.append(f"FailedBranches.parent_field_ids[{idx}] is not a JSON list")
                    continue
                bad = {x for x in parents if x not in v_ids}
                if bad:
                    errors.append(f"FailedBranches parent IDs absent from V_fields at row {idx}: {sorted(bad)}")
        illegal_ids = set()
        if "state" in v.column_names and "field_id" in v.column_names:
            field_ids = v.column("field_id").to_pylist()
            states = v.column("state").to_pylist()
            illegal_ids = {fid for fid, state in zip(field_ids, states, strict=False) if state == "illegal_excluded"}
        for table_name, table in [("ModelAssociations", model_assoc), ("ResidualAssociations", residual_assoc), ("Hypotheses", hypotheses)]:
            if not illegal_ids:
                break
            for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
                bad = _nonnull(_column_values(table, col)) & illegal_ids
                if bad:
                    errors.append(f"{table_name}.{col} references illegal_excluded fields: {sorted(bad)}")
        for table_name, table in [("QuarantinedFields", quarantined), ("ForcedFields", forced)]:
            for col in FIELD_REFERENCE_COLUMNS & set(table.column_names):
                bad = _nonnull(_column_values(table, col)) - v_ids
                if bad:
                    errors.append(f"{table_name}.{col} contains IDs absent from V_fields: {sorted(bad)}")
        _validate_race_bridge_contract(root=root, v=v, q=q, run_config=run_config, manifest=manifest, errors=errors)


    def validate_output_bundle(*, run_dir: str, schema_registry: OutputSchemaRegistry | None = None) -> OutputValidationResult:
        """Validate exact 17-key output bundle and mandatory cross-references."""
        schema_registry = schema_registry or OutputSchemaRegistry()
        root = Path(run_dir)
        errors: list[str] = []
        warnings: list[str] = []
        if not root.exists():
            return OutputValidationResult(ok=False, errors=[f"run_dir does not exist: {root}"], warnings=[])
        if not root.is_dir():
            return OutputValidationResult(ok=False, errors=[f"run_dir is not a directory: {root}"], warnings=[])
        _validate_first_class_keys(root, schema_registry, errors)
        if errors:
            return OutputValidationResult(ok=False, errors=errors, warnings=warnings)
        _user_intent, run_config, manifest = _validate_manifest_and_config(root=root, errors=errors, warnings=warnings)
        _validate_parquet_contracts(root=root, run_config=run_config, manifest=manifest, errors=errors)
        return OutputValidationResult(ok=not errors, errors=errors, warnings=warnings)
    ''')


def patch_compile() -> None:
    # Replace compile.py with the restored Slice 3B orchestrator plus a registry-backed downstream bridge hook.
    write("src/pegasus/workflows/compile.py", r'''
    from __future__ import annotations

    import json
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    from pydantic import ValidationError

    from pegasus.core.hashing import content_hash, sha256_file
    from pegasus.core.schemas import UserIntent
    from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
    from pegasus.output.maternal_child_compile_attach import attach_maternal_child_compile_fields
    from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.race_bridge import RaceBridgeRegistryError, resolve_race_bridge_plan
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.workflows.datasus import run_datasus_normalize_sim
    from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture
    from pegasus.workflows.race_bridge import run_attach_race_bridge
    from pegasus.workflows.sinasc import run_datasus_normalize_sinasc


    SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD: list[dict[str, str]] = [
        {
            "NC": "Nível Territorial (Código)",
            "NN": "Nível Territorial",
            "MC": "Unidade de Medida (Código)",
            "MN": "Unidade de Medida",
            "V": "Valor",
            "D1C": "Município (Código)",
            "D1N": "Município",
            "D2C": "Ano (Código)",
            "D2N": "Ano",
            "D3C": "Variável (Código)",
            "D3N": "Variável",
            "D4C": "Sexo (Código)",
            "D4N": "Sexo",
            "D5C": "Cor ou raça (Código)",
            "D5N": "Cor ou raça",
            "D6C": "Idade (Código)",
            "D6N": "Idade",
        },
        {
            "NC": "6",
            "NN": "Município",
            "MC": "45",
            "MN": "Pessoas",
            "V": "957916",
            "D1C": "2704302",
            "D1N": "Maceió (AL)",
            "D2C": "2022",
            "D2N": "2022",
            "D3C": "93",
            "D3N": "População residente",
            "D4C": "6794",
            "D4N": "Total",
            "D5C": "95251",
            "D5N": "Total",
            "D6C": "100362",
            "D6N": "Total",
        },
    ]


    SIDRA_POPULATION_MACEIO_CHUNK_REQUEST: dict[str, Any] = {
        "table_id": "9606",
        "variables": ["93"],
        "periods": ["2022"],
        "locality_level": "N6",
        "localities": ["2704302"],
        "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
    }


    def utc_stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


    def _load_intent(intent_path: Path) -> tuple[dict[str, Any], UserIntent]:
        payload = json.loads(intent_path.read_text(encoding="utf-8"))
        try:
            return payload, UserIntent.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid UserIntent file {intent_path}: {exc}") from exc


    def _registry_hashes() -> dict[str, str]:
        candidates = [
            Path("config/registries/registry_manifest.yaml"),
            Path("config/registries/sidra_views.yaml"),
            Path("config/registries/datasus/source_fields.yaml"),
            Path("config/registries/ontology/quality_permissions.yaml"),
            Path("config/registries/demographic/race_axis_registry.yaml"),
            Path("config/registries/demographic/race_bridge_priors.yaml"),
        ]
        return {str(path): sha256_file(path) for path in candidates if path.exists()}


    def _write_compile_manifest(*, run_id: str, intent_path: Path, data_root: Path, run_dir: Path, payload: dict[str, Any]) -> Path:
        path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "kind": "compile_smoke_manifest",
                    "run_id": run_id,
                    "intent_path": str(intent_path),
                    "run_dir": str(run_dir),
                    "intent": payload,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


    def _write_sidra_smoke_facts(*, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        facts = normalize_sidra_payload_to_facts(
            SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD,
            table_id="9606",
            request_hash=content_hash(SIDRA_POPULATION_MACEIO_CHUNK_REQUEST),
            metadata_hash=content_hash({"metadata": "compile_smoke_sidra_9606_maceio_total_v1"}),
            chunk_request=SIDRA_POPULATION_MACEIO_CHUNK_REQUEST,
            unit_by_variable=None,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        write_facts_parquet(facts, output_path=output_path)
        return output_path


    def _smoke_municipality_cod6(intent: UserIntent) -> str:
        if intent.execution_scale != "smoke":
            raise ValueError("Slice 4B compile supports execution_scale='smoke' only.")
        if intent.geography.level != "municipality":
            raise ValueError("Slice 4B compile smoke requires geography.level='municipality'.")
        if len(intent.geography.codes) != 1:
            raise ValueError("Slice 4B compile smoke requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(f"Could not convert intent municipality code to DATASUS cod6: {intent.geography.codes[0]!r}")
        return cod6


    def run_compile(
        *,
        intent_path: str | Path,
        run_dir: str | Path | None = None,
        data_root: str | Path = "data",
    ) -> dict[str, Any]:
        intent_path = Path(intent_path)
        data_root = Path(data_root)
        intent_payload, intent = _load_intent(intent_path)
        municipality_cod6 = _smoke_municipality_cod6(intent)
        try:
            race_bridge_plan = resolve_race_bridge_plan(intent=intent, municipality_cod6=municipality_cod6)
        except RaceBridgeRegistryError as exc:
            raise ValueError(f"Invalid Race Bridge registry plan for {intent_path}: {exc}") from exc
        if race_bridge_plan.status == "blocked":
            raise ValueError(f"Race Bridge mode is blocked for this compile slice: {race_bridge_plan.as_manifest()}")

        intent_hash = sha256_file(intent_path)
        run_id = f"compile_{intent_path.stem}_{utc_stamp()}_{intent_hash[:8]}"
        run_dir = Path(run_dir) if run_dir is not None else data_root / "runs" / run_id
        diagnostic_path = data_root / "diagnostics" / "compile" / f"{run_id}.telemetry.json"
        telemetry = RunTelemetry(run_id=run_id, diagnostic_path=diagnostic_path)

        source_hashes: dict[str, str] = {"intent": intent_hash}
        registry_hashes = _registry_hashes()
        if race_bridge_plan.registry_path is not None and race_bridge_plan.registry_hash is not None:
            registry_hashes[str(race_bridge_plan.registry_path)] = race_bridge_plan.registry_hash

        raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
        raw_sinasc_fixture_source = Path("tests/fixtures/datasus/sinasc_fixture.csv")
        if not raw_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")
        if not raw_sinasc_fixture_source.exists():
            raise FileNotFoundError(f"Missing SINASC smoke fixture: {raw_sinasc_fixture_source}")

        compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
        raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
        raw_sinasc_cache_path = data_root / "raw" / "datasus" / "SINASC" / "fixture" / "sinasc_fixture.csv"
        sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        sinasc_events_path = data_root / "processed" / "datasus" / "SINASC" / "fixture" / "sinasc_events.parquet"
        sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_maceio.parquet"

        with telemetry.stage("datasus_manifest"):
            compile_manifest_path = _write_compile_manifest(
                run_id=run_id,
                intent_path=intent_path,
                data_root=data_root,
                run_dir=run_dir,
                payload=intent_payload,
            )
            source_hashes["compile_manifest"] = sha256_file(compile_manifest_path)

        with telemetry.stage("datasus_acquire"):
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            raw_sinasc_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_fixture_source, raw_cache_path)
            shutil.copyfile(raw_sinasc_fixture_source, raw_sinasc_cache_path)
            source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)
            source_hashes["sinasc_raw_fixture"] = sha256_file(raw_sinasc_cache_path)

        with telemetry.stage("datasus_decode"):
            run_datasus_normalize_sim(
                input_path=raw_cache_path,
                output_path=sim_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            run_datasus_normalize_sinasc(
                input_path=raw_sinasc_cache_path,
                output_path=sinasc_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            source_hashes["sim_processed_events"] = sha256_file(sim_events_path)
            source_hashes["sinasc_processed_events"] = sha256_file(sinasc_events_path)

        telemetry.set_stage("sidra_metadata", "skipped", 0.0)
        telemetry.set_stage("sidra_plan", "skipped", 0.0)
        telemetry.set_stage("sidra_fetch", "skipped", 0.0)
        telemetry.flush()

        with telemetry.stage("sidra_normalize"):
            _write_sidra_smoke_facts(output_path=sidra_facts_path)
            source_hashes["sidra_facts"] = sha256_file(sidra_facts_path)

        with telemetry.stage("efg_build"):
            run_build_sim_fixture(
                sim_events_path=sim_events_path,
                run_dir=run_dir,
                municipality_cod6=municipality_cod6,
            )

        with telemetry.stage("she_build"):
            run_attach_sidra_denominator(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
            )
            attach_maternal_child_compile_fields(
                run_dir=run_dir,
                sinasc_events_path=sinasc_events_path,
                sim_events_path=sim_events_path,
                municipality_cod6=municipality_cod6,
            )
            source_hashes["maternal_child_linkage_summary"] = sha256_file(
                Path(run_dir) / "Tables" / "maternal_child_linkage_summary.parquet"
            )

        race_bridge_metadata: dict[str, Any] | None = None
        if race_bridge_plan.requires_attach:
            assert race_bridge_plan.prior_path is not None
            with telemetry.stage("race_bridge"):
                bridge_result = run_attach_race_bridge(
                    run_dir=run_dir,
                    sim_events_path=sim_events_path,
                    bridge_prior_path=race_bridge_plan.prior_path,
                    municipality_cod6=municipality_cod6,
                )
                if not bridge_result["validation"].ok:
                    raise RuntimeError("Race bridge attachment invalidated run bundle: " + "; ".join(bridge_result["validation"].errors))
                race_bridge_metadata = bridge_result.get("race_bridge")
                source_hashes["race_bridge_prior"] = sha256_file(race_bridge_plan.prior_path)
                summary_path = Path(run_dir) / "Tables" / "race_bridge_summary.parquet"
                if summary_path.exists():
                    source_hashes["race_bridge_summary"] = sha256_file(summary_path)
        else:
            telemetry.set_stage("race_bridge", "skipped", 0.0)
            telemetry.flush()

        telemetry.set_stage("geo_support", "success", 0.0)
        telemetry.set_stage("q_tensor", "success", 0.0)
        telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")
        telemetry.block("stdfm", reason="ST-DFM scaffold remains blocked for compile smoke")
        telemetry.block("pirs_model", reason="PIRS model stage is not invoked in compile smoke")
        telemetry.block("pirs_hsic", reason="PIRS HSIC stage is not invoked in compile smoke")
        telemetry.flush()

        with telemetry.stage("output_serialization"):
            (run_dir / "UserIntent.json").write_text(
                json.dumps(intent_payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            run_config_path = run_dir / "RunConfig.json"
            run_config_payload = {
                "schema_version": "1.0",
                "compile_mode": "smoke",
                "run_id": run_id,
                "intent_path": str(intent_path),
                "data_root": str(data_root),
                "support_policy": {
                    "geography_level": intent.geography.level,
                    "ibge_cod7": intent.geography.codes,
                    "datasus_cod6": [municipality_cod6],
                    "geo_mode": intent.geo_mode,
                },
                "population_mode": intent.population_mode,
                "race_tensor_mode": intent.race_tensor_mode,
                "race_bridge_plan": race_bridge_plan.as_manifest(),
                "registry_hashes": registry_hashes,
                "source_hashes": source_hashes,
            }
            existing_run_config = {}
            if run_config_path.exists():
                try:
                    existing_run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing_run_config = {}
            if existing_run_config.get("maternal_child_linkage"):
                run_config_payload["maternal_child_linkage"] = existing_run_config["maternal_child_linkage"]
            if race_bridge_metadata is None and existing_run_config.get("race_bridge"):
                race_bridge_metadata = existing_run_config["race_bridge"]
            if race_bridge_metadata is not None:
                run_config_payload["race_bridge"] = race_bridge_metadata
            run_config_path.write_text(
                json.dumps(run_config_payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            manifest_extras = {
                "compile_mode": "smoke",
                "compile_manifest": str(compile_manifest_path),
                "intent_path": str(intent_path),
                "maternal_child_linkage": True,
                "race_bridge_plan": race_bridge_plan.as_manifest(),
            }
            if race_bridge_metadata is not None:
                manifest_extras["race_bridge"] = race_bridge_metadata
            write_reproducibility_manifest(
                run_dir=run_dir,
                run_id=run_id,
                intent_hash=intent_hash,
                source_hashes=source_hashes,
                registry_hashes=registry_hashes,
                telemetry=telemetry,
                extras=manifest_extras,
            )

        with telemetry.stage("output_validation"):
            result = validate_output_bundle(run_dir=str(run_dir))
            if not result.ok:
                raise RuntimeError("Compile smoke produced invalid output bundle: " + "; ".join(result.errors))

        final_extras = {
            "compile_mode": "smoke",
            "compile_manifest": str(compile_manifest_path),
            "intent_path": str(intent_path),
            "maternal_child_linkage": True,
            "race_bridge_plan": race_bridge_plan.as_manifest(),
        }
        if race_bridge_metadata is not None:
            final_extras["race_bridge"] = race_bridge_metadata
        write_reproducibility_manifest(
            run_dir=run_dir,
            run_id=run_id,
            intent_hash=intent_hash,
            source_hashes=source_hashes,
            registry_hashes=registry_hashes,
            telemetry=telemetry,
            extras=final_extras,
        )

        validation = validate_output_bundle(run_dir=str(run_dir))
        return {
            "status": "success" if validation.ok else "failed",
            "run_id": run_id,
            "run_dir": run_dir,
            "intent": intent,
            "validation": validation,
            "source_hashes": source_hashes,
            "registry_hashes": registry_hashes,
            "telemetry": telemetry.model(),
            "race_bridge_plan": race_bridge_plan.as_manifest(),
            "race_bridge": race_bridge_metadata,
        }
    ''')


def patch_cli() -> None:
    path = rel("src/pegasus/cli.py")
    cli = path.read_text(encoding="utf-8")
    if "def efg_validate_race_bridge_prior" not in cli:
        block = r'''

@efg_app.command("validate-race-bridge-prior")
def efg_validate_race_bridge_prior(
    bridge_prior: Path = typer.Option(..., "--bridge-prior"),
) -> None:
    from pegasus.workflows.race_bridge import run_validate_race_bridge_prior

    result = run_validate_race_bridge_prior(bridge_prior_path=bridge_prior)
    print(f"[green]race bridge prior valid[/green] bridge_id={result['bridge_id']} hash={result['prior_hash']}")
'''
        cli = cli.rstrip() + "\n" + textwrap.dedent(block)
    if "def efg_plan_race_bridge" not in cli:
        block = r'''

@efg_app.command("plan-race-bridge")
def efg_plan_race_bridge(
    sim_events: Path = typer.Option(..., "--sim-events"),
    bridge_prior: Path | None = typer.Option(None, "--bridge-prior"),
    intent: Path | None = typer.Option(None, "--intent"),
    registry: Path = typer.Option(Path("config/registries/demographic/race_bridge_priors.yaml"), "--registry"),
    municipality_cod6: str | None = typer.Option(None, "--municipality-cod6"),
) -> None:
    from pegasus.workflows.race_bridge import run_plan_race_bridge

    result = run_plan_race_bridge(
        sim_events_path=sim_events,
        bridge_prior_path=bridge_prior,
        intent_path=intent,
        registry_path=registry,
        municipality_cod6=municipality_cod6,
    )
    print(result["summary_json"])
'''
        cli = cli.rstrip() + "\n" + textwrap.dedent(block)
    path.write_text(cli.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_registry_and_intent() -> None:
    write("config/registries/demographic/race_bridge_priors.yaml", r'''
    schema_version: "1.0"
    registry_version: "race_bridge_priors.v1.slice4b"
    created_at: "2026-06-10"
    updated_at: "2026-06-10"
    provenance: "Slice 4B smoke registry for downstream Bridge_R prior selection. The prior is a fixture for compiler validation and must not be interpreted as calibrated epidemiological truth."
    entries:
      - id: "fixedC_sim_admin_to_ibge_selfdeclared_smoke_v1"
        status: "experimental"
        description: "Smoke fixed-C dynamic-weight bridge from SIM administrative race/color to IBGE self-declared race axis."
        warnings:
          - "fixture_prior_not_calibrated"
          - "posterior_fields_are_sensitivity_observers"
          - "raw_administrative_counts_must_be_preserved"
        source_system: "SIM-DO"
        source_axis: "SIM_ADMIN_RACACOR"
        target_axis: "IBGE_SELF_DECLARED_RACE"
        mode: "fixedC_dynamic_weight"
        prior_path: "tests/fixtures/race_bridge/fixedC_valid.json"
        enabled_for_compile: true
        region_scope:
          - "AL"
        period_start: 2022
        period_end: 2022
        dashboard_policy: "downgrade_when_sensitive"
    ''')

    baseline_path = rel("config/intents/alagoas_smoke.json")
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload.pop("race_bridge_policy", None)
    payload["race_tensor_mode"] = "downstream_bridge"
    mandatory = list(payload.get("mandatory_fields") or [])
    for item in [
        "SIMRaceAdminRawCount_1",
        "SIMRaceBridgeMissingRaceObserver",
        "SIMRaceBridgePosteriorCount_parda",
        "SIMRaceBridgePosteriorCount_branca",
    ]:
        if item not in mandatory:
            mandatory.append(item)
    payload["mandatory_fields"] = mandatory
    context_policy = list(payload.get("context_policy") or [])
    for item in ["race_bridge_downstream", "raw_admin_race_preserved", "missing_race_observer_preserved"]:
        if item not in context_policy:
            context_policy.append(item)
    payload["context_policy"] = context_policy
    rel("config/intents/alagoas_smoke_race_bridge.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_tests_and_audit() -> None:
    write("tests/unit/test_race_bridge_registry.py", r'''
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


    def test_embedded_modes_block_until_population_tensor_bridge_exists():
        payload = json.loads(Path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))
        payload["race_tensor_mode"] = "embedded_fixedC"
        intent = UserIntent.model_validate(payload)
        plan = resolve_race_bridge_plan(intent=intent, municipality_cod6="270430")
        assert plan.status == "blocked"
        assert "embedded" in (plan.reason or "")
    ''')

    write("tests/unit/test_race_bridge_attach_idempotency.py", r'''
    import polars as pl

    from pegasus.workflows.compile import run_compile
    from pegasus.workflows.race_bridge import run_attach_race_bridge


    def test_race_bridge_attach_is_idempotent_on_compile_run(tmp_path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=tmp_path / "run",
            data_root=tmp_path / "data",
        )
        assert result["validation"].ok, result["validation"].errors
        sim_events = tmp_path / "data" / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        for _ in range(2):
            attach = run_attach_race_bridge(
                run_dir=tmp_path / "run",
                sim_events_path=sim_events,
                bridge_prior_path="tests/fixtures/race_bridge/fixedC_valid.json",
                municipality_cod6="270430",
            )
            assert attach["validation"].ok, attach["validation"].errors
        v = pl.read_parquet(tmp_path / "run" / "V_fields.parquet")
        assert v.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").height == 1
        assert v.filter(pl.col("field_id") == "SIMRaceBridgeMissingRaceObserver").height == 1
    ''')

    write("tests/integration/test_slice4b_compile_race_bridge_registry.py", r'''
    import json
    import shutil
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.compile import run_compile
    from pegasus.workflows.race_bridge import run_plan_race_bridge


    def test_baseline_compile_remains_decoupled_and_valid(tmp_path: Path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=tmp_path / "baseline_run",
            data_root=tmp_path / "baseline_data",
        )
        assert result["validation"].ok, result["validation"].errors
        assert result["race_bridge"] is None
        v = pl.read_parquet(tmp_path / "baseline_run" / "V_fields.parquet")
        assert "SIMRaceBridgePosteriorCount_parda" not in set(v["field_id"].to_list())
        manifest = json.loads((tmp_path / "baseline_run" / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        assert manifest["telemetry"]["stage_status"]["race_bridge"] == "skipped"


    def test_registry_backed_compile_attaches_downstream_bridge(tmp_path: Path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke_race_bridge.json",
            run_dir=tmp_path / "bridge_run",
            data_root=tmp_path / "bridge_data",
        )
        assert result["status"] == "success"
        assert result["validation"].ok, result["validation"].errors
        assert result["race_bridge"] is not None
        assert result["race_bridge_plan"]["status"] == "planned"
        v = pl.read_parquet(tmp_path / "bridge_run" / "V_fields.parquet")
        field_ids = set(v["field_id"].to_list())
        assert "SIMRaceAdminRawCount_4" in field_ids
        assert "SIMRaceBridgeMissingRaceObserver" in field_ids
        assert "SIMRaceBridgePosteriorCount_parda" in field_ids
        q = pl.read_parquet(tmp_path / "bridge_run" / "Q_tensor.parquet")
        parda_q = q.filter(pl.col("field_id") == "SIMRaceBridgePosteriorCount_parda").to_dicts()[0]
        assert parda_q["dashboard_safe"] == "False"
        assert parda_q["state"] == "fragile"
        if "bridge_mode" in q.columns:
            assert parda_q["bridge_mode"] == "fixedC_dynamic_weight"
        manifest = json.loads((tmp_path / "bridge_run" / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        run_config = json.loads((tmp_path / "bridge_run" / "RunConfig.json").read_text(encoding="utf-8"))
        assert manifest["race_bridge"]["raw_admin_counts_preserved"] is True
        assert run_config["race_bridge"]["missing_category_preserved"] is True
        assert "race_bridge_prior" in manifest["source_hashes"]
        assert "race_bridge_summary" in manifest["source_hashes"]
        assert manifest["telemetry"]["stage_status"]["race_bridge"] == "success"


    def test_registry_backed_plan_cli_workflow_payload_after_compile(tmp_path: Path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=tmp_path / "run",
            data_root=tmp_path / "data",
        )
        assert result["validation"].ok, result["validation"].errors
        sim_events = tmp_path / "data" / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        plan = run_plan_race_bridge(
            sim_events_path=sim_events,
            intent_path="config/intents/alagoas_smoke_race_bridge.json",
            registry_path="config/registries/demographic/race_bridge_priors.yaml",
            municipality_cod6="270430",
        )
        assert plan["summary"]["bridge_id"] == "fixedC_sim_admin_to_ibge_selfdeclared_smoke_v1"
        assert "posterior_counts" in plan["summary"]
        assert plan["summary"]["plan"]["status"] == "planned"


    def test_validator_rejects_race_bridge_fields_without_metadata(tmp_path: Path):
        result = run_compile(
            intent_path="config/intents/alagoas_smoke_race_bridge.json",
            run_dir=tmp_path / "run",
            data_root=tmp_path / "data",
        )
        assert result["validation"].ok, result["validation"].errors
        poisoned = tmp_path / "poisoned"
        shutil.copytree(tmp_path / "run", poisoned)
        run_config_path = poisoned / "RunConfig.json"
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config.pop("race_bridge", None)
        run_config_path.write_text(json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        validation = validate_output_bundle(run_dir=str(poisoned))
        assert not validation.ok
        assert any("race_bridge metadata" in error or "RunConfig.race_bridge" in error for error in validation.errors)
    ''')

    write("scripts/dev/audits/audit_slice4b_compile_race_bridge.py", r'''
    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        args = parser.parse_args()
        run = Path(args.run)
        failures: list[dict] = []
        validation = validate_output_bundle(run_dir=str(run))
        if not validation.ok:
            failures.append({"kind": "validate_output_bundle", "errors": validation.errors})
        v = pl.read_parquet(run / "V_fields.parquet")
        required_fields = {
            "SIMRaceAdminRawCount_1",
            "SIMRaceAdminRawCount_2",
            "SIMRaceAdminRawCount_3",
            "SIMRaceAdminRawCount_4",
            "SIMRaceAdminRawCount_5",
            "SIMRaceBridgeMissingRaceObserver",
            "SIMRaceBridgePosteriorCount_branca",
            "SIMRaceBridgePosteriorCount_preta",
            "SIMRaceBridgePosteriorCount_amarela",
            "SIMRaceBridgePosteriorCount_parda",
            "SIMRaceBridgePosteriorCount_indigena",
        }
        field_ids = set(v["field_id"].to_list())
        missing = sorted(required_fields - field_ids)
        if missing:
            failures.append({"kind": "missing_compile_race_bridge_fields", "missing": missing})
        manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
        for holder, name in [(manifest, "manifest"), (run_config, "run_config")]:
            bridge = holder.get("race_bridge")
            if not isinstance(bridge, dict):
                failures.append({"kind": f"missing_{name}_race_bridge"})
                continue
            for key in ["bridge_id", "mode", "prior_hash", "missing_race_share", "sensitivity_width", "raw_admin_counts_preserved", "missing_category_preserved", "attach_stage"]:
                if key not in bridge:
                    failures.append({"kind": f"{name}_race_bridge_missing_key", "key": key})
            if bridge.get("raw_admin_counts_preserved") is not True:
                failures.append({"kind": f"{name}_raw_admin_not_preserved"})
            if bridge.get("missing_category_preserved") is not True:
                failures.append({"kind": f"{name}_missing_not_preserved"})
        source_hashes = manifest.get("source_hashes") or {}
        for key in ["race_bridge_prior", "race_bridge_summary", "sim_processed_events"]:
            if key not in source_hashes:
                failures.append({"kind": "manifest_missing_source_hash", "key": key})
        telemetry = manifest.get("telemetry") or {}
        if telemetry.get("stage_status", {}).get("race_bridge") != "success":
            failures.append({"kind": "race_bridge_stage_not_success", "stage_status": telemetry.get("stage_status", {}).get("race_bridge")})
        q = pl.read_parquet(run / "Q_tensor.parquet")
        for fid in ["SIMRaceBridgePosteriorCount_parda", "SIMRaceBridgeMissingRaceObserver"]:
            row = q.filter(pl.col("field_id") == fid).to_dicts()
            if len(row) != 1:
                failures.append({"kind": "q_row_count", "field_id": fid, "count": len(row)})
                continue
            row = row[0]
            if row.get("dashboard_safe") == "True":
                failures.append({"kind": "dashboard_not_downgraded", "field_id": fid})
            if "sensitivity_width" in q.columns and row.get("sensitivity_width") is None:
                failures.append({"kind": "missing_sensitivity_width", "field_id": fid})
        table = run / "Tables" / "race_bridge_summary.parquet"
        if not table.exists():
            failures.append({"kind": "missing_race_bridge_summary"})
        if failures:
            fail({"status": "failed", "failures": failures})
        print("AUDIT PASSED: Slice 4B registry-backed compile Race Bridge fields validate with raw counts, missing observer, posterior metadata, and telemetry.")


    if __name__ == "__main__":
        main()
    ''')


def main() -> None:
    preflight()
    print("Slice 4B-redo updater preflight passed.")
    print("CREATE/REPLACE:")
    for p in CREATE_OR_REPLACE:
        print(f"  {p}")
    print("PATCH/REPLACE:")
    for p in PATCH:
        print(f"  {p}")
    write_registry_and_intent()
    write_race_bridge_registry()
    write_race_bridge_workflow()
    write_race_bridge_attach()
    write_validator()
    patch_compile()
    patch_cli()
    write_tests_and_audit()
    print("Applied Slice 4B-redo registry-backed downstream Race Bridge integration.")
    print("Baseline alagoas_smoke.json was not modified; use config/intents/alagoas_smoke_race_bridge.json for bridge-enabled compile.")


if __name__ == "__main__":
    main()
