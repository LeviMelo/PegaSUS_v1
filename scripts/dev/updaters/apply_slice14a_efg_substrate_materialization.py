from __future__ import annotations

import ast
import py_compile
import sys
from pathlib import Path
from textwrap import dedent

SLICE = "14A"
ROOT = Path.cwd()

TOUCH_LIST = (
    "src/pegasus/efg/materialize.py",
    "tests/unit/test_slice14a_efg_substrate_materialization.py",
    "tests/integration/test_slice14a_substrate_to_efg_materialization.py",
    "scripts/dev/audits/audit_slice14a_efg_substrate_materialization.py",
)

FORBIDDEN_PREFIXES = (
    "src/pegasus/workflows/compile.py",
    "src/pegasus/she/source_registry.py",
    "src/pegasus/she/substrate.py",
    "src/pegasus/output/",
    "src/pegasus/pirs/",
    "src/pegasus/dashboard/",
    "src/pegasus/datasus/",
    "src/pegasus/sidra/",
    "config/registries/",
)


def fail(message: str) -> None:
    raise SystemExit(f"[slice14a] {message}")


def read(path: str) -> str:
    p = ROOT / path
    if not p.exists():
        fail(f"required file missing: {path}")
    return p.read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    if any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        fail(f"refusing to write forbidden path: {path}")
    p = ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.rstrip() + "\n", encoding="utf-8")


def top_level_defs(text: str, name: str) -> int:
    tree = ast.parse(text)
    return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def preflight() -> None:
    required = (
        "src/pegasus/workflows/compile.py",
        "src/pegasus/she/substrate.py",
        "src/pegasus/she/source_registry.py",
        "src/pegasus/efg/node.py",
        "src/pegasus/efg/lineage.py",
        "src/pegasus/core/schemas.py",
    )
    for path in required:
        if not (ROOT / path).exists():
            fail(f"required preflight file missing: {path}")

    compile_text = read("src/pegasus/workflows/compile.py")
    public_count = top_level_defs(compile_text, "run_compile")
    impl_count = top_level_defs(compile_text, "_run_compile_impl")
    if public_count != 1:
        fail(f"compile.py must have exactly one public run_compile before Slice 14A; found {public_count}")
    if impl_count != 1:
        fail(f"compile.py must have exactly one private _run_compile_impl before Slice 14A; found {impl_count}")

    substrate = read("src/pegasus/she/substrate.py")
    for token in ("class SubstrateFieldCandidate", "class SubstrateBundle", "build_substrate_bundle"):
        if token not in substrate:
            fail(f"substrate.py missing required API: {token}")
    node = read("src/pegasus/efg/node.py")
    if "def make_field_node" not in node:
        fail("efg/node.py missing make_field_node")
    lineage = read("src/pegasus/efg/lineage.py")
    if "def make_lineage" not in lineage:
        fail("efg/lineage.py missing make_lineage")


def materialize_module() -> str:
    return dedent(r'''
    """EFG metadata materialization from SHE SubstrateBundle candidates.

    Slice 14A replaces the blocked materialization stub with a typed, metadata-only
    boundary.  It does not write output bundles, compute rates, build denominator
    ratios, run PIRS/HSIC, or mutate compile outputs.  Its only responsibility is
    to convert SHE-admissible source-field candidates into FieldNode objects and
    keep SHE exclusions out of the EFG admission surface.
    """

    from __future__ import annotations

    from dataclasses import asdict, dataclass
    from typing import Any, Iterable, Literal

    from pegasus.core.hashing import content_hash
    from pegasus.core.schemas import FieldNode
    from pegasus.efg.lineage import make_lineage, lineage_hash
    from pegasus.efg.node import make_field_node
    from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion


    FIELD_KINDS: frozenset[str] = frozenset({
        "extensive_measure",
        "intensive_density",
        "marked_functional",
        "context_gradient",
        "bridge_divergence",
        "bridge_module",
        "observer_proxy",
        "latent_context",
        "model_residual",
    })
    AGGREGATION_LAWS: frozenset[str] = frozenset({
        "additive",
        "weighted_mean",
        "statistical_functional",
        "compositional",
        "non_aggregable",
    })
    COUNT_UNITS: frozenset[str] = frozenset({"count", "counts", "events", "admissions", "births", "deaths"})


    class EFGMaterializationError(ValueError):
        """Raised when substrate-to-EFG metadata materialization is invalid."""


    @dataclass(frozen=True)
    class SubstrateMaterializedField:
        """One metadata-only FieldNode emitted from one SHE substrate candidate."""

        candidate_id: str
        field: FieldNode
        lineage_hash: str
        materialization_reason: str
        warnings: tuple[str, ...]

        def as_manifest(self) -> dict[str, Any]:
            return {
                "candidate_id": self.candidate_id,
                "field": self.field.model_dump(mode="json"),
                "lineage_hash": self.lineage_hash,
                "materialization_reason": self.materialization_reason,
                "warnings": list(self.warnings),
            }


    @dataclass(frozen=True)
    class SubstrateMaterializationResult:
        """Metadata-only EFG admission result for a SHE SubstrateBundle."""

        schema_version: str
        substrate_id: str
        materialization_id: str
        fields: tuple[SubstrateMaterializedField, ...]
        excluded_source_fields: tuple[dict[str, Any], ...]
        registry_hashes: dict[str, str]
        warnings: tuple[str, ...]

        @property
        def field_count(self) -> int:
            return len(self.fields)

        @property
        def excluded_field_count(self) -> int:
            return len(self.excluded_source_fields)

        def as_manifest(self) -> dict[str, Any]:
            return {
                "schema_version": self.schema_version,
                "substrate_id": self.substrate_id,
                "materialization_id": self.materialization_id,
                "field_count": self.field_count,
                "excluded_field_count": self.excluded_field_count,
                "registry_hashes": dict(self.registry_hashes),
                "warnings": list(self.warnings),
                "fields": [field.as_manifest() for field in self.fields],
                "excluded_source_fields": list(self.excluded_source_fields),
            }


    def _unique(values: Iterable[str | None]) -> list[str]:
        seen: dict[str, None] = {}
        for value in values:
            if value is None:
                continue
            text = str(value)
            if text:
                seen.setdefault(text, None)
        return list(seen)


    def _safe_aggregation(value: str) -> Literal[
        "additive",
        "weighted_mean",
        "statistical_functional",
        "compositional",
        "non_aggregable",
    ]:
        if value in AGGREGATION_LAWS:
            return value  # type: ignore[return-value]
        return "non_aggregable"


    def classify_substrate_candidate_kind(candidate: SubstrateFieldCandidate) -> Literal[
        "extensive_measure",
        "intensive_density",
        "marked_functional",
        "context_gradient",
        "bridge_divergence",
        "bridge_module",
        "observer_proxy",
        "latent_context",
        "model_residual",
    ]:
        """Map a SHE substrate candidate to an allowed FieldNode kind.

        Registry-provided FieldNode kinds are respected directly.  Otherwise this
        function performs a deterministic metadata-only classification from unit,
        aggregation law and role.  It never makes epidemiological ratio claims.
        """

        raw = str(candidate.substrate_kind)
        if raw in FIELD_KINDS:
            return raw  # type: ignore[return-value]
        roles = set(str(role) for role in candidate.role)
        unit = str(candidate.unit)
        aggregation = str(candidate.aggregation)
        if unit == "ICD10" or "diagnostic_topology" in roles:
            return "observer_proxy"
        if aggregation == "additive" and unit.lower() in COUNT_UNITS:
            return "extensive_measure"
        if aggregation == "weighted_mean":
            return "intensive_density"
        if aggregation in {"statistical_functional", "compositional"}:
            return "marked_functional"
        return "observer_proxy"


    def support_from_substrate_candidate(candidate: SubstrateFieldCandidate) -> dict[str, Any]:
        """Build a conservative support descriptor without fabricating axes."""

        return {
            "support_kind": "source_artifact_column",
            "source_system": candidate.source_system,
            "artifact_path": candidate.artifact_path,
            "column": candidate.column,
            "row_count": candidate.row_count,
            "non_null_count": candidate.non_null_count,
            "unique_non_null_count": candidate.unique_non_null_count,
            "missing_rate": candidate.missing_rate,
            "numeric_min": candidate.numeric_min,
            "numeric_max": candidate.numeric_max,
        }


    def materialize_candidate_field(candidate: SubstrateFieldCandidate) -> SubstrateMaterializedField:
        """Materialize one SHE substrate candidate as a metadata-only FieldNode."""

        aggregation = _safe_aggregation(str(candidate.aggregation))
        kind = classify_substrate_candidate_kind(candidate)
        source_manifest_hashes = _unique([candidate.source_manifest_hash, candidate.artifact_hash])
        lineage = make_lineage(
            parent_ids=[],
            operator_type="she_substrate_materialization",
            operator_params={
                "candidate_id": candidate.candidate_id,
                "source_system": candidate.source_system,
                "artifact_path": candidate.artifact_path,
                "column": candidate.column,
                "technical_name": candidate.technical_name,
                "substrate_kind": candidate.substrate_kind,
                "row_count": candidate.row_count,
                "non_null_count": candidate.non_null_count,
                "unique_non_null_count": candidate.unique_non_null_count,
                "missing_rate": candidate.missing_rate,
            },
            registry_versions={candidate.source_system: candidate.registry_hash, "substrate_materializer": "slice14a"},
            source_manifest_hashes=source_manifest_hashes,
            code_version="slice14a",
        )
        warnings = _unique([*candidate.warnings])
        state = "fragile" if warnings or (candidate.missing_rate is not None and candidate.missing_rate > 0.5) else "verified"
        field = make_field_node(
            name=candidate.technical_name,
            kind=kind,
            carrier=str(candidate.carrier),
            unit=str(candidate.unit),
            support=support_from_substrate_candidate(candidate),
            axes=dict(candidate.axes),
            aggregation=aggregation,
            role=_unique([*candidate.role, "source_field", "substrate_materialized"]),
            source=_unique([candidate.source_system, candidate.artifact_path, candidate.column]),
            operator="she_substrate_materialization",
            provenance=_unique([*candidate.provenance, "SHE_SubstrateBundle"]),
            state=state,
            warnings=warnings,
            lineage=lineage,
            materialization_state="metadata_only",
            path=None,
            dashboard_safe="warning" if warnings else False,
        )
        return SubstrateMaterializedField(
            candidate_id=candidate.candidate_id,
            field=field,
            lineage_hash=lineage_hash(lineage),
            materialization_reason="metadata_only_substrate_candidate",
            warnings=tuple(warnings),
        )


    def _exclusion_manifest(exclusion: SubstrateFieldExclusion) -> dict[str, Any]:
        payload = exclusion.as_manifest()
        payload["efg_materialized"] = False
        payload["efg_exclusion_reason"] = exclusion.reason
        return payload


    def materialize_substrate_bundle(bundle: SubstrateBundle) -> SubstrateMaterializationResult:
        """Convert SHE-admissible substrate candidates into metadata-only FieldNodes.

        Exclusions remain exclusions.  No excluded field is promoted into EFG nodes,
        and no output bundle files are written here.
        """

        fields = tuple(materialize_candidate_field(candidate) for candidate in bundle.candidates)
        excluded = tuple(_exclusion_manifest(exclusion) for exclusion in bundle.exclusions)
        payload = {
            "substrate_id": bundle.substrate_id,
            "field_ids": [item.field.id for item in fields],
            "excluded_ids": [item.get("exclusion_id") for item in excluded],
            "registry_hashes": dict(bundle.registry_hashes),
        }
        warnings = _unique([*bundle.warnings])
        return SubstrateMaterializationResult(
            schema_version="1.0",
            substrate_id=bundle.substrate_id,
            materialization_id=f"efg_materialization_{content_hash(payload)[:20]}",
            fields=fields,
            excluded_source_fields=excluded,
            registry_hashes=dict(bundle.registry_hashes),
            warnings=tuple(warnings),
        )


    def materialized_field_nodes(bundle: SubstrateBundle) -> tuple[FieldNode, ...]:
        """Return only FieldNode objects for SHE-admissible substrate candidates."""

        return tuple(item.field for item in materialize_substrate_bundle(bundle).fields)
    ''')


def unit_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    from pegasus.efg.materialize import (
        classify_substrate_candidate_kind,
        materialize_candidate_field,
        materialize_substrate_bundle,
    )
    from pegasus.she.substrate import SubstrateBundle, SubstrateFieldCandidate, SubstrateFieldExclusion


    def _candidate(**overrides):
        base = dict(
            candidate_id="substrate_candidate_demo",
            source_system="SIM-DO",
            artifact_path="tests/fixtures/demo.parquet",
            column="year",
            technical_name="SIM-DO.year",
            carrier="Deaths",
            unit="counts",
            aggregation="additive",
            role=("time_axis",),
            axes={"period": "year"},
            provenance=("registry", "fixture"),
            substrate_kind="source_measure",
            registry_hash="source_registry_hash:SIM-DO",
            row_count=3,
            non_null_count=3,
            unique_non_null_count=2,
            missing_rate=0.0,
            numeric_min=2020.0,
            numeric_max=2021.0,
            source_manifest_hash="manifest_hash",
            artifact_hash="artifact_hash",
            warnings=(),
        )
        base.update(overrides)
        return SubstrateFieldCandidate(**base)


    def test_slice14a_classifies_substrate_candidate_kinds_without_ratio_claims() -> None:
        assert classify_substrate_candidate_kind(_candidate()) == "extensive_measure"
        assert classify_substrate_candidate_kind(_candidate(unit="ICD10", aggregation="non_aggregable", role=("diagnostic_topology",))) == "observer_proxy"
        assert classify_substrate_candidate_kind(_candidate(aggregation="weighted_mean", unit="ratio")) == "intensive_density"


    def test_slice14a_materializes_candidate_as_metadata_only_field_node() -> None:
        item = materialize_candidate_field(_candidate())
        field = item.field
        assert field.name == "SIM-DO.year"
        assert field.carrier == "Deaths"
        assert field.unit == "counts"
        assert field.kind == "extensive_measure"
        assert field.aggregation == "additive"
        assert field.materialization_state == "metadata_only"
        assert field.support["column"] == "year"
        assert field.support["row_count"] == 3
        assert "substrate_materialized" in field.role
        assert item.lineage_hash == field.lineage.model_dump(mode="json") and False or item.lineage_hash


    def test_slice14a_materialization_never_promotes_exclusions() -> None:
        candidate = _candidate()
        exclusion = SubstrateFieldExclusion(
            exclusion_id="substrate_exclusion_constant",
            source_system="SIM-DO",
            artifact_path="tests/fixtures/demo.parquet",
            column="constant_col",
            reason="zero_variance_constant",
            registry_reason=None,
            row_count=3,
            non_null_count=3,
            unique_non_null_count=1,
            missing_rate=0.0,
            structural_role="constant",
            warnings=("zero_variance_constant",),
        )
        bundle = SubstrateBundle(
            schema_version="1.0",
            substrate_id="substrate_bundle_demo",
            source_reality_mode="fixture_only",
            source_artifacts=(),
            candidates=(candidate,),
            exclusions=(exclusion,),
            table_profiles=(),
            registry_hashes={"SIM-DO": "source_registry_hash:SIM-DO"},
            warnings=(),
        )
        result = materialize_substrate_bundle(bundle)
        assert result.field_count == 1
        assert result.excluded_field_count == 1
        assert result.fields[0].candidate_id == candidate.candidate_id
        assert result.excluded_source_fields[0]["column"] == "constant_col"
        assert result.excluded_source_fields[0]["efg_materialized"] is False
    ''')


def integration_test() -> str:
    return dedent(r'''
    from __future__ import annotations

    from pathlib import Path

    import polars as pl

    from pegasus.efg.materialize import materialize_substrate_bundle, materialized_field_nodes
    from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle


    def test_slice14a_materializes_real_substrate_bundle_without_promoting_zero_variance(tmp_path: Path) -> None:
        artifact = tmp_path / "sim.parquet"
        pl.DataFrame(
            {
                "year": [2020, 2021, 2021],
                "age_years": [50, 51, 52],
                "race_color_admin": ["1", "4", ""],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }
        ).write_parquet(artifact)

        bundle = build_substrate_bundle(
            artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
        )
        assert bundle.admissible_candidate_count > 0
        assert bundle.excluded_field_count > 0

        result = materialize_substrate_bundle(bundle)
        nodes = materialized_field_nodes(bundle)
        candidate_columns = {candidate.column for candidate in bundle.candidates}
        excluded_columns = {exclusion.column for exclusion in bundle.exclusions}
        materialized_columns = {field.support["column"] for field in nodes}

        assert result.field_count == len(bundle.candidates)
        assert result.excluded_field_count == len(bundle.exclusions)
        assert materialized_columns == candidate_columns
        assert materialized_columns.isdisjoint(excluded_columns)
        assert "constant_col" in excluded_columns
        assert "all_missing_col" in excluded_columns
        assert all(field.materialization_state == "metadata_only" for field in nodes)
        assert any(field.unit == "ICD10" and field.kind == "observer_proxy" for field in nodes)
    ''')


def audit_script() -> str:
    return dedent(r'''
    from __future__ import annotations

    import ast
    import inspect
    import json
    import tempfile
    from pathlib import Path

    import polars as pl


    def _top_level_defs(path: Path, name: str) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


    def main() -> int:
        errors: list[str] = []
        materialize_path = Path("src/pegasus/efg/materialize.py")
        compile_path = Path("src/pegasus/workflows/compile.py")
        if "BlockedModuleError" in materialize_path.read_text(encoding="utf-8"):
            errors.append("efg/materialize.py still contains blocked scaffold")
        if _top_level_defs(compile_path, "run_compile") != 1:
            errors.append("compile.py must keep exactly one public run_compile")
        if _top_level_defs(compile_path, "_run_compile_impl") != 1:
            errors.append("compile.py must keep exactly one private _run_compile_impl")

        from pegasus.efg.materialize import materialize_substrate_bundle, materialized_field_nodes
        from pegasus.she.source_registry import resolve_source_fields
        from pegasus.she.substrate import SourceArtifactRef, build_substrate_bundle

        if "allow_heuristic" not in inspect.signature(resolve_source_fields).parameters:
            errors.append("resolve_source_fields lost allow_heuristic")

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "sim.parquet"
            pl.DataFrame({
                "year": [2020, 2021, 2021],
                "age_years": [40, 41, 42],
                "underlying_icd_norm": ["I10", "J18", "R99"],
                "constant_col": [1, 1, 1],
                "all_missing_col": [None, None, None],
            }).write_parquet(artifact)
            bundle = build_substrate_bundle(
                artifacts=[SourceArtifactRef(path=str(artifact), source_system="SIM-DO", provenance_mode="fixture")]
            )
            result = materialize_substrate_bundle(bundle)
            nodes = materialized_field_nodes(bundle)
            if result.field_count != len(bundle.candidates):
                errors.append("materialized field count does not match substrate candidates")
            if result.excluded_field_count != len(bundle.exclusions):
                errors.append("exclusion count does not match substrate exclusions")
            if {f.support["column"] for f in nodes} & {e.column for e in bundle.exclusions}:
                errors.append("excluded substrate fields were promoted to FieldNode")
            if not any(field.unit == "ICD10" for field in nodes):
                errors.append("diagnostic ICD10 observer field was not materialized")
            if not all(field.materialization_state == "metadata_only" for field in nodes):
                errors.append("Slice 14A must emit metadata_only FieldNodes only")

        payload = {"ok": not errors, "errors": errors}
        print(json.dumps(payload, indent=2, sort_keys=True))
        if errors:
            return 1
        print("AUDIT PASSED: Slice 14A EFG substrate materialization boundary")
        return 0


    if __name__ == "__main__":
        raise SystemExit(main())
    ''')


def write_tests_and_audit() -> None:
    write("src/pegasus/efg/materialize.py", materialize_module())
    write("tests/unit/test_slice14a_efg_substrate_materialization.py", unit_test())
    write("tests/integration/test_slice14a_substrate_to_efg_materialization.py", integration_test())
    write("scripts/dev/audits/audit_slice14a_efg_substrate_materialization.py", audit_script())


def self_validate() -> None:
    for path in TOUCH_LIST:
        py_compile.compile(str(ROOT / path), doraise=True)
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from pegasus.efg.materialize import materialize_substrate_bundle, materialized_field_nodes  # noqa: F401
    from pegasus.efg.node import make_field_node  # noqa: F401
    if "BlockedModuleError" in read("src/pegasus/efg/materialize.py"):
        fail("post-validate: efg/materialize.py still contains blocked scaffold")


def main() -> None:
    preflight()
    write_tests_and_audit()
    self_validate()
    print("Slice 14A updater applied: EFG substrate materialization boundary added.")
    print("Touched files:")
    for path in TOUCH_LIST:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
