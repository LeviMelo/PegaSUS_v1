"""Auditable failed EFG expansion records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pegasus.core.hashing import content_hash
from pegasus.core.schemas import AlignmentResult, DeltaResult, FieldNode
from pegasus.efg.declaration import OperatorSpec


FailureDisposition = Literal["blocked", "illegal", "unsupported", "deferred"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FailedBranchRecord:
    failed_branch_id: str
    attempted_operator: str
    input_field_ids: tuple[str, ...]
    source_systems: tuple[str, ...]
    failure_stage: str
    failed_terms: tuple[str, ...]
    reason: str
    disposition: FailureDisposition
    support_axis_mismatch: dict[str, Any]
    required_module: str | None
    warnings: tuple[str, ...]
    created_at: str

    @property
    def parent_field_ids(self) -> list[str]:
        return list(self.input_field_ids)

    def as_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_field_ids"] = list(self.input_field_ids)
        payload["parent_field_ids"] = list(self.input_field_ids)
        payload["source_systems"] = list(self.source_systems)
        payload["failed_terms"] = list(self.failed_terms)
        payload["warnings"] = list(self.warnings)
        return payload


def _required_module(alignment: AlignmentResult | None, warnings: list[str]) -> str | None:
    values = [*(alignment.warnings if alignment else []), *warnings]
    for warning in values:
        if warning.startswith("required_module:"):
            return warning.split(":", 1)[1]
    return None


def _stage(failed_terms: list[str], alignment: AlignmentResult | None) -> str:
    if alignment is not None and not alignment.ok:
        return "alignment"
    allowed = {
        "support", "axes", "carrier", "unit", "aggregation", "provenance",
        "quality", "declaration", "materialization", "solver", "output_validation",
    }
    return next((term for term in failed_terms if term in allowed), "materialization")


def make_failed_branch_record(
    *,
    parents: list[FieldNode],
    operator: OperatorSpec,
    reason: str,
    delta: DeltaResult | None = None,
    alignment: AlignmentResult | None = None,
    disposition: FailureDisposition = "illegal",
    warnings: list[str] | None = None,
) -> FailedBranchRecord:
    failed_terms = list(delta.failed_terms if delta else [])
    if alignment is not None and not alignment.ok and "axes" not in failed_terms:
        failed_terms.append("axes")
    all_warnings = list(dict.fromkeys([
        *(delta.warnings if delta else []),
        *(alignment.warnings if alignment else []),
        *(warnings or []),
    ]))
    mismatch = dict((alignment.support_after_alignment if alignment else None) or {})
    payload = {
        "operator": operator.name,
        "parents": [parent.id for parent in parents],
        "failed_terms": failed_terms,
        "reason": reason,
        "disposition": disposition,
        "mismatch": mismatch,
    }
    return FailedBranchRecord(
        failed_branch_id=f"failed_{content_hash(payload)[:24]}",
        attempted_operator=operator.name,
        input_field_ids=tuple(parent.id for parent in parents),
        source_systems=tuple(sorted({source for parent in parents for source in parent.source[:1]})),
        failure_stage=_stage(failed_terms, alignment),
        failed_terms=tuple(dict.fromkeys(failed_terms)),
        reason=reason,
        disposition=disposition,
        support_axis_mismatch=mismatch,
        required_module=_required_module(alignment, all_warnings),
        warnings=tuple(all_warnings),
        created_at=_now(),
    )


def failed_exclusion(
    *,
    exclusion: dict[str, Any],
    disposition: FailureDisposition = "illegal",
) -> FailedBranchRecord:
    reason = str(exclusion.get("reason", "substrate_excluded"))
    payload = {
        "exclusion_id": exclusion.get("exclusion_id"),
        "source_system": exclusion.get("source_system"),
        "column": exclusion.get("column"),
        "reason": reason,
    }
    return FailedBranchRecord(
        failed_branch_id=f"failed_{content_hash(payload)[:24]}",
        attempted_operator="raw_field_admission",
        input_field_ids=(),
        source_systems=(str(exclusion.get("source_system", "unknown")),),
        failure_stage="materialization",
        failed_terms=("quality",),
        reason=reason,
        disposition=disposition,
        support_axis_mismatch={
            "artifact_path": exclusion.get("artifact_path"),
            "column": exclusion.get("column"),
            "registry_reason": exclusion.get("registry_reason"),
        },
        required_module=None,
        warnings=tuple(str(value) for value in exclusion.get("warnings", [])),
        created_at=_now(),
    )
