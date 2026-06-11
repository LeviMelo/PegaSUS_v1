from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SIDRAStitchingError(ValueError):
    """Raised when SIDRA longitudinal segments cannot be legally stitched."""


@dataclass(frozen=True)
class SIDRASegment:
    concept_id: str
    table_id: str
    variable_id: str
    segment_id: str
    periods: tuple[str, ...]
    unit: str
    classification_version: str
    source_hash: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SIDRASegment":
        periods = tuple(str(p) for p in payload.get("periods", []))
        if not periods:
            raise SIDRAStitchingError("SIDRA segment must declare at least one period.")
        return cls(
            concept_id=str(payload["concept_id"]),
            table_id=str(payload["table_id"]),
            variable_id=str(payload["variable_id"]),
            segment_id=str(payload.get("segment_id") or f"{payload['table_id']}:{payload['variable_id']}"),
            periods=periods,
            unit=str(payload.get("unit") or "unknown"),
            classification_version=str(payload.get("classification_version") or "unclassified"),
            source_hash=str(payload.get("source_hash")) if payload.get("source_hash") is not None else None,
        )

    def as_manifest(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "table_id": self.table_id,
            "variable_id": self.variable_id,
            "segment_id": self.segment_id,
            "periods": list(self.periods),
            "unit": self.unit,
            "classification_version": self.classification_version,
            "source_hash": self.source_hash,
        }


@dataclass(frozen=True)
class SIDRAStitchingResult:
    concept_id: str
    status: str
    stitched_periods: tuple[str, ...]
    segment_provenance: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    reason: str | None = None

    def as_manifest(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "status": self.status,
            "stitched_periods": list(self.stitched_periods),
            "segment_provenance": list(self.segment_provenance),
            "warnings": list(self.warnings),
            "reason": self.reason,
        }


def stitch_sidra_longitudinal_segments(segments: list[SIDRASegment]) -> SIDRAStitchingResult:
    if not segments:
        raise SIDRAStitchingError("Cannot stitch empty SIDRA segment list.")

    concept_ids = {s.concept_id for s in segments}
    if len(concept_ids) != 1:
        raise SIDRAStitchingError("SIDRA stitching cannot merge different concept_id values.")
    concept_id = next(iter(concept_ids))

    units = {s.unit for s in segments}
    if len(units) != 1:
        return SIDRAStitchingResult(
            concept_id=concept_id,
            status="blocked",
            stitched_periods=tuple(),
            segment_provenance=tuple(s.as_manifest() for s in segments),
            warnings=("sidra_stitch_unit_mismatch",),
            reason="SIDRA longitudinal stitching requires unit compatibility.",
        )

    by_period: dict[str, SIDRASegment] = {}
    warnings: list[str] = []
    for segment in segments:
        for period in segment.periods:
            if period in by_period and by_period[period].segment_id != segment.segment_id:
                warnings.append("sidra_stitch_overlap_segment_priority")
            by_period.setdefault(period, segment)

    if len({s.table_id for s in segments}) > 1 or len({s.variable_id for s in segments}) > 1:
        warnings.append("sidra_table_identity_not_concept_identity")
        warnings.append("sidra_stitch_segment_provenance")

    if len({s.classification_version for s in segments}) > 1:
        warnings.append("sidra_stitch_classification_version_change")

    return SIDRAStitchingResult(
        concept_id=concept_id,
        status="stitched" if len(segments) > 1 else "direct",
        stitched_periods=tuple(sorted(by_period.keys())),
        segment_provenance=tuple(s.as_manifest() for s in segments),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def stitch_from_payload(payload: dict[str, Any]) -> SIDRAStitchingResult:
    return stitch_sidra_longitudinal_segments([SIDRASegment.from_mapping(x) for x in payload.get("segments", [])])
