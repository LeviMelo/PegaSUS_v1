"""Disease Concept Registry (MSD-II §II.13.1).

A registry of provenance- and projection-typed **code -> concept** assertions, imported
from multiple groupers. Two contracts are load-bearing:

- **Multi-label preserved.** A code MAY hold several concepts (AHRQ CCSR is many-to-many);
  it is NEVER silently forced to one. Forcing a partition is a separate, explicit view
  (``icd_curated_groups.yaml`` remains one partition-view over this multi-label registry).
- **No silent cross-system truth.** CCSR/CCIR groupers are defined over ICD-10-CM, not
  Brazilian CID-10; assertions imported from them carry ``projection_status`` != ``exact``
  and a warning, so downstream fields enter at ``state <= fragile`` (§6). WHO ICD-10
  structural concepts (chapter/block) are exact for valid WHO codes; Brazilian custom lists
  are exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pegasus.core.config import load_yaml
from pegasus.disease import icd_adapter

ProjectionStatus = Literal[
    "exact", "parent_projection", "approximate", "unmappable", "source_system_specific"
]
Chronicity = Literal["chronic", "acute", "mixed", "unknown"]

REGISTRY_FILE = "disease/disease_concepts.yaml"


@dataclass(frozen=True)
class DiseaseConceptAssertion:
    """One code->concept assertion (§II.13.1 normative record)."""

    code: str                         # dot-normalized, e.g. "I21.9"
    code_system: str                  # CID-10 | ICD-10 | ICD-10-CM | ...
    concept_id: str                   # e.g. ccsr_CIR007 | chapter_IX | brazilian_icsap
    concept_family: str               # chronicity | body_system | clinical_category | pediatric_ccc | avoidable | custom
    source: str                       # icd_mappings@0.6.2 | who_icd10 | brazilian_list
    multi_label: bool
    projection_status: ProjectionStatus
    chronic: Chronicity = "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)


# icd-mappings grouper handling. ``kind='category'`` => the mapper returns a category id (or
# a list of them for a true multi-label source); ``kind='boolean_chronic'`` => the Chronic
# Condition Indicator returns True/False. CCSR/CCIR are ICD-10-CM (US) => approximate on
# CID-10. NOTE: icd-mappings@0.6.2's CCSR returns the PRIMARY category only, not the full
# AHRQ 1-to-6 multi-label set; the concept_family is still multi-label (a code CAN carry
# several), flagged with ``ccsr_primary_category_only``.
_GROUPERS: dict[str, dict[str, Any]] = {
    "ccsr": {"kind": "category", "family": "clinical_category", "prefix": "ccsr",
             "multi": True, "projection": "approximate",
             "warn": ("icd10cm_grouper_on_cid10", "ccsr_primary_category_only")},
    "ccir": {"kind": "boolean_chronic", "family": "chronicity", "prefix": "chronicity",
             "multi": False, "projection": "approximate", "warn": ("icd10cm_grouper_on_cid10",)},
    "ccc_category": {"kind": "category", "family": "pediatric_ccc", "prefix": "ccc",
                     "multi": False, "projection": "approximate", "warn": ("ccc_grouper_cross_system",)},
}


@lru_cache(maxsize=1)
def _mapper():
    from icdmappings import Mapper
    return Mapper()


def _map_raw(code_no_dot: str, grouper: str) -> Any:
    """Raw icd-mappings result for a code (dotless icd10 in); None on failure."""
    try:
        return _mapper().map(code_no_dot, source="icd10", target=grouper)
    except Exception:
        return None


def _as_categories(result: Any) -> list[str]:
    """Coerce a category-mapper result into a clean list of category ids."""
    if result is None or isinstance(result, bool):
        return []
    values = result if isinstance(result, (list, tuple, set)) else [result]
    return [str(v) for v in values if v is not None and str(v).lower() not in {"none", "nan", ""}]


@lru_cache(maxsize=1)
def _custom_concepts(root: str) -> tuple[dict[str, Any], ...]:
    """Brazilian custom concept definitions (code-range -> concept), loaded from the registry
    yaml. Each: {concept_id, concept_family, chronic, ranges:[3-char intervals], warnings}."""
    path = Path(root) / REGISTRY_FILE
    if not path.exists():
        return ()
    payload = load_yaml(path) or {}
    return tuple(payload.get("custom_concepts", []) or [])


def _in_ranges(category: str | None, ranges: list[str]) -> bool:
    if not category:
        return False
    cat = category.replace(".", "")[:3].upper()
    for interval in ranges:
        lo, _, hi = str(interval).partition("-")
        hi = hi or lo
        if lo[:1] == cat[:1] and lo.upper() <= cat <= hi.upper():
            return True
    return False


def assertions_for_code(
    code: str,
    *,
    code_system: str = "CID-10",
    registry_root: str | Path = "config/registries",
    enabled_groupers: tuple[str, ...] = ("ccsr", "ccir", "ccc_category"),
) -> list[DiseaseConceptAssertion]:
    """All concept assertions for a single DATASUS code — multi-label, provenance-typed.

    Structural concepts: the chapter is exact for every DATASUS code (CID-10 authoritative);
    the block is exact when the code is a WHO node. icd-mappings groupers (CCSR multi-label,
    CCI chronicity, CCC) are ICD-10-CM/ICD-10 US groupers => approximate on CID-10. Brazilian
    custom lists are exact. A code absent from the WHO tree (e.g. dengue A90) is still typed
    and gets its CID-10 chapter + custom + any grouper matches -- it is NOT dropped.
    """
    info = icd_adapter.code_info(code)
    out: list[DiseaseConceptAssertion] = []

    if info.status == "malformed":
        return [DiseaseConceptAssertion(
            code=code, code_system=code_system, concept_id="unmappable_malformed",
            concept_family="custom", source="pegasus", multi_label=False,
            projection_status="unmappable", warnings=("malformed_code",))]

    normalized = info.normalized or code
    who_absent = info.status == "source_system_specific"
    if who_absent:
        out.append(DiseaseConceptAssertion(
            code=normalized, code_system=code_system,
            concept_id=f"source_specific_{(info.category or code)}", concept_family="custom",
            source="pegasus", multi_label=False, projection_status="source_system_specific",
            warnings=("cid10_code_absent_from_who_icd10",)))

    # Chapter is authoritative for every code (CID-10 range table backs WHO-absent codes).
    if info.chapter:
        out.append(DiseaseConceptAssertion(
            code=normalized, code_system=code_system, concept_id=f"chapter_{info.chapter}",
            concept_family="body_system", source=("cid10_catalog" if who_absent else "who_icd10"),
            multi_label=False, projection_status="exact"))
    # Block requires the WHO tree.
    if info.block:
        out.append(DiseaseConceptAssertion(
            code=normalized, code_system=code_system, concept_id=f"block_{info.block}",
            concept_family="body_system", source="who_icd10", multi_label=False,
            projection_status="exact"))

    # icd-mappings groupers (cross-system approximate) — attempted for every code; a WHO-absent
    # CID-10 code may still map (icd-mappings uses a different code table).
    dotless = icd_adapter.remove_dot(normalized)
    for grouper in enabled_groupers:
        spec = _GROUPERS.get(grouper)
        if not spec:
            continue
        raw = _map_raw(dotless, grouper)
        if spec["kind"] == "boolean_chronic":
            if raw is True:  # Chronic Condition Indicator: this code is a chronic condition.
                out.append(DiseaseConceptAssertion(
                    code=normalized, code_system=code_system, concept_id="chronicity_chronic",
                    concept_family="chronicity", source="icd_mappings@0.6.2", multi_label=False,
                    projection_status=spec["projection"], chronic="chronic",
                    warnings=tuple(spec["warn"])))
            continue
        for category in _as_categories(raw):
            out.append(DiseaseConceptAssertion(
                code=normalized, code_system=code_system,
                concept_id=f"{spec['prefix']}_{category}", concept_family=spec["family"],
                source="icd_mappings@0.6.2", multi_label=spec["multi"],
                projection_status=spec["projection"], warnings=tuple(spec["warn"])))

    out.extend(_custom_assertions(info.category, code_system, registry_root))
    return out


def _custom_assertions(category: str | None, code_system: str,
                       registry_root: str | Path) -> list[DiseaseConceptAssertion]:
    out: list[DiseaseConceptAssertion] = []
    for concept in _custom_concepts(str(registry_root)):
        if _in_ranges(category, list(concept.get("ranges", []))):
            out.append(DiseaseConceptAssertion(
                code=icd_adapter.add_dot(category or ""), code_system=code_system,
                concept_id=str(concept["concept_id"]),
                concept_family=str(concept.get("concept_family", "custom")),
                source="brazilian_list", multi_label=bool(concept.get("multi_label", True)),
                projection_status="exact", chronic=str(concept.get("chronic", "unknown")),
                warnings=tuple(concept.get("warnings", []) or [])))
    return out


__all__ = [
    "DiseaseConceptAssertion",
    "ProjectionStatus",
    "Chronicity",
    "assertions_for_code",
    "REGISTRY_FILE",
]
