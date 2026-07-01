from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ICDGroup:
    id: str
    label: str
    start: str
    end: str
    kind: str


ICD_CHAPTERS = [
    ICDGroup("CHAPTER_01_A00_B99", "Certain infectious and parasitic diseases", "A00", "B99", "chapter"),
    ICDGroup("CHAPTER_02_C00_D48", "Neoplasms", "C00", "D48", "chapter"),
    ICDGroup("CHAPTER_03_D50_D89", "Diseases of the blood and immune mechanism", "D50", "D89", "chapter"),
    ICDGroup("CHAPTER_04_E00_E90", "Endocrine, nutritional and metabolic diseases", "E00", "E90", "chapter"),
    ICDGroup("CHAPTER_05_F00_F99", "Mental and behavioural disorders", "F00", "F99", "chapter"),
    ICDGroup("CHAPTER_06_G00_G99", "Diseases of the nervous system", "G00", "G99", "chapter"),
    ICDGroup("CHAPTER_07_H00_H59", "Diseases of the eye and adnexa", "H00", "H59", "chapter"),
    ICDGroup("CHAPTER_08_H60_H95", "Diseases of the ear and mastoid process", "H60", "H95", "chapter"),
    ICDGroup("CHAPTER_09_I00_I99", "Diseases of the circulatory system", "I00", "I99", "chapter"),
    ICDGroup("CHAPTER_10_J00_J99", "Diseases of the respiratory system", "J00", "J99", "chapter"),
    ICDGroup("CHAPTER_11_K00_K93", "Diseases of the digestive system", "K00", "K93", "chapter"),
    ICDGroup("CHAPTER_12_L00_L99", "Diseases of the skin and subcutaneous tissue", "L00", "L99", "chapter"),
    ICDGroup("CHAPTER_13_M00_M99", "Diseases of the musculoskeletal system", "M00", "M99", "chapter"),
    ICDGroup("CHAPTER_14_N00_N99", "Diseases of the genitourinary system", "N00", "N99", "chapter"),
    ICDGroup("CHAPTER_15_O00_O99", "Pregnancy, childbirth and puerperium", "O00", "O99", "chapter"),
    ICDGroup("CHAPTER_16_P00_P96", "Certain conditions originating in the perinatal period", "P00", "P96", "chapter"),
    ICDGroup("CHAPTER_17_Q00_Q99", "Congenital malformations and chromosomal abnormalities", "Q00", "Q99", "chapter"),
    ICDGroup("CHAPTER_18_R00_R99", "Symptoms, signs and abnormal clinical findings", "R00", "R99", "chapter"),
    ICDGroup("CHAPTER_19_S00_T98", "Injury, poisoning and external-cause consequences", "S00", "T98", "chapter"),
    ICDGroup("CHAPTER_20_V01_Y98", "External causes of morbidity and mortality", "V01", "Y98", "chapter"),
    ICDGroup("CHAPTER_21_Z00_Z99", "Factors influencing health status", "Z00", "Z99", "chapter"),
    ICDGroup("CHAPTER_22_U00_U99", "Codes for special purposes", "U00", "U99", "chapter"),
]

# Minimal stable block registry for common examples.
# This is intentionally not a full ICD catalog; the full catalog remains registry-backed.
ICD_BLOCKS = [
    ICDGroup("BLOCK_A30_A49", "Other bacterial diseases", "A30", "A49", "block"),
    ICDGroup("BLOCK_I10_I15", "Hypertensive diseases", "I10", "I15", "block"),
    ICDGroup("BLOCK_J09_J18", "Influenza and pneumonia", "J09", "J18", "block"),
    ICDGroup("BLOCK_P35_P39", "Infections specific to the perinatal period", "P35", "P39", "block"),
    ICDGroup("BLOCK_Q20_Q28", "Congenital malformations of the circulatory system", "Q20", "Q28", "block"),
    ICDGroup("BLOCK_R95_R99", "Ill-defined and unknown causes of mortality", "R95", "R99", "block"),
]


ICD_RE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]?$")


def _base_code(code: str | None) -> str | None:
    if code is None:
        return None
    normalized = str(code).strip().upper().replace(".", "")
    if not ICD_RE.fullmatch(normalized):
        return None
    return normalized[:3]


def _key(code3: str) -> tuple[int, int]:
    return (ord(code3[0]) - ord("A"), int(code3[1:3]))


def _in_range(code3: str, start: str, end: str) -> bool:
    return _key(start) <= _key(code3) <= _key(end)


def chapter_for_icd(code: str | None) -> ICDGroup | None:
    code3 = _base_code(code)
    if code3 is None:
        return None
    for group in ICD_CHAPTERS:
        if _in_range(code3, group.start, group.end):
            return group
    return None


def block_for_icd(code: str | None) -> ICDGroup | None:
    code3 = _base_code(code)
    if code3 is None:
        return None
    for group in ICD_BLOCKS:
        if _in_range(code3, group.start, group.end):
            return group
    return None


@lru_cache(maxsize=8)
def _curated_groups(registry_root: str) -> tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]:
    """Load curated cause groups (MSD §3.11 G_curated) as (id, label, ranges) tuples."""
    import yaml

    path = Path(registry_root) / "health/icd_curated_groups.yaml"
    if not path.exists():
        return ()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for entry in payload.get("entries", []) or []:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        ranges = tuple(
            (str(r[0]).upper(), str(r[1]).upper())
            for r in (entry.get("ranges") or [])
            if isinstance(r, (list, tuple)) and len(r) >= 2
        )
        if ranges:
            out.append((str(entry["id"]), str(entry.get("label") or entry["id"]), ranges))
    return tuple(out)


def curated_group_for_icd(code: str | None, *, registry_root: str | Path = "config/registries") -> ICDGroup | None:
    """Map an ICD-10 code to its curated cause group (registry-driven, §3.11 V_M04).

    Returns the matching group, or an explicit ``OTHER`` residual for well-formed codes that
    fall outside every curated group (so the σ_C partition stays exact). Unparseable codes
    return ``None`` (handled as UNCLASSIFIED upstream)."""
    code3 = _base_code(code)
    if code3 is None:
        return None
    for group_id, label, ranges in _curated_groups(str(registry_root)):
        for start, end in ranges:
            if _in_range(code3, start, end):
                return ICDGroup(group_id, label, start, end, "curated")
    return ICDGroup("OTHER", "Other curated cause", code3, code3, "curated")
