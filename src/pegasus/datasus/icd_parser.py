from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict


ICD_RE = re.compile(r"^[A-Z][0-9]{2}[0-9A-Z]?$")
ICD_ALLOWED_INITIALS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ICDParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw: str | None
    normalized: str | None
    parse_state: Literal["valid", "ill-defined", "blank", "invalid", "unparseable", "missing"]
    topology_role: str
    position: str | None
    source_field: str
    raw_marker: str | None = None
    warnings: list[str]


def normalize_icd(raw: str | None, *, strip_asterisk: bool = True) -> tuple[str | None, str | None]:
    if raw is None:
        return None, None

    value = str(raw).strip().upper()
    if value == "":
        return "", None

    marker = None
    if strip_asterisk and value.startswith("*"):
        marker = "*"
        value = value[1:].strip()

    value = value.replace(".", "")
    return value, marker


def parse_icd(
    raw: str | None,
    *,
    topology_role: str,
    source_field: str,
    position: str | None = None,
    strip_asterisk: bool = True,
) -> ICDParseResult:
    if raw is None:
        return ICDParseResult(
            raw=None,
            normalized=None,
            parse_state="missing",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            warnings=["missing_icd"],
        )

    normalized, marker = normalize_icd(raw, strip_asterisk=strip_asterisk)

    if normalized == "":
        return ICDParseResult(
            raw=str(raw),
            normalized=None,
            parse_state="blank",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=["blank_icd"],
        )

    if normalized is None:
        return ICDParseResult(
            raw=str(raw),
            normalized=None,
            parse_state="missing",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=["missing_icd"],
        )

    if not ICD_RE.fullmatch(normalized):
        return ICDParseResult(
            raw=str(raw),
            normalized=normalized,
            parse_state="unparseable",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=["unparseable_icd"],
        )

    if normalized[0] not in ICD_ALLOWED_INITIALS:
        return ICDParseResult(
            raw=str(raw),
            normalized=normalized,
            parse_state="invalid",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=["invalid_icd_initial"],
        )

    # Conservative quality routing: R-codes are syntactically valid but ill-defined
    # for disease-outcome use unless explicitly allowed by a registry.
    if normalized.startswith("R"):
        return ICDParseResult(
            raw=str(raw),
            normalized=normalized,
            parse_state="ill-defined",
            topology_role=topology_role,
            position=position,
            source_field=source_field,
            raw_marker=marker,
            warnings=["ill_defined_icd"],
        )

    return ICDParseResult(
        raw=str(raw),
        normalized=normalized,
        parse_state="valid",
        topology_role=topology_role,
        position=position,
        source_field=source_field,
        raw_marker=marker,
        warnings=[],
    )
