from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ParseState = Literal[
    "valid",
    "ill_defined",
    "blank",
    "invalid",
    "unparseable",
    "missing",
]

ValueState = Literal[
    "valid",
    "unknown",
    "missing",
    "blank",
    "invalid",
    "unparseable",
    "not_recorded",
    "not_applicable",
    "invalid_mark_state",
]


class ParsedValue(BaseModel):
    raw: object | None
    normalized: object | None
    state: ValueState
    warnings: list[str] = Field(default_factory=list)


class ParsedICD10(BaseModel):
    raw: object | None
    normalized: str | None
    state: ParseState
    warnings: list[str] = Field(default_factory=list)


class ParsedDate(BaseModel):
    raw: object | None
    normalized: str | None
    state: ValueState
    warnings: list[str] = Field(default_factory=list)


class ParsedAge(BaseModel):
    raw: object | None
    age_years: float | None
    age_days: float | None
    state: ValueState
    source: str
    warnings: list[str] = Field(default_factory=list)


class ParsedMunicipality(BaseModel):
    raw: object | None
    cod6: str | None
    cod7: str | None
    state: ValueState
    warnings: list[str] = Field(default_factory=list)