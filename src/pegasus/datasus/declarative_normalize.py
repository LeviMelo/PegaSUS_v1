"""Registry-routed DATASUS record normalization.

SHE authority lives in the Source Field Registry. This module does not
special-case source columns such as IDADE/DTOBITO/CODMUNRES. It reads the
registry resolution for each raw column, dispatches registered decoders and
parsers by name, and records excluded columns explicitly.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Callable

from pegasus.she.source_registry import resolve_source_field


Decoder = Callable[[Any], Any]


def _normalize_decoder_name(name: Any) -> str | None:
    if name is None:
        return None
    text = str(name).strip()
    if not text or text.lower() in {"none", "null", "na"}:
        return None
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def _resolve_decoder_callable(name: str | None) -> Decoder | None:
    if not name:
        return None

    candidates = [
        name,
        name.lower(),
        name.upper(),
        f"decode_{name}",
        f"decode_{name.lower()}",
    ]

    modules = (
        "pegasus.datasus.decoders",
        "pegasus.datasus.icd_parser",
    )

    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for candidate in candidates:
            fn = getattr(module, candidate, None)
            if callable(fn):
                return fn
    return None


def _spec_get(spec: Any, *names: str) -> Any:
    for name in names:
        if hasattr(spec, name):
            value = getattr(spec, name)
            if value is not None:
                return value
    metadata = getattr(spec, "metadata", None)
    if isinstance(metadata, dict):
        for name in names:
            if metadata.get(name) is not None:
                return metadata.get(name)
    return None


def _canonical_name(spec: Any, column: str) -> str:
    for name in (
        "canonical_field",
        "canonical_name",
        "target_field",
        "column_name",
        "field_id",
        "name",
    ):
        value = _spec_get(spec, name)
        if value:
            return str(value)
    return str(column)


def _route(spec: Any) -> str:
    value = _spec_get(spec, "route", "registry_route", "field_route", "action")
    if value:
        return str(value)
    if getattr(spec, "admissible", False):
        return "PreserveMark"
    return "Exclude"


def _declared_decoder(spec: Any) -> str | None:
    value = _spec_get(
        spec,
        "decoder",
        "decoder_name",
        "composite_decoder",
        "parser",
        "parser_name",
        "transform",
        "codec",
    )
    return _normalize_decoder_name(value)


def _parse_municipality(value: Any) -> tuple[str | None, str]:
    if value in {None, ""}:
        return None, "missing"
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) >= 6:
        return digits[:6], "parsed"
    return None, "invalid"


def _parse_icd10(value: Any) -> tuple[str | None, str]:
    if value in {None, ""}:
        return None, "missing"
    text = str(value).strip().upper()
    return (text or None), ("parsed" if text else "missing")


def _fallback_parse(value: Any, *, spec: Any) -> tuple[Any, str, str]:
    if value in {None, ""}:
        return None, "missing", "missingness_preserved"

    kind = " ".join(
        str(_spec_get(spec, key) or "")
        for key in ("field_kind", "quality_role", "unit", "semantic_type", "axis", "carrier")
    ).lower()

    if "municip" in kind or "geograph" in kind:
        parsed, state = _parse_municipality(value)
        return parsed, state, "municipality_parser"

    if "icd" in kind or "diagnostic" in kind:
        parsed, state = _parse_icd10(value)
        return parsed, state, "icd10_parser"

    return value, "valid", "preserve_mark"


def _decode_value(*, source_system: str, column: str, value: Any, spec: Any) -> dict[str, Any]:
    canonical = _canonical_name(spec, column)
    route = _route(spec)

    if route.lower() == "exclude" or not getattr(spec, "admissible", False):
        return {
            "canonical_field": canonical,
            "value": None,
            "state": "excluded",
            "decoder": "Exclude",
            "excluded": True,
        }

    decoder_name = _declared_decoder(spec)
    decoder = _resolve_decoder_callable(decoder_name)
    if decoder is not None:
        try:
            decoded = decoder(value)
            return {
                "canonical_field": canonical,
                "value": decoded,
                "state": "decoded",
                "decoder": decoder_name or decoder.__name__,
                "excluded": False,
            }
        except Exception as exc:
            return {
                "canonical_field": canonical,
                "value": None,
                "state": "invalid",
                "decoder": decoder_name or decoder.__name__,
                "excluded": False,
                "warning": f"decoder_failed:{type(exc).__name__}:{exc}",
            }

    parsed, state, fallback_decoder = _fallback_parse(value, spec=spec)
    return {
        "canonical_field": canonical,
        "value": parsed,
        "state": state,
        "decoder": decoder_name or fallback_decoder,
        "excluded": False,
    }


def normalize_record(
    row: dict[str, Any],
    *,
    source_system: str,
    registry_root: str = "config/registries",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "_source_system": source_system,
        "_registry_routed": True,
        "_excluded_columns": [],
        "_warnings": [],
    }

    for column, value in row.items():
        resolution = resolve_source_field(
            source_system=source_system,
            column_name=str(column),
            registry_root=registry_root,
        )
        spec = resolution.spec
        decoded = _decode_value(
            source_system=source_system,
            column=str(column),
            value=value,
            spec=spec,
        )

        warnings = list(resolution.warnings or [])
        if decoded.get("warning"):
            warnings.append(str(decoded["warning"]))

        if decoded.get("excluded"):
            out["_excluded_columns"].append(str(column))
            out["_warnings"].extend(warnings)
            continue

        canonical = str(decoded["canonical_field"])
        out[canonical] = decoded["value"]
        out[f"{canonical}__state"] = decoded["state"]
        out[f"{canonical}__decoder"] = decoded["decoder"]
        out["_warnings"].extend(warnings)

    out["_warnings"] = list(dict.fromkeys(str(item) for item in out["_warnings"]))
    return out


def normalize_sim_do_record(row: dict[str, Any], *, registry_root: str = "config/registries") -> dict[str, Any]:
    return normalize_record(row, source_system="SIM-DO", registry_root=registry_root)


def normalize_sinasc_record(row: dict[str, Any], *, registry_root: str = "config/registries") -> dict[str, Any]:
    return normalize_record(row, source_system="SINASC", registry_root=registry_root)


__all__ = ["normalize_record", "normalize_sim_do_record", "normalize_sinasc_record"]
