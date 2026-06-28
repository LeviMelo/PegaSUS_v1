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

from pegasus.she.source_registry import resolve_raw_source_fields


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


def _declared_transform(spec: Any) -> str | None:
    route = _route(spec).lower()
    parser = _spec_get(spec, "parser", "parser_name")
    decoder = _spec_get(spec, "decoder", "decoder_name", "composite_decoder")
    if route == "parse" and parser:
        return _normalize_decoder_name(parser)
    if decoder:
        return _normalize_decoder_name(decoder)
    if parser:
        return _normalize_decoder_name(parser)
    return None


def _output_key(spec: Any) -> str | None:
    value = _spec_get(spec, "output_key", "decoder_output_key", "value_key")
    return None if value is None else str(value)


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


def _decode_registered_value(name: str, value: Any, spec: Any, *, column: str, row: dict[str, Any]) -> Any:
    normalized = name.lower()
    if normalized == "decode_physical_scalar":
        axes = getattr(spec, "axes", {}) or {}
        measure = str(axes.get("measure") or _canonical_name(spec, column))
        if "birth_weight" in measure:
            return _resolve_decoder_callable("decode_physical_scalar")(
                value,
                unit="grams",
                lower=300,
                upper=7000,
                sentinels={"0", "9999"},
            )
        if "gestational" in measure:
            return _resolve_decoder_callable("decode_physical_scalar")(
                value,
                unit="weeks",
                lower=20,
                upper=45,
                sentinels={"0", "99"},
            )
        if "apgar" in measure:
            return _resolve_decoder_callable("decode_physical_scalar")(
                value,
                unit="score",
                lower=0,
                upper=10,
                sentinels={"99"},
            )
    if normalized == "decode_count2":
        return _resolve_decoder_callable("decode_count2")(value, sentinels={"99"})
    if normalized == "decode_sih_age":
        cod_idade = row.get("COD_IDADE", row.get("CODIDADE"))
        idade = row.get("IDADE", value)
        return _resolve_decoder_callable("decode_sih_age")(cod_idade, idade)
    if normalized == "filter_cnpj":
        return _resolve_decoder_callable("filter_cnpj")(value)
    if normalized == "parse_icd":
        fn = _resolve_decoder_callable("parse_icd")
        axes = getattr(spec, "axes", {}) or {}
        role = str(axes.get("icd_topology_role") or "diagnostic_code")
        return fn(value, topology_role=role, source_field=column)

    decoder = _resolve_decoder_callable(name)
    if decoder is None:
        raise LookupError(f"unknown_decoder:{name}")
    return decoder(value)


def _decoded_state(decoded: Any) -> str:
    for name in ("state", "parse_state"):
        value = getattr(decoded, name, None)
        if value is not None:
            return str(value)
    if isinstance(decoded, tuple) and len(decoded) >= 2:
        return str(decoded[1])
    return "decoded"


def _decoded_warning(decoded: Any) -> str | None:
    value = getattr(decoded, "warning", None)
    if value:
        return str(value)
    warnings = getattr(decoded, "warnings", None)
    if warnings:
        return ";".join(str(item) for item in warnings)
    return None


def _decoded_value(decoded: Any, output_key: str | None) -> Any:
    key = output_key or "value"
    if hasattr(decoded, key):
        return getattr(decoded, key)
    if key == "value":
        for fallback in ("normalized", "cnpj", "age_years", "value"):
            if hasattr(decoded, fallback):
                return getattr(decoded, fallback)
    if isinstance(decoded, dict):
        if key in decoded:
            return decoded[key]
        return decoded
    if isinstance(decoded, tuple):
        return decoded[0] if decoded else None
    return decoded


def _fallback_parse(value: Any, *, spec: Any) -> tuple[Any, str, str]:
    if value in {None, ""}:
        return None, "missing", "missingness_preserved"

    kind = " ".join(
        str(_spec_get(spec, key) or "")
        for key in ("field_kind", "quality_role", "unit", "semantic_type", "axis", "carrier")
    ).lower()
    axes = getattr(spec, "axes", {}) or {}
    axes_text = " ".join(str(k) + " " + str(v) for k, v in axes.items()).lower()

    if "municip" in kind or "geograph" in kind or "municip" in axes_text or "geograph" in axes_text:
        parsed, state = _parse_municipality(value)
        return parsed, state, "municipality_parser"

    if "icd" in kind or "diagnostic" in kind:
        parsed, state = _parse_icd10(value)
        return parsed, state, "icd10_parser"

    return value, "valid", "preserve_mark"


def _decode_value(*, source_system: str, column: str, value: Any, spec: Any, row: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_name(spec, column)
    route = _route(spec)

    if route.lower() == "exclude":
        return {
            "canonical_field": canonical,
            "value": None,
            "state": "excluded",
            "decoder": "Exclude",
            "excluded": True,
        }

    decoder_name = _declared_transform(spec)
    if decoder_name is not None:
        try:
            decoded = _decode_registered_value(decoder_name, value, spec, column=column, row=row)
            return {
                "canonical_field": canonical,
                "value": _decoded_value(decoded, _output_key(spec)),
                "state": _decoded_state(decoded),
                "decoder": decoder_name,
                "excluded": False,
                "warning": _decoded_warning(decoded),
            }
        except Exception as exc:
            return {
                "canonical_field": canonical,
                "value": None,
                "state": "invalid",
                "decoder": decoder_name,
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
        resolutions = resolve_raw_source_fields(
            source_system=source_system,
            raw_column_name=str(column),
            registry_root=registry_root,
        )
        handled = False
        for resolution in resolutions:
            spec = resolution.spec
            decoded = _decode_value(
                source_system=source_system,
                column=str(column),
                value=value,
                spec=spec,
                row=row,
            )

            warnings = list(resolution.warnings or [])
            if decoded.get("warning"):
                warnings.append(str(decoded["warning"]))

            if decoded.get("excluded"):
                out["_warnings"].extend(warnings)
                continue

            handled = True
            canonical = str(decoded["canonical_field"])
            out[canonical] = decoded["value"]
            out[f"{canonical}__state"] = decoded["state"]
            out[f"{canonical}__decoder"] = decoded["decoder"]
            out["_warnings"].extend(warnings)
        if not handled:
            out["_excluded_columns"].append(str(column))

    out["_warnings"] = list(dict.fromkeys(str(item) for item in out["_warnings"]))
    return out


def normalize_sim_do_record(row: dict[str, Any], *, registry_root: str = "config/registries") -> dict[str, Any]:
    return normalize_record(row, source_system="SIM-DO", registry_root=registry_root)


def normalize_sinasc_record(row: dict[str, Any], *, registry_root: str = "config/registries") -> dict[str, Any]:
    return normalize_record(row, source_system="SINASC", registry_root=registry_root)


__all__ = ["normalize_record", "normalize_sim_do_record", "normalize_sinasc_record"]
