"""Raw-column completeness guardrail for DATASUS normalization.

The normalizers decode **raw DBC codes** (`canonical_raw` in the R bridge), never
microdatasus's ``process_*`` output — that is deliberate: reading raw makes the
pipeline immune to ``process_*`` dropping columns or failing outright (e.g.
``process_cnes`` can return zero rows, yet the raw table still carries all ~209
columns). See ``r_scripts/fetch_process_microdatasus.R``:
``canonical_processed_role = "raw_codes_for_python_normalizer"``.

Because the vectorized primitives resolve an absent column to a null literal
(``Cols.first``), a genuinely missing raw column would be decoded *silently* to all
nulls rather than raising. This guardrail closes that gap: it names, per system, the
raw columns whose absence would silently corrupt a canonical field, and reports any
that are missing from a normalizer's input so the drop is visible instead of silent.
"""

from __future__ import annotations

import warnings

import polars as pl

# Primary DBC column names each normalizer depends on to produce a faithful record
# (identity + the key measures). These are the standard DATASUS names the normalizers
# read as the first coalesce alias; a real DBC extract always carries them, so any
# absence signals an upstream drop (a changed microdatasus version, a truncated file)
# rather than normal optionality.
REQUIRED_RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "SIM-DO": ("DTOBITO", "DTNASC", "IDADE", "SEXO", "RACACOR", "CODMUNRES",
               "CODMUNOCOR", "CODESTAB", "CAUSABAS", "LINHAA", "LINHAII"),
    "SINASC": ("DTNASC", "CODMUNRES", "CODMUNNASC", "SEXO", "PESO", "IDADEMAE",
               "SEMAGESTAC", "APGAR5", "PARTO", "CONSULTAS"),
    "SIH-RD": ("DT_INTER", "MUNIC_RES", "DIAG_PRINC", "SEXO", "COD_IDADE", "IDADE",
               "VAL_TOT", "MORTE", "CNES", "CGC_HOSP"),
    "CNES-ST": ("COMPETEN", "CODUFMUN", "CNES", "CPF_CNPJ"),
}


def missing_required_columns(df: "pl.DataFrame | pl.LazyFrame", system: str) -> list[str]:
    """Return the required raw columns for ``system`` that are absent from ``df``.

    Accepts a LazyFrame (schema-only, no data scan) so the streaming normalizer can
    check completeness without materializing the frame."""
    expected = REQUIRED_RAW_COLUMNS.get(system, ())
    present = set(df.collect_schema().names()) if isinstance(df, pl.LazyFrame) else set(df.columns)
    return [c for c in expected if c not in present]


def check_raw_completeness(df: pl.DataFrame, system: str) -> list[str]:
    """Report (and warn on) required raw columns missing from a normalizer's input.

    Returns the list of missing columns; also emits a ``UserWarning`` when non-empty
    so a silent upstream drop surfaces in logs. Callers should fold the returned list
    into their run summary so it lands in the manifest/telemetry.
    """
    missing = missing_required_columns(df, system)
    if missing:
        warnings.warn(
            f"{system}: normalizer input is missing expected raw columns {missing} — "
            "these canonical fields will be silently null. Check the DATASUS bridge "
            "(process_* drop / truncated DBC / microdatasus version).",
            UserWarning,
            stacklevel=2,
        )
    return missing
