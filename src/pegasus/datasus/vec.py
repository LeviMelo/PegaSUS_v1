"""Centralized vectorized decode primitives for DATASUS normalization.

Single source of truth for *how* a raw DATASUS column is decoded, expressed as
Polars column expressions instead of per-row Python. Every batch normalizer
(SIM-DO / SINASC / SIH-RD / CNES-ST) builds its canonical frame from these
primitives, so no system re-implements cleaning/date/municipality/CNPJ/ICD/count
logic and none of them loops over rows.

Design contract (why this module exists):

* **Computation is centralized here; schema is declared per system.** The fragile,
  easily-divergent part — parsing a DATASUS date, stripping a municipality sentinel,
  validating a CNPJ, classifying an ICD code — lives once, in this module. Each
  system's normalizer only says *which raw columns feed which primitive* and what
  its output columns are called. It never decides "per-row vs vectorized" and never
  re-derives a decode rule.
* **Each primitive mirrors the record-level authority in ``decoders.py`` /
  ``icd_parser.py``.** Those record decoders remain the correctness oracles the
  equivalence stress-checks pin against; the primitives here are their vectorized
  twins. Where a legacy per-row normalizer diverged from the authority (e.g. a
  missing municipality sentinel, or a CNPJ path that skipped check digits), the
  vectorized primitive follows the *authority*, which is the more-correct behavior.

The public surface is :class:`Cols`, a thin wrapper bound to a DataFrame that
resolves raw column names defensively (a source file that omits an optional column
yields a null literal rather than a crash) and exposes the primitives as methods.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

# Canonical blank/sentinel tokens. Matches ``decoders._none_or_blank`` (strip +
# case-insensitive {"", NA, NAN, NULL}) plus the historical "NONE" that the SIH/CNES
# substrate normalizers also treated as blank. These are never legitimate values in
# DATASUS coded/numeric fields, so nulling them uniformly is safe and correct.
_NULL_TOKENS = ["", "NA", "NAN", "NULL", "NONE"]

# Canonical ICD-10 shape and the normalized-code extractor (see icd_parser.ICD_RE).
_ICD_RE = r"^[A-Z][0-9]{2}[0-9A-Z]?$"

# DATASUS "município ignorado" sentinel: a real UF prefix followed by 0000
# (e.g. 270000 for Alagoas) is missingness, not geography (MSD §2.3).
_MUN_IGNORED_SUFFIX = "0000"


class Cols:
    """Vectorized decode primitives bound to a raw DATASUS DataFrame.

    All methods return Polars expressions (or lists of aliased expressions) and
    perform no materialization; the caller assembles them in a single
    ``with_columns`` pass. Raw column lookups are defensive: an absent column
    resolves to a typed null literal so an optional field never crashes the decode.
    """

    def __init__(self, df: pl.DataFrame):
        self._columns = set(df.columns)

    # -- raw access -------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._columns

    def raw(self, name: str) -> pl.Expr:
        """Raw column as Utf8, or a null Utf8 literal if the column is absent."""
        return pl.col(name).cast(pl.Utf8) if name in self._columns else pl.lit(None, dtype=pl.Utf8)

    def first(self, *names: str) -> pl.Expr:
        """First present column among ``names`` (coalesced), as Utf8."""
        present = [n for n in names if n in self._columns]
        if not present:
            return pl.lit(None, dtype=pl.Utf8)
        return pl.coalesce([pl.col(n).cast(pl.Utf8) for n in present])

    # -- cleaning ---------------------------------------------------------
    def clean(self, *names: str) -> pl.Expr:
        """Strip whitespace and null the canonical blank/sentinel tokens."""
        t = self.first(*names).str.strip_chars()
        return pl.when(t.str.to_uppercase().is_in(_NULL_TOKENS)).then(None).otherwise(t)

    def digits(self, *names: str) -> pl.Expr:
        """Cleaned value reduced to its digit run, or null when no digit remains."""
        d = self.clean(*names).str.replace_all(r"\D", "")
        return pl.when(d == "").then(None).otherwise(d)

    def int_digits(self, *names: str) -> pl.Expr:
        """Digit run parsed as Int64 (null-safe). Sign is discarded — use
        :meth:`nonneg_int` when a leading '-' must invalidate the value."""
        return self.digits(*names).cast(pl.Int64, strict=False)

    # -- dates ------------------------------------------------------------
    def date(self, *names: str) -> pl.Expr:
        """Parse a DATASUS date to a Polars ``Date``.

        Accepts every encoding the record authorities do: 8-digit ``YYYYMMDD`` or
        ``DDMMYYYY`` runs, and slash/ISO ``DD/MM/YYYY`` / ``YYYY-MM-DD`` text.
        """
        d = self.digits(*names)
        # An 8-digit DDMMYYYY date stored as an integer loses its leading zero
        # (e.g. 1012022 for 01012022); zero-pad a 7-digit run back to 8 so the
        # numeric-column case parses identically to the string case.
        d = pl.when(d.str.len_chars() == 7).then(pl.lit("0") + d).otherwise(d)
        c = self.clean(*names)
        return pl.coalesce([
            d.str.strptime(pl.Date, "%Y%m%d", strict=False),
            d.str.strptime(pl.Date, "%d%m%Y", strict=False),
            c.str.strptime(pl.Date, "%d/%m/%Y", strict=False),
            c.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        ])

    def date_state(self, *names: str) -> pl.Expr:
        """§2.3 state for :meth:`date`: missing (blank) / valid / invalid."""
        return (
            pl.when(self.clean(*names).is_null()).then(pl.lit("missing"))
            .when(self.date(*names).is_null()).then(pl.lit("invalid"))
            .otherwise(pl.lit("valid"))
        )

    # -- municipality -----------------------------------------------------
    def municipality(self, prefix: str, *names: str, crosswalk: dict[str, str] | None = None) -> list[pl.Expr]:
        """Decode a municipality code into ``{prefix}_cod6``, ``{prefix}_cod7`` and
        ``{prefix}_state`` (matches ``decode_municipality_cod6`` + the cod6→cod7
        crosswalk derivation).

        A 7-digit IBGE code carries its own cod7; a 6-digit DATASUS code is mapped
        via ``crosswalk`` when supplied. The UF+0000 "ignored" sentinel is nulled to
        missingness, not kept as geography.
        """
        d = self.digits(*names)
        cod6 = (
            pl.when(d.str.len_chars() == 6).then(d)
            .when(d.str.len_chars() == 7).then(d.str.slice(0, 6))
            .otherwise(None)
        )
        ignored = cod6.str.slice(2, 4) == _MUN_IGNORED_SUFFIX
        valid = cod6.is_not_null() & ~ignored
        cod6v = pl.when(valid).then(cod6).otherwise(None)
        if crosswalk is not None:
            cod7_from6 = cod6v.replace_strict(crosswalk, default=None)
        else:
            cod7_from6 = pl.lit(None, dtype=pl.Utf8)
        return [
            cod6v.alias(f"{prefix}_cod6"),
            (
                pl.when(valid & (d.str.len_chars() == 7)).then(d)
                .when(valid).then(cod7_from6)
                .otherwise(None)
                .alias(f"{prefix}_cod7")
            ),
            (
                pl.when(d.is_null()).then(pl.lit("missing"))
                .when(ignored).then(pl.lit("ignored_municipality"))
                .when(d.str.len_chars() == 7).then(pl.lit("ibge_cod7"))
                .when(d.str.len_chars() == 6).then(pl.lit("datasus_cod6"))
                .otherwise(pl.lit("invalid"))
                .alias(f"{prefix}_state")
            ),
        ]

    # -- sex / race -------------------------------------------------------
    def sex(self, *names: str) -> tuple[pl.Expr, pl.Expr]:
        """Decode SEXO (matches ``decode_datasus_sex``): 1→male, 2→female (valid);
        9→unknown; blank→missing; else invalid. Returns ``(value, state)``."""
        d = self.digits(*names)
        value = pl.when(d == "1").then(pl.lit("male")).when(d == "2").then(pl.lit("female")).otherwise(None)
        state = (
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(d.is_in(["1", "2"])).then(pl.lit("valid"))
            .when(d == "9").then(pl.lit("unknown"))
            .otherwise(pl.lit("invalid"))
        )
        return value, state

    def race_admin(self, *names: str) -> tuple[pl.Expr, pl.Expr]:
        """Decode administrative race/color (matches ``decode_race_admin``): the
        code is preserved as the value for 1–5; 9/99 are the unknown sentinel.
        Returns ``(code_value, state)`` where state ∈ {missing, valid, unknown,
        invalid}. Callers may relabel to their schema vocabulary."""
        d = self.digits(*names)
        value = pl.when(d.is_in(["1", "2", "3", "4", "5"])).then(d).when(d.is_in(["9", "99"])).then(d).otherwise(None)
        state = (
            pl.when(d.is_null()).then(pl.lit("missing"))
            .when(d.is_in(["1", "2", "3", "4", "5"])).then(pl.lit("valid"))
            .when(d.is_in(["9", "99"])).then(pl.lit("unknown"))
            .otherwise(pl.lit("invalid"))
        )
        return value, state

    # -- numbers ----------------------------------------------------------
    def nonneg_int(self, *names: str) -> tuple[pl.Expr, pl.Expr]:
        """Sign-preserving non-negative integer (matches ``_int_nonnegative``): a
        value parsed via float; non-integer or negative → invalid. Returns
        ``(value, state)`` with state ∈ {missing, valid, invalid}."""
        rawv = self.clean(*names)
        val = rawv.str.replace_all(",", ".").cast(pl.Float64, strict=False)
        ok = val.is_not_null() & (val >= 0) & (val == val.floor())
        state = pl.when(rawv.is_null()).then(pl.lit("missing")).when(ok).then(pl.lit("valid")).otherwise(pl.lit("invalid"))
        return pl.when(ok).then(val.cast(pl.Int64)).otherwise(None), state

    def money(self, *names: str) -> pl.Expr:
        """Non-negative decimal (comma or dot). Negative/unparseable → null."""
        v = self.clean(*names).str.replace_all(",", ".").cast(pl.Float64, strict=False)
        return pl.when(v >= 0).then(v).otherwise(None)

    def money_state(self, *names: str) -> pl.Expr:
        rawv = self.clean(*names)
        v = rawv.str.replace_all(",", ".").cast(pl.Float64, strict=False)
        return pl.when(rawv.is_null()).then(pl.lit("missing")).when(v >= 0).then(pl.lit("valid")).otherwise(pl.lit("invalid"))

    # -- boolean service flags --------------------------------------------
    def flag_int(self, *names: str) -> tuple[pl.Expr, pl.Expr]:
        """Decode a boolean service flag to 0/1 (matches ``clamp_bool``): textual
        false/true tokens, then a numeric path where a non-integer float is
        Unparseable and an integer outside {0,1} is InvalidFlagState. Returns
        ``(value, state)`` with the DecodedBoolean state vocabulary."""
        c = self.clean(*names)
        low = c.str.to_lowercase()
        f = c.str.replace_all(",", ".").cast(pl.Float64, strict=False)
        is_false = low.is_in(["0", "não", "nao", "no", "false", "f"])
        is_true = low.is_in(["1", "sim", "yes", "true", "t"])
        int_ok = f.is_not_null() & (f == f.floor())
        value = (
            pl.when(is_true).then(1)
            .when(is_false).then(0)
            .when(int_ok & f.is_in([0.0, 1.0])).then(f.cast(pl.Int64))
            .otherwise(None)
        )
        state = (
            pl.when(c.is_null()).then(pl.lit("MissingFlag"))
            .when(is_true).then(pl.lit("ValidTrue"))
            .when(is_false).then(pl.lit("ValidFalse"))
            .when(~int_ok).then(pl.lit("UnparseableFlag"))
            .when(f == 1.0).then(pl.lit("ValidTrue"))
            .when(f == 0.0).then(pl.lit("ValidFalse"))
            .otherwise(pl.lit("InvalidFlagState"))
        )
        return value, state

    # -- CNPJ (the §2.4.0.5 linkage gate) ---------------------------------
    def cnpj(self, *names: str) -> tuple[pl.Expr, pl.Expr]:
        """Validate a 14-digit CNPJ *including mod-11 check digits* (matches
        ``filter_cnpj``). Returns ``(value, state)`` with the full state vocabulary
        {MissingCNPJ, UnparseableCNPJ, NullifiedZeroCNPJ, InvalidCNPJLength,
        InvalidCNPJDigits, ValidCNPJ}."""
        clean = self.clean(*names)
        d = clean.str.replace_all(r"\D", "")
        d = pl.when(d == "").then(None).otherwise(d)
        is_zero = d.str.replace_all("0", "") == ""
        len14 = d.str.len_chars() == 14
        check_ok = _cnpj_check_ok(d)
        valid = len14 & check_ok
        value = pl.when(valid).then(d).otherwise(None)
        state = (
            pl.when(clean.is_null()).then(pl.lit("MissingCNPJ"))
            .when(d.is_null()).then(pl.lit("UnparseableCNPJ"))
            .when(is_zero).then(pl.lit("NullifiedZeroCNPJ"))
            .when(~len14).then(pl.lit("InvalidCNPJLength"))
            .when(~check_ok).then(pl.lit("InvalidCNPJDigits"))
            .otherwise(pl.lit("ValidCNPJ"))
        )
        return value, state

    # -- ICD --------------------------------------------------------------
    def icd_norm(self, *names: str) -> pl.Expr:
        """Normalized ICD code (matches ``normalize_icd``): upper, strip a leading
        ``*`` marker, drop dots. Null when blank/missing."""
        c = self.clean(*names).str.to_uppercase().str.strip_chars()
        c = pl.when(c.str.starts_with("*")).then(c.str.slice(1).str.strip_chars()).otherwise(c)
        c = c.str.replace_all(r"\.", "")
        return pl.when(c == "").then(None).otherwise(c)

    def icd_state(self, *names: str) -> pl.Expr:
        """§ ICD parse state (matches ``parse_icd``): missing / blank / unparseable /
        ill-defined (R-codes) / valid. (The 'invalid initial' branch of the record
        parser is unreachable — the ICD regex already forces an A–Z initial.)"""
        # parse_icd distinguishes raw-is-None (missing) from raw-empty-after-strip
        # (blank); clean() nulls both, so key the missing test on the raw value.
        c = self.first(*names).str.to_uppercase().str.strip_chars()
        stripped = pl.when(c.str.starts_with("*")).then(c.str.slice(1).str.strip_chars()).otherwise(c)
        norm = stripped.str.replace_all(r"\.", "")
        return (
            pl.when(self.first(*names).is_null()).then(pl.lit("missing"))
            .when(norm == "").then(pl.lit("blank"))
            .when(~norm.str.contains(_ICD_RE)).then(pl.lit("unparseable"))
            .when(norm.str.starts_with("R")).then(pl.lit("ill-defined"))
            .otherwise(pl.lit("valid"))
        )

    def icd_marker(self, *names: str) -> pl.Expr:
        """The stripped ``*`` marker (``"*"`` or null) — SIM cause-chain provenance."""
        c = self.clean(*names).str.to_uppercase().str.strip_chars()
        return pl.when(c.str.starts_with("*")).then(pl.lit("*")).otherwise(None)


def struct_json(exprs: Sequence[pl.Expr]) -> pl.Expr:
    """Encode a set of aliased expressions as one JSON object column (sorted keys
    are the natural order of ``exprs``). Empty input → the literal ``"{}"``."""
    exprs = list(exprs)
    if not exprs:
        return pl.lit("{}")
    return pl.struct(exprs).struct.json_encode()


def row_hash(source_manifest_hash: str, index_col: str = "_i") -> pl.Expr:
    """Stable per-row hash keyed by the manifest hash and the row index.

    A deterministic surrogate for the record path's raw-payload SHA-256: distinct
    per (file, row) and reproducible across runs of the same input, without the
    cost of JSON-encoding every raw row."""
    return pl.concat_str([pl.lit(source_manifest_hash), pl.lit(":"), pl.col(index_col).cast(pl.Utf8)]).hash().cast(pl.Utf8)


def _cnpj_check_ok(d: pl.Expr) -> pl.Expr:
    """Boolean expr: the 14-digit string ``d`` has valid mod-11 CNPJ check digits
    and is not a single repeated digit. Mirrors ``_valid_cnpj_check_digits``.

    Each digit position is sliced once and reused across both weighted sums so the
    expression stays cheap on large panels."""
    dig = [d.str.slice(i, 1).cast(pl.Int64, strict=False) for i in range(14)]

    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    s1 = sum((dig[i] * w for i, w in enumerate(w1)), pl.lit(0))
    r1 = s1 % 11
    dv1 = pl.when(r1 < 2).then(0).otherwise(11 - r1)

    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3]
    s2 = sum((dig[i] * w for i, w in enumerate(w2)), pl.lit(0)) + 2 * dv1
    r2 = s2 % 11
    dv2 = pl.when(r2 < 2).then(0).otherwise(11 - r2)

    # Reject an all-same-digit CNPJ (matches _valid_cnpj_check_digits' set()==1
    # guard). The Rust regex engine has no backreferences, so compare against the
    # first character repeated 14×.
    uniform = d == pl.concat_str([d.str.slice(0, 1)] * 14)
    return (d.str.len_chars() == 14) & ~uniform & (dig[12] == dv1) & (dig[13] == dv2)
