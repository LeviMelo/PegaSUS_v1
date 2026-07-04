"""T0-1 — Source-reality contract tests (MSD-III §II.2, X.1; TDD Phase 0).

Pins the landed S0 substrate remediations against the multi-state DATASUS fixtures.
A red test here means the *decode path* has regressed, never that the test is wrong.

The seven contracts (TDD §5.2):
    1. no admissible (fixture-fed) canonical column is silently all-null
    2. SIM structural age decode is exact (decoder is the oracle)
    3. SIM cause chain carries 4 ordered positions (A-D) + per-position parse state
    4. SINASC emits scalar *primitives*, never baked threshold flags (SHE-SINASC-01)
    5. CNES carries per-index bed carriers and NO forbidden bed total (SHE-CNES-01)
    6. all-zero CNPJ is nullified with an explicit audit state (SHE-CNES-02)
    7. every *_state value is a recognized MSD-I §2.3 state family
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from pegasus.datasus.decoders import decode_sim_idade

# --------------------------------------------------------------------------- #
# MSD-I §2.3 state taxonomy: every emitted *_state literal MUST map to one of
# the five canonical families (plus the legitimate `sentinel` typed state).
# Unrecognized literals fail the vocabulary contract by design — new states
# must be registered here, never leaked silently.
# --------------------------------------------------------------------------- #
_STATE_FAMILY_EXACT: dict[str, str] = {
    # valid family
    "valid": "valid",
    "valid_admin_race": "valid",
    "valid_absent": "valid",
    "valid_present": "valid",
    "valid_q_anomaly": "valid",
    "ValidCNPJ": "valid",
    "ValidTrue": "valid",
    "ValidFalse": "valid",
    "datasus_cod6": "valid",
    "datasus_cod7": "valid",
    # missing family
    "missing": "missing",
    "MissingCNPJ": "missing",
    "MissingCount": "missing",
    "MissingFlag": "missing",
    "MissingWeight": "missing",
    "blank": "missing",
    # invalid family
    "invalid": "invalid",
    "invalid_identity": "invalid",
    "ill-defined": "invalid",
    "InvalidCNPJDigits": "invalid",
    "InvalidCNPJLength": "invalid",
    "InvalidCount": "invalid",
    "InvalidFlagState": "invalid",
    "InvalidWeight": "invalid",
    "OutOfRangeWeight": "invalid",
    # unparseable family
    "unparseable": "unparseable",
    "UnparseableCNPJ": "unparseable",
    "UnparseableFlag": "unparseable",
    # unknown family
    "unknown": "unknown",
    "UnknownAge": "unknown",
    "UnknownAgeUnit": "unknown",
    "UnknownCount": "unknown",
    "flag_present_code_missing": "unknown",
    # sentinel family (a legitimate typed state per MSD-I §2.4.0.2/§2.4.0.4)
    "sentinel": "sentinel",
    "ignored_sentinel": "sentinel",
    "NullifiedZeroCNPJ": "sentinel",
}

_REQUIRED_FAMILIES = {"valid", "missing", "invalid", "unparseable", "unknown"}

# Canonical columns each fixture DOES feed raw input for; these must not be
# silently all-null. (Columns whose raw source is absent from the tiny fixture
# — and which the normalizer explicitly warns about — are excluded here.)
_CORE_NONNULL: dict[str, list[str]] = {
    "sim": [
        "event_id", "death_date", "age_years", "sex",
        "mun_residence_cod6", "underlying_icd_norm", "cause_chain_norm",
    ],
    "sih": [
        "admission_id", "admission_date", "age_years", "sex",
        "mun_residence_cod6", "principal_icd_norm", "total_admission_cost_real",
    ],
    "sinasc": [
        "event_id", "birth_date", "mun_residence_cod6",
        "birth_weight_grams", "gestational_weeks", "apgar_5min",
    ],
    "cnes": [
        "facility_id", "mun_facility_cod6", "clinical_bed_capacity",
        "facility_cnpj_state",
    ],
}


def _state_family(value: str) -> str | None:
    return _STATE_FAMILY_EXACT.get(value)


# --------------------------------------------------------------------------- #
# 1. No admissible (fixture-fed) canonical column is all-null.
# --------------------------------------------------------------------------- #
def test_no_admissible_column_all_null(all_df: dict) -> None:
    for system, cols in _CORE_NONNULL.items():
        df = all_df[system]
        for col in cols:
            assert col in df.columns, f"{system}: missing canonical column {col!r}"
            nonnull = df[col].drop_nulls().len()
            assert nonnull > 0, (
                f"{system}.{col} is silently all-null on a fixture that feeds it — "
                "anti-silence regression (MSD-I §2.3)."
            )


# --------------------------------------------------------------------------- #
# 2. Structural age decode is exact — the decoder is the oracle.
# --------------------------------------------------------------------------- #
def test_age_decode_exact(sim_df: pl.DataFrame) -> None:
    # (a) the decoder itself is exact on canonical IDADE codes
    assert decode_sim_idade("474").age_years == 74.0
    assert decode_sim_idade("402").age_years == 2.0
    assert decode_sim_idade("201").age_unit == "days" and decode_sim_idade("201").age_days == 1.0
    assert decode_sim_idade("501").age_unit == "years_100_plus" and decode_sim_idade("501").age_years == 101.0
    assert decode_sim_idade("674").state == "UnknownAge"  # unit code outside {1..5}

    # (b) where the frame's age came from the IDADE code, it MUST equal the decode
    for row in sim_df.iter_rows(named=True):
        if row["age_source"] == "IDADE":
            expected = decode_sim_idade(row["raw_age_code"]).age_years
            assert row["age_years"] == pytest.approx(expected), (
                f"IDADE-sourced age mismatch for code {row['raw_age_code']}"
            )


# --------------------------------------------------------------------------- #
# 3. Cause chain: 4 ordered positions (A-D) + per-position parse state.
# --------------------------------------------------------------------------- #
def test_cause_chain_positions(sim_df: pl.DataFrame) -> None:
    for col in ("cause_chain_norm", "cause_chain_parse_states"):
        assert col in sim_df.columns

    ordered = {"A", "B", "C", "D"}
    saw_state = False
    for norm_json, state_json in zip(
        sim_df["cause_chain_norm"].to_list(),
        sim_df["cause_chain_parse_states"].to_list(),
    ):
        norm = json.loads(norm_json) if norm_json else {}
        states = json.loads(state_json) if state_json else {}
        # positions are drawn from the ordered LINHAA-D alphabet only
        assert set(norm).issubset(ordered), f"unexpected chain position(s): {set(norm) - ordered}"
        assert set(states).issubset(ordered)
        for pos, st in states.items():
            assert _state_family(st) is not None, f"unrecognized cause-chain state {st!r}"
            saw_state = True
    assert saw_state, "no cause-chain parse state emitted on the fixture"


# --------------------------------------------------------------------------- #
# 4. SINASC emits scalar primitives, never baked threshold flags.
# --------------------------------------------------------------------------- #
def test_sinasc_emits_primitives(sinasc_df: pl.DataFrame) -> None:
    for primitive in ("birth_weight_grams", "apgar_5min", "gestational_weeks"):
        assert primitive in sinasc_df.columns, f"missing SINASC primitive {primitive!r}"

    forbidden = {
        "low_birth_weight_flag", "low_birth_weight", "lbw_flag", "is_lbw",
        "prematurity_flag", "premature_flag", "is_premature", "preterm_flag",
        "apgar_asphyxia_flag", "cesarean_flag",
    }
    leaked = forbidden.intersection(sinasc_df.columns)
    assert not leaked, f"SHE-SINASC-01: baked threshold flags leaked into SHE output: {leaked}"


# --------------------------------------------------------------------------- #
# 5. CNES carries per-index bed carriers and NO forbidden bed total.
# --------------------------------------------------------------------------- #
def test_cnes_no_bed_total(cnes_df: pl.DataFrame) -> None:
    for carrier in ("clinical_bed_capacity", "surgical_bed_capacity", "obstetric_bed_capacity"):
        assert carrier in cnes_df.columns, f"missing per-index bed carrier {carrier!r}"

    lowered = {c.lower() for c in cnes_df.columns}
    forbidden_substrings = ("capacity_total", "bed_total", "total_bed", "leito_total", "total_leito")
    offenders = [c for c in lowered if any(s in c for s in forbidden_substrings)]
    assert not offenders, f"SHE-CNES-01: forbidden aggregate bed-total column(s): {offenders}"


# --------------------------------------------------------------------------- #
# 6. All-zero CNPJ is nullified with an explicit audit state.
# --------------------------------------------------------------------------- #
def test_cnpj_gate(cnes_df: pl.DataFrame) -> None:
    zeroed = cnes_df.filter(pl.col("facility_cnpj_state") == "NullifiedZeroCNPJ")
    assert zeroed.height > 0, "fixture must include an all-zero facility CNPJ row"
    # the nullified link carries NO fabricated identifier
    assert zeroed["facility_cnpj"].drop_nulls().len() == 0, (
        "SHE-CNES-02: nullified CNPJ must not emit a fabricated identifier"
    )


# --------------------------------------------------------------------------- #
# 7. Every *_state value is a recognized MSD-I §2.3 family.
# --------------------------------------------------------------------------- #
def test_state_vocab(all_df: dict) -> None:
    seen_families: set[str] = set()
    for system, df in all_df.items():
        for col in (c for c in df.columns if c.endswith("_state")):
            for value in df[col].drop_nulls().unique().to_list():
                fam = _state_family(value)
                assert fam is not None, (
                    f"{system}.{col} emitted unrecognized state {value!r} — "
                    "register it in the §2.3 family map or fix the decoder."
                )
                seen_families.add(fam)

    # the taxonomy as exercised must be able to distinguish the five families.
    missing_expressive = _REQUIRED_FAMILIES - (seen_families | set(_STATE_FAMILY_EXACT.values()))
    assert not missing_expressive, f"state taxonomy cannot express: {missing_expressive}"
