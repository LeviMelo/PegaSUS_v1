"""REFACTOR-01 — decode single-source contract (XCUT-02; MSD-III §II.2, X.2).

Reality check first: the flat duplicated normalizers XCUT-02 named
(``declarative_normalize.py`` + four hand-written vectorized files) NO LONGER
EXIST. They were consolidated into ``pegasus.datasus.normalize`` — a package whose
per-system modules only *declare* which raw columns feed shared decode primitives
(``primitives.Cols``, the vectorized single source), with the scalar
``pegasus.datasus.decoders`` functions retained as independent record-level oracles.

The residual XCUT-02 risk is therefore *drift* between the vectorized ``Cols``
primitives and the scalar oracle they claim to "match". This contract closes that
risk: for every decode primitive, the vectorized polars expression MUST produce the
same ``(value, state)`` as the scalar decoder across the full §2.3 state space.
A red test here means the two decode implementations have diverged — a real bug.
"""

from __future__ import annotations

import polars as pl
import pytest

from pegasus.datasus.decoders import (
    clamp_bool,
    decode_datasus_sex,
    decode_race_admin,
    filter_cnpj,
)
from pegasus.datasus.normalize.primitives import Cols


def _eval_pair(method_name: str, raw_values: list) -> tuple[list, list]:
    """Evaluate a Cols (value, state) primitive over a single raw column."""
    df = pl.DataFrame({"C": pl.Series(raw_values, dtype=pl.Utf8)})
    value_expr, state_expr = getattr(Cols(df), method_name)("C")
    res = df.select(v=value_expr, s=state_expr)
    return res["v"].to_list(), res["s"].to_list()


# Inputs chosen to exercise every documented decode state per primitive.
_SEX_INPUTS = ["1", "2", "9", "", "7", None]
_RACE_INPUTS = ["1", "3", "5", "9", "99", "", "8", None]
_BOOL_INPUTS = ["0", "1", "Sim", "Não", "", "2", "-1", "1.5", None]
_CNPJ_INPUTS = ["11222333000181", "12345678000190", "00000000000000", "123", "", None]


def test_sex_vectorized_matches_scalar_oracle() -> None:
    values, states = _eval_pair("sex", _SEX_INPUTS)
    for raw, value, state in zip(_SEX_INPUTS, values, states):
        oracle = decode_datasus_sex(raw)
        assert (value, state) == (oracle.value, oracle.state), f"sex drift on {raw!r}"


def test_race_admin_vectorized_matches_scalar_oracle() -> None:
    values, states = _eval_pair("race_admin", _RACE_INPUTS)
    for raw, value, state in zip(_RACE_INPUTS, values, states):
        oracle = decode_race_admin(raw)
        assert (value, state) == (oracle.value, oracle.state), f"race_admin drift on {raw!r}"


def test_boolean_flag_vectorized_matches_scalar_oracle() -> None:
    values, states = _eval_pair("flag_int", _BOOL_INPUTS)
    for raw, value, state in zip(_BOOL_INPUTS, values, states):
        oracle = clamp_bool(raw)
        assert (value, state) == (oracle.value, oracle.state), f"boolean flag drift on {raw!r}"


def test_cnpj_vectorized_matches_scalar_oracle() -> None:
    values, states = _eval_pair("cnpj", _CNPJ_INPUTS)
    for raw, value, state in zip(_CNPJ_INPUTS, values, states):
        oracle = filter_cnpj(raw)
        assert (value, state) == (oracle.cnpj, oracle.state), f"cnpj drift on {raw!r}"


def test_flat_duplicated_normalizers_are_gone() -> None:
    """The pre-consolidation flat files must not reappear (XCUT-02 regression guard)."""
    import pegasus.datasus as ds

    pkg_dir = ds.__path__[0]
    from pathlib import Path

    forbidden = ["declarative_normalize.py", "sinasc_normalize.py", "sih_normalize.py", "cnes_normalize.py"]
    present = [name for name in forbidden if (Path(pkg_dir) / name).exists()]
    assert not present, f"flat duplicated normalizer(s) reappeared: {present}"
