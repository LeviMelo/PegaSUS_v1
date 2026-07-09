"""Guard the two bundle-schema authorities against silent drift.

schema_seed.TABLE_SCHEMAS (PyArrow, the canonical column set the compile emits) and
validate.REQUIRED_*_COLUMNS (the sets the output validator enforces) are maintained separately.
If validate requires a column the seed schema never produces, validation fails on every real bundle;
if the seed drops a column validate still requires, same. This pins the invariant: every required
column must exist in the seed schema (the seed may carry extra/optional columns — that's fine).
"""
from __future__ import annotations

import pytest

from pegasus.output import validate as V
from pegasus.output.schema_seed import TABLE_SCHEMAS

CASES = [
    ("V_fields.parquet", V.REQUIRED_V_FIELDS_COLUMNS),
    ("E_DAG.parquet", V.REQUIRED_E_DAG_COLUMNS),
    ("Q_tensor.parquet", V.REQUIRED_Q_TENSOR_COLUMNS),
    ("VariableDictionary.parquet", V.REQUIRED_VARIABLE_DICTIONARY_COLUMNS),
]


@pytest.mark.parametrize("filename,required", CASES)
def test_required_columns_exist_in_seed_schema(filename, required):
    seed_cols = set(TABLE_SCHEMAS[filename].names)
    missing = set(required) - seed_cols
    assert not missing, (
        f"{filename}: validate.REQUIRED_*_COLUMNS requires columns absent from "
        f"schema_seed.TABLE_SCHEMAS (schema authority drift): {sorted(missing)}"
    )
