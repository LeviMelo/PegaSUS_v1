"""DATASUS raw→canonical normalization.

All four systems (SIM-DO, SINASC, SIH-RD, CNES-ST) normalize **raw DBC codes** into
the canonical SHE schema, fully vectorized on the shared ``primitives.Cols`` decode
library (single source of truth for every decode computation). Each system module
only declares which raw columns feed which primitive; none loops over rows or
re-implements a decode rule. The record-level ``normalize_*_record`` functions are
retained as independent correctness oracles.

This ``__init__`` is the public surface — import the batch entry points from
``pegasus.datasus.normalize`` regardless of which system module they live in.
"""

from __future__ import annotations

from pegasus.datasus.normalize.cnes import normalize_cnes_st_events, normalize_cnes_st_record
from pegasus.datasus.normalize.completeness import check_raw_completeness, missing_required_columns
from pegasus.datasus.normalize.primitives import Cols, read_raw_table, row_hash, struct_json
from pegasus.datasus.normalize.records import normalize_record
from pegasus.datasus.normalize.sih import normalize_sih_rd_events, normalize_sih_rd_record
from pegasus.datasus.normalize.sim import (
    SIM_DO_NORMALIZED_COLUMNS,
    normalize_sim_do_events,
    normalize_sim_do_record,
)
from pegasus.datasus.normalize.sinasc import (
    decode_anomaly_flag,
    normalize_anomaly_icd,
    normalize_sinasc_events,
    normalize_sinasc_record,
)

__all__ = [
    "Cols",
    "read_raw_table",
    "row_hash",
    "struct_json",
    "check_raw_completeness",
    "missing_required_columns",
    "normalize_record",
    "normalize_sim_do_events",
    "normalize_sim_do_record",
    "SIM_DO_NORMALIZED_COLUMNS",
    "normalize_sih_rd_events",
    "normalize_sih_rd_record",
    "normalize_sinasc_events",
    "normalize_sinasc_record",
    "decode_anomaly_flag",
    "normalize_anomaly_icd",
    "normalize_cnes_st_events",
    "normalize_cnes_st_record",
]
