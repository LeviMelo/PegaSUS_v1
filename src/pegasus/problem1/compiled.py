from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from pegasus.problem1.contracts import FieldNode, QState, WarningRecord


@dataclass(frozen=True)
class CompiledField:
    field: FieldNode
    data: pl.DataFrame
    q_state: QState
    warnings: list[WarningRecord]