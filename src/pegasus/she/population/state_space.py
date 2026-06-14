"""Sparse population state-space indexing and transition plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from pegasus.she.population.schema import PopulationTensorProblem


@dataclass(frozen=True)
class PopulationStateSpace:
    shape: tuple[int, int, int, int, int]
    anchor_indices: tuple[int, ...]
    aging_edges: tuple[tuple[int, int], ...]
    birth_edges: tuple[tuple[int, int], ...]
    migration_indices: tuple[int, ...]
    race_groups: tuple[tuple[int, ...], ...]
    closure_groups: tuple[tuple[int, ...], ...]

    @property
    def n_cells(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def as_manifest(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "n_cells": self.n_cells,
            "anchor_count": len(self.anchor_indices),
            "aging_edge_count": len(self.aging_edges),
            "birth_edge_count": len(self.birth_edges),
            "migration_cell_count": len(self.migration_indices),
            "race_group_count": len(self.race_groups),
            "closure_group_count": len(self.closure_groups),
        }


def flat_index(shape: tuple[int, int, int, int, int], s: int, t: int, a: int, x: int, r: int) -> int:
    _, periods, ages, sexes, races = shape
    return ((((s * periods) + t) * ages + a) * sexes + x) * races + r


def iter_coordinates(shape: tuple[int, int, int, int, int]) -> Iterator[tuple[int, int, int, int, int]]:
    localities, periods, ages, sexes, races = shape
    for s in range(localities):
        for t in range(periods):
            for a in range(ages):
                for x in range(sexes):
                    for r in range(races):
                        yield s, t, a, x, r


def build_population_state_space(problem: PopulationTensorProblem) -> PopulationStateSpace:
    shape = problem.shape
    localities, periods, ages, sexes, races = shape
    anchors = tuple(i for i, value in enumerate(problem.anchors) if value is not None)
    aging: list[tuple[int, int]] = []
    births: list[tuple[int, int]] = []
    race_groups: list[tuple[int, ...]] = []
    closures: list[tuple[int, ...]] = []
    for s in range(localities):
        for t in range(periods):
            closures.append(tuple(flat_index(shape, s, t, a, x, r) for a in range(ages) for x in range(sexes) for r in range(races)))
            for a in range(ages):
                for x in range(sexes):
                    race_groups.append(tuple(flat_index(shape, s, t, a, x, r) for r in range(races)))
            if t == 0:
                continue
            for x in range(sexes):
                for r in range(races):
                    births.append((flat_index(shape, s, t - 1, 0, x, r), flat_index(shape, s, t, 0, x, r)))
                    for a in range(1, ages):
                        aging.append((flat_index(shape, s, t - 1, a - 1, x, r), flat_index(shape, s, t, a, x, r)))
    return PopulationStateSpace(
        shape=shape,
        anchor_indices=anchors,
        aging_edges=tuple(aging),
        birth_edges=tuple(births),
        migration_indices=tuple(range(problem.n_cells)),
        race_groups=tuple(race_groups),
        closure_groups=tuple(closures),
    )
