"""DiseaseGraph — structural prior over the disease/variable dimension (MSD-II §II.13.3).

The disease-axis analogue of the SpatialWeightGraph: a structural (data-independent) graph
over a set of disease codes, exposing the Laplacian ``L_D`` that enters the LDO's
variable-dependency operator ``Omega_var`` as a smoothness/fused prior — exactly parallel to
how the spatial ``L_W`` enters the cell-precision (§II.13.3 / §5.2).

Legality class (reuses §II.4.1's circularity guard):
- ``structural`` graphs (hierarchy, external concept membership, label-text embeddings) are
  data-independent and SAFE as an LDO prior and as a search policy.
- ``context_derived`` graphs (empirical co-occurrence learned from the DATASUS events being
  tested) MUST NOT be used as a prior for an edge whose variables share the graph's
  provenance — that smuggles the hypothesis into the null. ``as_prior()`` refuses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import scipy.sparse as sp

from pegasus.disease import icd_adapter
from pegasus.disease import concept_registry


LegalityClass = Literal["structural", "context_derived"]


class DiseaseGraphCircularityError(ValueError):
    """§II.13.3 abort: context-derived disease structure shares provenance with the tested
    variables and cannot be used as a same-data prior."""


def _structural_weight(a: str, b: str) -> float:
    """Structural closeness of two codes in the ICD/CID hierarchy, in [0,1].

    1.0 same 3-char category; 0.5 same block; 0.25 same chapter; 0 otherwise. Deterministic
    and data-independent (§II.13.3 structural), so safe as a prior.
    """
    ia, ib = icd_adapter.code_info(a), icd_adapter.code_info(b)
    if ia.category and ia.category == ib.category:
        return 1.0
    if ia.block and ia.block == ib.block:
        return 0.5
    if ia.chapter and ia.chapter == ib.chapter:
        return 0.25
    return 0.0


@dataclass(frozen=True)
class DiseaseGraph:
    """A structural graph over an ordered disease-code variable set."""

    codes: tuple[str, ...]
    weights: sp.csr_matrix           # symmetric, zero diagonal
    legality_class: LegalityClass
    provenance: tuple[str, ...]

    @classmethod
    def hierarchy(cls, codes: Iterable[str]) -> "DiseaseGraph":
        """Build the CID-10/ICD-10 hierarchy graph over ``codes`` (structural, §II.13.3).

        Sparse by construction: only same-chapter pairs are nonzero, so a national disease
        variable set stays sparse exactly like the queen-contiguity spatial graph."""
        ordered = tuple(dict.fromkeys(str(c) for c in codes))  # de-dup, keep order
        n = len(ordered)
        chapter = [icd_adapter.code_info(c).chapter for c in ordered]
        rows, cols, data = [], [], []
        for i in range(n):
            if chapter[i] is None:
                continue
            for j in range(i + 1, n):
                if chapter[j] != chapter[i]:
                    continue
                w = _structural_weight(ordered[i], ordered[j])
                if w > 0.0:
                    rows += [i, j]
                    cols += [j, i]
                    data += [w, w]
        weights = sp.csr_matrix((data, (rows, cols)), shape=(n, n)) if data else sp.csr_matrix((n, n))
        return cls(codes=ordered, weights=weights, legality_class="structural",
                   provenance=("cid10_hierarchy", "who_icd10"))

    def laplacian(self) -> sp.csr_matrix:
        """Graph Laplacian ``L_D = D - W`` (the disease smoothness operator for the LDO prior)."""
        degree = np.asarray(self.weights.sum(axis=1)).ravel()
        return (sp.diags(degree) - self.weights).tocsr()

    def adjacency(self) -> sp.csr_matrix:
        return self.weights

    def distance(self, a: str, b: str) -> float | None:
        """Hierarchy distance via nearest-common-ancestor depth (None if incomparable)."""
        if icd_adapter.code_info(a).category == icd_adapter.code_info(b).category:
            return 0.0
        nca = icd_adapter.nearest_common_ancestor(a, b)
        if nca is None:
            return None
        return float(len(icd_adapter.ancestors(a)) + len(icd_adapter.ancestors(b))
                     - 2 * len(icd_adapter.ancestors(nca)))

    def groups(self, *, registry_root: str = "config/registries") -> dict[str, list[str]]:
        """Overlapping concept groups over the codes (for group penalties / overlap handling,
        §5.3). concept_id -> list of member codes. Multi-label preserved."""
        out: dict[str, list[str]] = {}
        for code in self.codes:
            for assertion in concept_registry.assertions_for_code(code, registry_root=registry_root):
                out.setdefault(assertion.concept_id, []).append(code)
        return out

    def as_prior(self) -> sp.csr_matrix:
        """The Laplacian, guarded: a context-derived graph is refused as a same-data prior."""
        if self.legality_class == "context_derived":
            raise DiseaseGraphCircularityError(
                "context-derived disease structure shares provenance with the tested variables"
            )
        return self.laplacian()


__all__ = ["DiseaseGraph", "DiseaseGraphCircularityError", "LegalityClass"]
