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
from typing import Iterable, Literal, Mapping

import numpy as np
import scipy.sparse as sp

from pegasus.disease import icd_adapter
from pegasus.disease import concept_registry


LegalityClass = Literal["structural", "context_derived"]


class DiseaseGraphCircularityError(ValueError):
    """§II.13.3 abort: context-derived disease structure shares provenance with the tested
    variables and cannot be used as a same-data prior."""


def _weight_from_info(ia: "icd_adapter.ICDCodeInfo", ib: "icd_adapter.ICDCodeInfo") -> float:
    """Structural closeness of two already-resolved codes, in [0,1] (see ``_structural_weight``).

    Kept separate so callers building an O(n^2) graph can resolve ``code_info`` once per code
    (O(n)) and reuse the resolved view in the inner loop, rather than re-resolving both codes
    on every pair.
    """
    if ia.category and ia.category == ib.category:
        return 1.0
    if ia.block and ia.block == ib.block:
        return 0.5
    if ia.chapter and ia.chapter == ib.chapter:
        return 0.25
    return 0.0


def _structural_weight(a: str, b: str) -> float:
    """Structural closeness of two codes in the ICD/CID hierarchy, in [0,1].

    1.0 same 3-char category; 0.5 same block; 0.25 same chapter; 0 otherwise. Deterministic
    and data-independent (§II.13.3 structural), so safe as a prior.
    """
    return _weight_from_info(icd_adapter.code_info(a), icd_adapter.code_info(b))


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
        # Resolve each code's hierarchy view ONCE (O(n)); the O(n^2) inner loop then
        # reads the resolved struct instead of re-calling code_info per pair.
        info = [icd_adapter.code_info(c) for c in ordered]
        chapter = [ci.chapter for ci in info]
        rows, cols, data = [], [], []
        for i in range(n):
            if chapter[i] is None:
                continue
            info_i = info[i]
            for j in range(i + 1, n):
                if chapter[j] != chapter[i]:
                    continue
                w = _weight_from_info(info_i, info[j])
                if w > 0.0:
                    rows += [i, j]
                    cols += [j, i]
                    data += [w, w]
        weights = sp.csr_matrix((data, (rows, cols)), shape=(n, n)) if data else sp.csr_matrix((n, n))
        return cls(codes=ordered, weights=weights, legality_class="structural",
                   provenance=("cid10_hierarchy", "who_icd10"))

    @classmethod
    def from_variable_code_sets(
        cls, code_sets: Mapping[str, Iterable[str]]
    ) -> "DiseaseGraph":
        """Build a structural graph keyed by *variable id* from each variable's code set.

        The LDO's variables are field ids (``σ_C`` restriction counts like
        ``DengueHospitalAdmissions``), not bare ICD codes, so the plain code-keyed
        ``hierarchy`` graph never intersects ``gf.variables`` and the disease penalty would
        be a silent no-op. This graph's nodes ARE the variable ids; the edge weight between
        two variables is the *strongest* CID-10 structural closeness across the cross-product
        of their member codes (1.0 same 3-char category, 0.5 same block, 0.25 same chapter).
        Still data-independent (``structural``), so admissible as an LDO prior.
        """
        ordered = tuple(dict.fromkeys(str(v) for v in code_sets))
        members = {
            v: tuple(str(c) for c in (code_sets[v] or ()))
            for v in ordered
        }
        n = len(ordered)
        # Resolve code_info ONCE per distinct member code (O(distinct_codes)), then reuse the
        # resolved views in the O(variables^2 x codes_i x codes_j) closeness scan below. This
        # avoids two code_info lookups per (a, b) code pair (previously the dominant cost when a
        # variable spans many CID-10 subcodes at national resolution).
        info_by_code = {
            c: icd_adapter.code_info(c)
            for c in {code for codes in members.values() for code in codes}
        }
        member_info = {v: tuple(info_by_code[c] for c in members[v]) for v in ordered}
        rows, cols, data = [], [], []
        for i in range(n):
            ci = member_info[ordered[i]]
            if not ci:
                continue
            for j in range(i + 1, n):
                cj = member_info[ordered[j]]
                if not cj:
                    continue
                w = max((_weight_from_info(a, b) for a in ci for b in cj), default=0.0)
                if w > 0.0:
                    rows += [i, j]
                    cols += [j, i]
                    data += [w, w]
        weights = sp.csr_matrix((data, (rows, cols)), shape=(n, n)) if data else sp.csr_matrix((n, n))
        return cls(codes=ordered, weights=weights, legality_class="structural",
                   provenance=("cid10_hierarchy", "who_icd10", "variable_code_sets"))

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
