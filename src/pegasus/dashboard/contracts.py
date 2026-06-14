"""Read-only dashboard contract for completed PegaSUS run bundles.

The dashboard layer is allowed to inspect already-materialized output bundles.
It is forbidden to fetch sources, invoke SHE/EFG/PIRS/ST-DFM/HSIC/population
solvers, mutate run artifacts, or create new analytic fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class DashboardContractError(PermissionError):
    """Raised when a dashboard request attempts a forbidden computation."""


READ_ONLY_OPERATIONS: frozenset[str] = frozenset({
    "inspect_run",
    "list_tables",
    "table_head",
    "variable_dictionary",
    "hypotheses",
    "warnings",
    "q_tensor",
    "bundle_summary",
    "bundle_overview",
    "audit_policy",
})

FORBIDDEN_OPERATIONS: frozenset[str] = frozenset({
    "datasus_fetch",
    "datasus_ingest",
    "sidra_fetch",
    "sidra_extract",
    "she_build",
    "efg_build",
    "pirs_model",
    "pirs_hsic",
    "stdfm_solve",
    "population_solver",
    "race_bridge",
    "bridge_r",
    "compile",
    "write_bundle",
    "mutate_run",
})


@dataclass(frozen=True)
class DashboardPolicy:
    read_only: bool
    allowed_operations: tuple[str, ...]
    forbidden_operations: tuple[str, ...]
    computation_triggers_forbidden: bool
    external_fetch_forbidden: bool
    mutation_forbidden: bool

    def as_manifest(self) -> dict[str, Any]:
        return {
            "read_only": self.read_only,
            "allowed_operations": list(self.allowed_operations),
            "forbidden_operations": list(self.forbidden_operations),
            "computation_triggers_forbidden": self.computation_triggers_forbidden,
            "external_fetch_forbidden": self.external_fetch_forbidden,
            "mutation_forbidden": self.mutation_forbidden,
        }


def dashboard_policy() -> DashboardPolicy:
    return DashboardPolicy(
        read_only=True,
        allowed_operations=tuple(sorted(READ_ONLY_OPERATIONS)),
        forbidden_operations=tuple(sorted(FORBIDDEN_OPERATIONS)),
        computation_triggers_forbidden=True,
        external_fetch_forbidden=True,
        mutation_forbidden=True,
    )


def dashboard_policy_manifest() -> dict[str, Any]:
    return dashboard_policy().as_manifest()


def assert_read_only_operation(operation: str) -> None:
    if operation in FORBIDDEN_OPERATIONS:
        raise DashboardContractError(
            f"Dashboard operation {operation!r} is forbidden because it would trigger computation, fetch, or mutation."
        )
    if operation not in READ_ONLY_OPERATIONS:
        raise DashboardContractError(
            f"Dashboard operation {operation!r} is not in the explicit read-only allowlist."
        )


def assert_no_compute_trigger(operation: str) -> None:
    assert_read_only_operation(operation)
