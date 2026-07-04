"""Adaptive Precision Controller (MSD-III §V.5, APC-01).

Uncertainty is an *actionable control signal*, not a report annotation. The engine runs
**anytime**: always a current best answer with current uncertainties, monotonically
tightened by more compute. Every approximation knob (SVD rank, Hutchinson probes, sketch
size, CG iterations, bootstrap count) is a *dial* trading compute for precision; the
scheduler spends the budget where expected decision-relevant uncertainty-reduction per
unit cost is highest.

The self-correcting property: **when the *numerical* approximation is the dominant
uncertainty for a decision-relevant quantity, the scheduler's best action is to escalate
that quantity to exact computation** — a bad approximation triggers its own correction
exactly where it matters, and is left alone where it doesn't. A quantity limited by
*statistical* (data) uncertainty is never escalated — more compute cannot help it, and it
is reported as data-limited rather than approximation-limited. On stop, the controller
reports which quantities met the target and which remain approximation-limited (typed,
never silent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace


@dataclass
class Budget:
    total: float
    _spent: float = 0.0

    def remaining(self) -> float:
        return max(0.0, self.total - self._spent)

    def spend(self, cost: float) -> None:
        self._spent += float(cost)

    @property
    def spent(self) -> float:
        return self._spent


@dataclass(frozen=True)
class Quantity:
    """A decision-relevant estimate with separable uncertainty sources.

    ``statistical_uncertainty`` is data-limited (irreducible by compute);
    ``numerical_uncertainty`` is the reducible approximation error;
    ``exact_cost`` is the compute to escalate this quantity to exact (numerical → 0).
    """

    name: str
    statistical_uncertainty: float
    numerical_uncertainty: float
    exact_cost: float
    decision_relevant: bool = True

    @property
    def total_uncertainty(self) -> float:
        return math.hypot(self.statistical_uncertainty, self.numerical_uncertainty)

    @property
    def numerical_dominates(self) -> bool:
        return self.numerical_uncertainty > self.statistical_uncertainty


@dataclass(frozen=True)
class ControllerReport:
    met: tuple[str, ...]                     # reached the target uncertainty
    approximation_limited: tuple[str, ...]   # miss the target and numerical error dominates (needs compute)
    data_limited: tuple[str, ...]            # miss the target but data-limited (compute cannot help)
    spent: float
    quantities: dict[str, Quantity]


def adaptive_precision_run(
    quantities: list[Quantity], *, target: float, budget: Budget
) -> ControllerReport:
    """Spend ``budget`` to bring decision-relevant quantities under ``target`` uncertainty.

    Anytime: callable at any stop point; the returned report is always valid. Each step
    escalates the decision-relevant, numerically-dominated quantity with the best expected
    uncertainty-reduction per unit cost. Terminates when the target is met for all such
    quantities, no affordable/useful action remains, or the budget is exhausted.
    """
    state = {q.name: q for q in quantities}

    while budget.remaining() > 0:
        # Actionable = decision-relevant, still above target, numerical error dominant
        # (so compute can help), and affordable. This is the self-correction filter: a
        # data-limited quantity is excluded — escalating it would waste budget.
        actionable = [
            q for q in state.values()
            if q.decision_relevant
            and q.total_uncertainty > target
            and q.numerical_dominates
            and q.exact_cost <= budget.remaining()
        ]
        if not actionable:
            break
        # value-of-computation: reduce the most numerical uncertainty per unit cost.
        chosen = max(actionable, key=lambda q: q.numerical_uncertainty / max(q.exact_cost, 1e-12))
        budget.spend(chosen.exact_cost)
        state[chosen.name] = replace(chosen, numerical_uncertainty=0.0)  # exact escalation

    met = tuple(sorted(n for n, q in state.items() if q.total_uncertainty <= target))
    approx = tuple(sorted(
        n for n, q in state.items()
        if q.total_uncertainty > target and q.numerical_dominates
    ))
    data = tuple(sorted(
        n for n, q in state.items()
        if q.total_uncertainty > target and not q.numerical_dominates
    ))
    return ControllerReport(
        met=met, approximation_limited=approx, data_limited=data,
        spent=budget.spent, quantities=dict(state),
    )


__all__ = ["Budget", "Quantity", "ControllerReport", "adaptive_precision_run"]
