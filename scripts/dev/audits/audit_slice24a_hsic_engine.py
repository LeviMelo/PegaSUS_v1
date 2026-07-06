from __future__ import annotations

import json

from pegasus.ldo.hsic import run_hsic_scan


def main() -> int:
    x = [float(index) / 30.0 for index in range(120)]
    residuals = [value * value + 0.05 * (index % 3) for index, value in enumerate(x)]
    kwargs = {
        "outcome_residual_field_id": "audit_residual",
        "covariate_field_id": "audit_covariate",
        "residuals": residuals,
        "covariate": x,
        "support_intersection": {"n_eff": 120},
        "budget": "standard",
        "null_strategy": "unrestricted_permutation",
        "fdr_method": "BY",
        "permutations": 99,
        "seed": 24,
    }
    first = run_hsic_scan(**kwargs)
    second = run_hsic_scan(**kwargs)
    checks = {
        "exact_kernel": first.hsic_mode == "exact",
        "nonlinear_signal": first.statistic is not None and first.statistic > 0.0,
        "empirical_p_value": first.p_value is not None and first.p_value <= 0.05,
        "permutations_executed": first.approximation_diagnostics.get("permutations_executed") == 99,
        "seed_reproducible": first.p_value == second.p_value and first.statistic == second.statistic,
    }
    print(json.dumps({"slice": "24A", "checks": checks, "result": first.as_manifest()}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
