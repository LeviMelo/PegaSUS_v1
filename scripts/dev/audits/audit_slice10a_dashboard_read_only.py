from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.dashboard.contracts import DashboardContractError, assert_no_compute_trigger, dashboard_policy_manifest
from pegasus.dashboard.read_only import inspect_run, read_table_head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run)
    policy = dashboard_policy_manifest()
    if not policy.get("read_only"):
        raise SystemExit("dashboard policy is not read-only")
    for forbidden in ["datasus_fetch", "sidra_fetch", "she_build", "efg_build", "pirs_model", "pirs_hsic", "stdfm_solve", "population_solver", "race_bridge", "compile", "mutate_run"]:
        try:
            assert_no_compute_trigger(forbidden)
        except DashboardContractError:
            pass
        else:
            raise SystemExit(f"dashboard allowed forbidden operation: {forbidden}")
    summary = inspect_run(run_dir=run_dir, validate=True)
    if not summary["validation_ok"]:
        raise SystemExit("dashboard inspected run does not validate: " + json.dumps(summary["validation_errors"]))
    if summary["first_class_keys_missing"]:
        raise SystemExit("dashboard did not see all 17 keys")
    v_head = read_table_head(run_dir=run_dir, table_name="V_fields", limit=1)
    q_head = read_table_head(run_dir=run_dir, table_name="Q_tensor", limit=1)
    if "field_id" not in v_head["columns"] or "field_id" not in q_head["columns"]:
        raise SystemExit("dashboard table heads do not expose schema columns")
    print("AUDIT PASSED: Slice 10A read-only dashboard contract, local run inspection, table-head view, and computation-trigger forbiddance validated.")


if __name__ == "__main__":
    main()

