
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pegasus.acceptance.contracts import acceptance_plan, summarize_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Slice 11A milestone acceptance contract surfaces.")
    parser.add_argument("--run", action="append", default=[], help="Run directory to validate. May be repeated.")
    parser.add_argument("--require-non-scaffold", action="store_true")
    args = parser.parse_args()

    plan = acceptance_plan()
    print(json.dumps({"acceptance_plan": plan}, indent=2, sort_keys=True))
    errors: list[str] = []
    for run in args.run:
        summary = summarize_run(Path(run), require_non_scaffold=args.require_non_scaffold)
        print(json.dumps(summary.as_manifest(), indent=2, sort_keys=True))
        if not summary.ok:
            errors.extend(f"{run}: {error}" for error in summary.errors)
    if errors:
        raise SystemExit("AUDIT FAILED: " + "; ".join(errors))
    print("AUDIT PASSED: Slice 11A milestone acceptance contract and run-bundle summaries validated.")


if __name__ == "__main__":
    main()
