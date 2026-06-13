from __future__ import annotations

import argparse
import json
import sys

from pegasus.acceptance.contracts import evaluate_level3_acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    result = evaluate_level3_acceptance(args.run)
    print(json.dumps(result.as_manifest(), ensure_ascii=False, sort_keys=True, indent=2))
    if not result.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
