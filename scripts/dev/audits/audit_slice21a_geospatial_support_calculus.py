from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from pegasus.geo.amc import contract_to_amc
from pegasus.geo.geneallocation import geneallocate_measure


def main() -> int:
    Path(".tmp").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="pegasus_slice21a_", dir=".tmp") as temp:
        root = Path(temp)
        crosswalk = root / "amc.csv"
        weights = root / "weights.csv"
        pl.DataFrame({"municipality_id": ["a", "b"], "year": [2020, 2020], "amc_id": ["x", "x"]}).write_csv(crosswalk)
        pl.DataFrame({"source_id": ["x", "x"], "year": [2020, 2020], "target_id": ["a", "b"], "weight": [0.4, 0.6]}).write_csv(weights)
        amc = contract_to_amc(
            pl.DataFrame({"municipality_id": ["a", "b"], "year": [2020, 2020], "value": [40.0, 60.0]}),
            crosswalk_path=crosswalk,
            value_column="value",
        )
        allocated = geneallocate_measure(
            amc.frame.rename({"amc_id": "source_id"}),
            weights_path=weights,
            value_column="value",
        )
        checks = {"amc_conserved": amc.conserved, "geneallocation_conserved": allocated.conserved, "roundtrip_total": allocated.output_total == 100.0}
        print(json.dumps({"slice": "21A", "checks": checks}, indent=2, sort_keys=True))
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
