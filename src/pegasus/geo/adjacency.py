"""External municipality adjacency contract."""

from __future__ import annotations

from pathlib import Path

from pegasus.geo.geodata import GeoArtifactError, load_geo_artifact


def load_adjacency(path: str | Path, *, require_symmetric: bool = True) -> dict[str, tuple[str, ...]]:
    frame, _artifact = load_geo_artifact(path, required_columns={"left_id", "right_id"})
    pairs = {(str(row["left_id"]), str(row["right_id"])) for row in frame.to_dicts()}
    if any(left == right for left, right in pairs):
        raise GeoArtifactError("adjacency artifact contains self-neighbor edges")
    if require_symmetric:
        missing = {(right, left) for left, right in pairs if (right, left) not in pairs}
        if missing:
            raise GeoArtifactError("adjacency artifact is not symmetric")
    output: dict[str, set[str]] = {}
    for left, right in pairs:
        output.setdefault(left, set()).add(right)
    return {key: tuple(sorted(values)) for key, values in sorted(output.items())}
