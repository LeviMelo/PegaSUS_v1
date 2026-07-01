"""Build the structural municipality queen-contiguity graph from IBGE geometry.

This is the *structural* spatial base graph for the SpatialWeightGraph
(MSD-II §II.4 / MII-SPG-01). It is contiguity — shared boundary — and therefore
provenance-disjoint from every substantive variable (population, GDP, flows), so
it is legal as the default spatial structure for all inference (the §II.4.1
circularity guard requires a `structural` default).

Source: IBGE malhas v3 API, municipality-level polygons per UF. All UFs are
combined into one GeoDataFrame before contiguity is computed, so cross-UF border
neighbours are captured (a per-UF computation would miss them).

Output: a parquet edge list ``(left_id, right_id)`` of IBGE cod7 pairs, symmetric,
no self-loops — the artifact shape ``pegasus.geo.adjacency.load_adjacency`` reads.

Usage:
    python scripts/dev/geo/build_municipality_contiguity.py            # national
    python scripts/dev/geo/build_municipality_contiguity.py --uf 27    # one UF (AL)
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
import zlib
from pathlib import Path

import geopandas as gpd
import polars as pl
from shapely.geometry import shape

UFS = [
    11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53,
]
MALHA_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}"
    "?formato=application/vnd.geo+json&intrarregiao=municipio"
)
OUT_PARQUET = Path("config/registries/spatial/municipality_contiguity_queen.parquet")
OUT_MANIFEST = Path("config/registries/spatial/municipality_contiguity_queen.manifest.json")


def _fetch_json(url: str, *, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0 pegasus-spg", "Accept-Encoding": "gzip, deflate"}
            )
            resp = urllib.request.urlopen(req, timeout=120)
            body = resp.read()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            if enc == "gzip" or body[:2] == b"\x1f\x8b":
                body = gzip.decompress(body)
            elif enc == "deflate":
                body = zlib.decompress(body)
            return json.loads(body)
        except Exception as exc:  # noqa: BLE001 - transient API, retry with backoff
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _load_uf_municipalities(uf: int) -> gpd.GeoDataFrame:
    gj = _fetch_json(MALHA_URL.format(uf=uf))
    feats = gj.get("features", [])
    codes = [str(f["properties"]["codarea"]) for f in feats]
    geoms = [shape(f["geometry"]).buffer(0) for f in feats]  # buffer(0) repairs invalid rings
    return gpd.GeoDataFrame({"cod7": codes}, geometry=geoms, crs="EPSG:4674")


def build(ufs: list[int]) -> tuple[pl.DataFrame, dict]:
    frames = []
    for uf in ufs:
        t = time.time()
        gdf = _load_uf_municipalities(uf)
        frames.append(gdf)
        print(f"  UF {uf}: {len(gdf)} municipalities in {time.time() - t:.1f}s", flush=True)
    national = gpd.GeoDataFrame(
        gpd.pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4674"
    )

    # Vectorized queen contiguity: polygons that intersect but are not identical.
    joined = gpd.sjoin(national, national, predicate="intersects", how="inner")
    left = national["cod7"].to_numpy()
    edges: set[tuple[str, str]] = set()
    isolated_guard: set[str] = set(national["cod7"])
    for li, rj in zip(joined.index, joined["index_right"]):
        a = str(national["cod7"].iloc[li])
        b = str(national["cod7"].iloc[int(rj)])
        if a == b:
            continue
        edges.add((a, b))
        edges.add((b, a))
        isolated_guard.discard(a)
        isolated_guard.discard(b)

    rows = sorted(edges)
    df = pl.DataFrame({"left_id": [a for a, _ in rows], "right_id": [b for _, b in rows]})
    manifest = {
        "graph_id": "contiguity_queen",
        "legality_class": "structural",
        "provenance": "IBGE malhas v3 API (municipality polygons); queen contiguity (shared boundary).",
        "crs": "EPSG:4674",
        "municipalities": int(len(national)),
        "undirected_edges": len(edges) // 2,
        "isolated_municipalities": sorted(isolated_guard),
        "ufs": ufs,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return df, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uf", type=int, default=None, help="Single UF code (default: national).")
    args = ap.parse_args()
    ufs = [args.uf] if args.uf else UFS
    df, manifest = build(ufs)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PARQUET)
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUT_PARQUET} ({df.height} directed edges)")
    print(f"wrote {OUT_MANIFEST}")
    print(f"municipalities={manifest['municipalities']} edges={manifest['undirected_edges']} "
          f"isolated={len(manifest['isolated_municipalities'])}")


if __name__ == "__main__":
    main()
