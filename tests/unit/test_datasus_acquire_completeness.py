"""T1.2 — DATASUS acquisition must not silently lose chunks.

Availability windows already drop (system,year) that DATASUS never published, so a chunk that
reaches fetch and does not succeed is data that SHOULD exist. `_acquire_datasus` therefore
hard-fails on a BLOCKED R bridge (an environment failure, never a coverage gap) and fails closed
on any non-success chunk when `require_complete` is set. These paths short-circuit before the
real combine/normalize, so a lightweight fake client suffices.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from pegasus.workflows import pipeline as P


class _FakeClient:
    def __init__(self, batches: dict) -> None:
        self._batches = batches

    def fetch_systems(self, *, systems, uf, years):  # matches MicrodatasusClient.fetch_systems
        return self._batches


def _req(status: str, err: str | None = None) -> NS:
    return NS(status=status, error_message=err)


def _batch(reqs: list[NS]) -> NS:
    return NS(requests=tuple(reqs), manifest_paths=())


def test_acquire_datasus_hard_fails_on_blocked_bridge() -> None:
    client = _FakeClient({"SIM": _batch([_req("blocked", "R bridge script missing")])})
    with pytest.raises(P.LivePipelineError, match="BLOCKED"):
        P._acquire_datasus(systems=["SIM"], uf="AL", years="2020", data_root=Path("data"), client=client)


def test_acquire_datasus_require_complete_fails_closed_on_partial() -> None:
    # one success + one transient timeout: tolerated by default, but require_complete=True refuses.
    client = _FakeClient({"SIM": _batch([_req("success"), _req("timeout", "R subprocess timed out")])})
    with pytest.raises(P.LivePipelineError, match="incomplete"):
        P._acquire_datasus(
            systems=["SIM"], uf="AL", years="2020-2021", data_root=Path("data"),
            client=client, require_complete=True,
        )


def test_acquire_datasus_partial_is_loud_not_silent() -> None:
    # Default (require_complete=False) tolerates a partial batch but MUST warn loudly (never a
    # silent skip). A system with zero usable chunks alongside is fine as long as ≥1 system is usable;
    # here the single system has a usable chunk, so it proceeds to combine — which we don't exercise.
    # We only assert the PARTIAL-COVERAGE warning fires before any combine work.
    client = _FakeClient({"SIM": _batch([_req("failed", "processed.parquet missing")])})
    # zero usable chunks for the only system -> hard error (no silent empty run), and the warning fired.
    with pytest.warns(RuntimeWarning, match="PARTIAL COVERAGE"):
        with pytest.raises(P.LivePipelineError, match="no available chunks"):
            P._acquire_datasus(systems=["SIM"], uf="AL", years="2020", data_root=Path("data"), client=client)
