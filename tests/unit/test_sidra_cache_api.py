from pathlib import Path

from pegasus.sidra.api import SidraClient, SidraClientConfig, classification_expr, locality_expr
from pegasus.sidra.cache import SidraJsonCache


def test_sidra_expression_builders():
    assert classification_expr({"86": ["2776", "2777"], "2": ["4", "5"]}) == "2[4,5]|86[2776,2777]"
    assert locality_expr("N6", ["2704302", "2700300"]) == "N6[2704302,2700300]"


def test_sidra_client_cache_hit(tmp_path: Path):
    calls = {"n": 0}

    def transport(url, params, timeout):
        calls["n"] += 1
        return 200, [{"ok": True, "params": params}]

    client = SidraClient(
        config=SidraClientConfig(max_retries=0),
        cache=SidraJsonCache(tmp_path),
        transport=transport,
    )

    first = client.values(
        table_code="9606",
        periods=["2022"],
        variables=["93"],
        localities=["2704302"],
        locality_level="N6",
        classifications={"86": ["95251"]},
    )
    second = client.values(
        table_code="9606",
        periods=["2022"],
        variables=["93"],
        localities=["2704302"],
        locality_level="N6",
        classifications={"86": ["95251"]},
    )

    assert first.from_cache is False
    assert second.from_cache is True
    assert calls["n"] == 1


def test_sidra_client_retries_transient_status(tmp_path: Path):
    calls = {"n": 0}

    def transport(url, params, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 500, {"error": "server"}
        return 200, [{"ok": True}]

    client = SidraClient(
        config=SidraClientConfig(max_retries=1, backoff_initial_seconds=0, backoff_max_seconds=0),
        cache=SidraJsonCache(tmp_path),
        transport=transport,
    )

    response = client.metadata("9606")
    assert response.status_code == 200
    assert response.attempt == 2
