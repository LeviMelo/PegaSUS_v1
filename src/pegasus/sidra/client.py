from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from pegasus.core.hashing import sha256_json
from pegasus.sidra.models import SIDRAQuerySpec


class SIDRAClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class SIDRAHTTPResult:
    url: str
    payload: Any
    elapsed_seconds: float
    download_bytes: int
    status_code: int
    attempts: int


class SIDRAClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int = 60,
        max_retries: int = 5,
        backoff_initial_seconds: float = 0.25,
        backoff_max_seconds: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_initial_seconds = backoff_initial_seconds
        self.backoff_max_seconds = backoff_max_seconds

    def build_url(self, spec: SIDRAQuerySpec) -> str:
        periods = ",".join(spec.periods)
        variables = ",".join(spec.variables)

        url = (
            f"{self.base_url}/{spec.table_id}"
            f"/periodos/{periods}"
            f"/variaveis/{variables}"
        )

        params: dict[str, str] = {
            "localidades": spec.localities,
            "view": spec.view,
        }

        if spec.classifications:
            params["classificacao"] = "|".join(spec.classifications)

        return f"{url}?{urlencode(params)}"

    def cache_key(self, spec: SIDRAQuerySpec) -> str:
        return sha256_json(spec.model_dump(mode="json"))

    def fetch_json(self, spec: SIDRAQuerySpec) -> SIDRAHTTPResult:
        url = self.build_url(spec)
        last_error: Exception | None = None
        start = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.get(url, timeout=self.timeout_seconds)
                elapsed = time.perf_counter() - start
                download_bytes = len(response.content)

                if response.status_code == 200:
                    try:
                        payload = response.json()
                    except json.JSONDecodeError as exc:
                        raise SIDRAClientError(f"SIDRA returned non-JSON response: {url}") from exc

                    return SIDRAHTTPResult(
                        url=url,
                        payload=payload,
                        elapsed_seconds=elapsed,
                        download_bytes=download_bytes,
                        status_code=response.status_code,
                        attempts=attempt + 1,
                    )

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise SIDRAClientError(
                        f"SIDRA transient HTTP {response.status_code}: {response.text[:500]}"
                    )

                raise SIDRAClientError(
                    f"SIDRA HTTP {response.status_code}: {response.text[:1000]}"
                )

            except Exception as exc:
                last_error = exc

                if attempt >= self.max_retries:
                    break

                delay = min(
                    self.backoff_initial_seconds * (2**attempt),
                    self.backoff_max_seconds,
                )
                time.sleep(delay)

        raise SIDRAClientError(f"SIDRA fetch failed after retries: {last_error}") from last_error


def write_raw_sidra_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_raw_sidra_json(path: str | Path) -> Any:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)