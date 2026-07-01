from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pegasus.core.config import load_yaml
from pegasus.sidra.cache import SidraJsonCache
from pegasus.sidra.schemas import SIDRAChunk


DEFAULT_BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"

Transport = Callable[[str, dict[str, Any], int], tuple[int, Any]]


@dataclass(frozen=True)
class SidraClientConfig:
    base_url: str = DEFAULT_BASE_URL
    view_mode: str = "flat"
    timeout_seconds: int = 60
    # 599 is the client-synthesized code for a transport-level failure (connection
    # reset / timeout / DNS) -- see get_json's `except: status = 599`. SIDRA's front
    # end throws these intermittently under heavy concurrency, so they MUST be
    # retried (they are transient), not surfaced as a hard failure that aborts a
    # 94-table metadata fetch.
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504, 599)
    max_retries: int = 5
    backoff_initial_seconds: float = 0.25
    backoff_max_seconds: float = 10.0

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SidraClientConfig":
        return cls(
            base_url=str(payload.get("base_url", DEFAULT_BASE_URL)),
            view_mode=str(payload.get("view_mode", "flat")),
            timeout_seconds=int(payload.get("timeout_seconds", 60)),
            retry_status_codes=tuple(int(x) for x in payload.get("retry_status_codes", [429, 500, 502, 503, 504, 599])),
            max_retries=int(payload.get("max_retries", 5)),
            backoff_initial_seconds=float(payload.get("backoff_initial_seconds", 0.25)),
            backoff_max_seconds=float(payload.get("backoff_max_seconds", 10.0)),
        )

    @classmethod
    def from_file(cls, path: str | Path = "config/sidra.yaml") -> "SidraClientConfig":
        data = load_yaml(path)
        return cls.from_mapping(data.get("sidra", data))


@dataclass(frozen=True)
class SidraResponse:
    url: str
    params: dict[str, Any]
    payload: Any
    status_code: int
    from_cache: bool
    attempt: int
    seconds: float
    sidecar: dict[str, Any] | None = None


def _decode_body(raw_bytes: bytes, content_encoding: str | None) -> str:
    """Decode a possibly gzip/deflate-compressed SIDRA response body.

    SIDRA intermittently returns gzip-compressed payloads (and we request gzip to make
    large national transfers smaller/faster). urllib does not auto-decompress, so we
    handle Content-Encoding explicitly, with a magic-byte fallback (0x1f 0x8b = gzip).
    """
    import gzip
    import zlib

    encoding = (content_encoding or "").lower()
    if encoding == "gzip" or raw_bytes[:2] == b"\x1f\x8b":
        raw_bytes = gzip.decompress(raw_bytes)
    elif encoding == "deflate":
        try:
            raw_bytes = zlib.decompress(raw_bytes)
        except zlib.error:
            raw_bytes = zlib.decompress(raw_bytes, -zlib.MAX_WBITS)
    return raw_bytes.decode("utf-8", errors="replace")


def _default_transport(url: str, params: dict[str, Any], timeout_seconds: int) -> tuple[int, Any]:
    query = urllib.parse.urlencode(params, doseq=False)
    full_url = f"{url}?{query}" if query else url
    request = urllib.request.Request(
        full_url,
        headers={
            "User-Agent": "PegaSUS/0.1 SIDRA client",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = _decode_body(response.read(), response.headers.get("Content-Encoding"))
            return status, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = _decode_body(exc.read(), exc.headers.get("Content-Encoding") if exc.headers else None)
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"error": raw}
        return int(exc.code), payload


def classification_expr(classifications: dict[str, list[str]] | None) -> str | None:
    if not classifications:
        return None
    parts = []
    for cls_id in sorted(classifications, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        categories = classifications[cls_id]
        parts.append(f"{cls_id}[{','.join(str(x) for x in categories)}]")
    return "|".join(parts)


def locality_expr(level: str, localities: list[str] | str) -> str:
    if isinstance(localities, str):
        return localities

    if len(localities) == 1 and localities[0] in {"BR", "N1", "N2", "N3", "N6"}:
        return localities[0]

    return f"{level}[{','.join(str(x) for x in localities)}]"


def pipe_expr(values: list[str]) -> str:
    return "|".join(str(x) for x in values)


class SidraClient:
    def __init__(
        self,
        *,
        config: SidraClientConfig | None = None,
        cache: SidraJsonCache | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or SidraClientConfig.from_file()
        self.cache = cache or SidraJsonCache()
        self.transport = transport or _default_transport

    def endpoint(self, suffix: str = "") -> str:
        base = self.config.base_url.rstrip("/")
        suffix = suffix.strip("/")
        return base if not suffix else f"{base}/{suffix}"

    def catalog(self, **params: Any) -> SidraResponse:
        return self.get_json(self.endpoint(), params={k: v for k, v in params.items() if v is not None}, namespace="http")

    def metadata(self, table_code: str) -> SidraResponse:
        return self.get_json(self.endpoint(f"{table_code}/metadados"), namespace="http")

    def periods(self, table_code: str) -> SidraResponse:
        return self.get_json(self.endpoint(f"{table_code}/periodos"), namespace="http")

    def localities(self, table_code: str, level: str) -> SidraResponse:
        return self.get_json(self.endpoint(f"{table_code}/localidades/{level}"), namespace="http")

    def values(
        self,
        *,
        table_code: str,
        periods: list[str],
        variables: list[str],
        localities: list[str] | str,
        locality_level: str,
        classifications: dict[str, list[str]] | None = None,
        view: str | None = None,
    ) -> SidraResponse:
        params: dict[str, Any] = {
            "localidades": locality_expr(locality_level, localities),
            "view": view or self.config.view_mode,
        }
        cls_expr = classification_expr(classifications)
        if cls_expr:
            params["classificacao"] = cls_expr

        return self.get_json(
            self.endpoint(f"{table_code}/periodos/{pipe_expr(periods)}/variaveis/{pipe_expr(variables)}"),
            params=params,
            namespace="values",
        )

    def values_from_chunk(self, chunk: SIDRAChunk, *, view: str | None = None) -> SidraResponse:
        return self.values(
            table_code=chunk.table_id,
            periods=chunk.periods,
            variables=chunk.variables,
            localities=chunk.localities,
            locality_level=chunk.locality_level,
            classifications=chunk.classifications,
            view=view,
        )

    def fetch_chunks_parallel(
        self,
        chunks: list[SIDRAChunk],
        *,
        max_workers: int = 12,
        view: str | None = None,
    ) -> dict[str, SidraResponse]:
        """Fetch many cell-budgeted chunks concurrently (MSD national-scale acquisition).

        SIDRA throttles by *cells per request*, not by request rate, so the optimal
        strategy is wide concurrency over chunks already sized just under the cell
        ceiling. There is deliberately NO inter-request sleep — only per-request
        exponential backoff on transient 429/5xx (handled in get_json). Distinct chunks
        write distinct cache keys, so the file cache is safe under concurrency.

        Returns ``{chunk_id: SidraResponse}``; callers inspect ``status_code`` per chunk.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not chunks:
            return {}
        workers = max(1, min(int(max_workers), len(chunks)))
        results: dict[str, SidraResponse] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_chunk = {
                executor.submit(self.values_from_chunk, chunk, view=view): chunk
                for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    results[chunk.chunk_id] = future.result()
                except Exception as exc:  # pragma: no cover - defensive; transport already guards
                    results[chunk.chunk_id] = SidraResponse(
                        url=chunk.request_url,
                        params=dict(chunk.request_params),
                        payload={"error": str(exc)},
                        status_code=599,
                        from_cache=False,
                        attempt=0,
                        seconds=0.0,
                    )
        return results

    def ping(self) -> SidraResponse:
        return self.catalog(nivel="N1")

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        namespace: str,
        use_cache: bool = True,
    ) -> SidraResponse:
        params = params or {}
        if use_cache:
            cached = self.cache.read(namespace=namespace, url=url, params=params)
            if cached.hit:
                return SidraResponse(
                    url=url,
                    params=params,
                    payload=cached.payload,
                    status_code=int((cached.sidecar or {}).get("status_code", 200)),
                    from_cache=True,
                    attempt=int((cached.sidecar or {}).get("attempt", 0)),
                    seconds=0.0,
                    sidecar=cached.sidecar,
                )

        last_status = 0
        last_payload: Any = None
        start_all = time.time()

        for attempt in range(1, self.config.max_retries + 2):
            started = time.time()
            try:
                status, payload = self.transport(url, params, self.config.timeout_seconds)
            except Exception as exc:
                status = 599
                payload = {"error": str(exc)}

            seconds = time.time() - started
            last_status = status
            last_payload = payload

            if status < 400 or status not in self.config.retry_status_codes or attempt > self.config.max_retries:
                _, _, sidecar = self.cache.write(
                    namespace=namespace,
                    url=url,
                    params=params,
                    payload=payload,
                    status_code=status,
                    attempt=attempt,
                    seconds=seconds,
                )
                return SidraResponse(
                    url=url,
                    params=params,
                    payload=payload,
                    status_code=status,
                    from_cache=False,
                    attempt=attempt,
                    seconds=round(time.time() - start_all, 6),
                    sidecar=sidecar,
                )

            sleep_s = min(
                self.config.backoff_max_seconds,
                self.config.backoff_initial_seconds * (2 ** (attempt - 1)),
            )
            time.sleep(sleep_s)

        # Defensive fallback; loop always returns.
        return SidraResponse(
            url=url,
            params=params,
            payload=last_payload,
            status_code=last_status,
            from_cache=False,
            attempt=self.config.max_retries + 1,
            seconds=round(time.time() - start_all, 6),
        )
