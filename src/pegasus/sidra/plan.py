from __future__ import annotations

from dataclasses import replace
from typing import Any

from pegasus.core.hashing import content_hash
from pegasus.sidra.schemas import SIDRAChunk, SIDRAMetadata, SIDRARequest


DEFAULT_BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"


def estimate_cells(
    *,
    localities: list[str],
    periods: list[str],
    variables: list[str],
    classifications: dict[str, list[str]],
) -> int:
    cells = len(localities) * len(periods) * len(variables)
    for categories in classifications.values():
        cells *= max(1, len(categories))
    return cells


def _chunk_id(spec: dict[str, Any]) -> str:
    return content_hash(spec)


def _request_url(base_url: str, table_id: str) -> str:
    return f"{base_url.rstrip('/')}/{table_id}/periodos/{{periods}}/variaveis/{{variables}}"


def _make_chunk(
    request: SIDRARequest,
    *,
    localities: list[str],
    periods: list[str],
    variables: list[str],
    classifications: dict[str, list[str]],
    base_url: str,
) -> SIDRAChunk:
    estimated = estimate_cells(
        localities=localities,
        periods=periods,
        variables=variables,
        classifications=classifications,
    )
    params = {
        "table_id": request.table_id,
        "variables": variables,
        "periods": periods,
        "locality_level": request.locality_level,
        "localities": localities,
        "classifications": classifications,
    }
    return SIDRAChunk(
        chunk_id=_chunk_id(params),
        table_id=request.table_id,
        variables=variables,
        periods=periods,
        locality_level=request.locality_level,
        localities=localities,
        classifications=classifications,
        estimated_cells=estimated,
        request_url=_request_url(base_url, request.table_id),
        request_params=params,
    )


def _split_list(values: list[str]) -> tuple[list[str], list[str]]:
    if len(values) <= 1:
        raise ValueError("Cannot split singleton list.")
    mid = max(1, len(values) // 2)
    return values[:mid], values[mid:]


def _largest_classification(classifications: dict[str, list[str]]) -> str | None:
    splittable = {k: v for k, v in classifications.items() if len(v) > 1}
    if not splittable:
        return None
    return sorted(splittable, key=lambda k: (-len(splittable[k]), k))[0]


def _split_request(request: SIDRARequest) -> list[SIDRARequest]:
    # TDD order: localities, highest-cardinality classification, periods, variables.
    if len(request.localities) > 1:
        left, right = _split_list(request.localities)
        return [
            request.model_copy(update={"localities": left}),
            request.model_copy(update={"localities": right}),
        ]

    cls = _largest_classification(request.classifications)
    if cls is not None:
        left, right = _split_list(request.classifications[cls])
        left_cls = dict(request.classifications)
        right_cls = dict(request.classifications)
        left_cls[cls] = left
        right_cls[cls] = right
        return [
            request.model_copy(update={"classifications": left_cls}),
            request.model_copy(update={"classifications": right_cls}),
        ]

    if len(request.periods) > 1:
        left, right = _split_list(request.periods)
        return [
            request.model_copy(update={"periods": left}),
            request.model_copy(update={"periods": right}),
        ]

    if len(request.variables) > 1:
        left, right = _split_list(request.variables)
        return [
            request.model_copy(update={"variables": left}),
            request.model_copy(update={"variables": right}),
        ]

    raise ValueError("Cannot split SIDRA request below cell ceiling.")


def validate_request_against_metadata(
    request: SIDRARequest,
    metadata: SIDRAMetadata,
) -> list[str]:
    errors: list[str] = []
    table = metadata.tables.get(request.table_id)
    if table is None:
        return [f"Unknown SIDRA table: {request.table_id}"]

    for variable in request.variables:
        if variable not in table.variables:
            errors.append(f"Unknown variable for table {request.table_id}: {variable}")

    for period in request.periods:
        if period not in table.periods:
            errors.append(f"Unknown period for table {request.table_id}: {period}")

    if request.locality_level not in table.locality_levels:
        errors.append(f"Unknown locality level for table {request.table_id}: {request.locality_level}")
    else:
        allowed_locs = set(table.localities_by_level.get(request.locality_level, []))
        for locality in request.localities:
            if locality not in allowed_locs:
                errors.append(f"Unknown locality {locality} at level {request.locality_level}")

    for cls_id, categories in request.classifications.items():
        if cls_id not in table.classifications:
            errors.append(f"Unknown classification for table {request.table_id}: {cls_id}")
            continue
        allowed = set(table.classifications[cls_id])
        for category in categories:
            if category not in allowed:
                errors.append(f"Unknown category {category} for classification {cls_id}")

    return errors


def plan_sidra_chunks(
    request: SIDRARequest,
    metadata: SIDRAMetadata,
    *,
    max_cells_per_request: int = 49900,
    max_localities_per_request: int = 200,
    base_url: str = DEFAULT_BASE_URL,
) -> list[SIDRAChunk]:
    errors = validate_request_against_metadata(request, metadata)
    if errors:
        raise ValueError("; ".join(errors))

    pending = [request]
    chunks: list[SIDRAChunk] = []

    while pending:
        current = pending.pop(0)
        cells = estimate_cells(
            localities=current.localities,
            periods=current.periods,
            variables=current.variables,
            classifications=current.classifications,
        )

        # Cap localities/request independently of cells: IBGE's gateway times out (HTTP 599,
        # empty body) on very long locality URLs even when the cell count is small — e.g. a
        # census-total request for MG's 853 municipalities is <1k cells but one oversized URL.
        # _split_request bisects localities first, so this splits such a chunk into servable
        # batches (throughput is recovered by concurrency, not by fatter requests).
        if cells <= max_cells_per_request and len(current.localities) <= max_localities_per_request:
            chunks.append(
                _make_chunk(
                    current,
                    localities=current.localities,
                    periods=current.periods,
                    variables=current.variables,
                    classifications=current.classifications,
                    base_url=base_url,
                )
            )
            continue

        try:
            pending = _split_request(current) + pending
        except ValueError as exc:
            raise ValueError(
                f"SIDRA request cannot be split below ceiling {max_cells_per_request}; "
                f"minimum unsplittable cell count={cells}"
            ) from exc

    return chunks
