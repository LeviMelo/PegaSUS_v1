# SIDRA Subsystem

SIDRA ingestion validates requests against metadata, splits them under the cell ceiling, retries transient HTTP failures, caches responses, normalizes flat payloads to fact Parquet, and writes plan/extraction logs.

The workflow never substitutes fixtures after a live request failure. Successful fact and raw artifacts are hashed into a `materialized_external` source manifest.
