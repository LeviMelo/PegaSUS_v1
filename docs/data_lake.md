# Data Lake

Source data is separated into raw, processed, cache, manifest, diagnostic, and run locations. External DATASUS and SIDRA workflows materialize immutable hashed artifacts; compile runs reference those artifacts through source manifests.

Parquet access uses `pegasus.storage` for schema-aware writes, scans, row counts, hashes, and optional DuckDB queries. Run outputs remain confined to the exact 17-key bundle.
