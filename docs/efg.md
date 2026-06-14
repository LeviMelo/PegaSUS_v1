# EFG

The Evidence Field Graph is registry-backed and proof-carrying. Field declarations are canonicalized before Delta legality checks, and legality warnings include registry evidence. Illegal attempts are materialized in `FailedBranches.parquet`.

Topological precompression runs before graph materialization. Its manifest records equivalence groups and protected non-equivalences. Geospatial pushforwards must carry transform evidence; direct rate allocation is forbidden.
