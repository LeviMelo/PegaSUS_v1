# DATASUS Subsystem

DATASUS acquisition is isolated behind `MicrodatasusClient` and an R subprocess. The boundary supports SIM-DO, SINASC, SIH-RD, and CNES-ST, checks required R packages, emits heartbeat/row progress, enforces timeout and stale-heartbeat termination, and materializes raw RDS plus processed Parquet.

Successful requests record hashes, row/column metadata, package versions, logs, and request manifests. Missing Rscript/packages and subprocess failures remain explicit blocked/failed results.
