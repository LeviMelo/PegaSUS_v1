# Architecture

PegaSUS compiles validated source artifacts into an exact 17-key output bundle. The canonical graph authority is `pegasus.efg.dag.build_efg`; legacy bundle builders are fixture-only compatibility modules and are not production runtime authorities.

The compile path is source-reality aware. Missing source manifests produce `fixture_only` runs. Strict runs resolve SIM, SINASC, SIDRA, and optional CNES/SIH inputs from a validated `materialized_external` manifest. Stage telemetry records terminal status and duration for every compiler stage.

Numerical services use the shared compute, storage, population, ST-DFM, and PIRS/HSIC layers. Dashboard and report modules only read compiled artifacts.
