# Output Bundle

Every run has exactly 17 first-class keys: the eleven canonical parquet tables, `Tables`, `Maps`, and four JSON manifests/configuration files. Additional artifacts belong under `Tables` or `Maps`; no workflow may add another root key.

`V_fields`, `E_DAG`, and `Q_tensor` are the graph/numerical core. Warnings, failed branches, quarantined fields, forced fields, model associations, residual associations, hypotheses, and the variable dictionary carry evidence and safety state.
