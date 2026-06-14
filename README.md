# PegaSUS

PegaSUS is an evidence-field epidemiological compiler for DATASUS and SIDRA sources. It builds a registry-backed Evidence Field Graph, Scientific Health Engine artifacts, PIRS models and residual scans, and an exact 17-key output bundle.

The canonical compiler distinguishes fixture validation from production candidacy through hashed source manifests and the strict command:

```powershell
pegasus acceptance level3 --run <run-directory>
```

Development validation:

```powershell
conda run -n pegasus python -m pytest tests/unit tests/integration
conda run -n pegasus python scripts/dev/audits/audit_slice26a26b_runtime_authority_stage_plan.py
conda run -n pegasus python scripts/dev/audits/audit_slice27a27b_registry_canonical_delta.py
```

The MSD and TDD remain the authoritative architecture and technical contracts. See `docs/architecture.md`, `docs/output_bundle.md`, and `docs/level3_acceptance.md` for implemented boundaries.
