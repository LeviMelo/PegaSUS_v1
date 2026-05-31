# PegaSUS

PegaSUS is a compiler-like epidemiological engine for Brazilian public-health data.

It ingests DATASUS, SIDRA/IBGE, and CNES substrates; preserves raw/provenance layers; compiles typed epidemiological fields; enforces legality predicates; emits `Q_tensor`; and serializes a reproducible 17-key run bundle.

This repository follows the PegaSUS v1.0/v1.0.1/v1.0.2 production contract and implementation doctrine.

## First execution target

The first vertical slice is:

```text
SIM-DO
→ raw/processed cache
→ profile
→ normalize
→ typed mortality fields
→ legality gate
→ Q(v)
→ 17-key output bundle