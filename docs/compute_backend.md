# Compute Backend

`pegasus.compute` centralizes device selection, dtype, deterministic seeds, CUDA-required policy, and RAM/VRAM preflight. Numerical modules must request a `ComputeDevicePlan` and include it in diagnostics rather than choosing Torch devices locally.

Memory failures and unavailable required CUDA raise typed errors. CPU fallback is allowed only when CUDA is preferred, not required.
