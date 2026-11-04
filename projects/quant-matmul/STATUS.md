# Quantized matmul: status

## Current state
Active: settling the open questions before the first code.

## Next step
Settle the open questions below, then propose a plan.

## Open questions
- Matrix sizes and shapes to benchmark.
- Quantization scheme: per-tensor or per-row scales, symmetric or asymmetric.
- Whether to try multithreading, or stay single-threaded.

## Decisions
Record each as: date, decision, reason.
- 2026-11-03: Builds as part of the root CMake project, added with
  `add_subdirectory`. Reason: the repo-wide checks and CI then cover it.
