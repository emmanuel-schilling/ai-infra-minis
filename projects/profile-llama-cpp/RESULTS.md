# Profile llama.cpp: results

Each profiling run records the llama.cpp commit, the model, the build flags,
the CPU features, the hardware and the exact command, and keeps its raw output
in a dated folder under `results/` in this project. Never overwrite an earlier
run; add a new one and summarize it here.

## Runs
All on the MacBook Pro (Intel Core i9-8950HK), llama.cpp b12377. The folder
dates are UTC; STATUS.md has each session's validity checks and notes.

| Folder | Session | What |
|---|---|---|
| `2026-11-16_i9-8950hk` | A | Baseline: both models, 1, 6 and 12 threads, and the ceilings (`report.md`) |
| `2026-11-17_i9-8950hk_accelerate` | A | The Accelerate build at 6 threads (`report.md`) |
| `2026-11-17_i9-8950hk_profile` | B | Time Profiler, Q8_0 |
| `2026-11-17_i9-8950hk_profile_2` | B | Time Profiler, Q4_K_M |
| `2026-11-17_i9-8950hk_kernels` | B | ggml's `MUL_MAT` on GPT-2's shapes against Quantized matmul (`comparison.md`) |
| `2026-11-18_i9-8950hk_profile_<model>_ports` | B | CPU counters: execution ports 0, 1 and 5, per model |
| `2026-11-18_i9-8950hk_profile_<model>_memory` | B | CPU counters: L2 and L3 misses and back-end stalls, per model |
| `2026-11-17_i9-8950hk_scaling` | C | Q8_0 from 1 to 12 threads (`report.md`) |

The write-up follows the outline in STATUS.md.
