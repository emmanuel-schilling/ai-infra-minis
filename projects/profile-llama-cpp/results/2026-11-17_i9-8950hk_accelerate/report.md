# Baseline on the roofline

## Ceilings measured

| Threads | Read bandwidth (GB/s) | fp32 FMA peak (GFLOP/s) |
|---:|---:|---:|
| 1 | 17.9 | 159 |
| 6 | 30.6 | 743 |
| 12 | 31.5 | 811 |

## Token generation against bandwidth

| Model | Threads | Tokens/s | Weight reads (GB/s) | Same threads | Machine |
|---|---:|---:|---:|---:|---:|
| q4_k_m | 6 | 27.7 | 27.1 | 89% | 86% |
| q8_0 | 6 | 17.0 | 28.0 | 91% | 89% |

## Prompt processing against compute

| Model | Threads | Tokens/s | GFLOP/s | Same threads | Machine | Q8 estimate |
|---|---:|---:|---:|---:|---:|---:|
| q4_k_m | 6 | 186.3 | 575 | 77% | 71% | — |
| q8_0 | 6 | 109.8 | 339 | 46% | 42% | 63% |

## Thread scaling

| Model | Phase | Threads | Speedup | Per thread |
|---|---|---:|---:|---:|
| q4_k_m | pp | 6 | 1.00× | 100% |
| q4_k_m | tg | 6 | 1.00× | 100% |
| q8_0 | pp | 6 | 1.00× | 100% |
| q8_0 | tg | 6 | 1.00× | 100% |

Ceilings: the higher of `ceilings_after.json`, `ceilings_before.json` in `projects/profile-llama-cpp/results/2026-11-16_i9-8950hk`.

Readings more than 10% apart: flops on 6 threads: 7.43e+11, 6.39e+11.
