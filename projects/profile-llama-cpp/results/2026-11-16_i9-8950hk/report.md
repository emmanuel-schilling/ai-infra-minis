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
| q4_k_m | 1 | 10.6 | 10.4 | 58% | 33% |
| q4_k_m | 6 | 27.6 | 27.1 | 88% | 86% |
| q4_k_m | 12 | 27.7 | 27.2 | 86% | 86% |
| q8_0 | 1 | 7.0 | 11.5 | 65% | 37% |
| q8_0 | 6 | 17.1 | 28.1 | 92% | 89% |
| q8_0 | 12 | 17.8 | 29.2 | 93% | 93% |

## Prompt processing against compute

| Model | Threads | Tokens/s | GFLOP/s | Same threads | Machine | Q8 estimate |
|---|---:|---:|---:|---:|---:|---:|
| q4_k_m | 1 | 48.4 | 150 | 94% | 18% | — |
| q4_k_m | 6 | 218.4 | 674 | 91% | 83% | — |
| q4_k_m | 12 | 216.2 | 667 | 82% | 82% | — |
| q8_0 | 1 | 24.3 | 75 | 47% | 9% | 14% |
| q8_0 | 6 | 104.2 | 322 | 43% | 40% | 59% |
| q8_0 | 12 | 106.9 | 330 | 41% | 41% | 61% |

## Thread scaling

| Model | Phase | Threads | Speedup | Per thread |
|---|---|---:|---:|---:|
| q4_k_m | pp | 1 | 1.00× | 100% |
| q4_k_m | pp | 6 | 4.51× | 75% |
| q4_k_m | pp | 12 | 4.46× | 37% |
| q4_k_m | tg | 1 | 1.00× | 100% |
| q4_k_m | tg | 6 | 2.60× | 43% |
| q4_k_m | tg | 12 | 2.61× | 22% |
| q8_0 | pp | 1 | 1.00× | 100% |
| q8_0 | pp | 6 | 4.29× | 72% |
| q8_0 | pp | 12 | 4.41× | 37% |
| q8_0 | tg | 1 | 1.00× | 100% |
| q8_0 | tg | 6 | 2.44× | 41% |
| q8_0 | tg | 12 | 2.53× | 21% |

Ceilings: the higher of `ceilings_after.json`, `ceilings_before.json`.

Readings more than 10% apart: flops on 6 threads: 7.43e+11, 6.39e+11.
