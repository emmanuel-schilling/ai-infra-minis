# Baseline on the roofline

## Ceilings measured

| Threads | Read bandwidth (GB/s) | fp32 FMA peak (GFLOP/s) |
|---:|---:|---:|
| 1 | 17.8 | 159 |
| 6 | 29.1 | 772 |
| 12 | 33.2 | 852 |

## Token generation against bandwidth

| Model | Threads | Tokens/s | Weight reads (GB/s) | Same threads | Machine |
|---|---:|---:|---:|---:|---:|
| q8_0 | 1 | 7.0 | 11.5 | 65% | 35% |
| q8_0 | 2 | 12.9 | 21.2 | — | 64% |
| q8_0 | 3 | 15.5 | 25.4 | — | 77% |
| q8_0 | 4 | 16.3 | 26.7 | — | 81% |
| q8_0 | 6 | 17.1 | 28.1 | 96% | 85% |
| q8_0 | 8 | 17.4 | 28.6 | — | 86% |
| q8_0 | 10 | 17.7 | 29.1 | — | 88% |
| q8_0 | 12 | 17.7 | 29.0 | 88% | 88% |

## Prompt processing against compute

| Model | Threads | Tokens/s | GFLOP/s | Same threads | Machine | Q8 estimate |
|---|---:|---:|---:|---:|---:|---:|
| q8_0 | 1 | 24.1 | 74 | 47% | 9% | 13% |
| q8_0 | 2 | 44.0 | 136 | — | 16% | 24% |
| q8_0 | 3 | 60.9 | 188 | — | 22% | 33% |
| q8_0 | 4 | 77.2 | 238 | — | 28% | 42% |
| q8_0 | 6 | 105.1 | 325 | 42% | 38% | 57% |
| q8_0 | 8 | 94.1 | 290 | — | 34% | 51% |
| q8_0 | 10 | 99.0 | 306 | — | 36% | 54% |
| q8_0 | 12 | 105.5 | 326 | 38% | 38% | 57% |

## Thread scaling

| Model | Phase | Threads | Speedup | Per thread |
|---|---|---:|---:|---:|
| q8_0 | pp | 1 | 1.00× | 100% |
| q8_0 | pp | 2 | 1.82× | 91% |
| q8_0 | pp | 3 | 2.52× | 84% |
| q8_0 | pp | 4 | 3.20× | 80% |
| q8_0 | pp | 6 | 4.36× | 73% |
| q8_0 | pp | 8 | 3.90× | 49% |
| q8_0 | pp | 10 | 4.11× | 41% |
| q8_0 | pp | 12 | 4.38× | 36% |
| q8_0 | tg | 1 | 1.00× | 100% |
| q8_0 | tg | 2 | 1.84× | 92% |
| q8_0 | tg | 3 | 2.21× | 74% |
| q8_0 | tg | 4 | 2.31× | 58% |
| q8_0 | tg | 6 | 2.44× | 41% |
| q8_0 | tg | 8 | 2.48× | 31% |
| q8_0 | tg | 10 | 2.52× | 25% |
| q8_0 | tg | 12 | 2.51× | 21% |

Ceilings: the higher of `ceilings_after.json`, `ceilings_before.json`.
