# Quantized matmul: results

Each benchmark run records the hardware, compiler flags and exact command, and
keeps its raw output in a dated folder under `results/` in this project. Never
overwrite an earlier run; add a new one and summarize it here.

A round is excluded when its speed check, timed before and after the round,
differs by more than 5% from the run's first check (`slow_rounds` in
`metadata.json`). Excluded runs are kept, with an `INVALID.md` that gives the
reason, and are never summarized.

## Benchmarks

### fp32: scalar, AVX2 v1 to v3, Accelerate (2026-11-11)
Run `results/2026-11-11_i9-8950hk/`: 3 rounds × 5 repetitions, all valid (no
throttling; speed checks 66.7–69.7 ms, within 3% of the first).

- Hardware: Intel Core i9-8950HK (6 cores, AVX2 + FMA), 32 GB, on AC power,
  Low Power Mode off; single-threaded, Accelerate with
  `VECLIB_MAXIMUM_THREADS=1`.
- Software: macOS 15.8.2, Apple clang 17.0.0, Release (`-O3 -DNDEBUG
  -std=c++20`; the AVX2 file adds `-mavx2 -mfma`).
- Source: commit `85a4f2d` plus uncommitted v3, saved as `source.patch` in the
  run folder.
- Command: `uv run python projects/quant-matmul/scripts/run_bench.py`.
- Measured ceilings: peak FMA 152 GFLOP/s, read bandwidth 16.5 GiB/s.

GEMM (prefill), GFLOP/s, median over 15 repetitions:

| Shape (n × k), tokens | scalar | v1 | v2 | v3 | Accelerate |
|---|---|---|---|---|---|
| attn_out 768 × 768, 128 | 2.3 | 20.6 | 27.1 | 78.7 | 85.3 |
| attn_out 768 × 768, 1,024 | 2.3 | 20.5 | 27.4 | 81.7 | 108.3 |
| mlp_up 3,072 × 768, 128 | 2.1 | 12.8 | 14.2 | 45.1 | 83.7 |
| mlp_up 3,072 × 768, 1,024 | 2.2 | 12.8 | 14.3 | 47.1 | 99.0 |
| mlp_down 768 × 3,072, 128 | 2.1 | 12.2 | 14.4 | 50.7 | 84.0 |
| mlp_down 768 × 3,072, 1,024 | 2.1 | 12.0 | 14.2 | 53.4 | 107.3 |
| lm_head 50,257 × 768, 128 | 2.1 | 8.7 | 9.0 | 21.7 | 77.8 |

GEMV (decode, 1 token), GiB/s of weights read:

| Shape | scalar | v1 | v2 | v3 | Accelerate |
|---|---|---|---|---|---|
| attn_out | 3.9 | 16.5 | 17.3 | 16.2 | 15.6 |
| mlp_up | 3.9 | 16.5 | 17.3 | 16.1 | 14.3 |
| mlp_down | 3.8 | 15.9 | 17.3 | 15.9 | 18.5 |
| lm_head | 3.9 | 16.2 | 17.0 | 15.9 | 13.1 |

## Conclusions

From the 2026-11-11 run (one machine, one run; to be confirmed by later runs):

- **GEMV is memory-bound, and every AVX2 version reaches the bandwidth.** v1
  to v3 read weights at 15.9–17.3 GiB/s, at or slightly above the roofline's
  16.5 GiB/s, so that benchmark slightly understates what the memory system
  delivers. The scalar kernel, at 3.9 GiB/s, is limited by its add latency
  instead. Only reading fewer bytes, as Q8 will, can speed GEMV up.
- **v3's fallback didn't slow GEMV.** With m = 1, v3 runs entirely through its
  one-accumulator fallback, and matches v1.
- **v2's extra accumulators helped GEMM only 1.0–1.3×.** Each FMA still needs
  two fresh loads, and W streams from L3 for every row of A, so the FMA
  latency v2 hides wasn't the main limit.
- **Register blocking (v3) gave 2.4–3.8× over v2,** and reaches 78.7–81.7
  GFLOP/s on attn_out, 92% and 75% of Accelerate (52–54% of the FMA peak).
- **The larger W matrices are the remaining gap, which is v4's target.**
  On mlp_up and mlp_down, v3 reaches 48–60% of Accelerate. On lm_head, whose
  W is 154 MB, v3 reads W about 44 times per call (once per 3-row tile, plus
  the 2 leftover rows), about 6.8 GB in 0.46 s: roughly 14 GiB/s, close to the
  read bandwidth. So lm_head is memory-bound even as a GEMM, and cache blocking
  should help it most.
- **The leftover rows may cost a few percent.** v3 is 4–5% faster at 1,024
  tokens (1 leftover row) than at 128 (2 leftover rows), consistent with the
  estimate, though other effects differ between the two sizes too.

## Measurement notes

### The screen saver halved the CPU's speed (2026-11-10)
The first baseline run, `results/2026-11-10_i9-8950hk/`, is invalid. The Drift
screen saver started during the cooldown after round 1, and all 24 benchmarks
slowed by 1.3–3.4× in the later rounds. macOS still reported no throttling. The run's
own per-round numbers showed it: the scalar kernel took 65.6 ms on
`attn_out/tokens:128` in round 1, and 122.6 ms and 124.0 ms in rounds 2 and 3.

A controlled test confirmed the cause. The same benchmark took 118–126 ms while
the screen saver ran, and 71.3–71.6 ms seconds after it was stopped. The run
folder's `INVALID.md` has the timeline and evidence.

The cause was a gap in `run_bench.py`: each round had its own `caffeinate`, so
the cooldowns weren't protected. The script now holds one `caffeinate` for the
whole run, and the speed check above would have flagged rounds 2 and 3.
