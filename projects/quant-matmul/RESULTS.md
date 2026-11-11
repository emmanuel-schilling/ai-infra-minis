# Quantized matmul: results

No valid results yet.

Each benchmark run records the hardware, compiler flags and exact command, and
keeps its raw output in a dated folder under `results/` in this project. Never
overwrite an earlier run; add a new one and summarize it here.

A round is excluded when its speed check, timed before and after the round,
differs by more than 5% from the run's first check (`slow_rounds` in
`metadata.json`). Excluded runs are kept, with an `INVALID.md` that gives the
reason, and are never summarized.

## Benchmarks

## Conclusions

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
