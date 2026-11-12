# Invalid run: excluded from all results

The speed check flagged rounds 2 and 3 (`slow_rounds: [2, 3]` in
`metadata.json`), so by the exclusion rule in RESULTS.md this run is invalid.
It was rerun the same morning from the same commit as
`results/2026-11-12_i9-8950hk_2/`, which passed. The raw files are kept
unchanged as the record of why.

## What the speed check saw
Median of 3 repetitions of `fp32/scalar/attn_out/tokens:128`, in ms:

| Round | Before | After |
|---|---|---|
| 1 | 71.1 (the reference) | 69.8 |
| 2 | 64.6 (−9%) | 66.2 (−7%) |
| 3 | 66.2 (−7%) | 66.3 (−7%) |

Rounds 2 and 3 were flagged for being faster than round 1's first check, not
slower. macOS reported a CPU speed limit of 100% before and after every round.

## Evidence that the flag reflects the check, not the rounds
- The benchmarks themselves agree across rounds. Over all benchmarks, the
  median of (time in round 2 ÷ time in round 1) is 0.991, and for round 3 it is
  0.996. The scalar, v3, v4 and Accelerate benchmarks are each within 1.1%.
- Run back to back, the check is stable. Six later runs of 15 repetitions each
  gave medians of 3 repetitions between 68.4 and 69.5 ms (within 1.5%).
- So the 10% spread above is most likely the CPU's speed at the moment each
  check ran (not measured directly):
  the turbo clock depends on what ran just before (the build, a cooldown or a
  benchmark round). Each round runs for about 7 minutes and averages over
  those states; a check of a few seconds doesn't.

The run is excluded anyway: the rule was fixed before the run, and changing it
afterwards to keep a run would make exclusions depend on the numbers. Later
runs compare the rounds' own results instead (see RESULTS.md); this run keeps
the verdict of the rule it ran under.
