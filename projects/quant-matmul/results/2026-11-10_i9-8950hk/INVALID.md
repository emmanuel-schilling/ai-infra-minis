# Invalid run: excluded from all results

The macOS screen saver started during this run and slowed rounds 2 and 3 by
about 2×. Rounds 2 and 3 are invalid. Round 1 isn't reported either: it
completed before the slowdown, but one round alone isn't a baseline. The raw
files are kept unchanged as the record of why.

## What happened
Times are local (UTC−8).

| Time | Event |
|---|---|
| 14:56:18 | Last keyboard input |
| 14:57:11–14:59:19 | Round 1, with `caffeinate` holding the display awake |
| 14:59:19 | Round 1 ends; `caffeinate` exits; the 60 s cooldown runs unprotected |
| 14:59:46 | The Drift screen saver starts (its idle delay was 2 minutes) |
| 15:00:19–15:07:50 | Rounds 2 and 3, each 3 min 15 s against round 1's 2 min 7 s |
| 15:11:26 | The screen saver stops |

All 24 benchmarks slowed in rounds 2 and 3, by 1.3× (`roofline/peak_fma`) to
3.4× (`fp32/accelerate/mlp_down/tokens:1`). Per-round medians for four of
them (ms):

| Benchmark | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| `fp32/scalar/attn_out/tokens:128` | 65.6 | 122.6 | 124.0 |
| `fp32/scalar/attn_out/tokens:1` | 0.55 | 1.28 | 1.27 |
| `roofline/peak_fma` | 5.24 | 6.86 | 7.16 |
| `fp32/accelerate/attn_out/tokens:1024` | 10.9 | 24.3 | 23.8 |

macOS reported a CPU speed limit of 100% before and after every round, so the
script's throttling check didn't flag anything. `summary.json` mixes the fast
round with the slow ones and describes neither.

## How it was confirmed
The times above come from `pmset -g log` (power assertions) and the system log
(the Drift process). In a later test, `fp32/scalar/attn_out/tokens:128` took
118–126 ms while the screen saver ran, and 71.3–71.6 ms seconds after it was
stopped. Only stopping it was done on purpose; it started by itself twice,
during this run and again before the test. Two other suspects were ruled out by
their timing: a Bitdefender update was installed at 15:29, after the run, and
a macOS update had been downloaded at about 13:00, before it.

## Why the script missed it
`run_bench.py` at revision `76010d5` wrapped each round in its own `caffeinate`,
so nothing kept the Mac awake during the cooldowns. Once the screen saver had
started, the next round's `caffeinate` didn't stop it. The fix holds one
`caffeinate` for the whole run, and adds a speed check before and after every
round that flags any round more than 5% away from the first.

`metadata.json` also records `uncommitted_changes: true`, which is wrong: the
working tree was clean, and the flag came from this run's own `results/`
folder. That was fixed at the same time.
