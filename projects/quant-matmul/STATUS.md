# Quantized matmul: status

## Current state
Active. The scaffolding is in place: the build, the kernel interface, the
tests, the benchmarks (including the roofline) and `scripts/run_bench.py`.
The scalar fp32 kernel, `matmul_scalar`, is done: a plain triple loop, the
reference every other version is tested against. Its pending switch is off,
so its tests now check it against the double-precision reference. The first
AVX2 + FMA kernel, `matmul_avx2`, is done and tested for real too: one 8-wide
FMA accumulator per output, a horizontal sum, and a scalar loop for the last
k % 8 values. That kernel is now step v1 (`avx2_steps::v1_one_accumulator`).
Step v2, `v2_accumulators`, is done and tested for real: 8 independent
accumulators over 64 floats per pass, a tree reduction and one horizontal sum.
Step v3, `v3_register_blocking`, is done and tested for real too: a 3 × 4 tile
of outputs with 12 accumulators, each loaded W vector feeding 3 FMAs, and a
one-accumulator fallback for leftover rows and columns. Its compiled inner
loop keeps all 16 YMM registers in use with no spills to the stack. Step v4,
`v4_cache_blocking`, is done, tested for real and measured as the fastest
version, so `matmul_avx2` now calls it. All fp32 kernels accept empty sizes:
k = 0 sets C to zero, and the BLAS wrappers handle it themselves, because
BLAS rejects a leading dimension below 1. `quantize_q8` and `dequantize_q8`
are done and tested for real: amax / 127 as each block's fp32 scale, each
value divided by the scale, rounded to nearest with `std::nearbyint` and
clamped to ±127, and a zero scale, including one that underflows from a tiny
nonzero amax, giving an all-zero block. The scalar Q8 kernel,
`matmul_q8_scalar`, is done and tested for real, the reference for the other
Q8 kernels: each row of A is quantized once into a reused buffer of k / 32
blocks, each block pair's dot product is exact in int32, and the scaled
results are summed across blocks in fp32. The AVX2 Q8 kernel,
`matmul_q8_avx2`, is done and tested for real, one output at a time: A is
quantized once up front, and each block pair's dot product uses
`_mm256_maddubs_epi16` with the `_mm256_sign_epi8` trick, is widened to int32
with `_mm256_madd_epi16`, converted to fp32 and scaled into one 8-lane
accumulator per output. That version is now step v1,
`q8_avx2_steps::v1_one_accumulator` (declared in `include/qmm/q8_avx2_steps.hpp`),
and `matmul_q8_avx2` calls it. Step v2, `v2_register_blocking`, is done and
tested for real: blocks of 64 rows of W, a 2 × 4 tile of outputs in 8
accumulators, so each loaded A block feeds 4 block products and each W block
2, and `dot_q8_rows` for the leftover rows and columns. With 1 token there is
no pair of A rows to tile, so v2 then does v1's work. Both steps share the
per-block step `accumulate_block`, which is inlined. v2's compiled inner loop
keeps 5 of the 8 accumulators in registers and spills 3 to the stack (3 loads
and 3 stores per block step); whether that costs time is for the benchmark
run. `matmul_q8_avx2` keeps calling v1 until a benchmark shows v2 is faster.
No Q8 kernel is benchmarked yet. The AVX2 files
share `hsum256_ps` through the internal header `src/hsum_avx2.hpp`. The
other kernels you write are still stubs behind pending switches in
`include/qmm/pending.hpp`, so their tests pass without checking anything. The
BLAS baselines are done, and their tests run for real: Accelerate on macOS,
OpenBLAS on Linux and Windows. The NEON Q8 kernel (a stretch goal) is a stub
too, built and tested on Arm only.

The latest valid run is `results/2026-11-14_i9-8950hk/` (dated in UTC; run on
the evening of 2026-11-13), from commit `cc0e71e`, the first with the Q8
kernels. Q8 v2 is 1.5× faster than v1 on GEMM and 3.2× on lm_head, and the
same at 1 token. Q8 speeds up GEMV 2.6–2.7× over fp32 v4, but reaches only
0.64–0.76× of v4's speed on GEMM: without VNNI, 32 int8 products take 6
operations on the FMA ports against fp32's 4 FMAs (an estimate from the
instruction tables, not counters). Q8 v2 shows no mlp_down gap, since its
block of W fits L2 at k = 3,072, which supports the L2 hypothesis for v4. The
scalar Q8 kernel is auto-vectorized with SSE, so it is 11× the scalar fp32
kernel. The fp32 steps kept their speed after sharing `dot_fp32_rows`.

The run before it is `results/2026-11-12_i9-8950hk_2/`, on macOS 15.8.3
from commit `f454d55`, summarized in RESULTS.md. v4 is 1.3–4.8× faster than
v3 on every GEMM shape and beats single-threaded Accelerate on four of the
five (by 20–24% at 128 tokens, 3–6% at 1,024). mlp_down (k = 3,072) is the
exception, at 66–83% of Accelerate: v4's 64-row block of W is 768 KB at that
k, too big for the 256 KB L2 (a hypothesis, not yet measured). GEMV is
unchanged at the bandwidth. The run before it that morning,
`results/2026-11-12_i9-8950hk/`, is invalid: the speed check flagged rounds
that were faster than its first check (see its `INVALID.md`).

The first valid run is `results/2026-11-11_i9-8950hk/`. Every AVX2 version
reaches the read bandwidth for GEMV (about 16 GiB/s). For GEMM, v2 is only 1.0–1.3× faster than v1, while v3 is 2.4–3.8×
faster than v2: 79–82 GFLOP/s on attn_out, 75–92% of Accelerate, but 48–60% of
it on the MLP shapes and 28% on lm_head, where W doesn't stay in cache. It ran
on macOS 15.8.2 with v3 not yet committed; its exact source is in the run
folder's `source.patch`. The earlier run (`results/2026-11-10_i9-8950hk/`) is
invalid: the screen saver started during it (see its `INVALID.md`).

## Next step
In order:
1. Point `matmul_q8_avx2` at Q8 v2, which the 2026-11-13 run measured 1.5–3.2×
   faster on GEMM and equal at 1 token.
2. The accuracy comparison, the last part of the done criteria: relative
   error (‖C − C_ref‖ / ‖C_ref‖) and maximum absolute error of each kernel
   against the scalar fp32 result, on the benchmarked GPT-2 shapes, in
   RESULTS.md. The tests check bounds but don't report the errors.
3. Place each version on the roofline in RESULTS.md (the run has the peak and
   bandwidth; each kernel needs its arithmetic intensity).

For benchmark runs: close other apps, Chrome included, turn Low Power Mode
off, and start the run while the screen saver isn't running: `caffeinate`
keeps it from starting but can't stop it.

Optional work, after the required kernels:
- v5, cache blocking that adapts to k, to test whether L2 explains the
  mlp_down gap. v4 keeps 64 full rows of W per block: 192 KB at k = 768,
  which fits the 256 KB L2, but 768 KB at k = 3,072, which doesn't. Three
  rows of A grow from 9 KB to 36 KB, past the 32 KB L1. Two fixes, simpler
  first:
  1. Size the block of W from k: rows = 192 KB ÷ (k × 4 bytes), rounded down
     to a multiple of 4 for the 3 × 4 tile (64 rows at k = 768, 16 at
     k = 3,072). W then stays in L2 at any k; A still doesn't fit L1.
  2. Block along k as well, as BLAS libraries do (GotoBLAS, BLIS): split k
     into chunks of a few hundred values and run the blocks and tiles over
     one chunk at a time. A 64 × 256 block of W is 64 KB and 3 rows of A
     over 256 values are 3 KB. The cost: C holds partial sums, written by the
     first chunk and added to by the rest (k / chunk passes over C), and each
     tile does its horizontal sums once per chunk, so chunks must stay long.
     Whether small k should skip chunking is for the benchmark to decide.

  Write it as v5 next to v4, with the same signature, tests and benchmark. If
  the hypothesis holds, mlp_down improves clearly while the k = 768 shapes and
  GEMV stay about the same. Optionally, count L2 misses for v4 and v5 on
  mlp_down with Instruments (`xctrace`). Take block and chunk sizes from the
  cache sizes and confirm 2 or 3 values by measurement, rather than tuning
  for the best number. This is unrelated to fp16 scales, which change the Q8
  format, not the fp32 kernel.
- fp16 block scales, as in llama.cpp's Q8_0, converted with F16C's
  `_mm256_cvtph_ps`. A block shrinks from 36 to 34 bytes, about 6% less to
  read for memory-bound GEMV, and the format then matches llama.cpp's
  exactly. It needs the quantization test's scale tolerance widened for
  fp16's rounding, and a benchmark against the fp32 version.
- A 1 × 4 tile for the leftover rows in v3 and v4, if a measurement shows the
  current single-accumulator fallback costs enough (it may cost GEMM a few
  percent at 128 tokens).
- The NEON Q8 kernel (stretch goal), `src/q8_neon.cpp`, tested only by CI's
  Arm runners.
- Multithreading with `std::thread` (stretch goal), once all versions are
  compared single-threaded.

The AVX2 fp32 kernel was optimized in steps, each a separate version declared
in `include/qmm/avx2_steps.hpp` and written in `src/avx2.cpp`: v1 one
accumulator, v2 several accumulators, v3 register blocking (a 3 × 4 tile), v4
cache blocking. `matmul_avx2` calls the fastest measured version, v4 since
the 2026-11-12 run.

## Open questions
None.

## Decisions
Record each as: date, decision, reason.
- 2026-11-03: Builds as part of the root CMake project, added with
  `add_subdirectory`. Reason: the repo-wide checks and CI then cover it.
- 2026-11-03: Benchmark GPT-2 (124M) weight shapes: 768×768, 768×3072,
  3072×768 and 768×50257, each as GEMV (1 token) and GEMM (128 to 1,024
  tokens). Tests also use odd sizes such as 1, 7, 9 and 33. Reason: the
  results carry over to a GPT-2 forward pass, and GEMV against GEMM separates
  memory-bound decode from compute-bound prefill.
- 2026-11-03: Symmetric int8 quantization in [−127, 127], with one scale per
  block of 32 values along K, as in llama.cpp's Q8_0. Weights are quantized
  ahead of time and activations as each multiply starts, since AVX2's integer
  dot product needs both operands in int8. Reason: per-block scales are more
  accurate than per-row or per-tensor ones for little extra work, the format
  matches llama.cpp's own, so its kernels compare directly, and leaving out −128 keeps
  `_mm256_maddubs_epi16` from saturating.
- 2026-11-03: Single-threaded first; multithreading is a stretch goal once
  all four versions are compared. Reason: it keeps per-core kernel quality
  separate from thread scaling.
- 2026-11-03: Accuracy is measured as relative error (‖C − C_ref‖ / ‖C_ref‖)
  and maximum absolute error against the scalar fp32 result. Reason: the
  relative error summarizes the whole output, and the maximum catches single
  bad elements that an average hides.
- 2026-11-03: Measure this CPU's roofline (peak fp32 FLOP/s and memory
  bandwidth) and place every kernel on it. GEMV benchmarks cycle through more
  data than the 12 MB L3 holds. Reason: a kernel's speed means little without
  its ceiling, and one GPT-2 matrix fits in L3, which would overstate
  memory-bound performance.
- 2026-11-07: Test-driven switches, one per kernel, in `qmm/pending.hpp`.
  While a switch is on, that kernel's tests pass without checking it and the
  benchmarks leave it out. Each kernel then goes pass, fail, pass: turn the
  switch off, see the tests fail, write the kernel until they pass. Reason:
  `check` gates every commit, so the scaffolding could be committed before
  any kernel exists.
- 2026-11-07: The interface computes C = A · Wᵀ with W stored n × k, row-major,
  one row per output as in PyTorch's `Linear`. Q8 kernels require k to be a
  multiple of 32, quantize A with `quantize_q8`, and compute each block's dot
  product exactly in integers before scaling. Reason: every output becomes a
  dot product along k, the axis the Q8 blocks run along. GPT-2's checkpoint
  stores weights k × n, so a model loader transposes them.
- 2026-11-07: fp32 kernels are tested against a double-precision triple loop,
  with the classic rounding bound γ(k) · Σ|aᵢwᵢ| (Higham, section 3.1) as the
  tolerance. Q8 kernels are tested against the exact Q8 result, with the
  bound for k/32 + 2 operations, and must stay within 2% relative error of the
  fp32 product. Reason: a bound derived from the arithmetic accepts any
  correct summation order and catches real errors, where a fixed tolerance
  is either too loose or too tight.
- 2026-11-07: The benchmarks skip the output layer at 1,024 tokens. Reason:
  it is 79 GFLOP per call, which would dominate the run time; 128 tokens
  already measures it as a GEMM.
- 2026-11-07: Accelerate is benchmarked single-threaded
  (`VECLIB_MAXIMUM_THREADS=1`, set by `run_bench.py`). The multithreading
  stretch goal will use `std::thread`, not OpenMP. Reason: the comparison is
  per core, and Apple clang has no OpenMP.
- 2026-11-07: The roofline's peak-FMA loop runs 12 independent FMA chains,
  written by Claude at your choice. Reason: FMA's 4-cycle latency on two FMA
  units needs at least 8 chains, and 12 leaves a margin within 16 registers.
- 2026-11-07: Dependencies are pinned: vcpkg baseline `d186940b` with
  GoogleTest 1.18.0 and Google Benchmark 1.9.5; CMake 4.0 or later. CI
  installs CMake 4.4.4, Ninja 1.13.2, clang-format 23.1.3 and clang-tidy
  23.1.4, the versions on this Mac. Reason: local and CI builds match.
- 2026-11-08: OpenBLAS (0.3.34, through vcpkg, single-threaded) is the
  baseline on Linux and Windows, behind the same interface as Accelerate.
  macOS keeps Accelerate alone. Linux builds OpenBLAS with `dynamic-arch`,
  which picks kernels for the CPU at run time; MSVC can't, so CI keys its
  package cache by CPU model. Reason: every platform gets a tuned baseline,
  and both libraries define `cblas_sgemm`, so a static OpenBLAS linked on
  macOS would silently replace Accelerate's.
- 2026-11-08: A NEON Q8 kernel using the dot-product extension (`SDOT`) is a
  stretch goal, scaffolded with a pending switch and tested in CI on Arm
  runners. Reason: Arm's signed int8 dot product is the clearest contrast with
  AVX2's `maddubs` workaround, and CI's macOS runners are Arm.
- 2026-11-08: CMake's C++20 module scanning is off. Reason: it put
  module-map arguments into the compile commands that exist only after a
  build, which broke CI's clang-tidy job, which only configures.
- 2026-11-10: Kernels check their documented size preconditions with `assert`
  on entry, not inside their loops. Reason: the dev build, which runs the
  tests, catches a wrong size at the call, and the release build defines
  `NDEBUG`, so the checks cost the benchmarks nothing.
- 2026-11-10: `run_bench.py` holds one `caffeinate` for the whole run, and
  times `fp32/scalar/attn_out/tokens:128` three times before and after every
  round. A round whose check differs from the run's first by more than 5% is
  excluded. Reason: the screen saver started in a cooldown that wasn't
  protected and halved the CPU's speed while macOS reported no throttling. The
  scalar benchmark is in every build and its time follows the clock speed;
  comparing with the first check catches a slowdown that lasts whole rounds.
- 2026-11-10: Invalid runs are kept and committed, with an `INVALID.md`
  giving the reason, and are never summarized. Reason: keeping them shows the
  exclusions follow a rule rather than the numbers.
- 2026-11-10: The AVX2 kernel is optimized in steps (several accumulators,
  register blocking, cache blocking), each kept as its own version with the
  same signature, tested, and benchmarked in the same run as the others.
  `matmul_avx2` stays the public interface and points at the best version. A
  separate GEMV path is added only if a measurement shows a GEMM step slowing
  GEMV. Reason: versions timed side by side compare fairly, where numbers from
  different days don't, and each step's measured gain is the lesson.
- 2026-11-11: Every size may be 0. With k = 0 each output is an empty sum and C
  is set to zero; with m = 0 or n = 0 nothing is written. Kernels must not index
  an empty span (use `data()` plus an offset). Reason: it matches BLAS and
  PyTorch, and costs the loops nothing. The tests found that Accelerate ends
  the process on a leading dimension of 0, so the BLAS wrappers return early.
- 2026-11-11: Q8 block scales are fp32; fp16 is optional later work. Reason:
  fp32 is simpler, adds no rounding to the scale, and is what the quantization
  test's tolerance assumes. fp16 would save about 6% of GEMV's memory traffic
  and match llama.cpp's format exactly, which is worth measuring once the fp32
  kernels work.
- 2026-11-11: The macOS update and the next benchmark run are postponed, and
  the Q8 code continues meanwhile. Reason: the code doesn't depend on the
  measurements, and one later run can measure v4 and the Q8 kernels together.
- 2026-11-11: GPT-2's attention input projection, `attn_qkv` (n = 2,304,
  k = 768: query, key and value together), joins the benchmarked shapes, and
  the fp32 and Q8 tests run it with 4 tokens. Reason: it's the last of
  GPT-2's per-layer matrices, so the five shapes now cover a whole layer. Its
  7.1 MB W sits between attn_out's 2.4 MB and mlp_up's 9.4 MB, which helps
  show where W stops fitting in cache.
- 2026-11-11: `quantize_q8` divides each value by the scale rather than
  multiplying by 127 / amax, tests the scale (not amax) for zero, rounds with
  `std::nearbyint`, and clamps to ±127. Reason: 127 / amax overflows to
  infinity for amax below about 3.7e-37, a normal float; amax / 127 rounds to
  zero for amax below about 9e-44, so a test on amax still divides by zero;
  and adding 0.5 before truncating rounds 0.49999997 up to 1. Each case made
  an infinity or NaN that the int8 cast turns into undefined behaviour, and
  each now has a test. A subnormal scale keeps few bits, so such a block may
  peak below 127; the clamp keeps it in range.
- 2026-11-12: `run_bench.py` excludes a round when its results differ from
  round 1's by more than 5%: the median, over all benchmarks, of each
  benchmark's time in that round divided by its time in round 1. The speed
  check before and after each round stays, recorded as a diagnostic only.
  Earlier runs keep the verdicts of the rule they ran under. Reason: the check
  lasts a few seconds and followed the clock speed of that moment, varying by
  10% within a run while the rounds agreed within 1%, so it flagged the first
  2026-11-12 run. Back to back it agrees within 1.5%. The new rule passes that
  run (0.991 and 0.996) and the valid one after it (1.009 and 1.002), and
  flags the screen saver run (2.28 and 2.26). Like the old rule, it can't see
  a slowdown that lasts the whole run.
- 2026-11-13: The AVX2 Q8 kernel is optimized in steps, like the fp32 one:
  `q8_avx2_steps::v1_one_accumulator` and `v2_register_blocking`, each tested
  and benchmarked, with `matmul_q8_avx2` calling the fastest measured one.
  Reason: the same as for fp32 (2026-11-10): versions timed side by side
  compare fairly, and each step's measured gain is the lesson.
- 2026-11-13: The Q8 exactness test's bound is now γ(k + 1) times the sum of
  scale_a · scale_w · |qa · qw| over every product, replacing γ(k/32 + 2)
  times the sum of |scale_a · scale_w · dot| per block. Reason: the old bound
  assumed each block's dot product is formed exactly before any fp32
  rounding, which holds for the scalar kernel but not for AVX2 v1, which
  scales 8 partial dot products per block separately, nor for NEON's 4-lane
  SDOT. Partial sums can exceed their total, so v1 missed the old bound by
  7% on the new {7, 131, 96} shape. The new bound holds in any summation
  order, as the fp32 test's does. Checked that it still catches a real bug:
  with block 0's scales used for every block, v1 fails on all 8 shapes with
  more than one block.
