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
`v4_cache_blocking`, is done and tested for real but not yet measured, so
`matmul_avx2` still calls v3. All fp32 kernels now accept empty sizes:
k = 0 sets C to zero, and the BLAS wrappers handle it themselves, because
BLAS rejects a leading dimension below 1. `quantize_q8` and `dequantize_q8`
are done and tested for real: amax / 127 as each block's fp32 scale, each
value divided by the scale, rounded to nearest with `std::nearbyint` and
clamped to ±127, and a zero scale, including one that underflows from a tiny
nonzero amax, giving an all-zero block. The other kernels you write are
still stubs behind pending switches in
`include/qmm/pending.hpp`, so their tests pass without checking anything. The
BLAS baselines are done, and their tests run for real: Accelerate on macOS,
OpenBLAS on Linux and Windows. The NEON Q8 kernel (a stretch goal) is a stub
too, built and tested on Arm only.

The first valid run is `results/2026-11-11_i9-8950hk/`, summarized in
RESULTS.md. Every AVX2 version reaches the read bandwidth for GEMV (about 16
GiB/s). For GEMM, v2 is only 1.0–1.3× faster than v1, while v3 is 2.4–3.8×
faster than v2: 79–82 GFLOP/s on attn_out, 75–92% of Accelerate, but 48–60% of
it on the MLP shapes and 28% on lm_head, where W doesn't stay in cache. It ran
on macOS 15.8.2 with v3 not yet committed; its exact source is in the run
folder's `source.patch`. The earlier run (`results/2026-11-10_i9-8950hk/`) is
invalid: the screen saver started during it (see its `INVALID.md`).

## Next step
The code continues while the benchmarks wait. In order:
1. The scalar Q8 kernel, `matmul_q8_scalar` in `src/q8_scalar.cpp`
   (`pending::q8_scalar`), the reference for the other Q8 kernels.
2. The AVX2 Q8 kernel, `matmul_q8_avx2` in `src/q8_avx2.cpp`
   (`pending::q8_avx2`), using `_mm256_maddubs_epi16` with the
   `_mm256_sign_epi8` trick.

Postponed, to be done together later:
- Install the pending macOS update.
- Run `run_bench.py` on the new OS: v4 against v3 (point `matmul_avx2` at v4
  only if it's faster), and the Q8 kernels written by then. Close other apps,
  Chrome included, and start the run while the screen saver isn't running:
  `caffeinate` keeps it from starting but can't stop it. Turn Low Power Mode
  off first.

Optional work, after the required kernels:
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
cache blocking. `matmul_avx2` calls the fastest measured version.

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
