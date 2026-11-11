# Quantized matmul: status

## Current state
Active. The scaffolding is in place: the build, the kernel interface, the
tests, the benchmarks (including the roofline) and `scripts/run_bench.py`.
The scalar fp32 kernel, `matmul_scalar`, is done: a plain triple loop, the
reference every other version is tested against. Its pending switch is off,
so its tests now check it against the double-precision reference. The other
kernels you write are still stubs behind pending switches in
`include/qmm/pending.hpp`, so their tests pass without checking anything. The
BLAS baselines are done, and their tests run for real: Accelerate on macOS,
OpenBLAS on Linux and Windows. The NEON Q8 kernel (a stretch goal) is a stub
too, built and tested on Arm only.

No valid benchmark results exist yet. The first baseline run
(`results/2026-11-10_i9-8950hk/`) is invalid: the screen saver started during
it and slowed rounds 2 and 3 by about 2× (see its `INVALID.md` and RESULTS.md).
`run_bench.py` now keeps the Mac awake for the whole run and checks the CPU's
speed before and after every round.

## Next step
Record a first baseline with `run_bench.py`: the roofline, the scalar kernel
and Accelerate. Run it after the pending macOS update is installed, so the
baseline matches the OS used from now on, and after Spotlight and Bitdefender
have finished their post-update work. Turn Low Power Mode off first;
`run_bench.py` warns while it's on. Then write the AVX2 kernel: set `pending::avx2` to false (its tests fail),
then write `matmul_avx2` in `src/avx2.cpp` until they pass. After that, in
order: `quantize_q8` and `dequantize_q8`, the scalar Q8 kernel, then the AVX2
Q8 kernel. The NEON Q8 kernel comes last, if at all; it can be tested only by
CI's Arm runners.

## Open questions
- How to store each block's scale: fp32, or fp16 as in llama.cpp's Q8_0
  (converted with F16C's `_mm256_cvtph_ps`). fp16 makes a block 34 bytes
  instead of 36, about 6% less memory traffic for memory-bound GEMV, and
  matches llama.cpp exactly; fp32 is simpler and adds no rounding to the scale.
  Decide before writing `quantize_q8`; the quantization test's scale tolerance
  assumes fp32.
- Whether to add GPT-2's attention input projection (768 to 2,304: query, key
  and value together) to the benchmarked shapes.

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
