# Profile llama.cpp: status

## Current state
Started 2026-11-13. `scripts/setup_llama_cpp.py` sets up llama.cpp release
b12377 and the two Qwen2.5-1.5B models outside the repository (see Decisions),
builds them and runs a smoke test. It ran on 2026-11-14 into ~/src/llama.cpp
and ~/models: both builds succeeded, and both models generate coherent text.

The scripts for steps 3–8 are written and smoke-tested on tiny settings
(2026-11-15), but haven't run for real: no measurements yet.
- `run_baseline.py` (steps 3, 7, 8): llama-bench rounds with cooldowns, and
  the ceilings from Quantized matmul's `qmm_bench`, measured before and after
  the rounds. `qmm_bench` gained 6- and 12-thread variants of its bandwidth
  and FMA-peak benchmarks for this.
- `report_baseline.py` (steps 4, 7): the roofline shares and thread scaling.
- `run_profile.py` (step 5): `xctrace` Time Profiler per phase, summarized by
  ggml operation, leaf function, stack function and thread.
- `run_kernel_compare.py` (step 6): ggml's `MUL_MAT` on GPT-2's shapes with
  Q8_0 and fp32 weights, single-threaded, against Quantized matmul's latest
  run.

What the b12377 source shows about this machine's kernel paths, to confirm in
the profiles:
- Llamafile's sgemm is on by default in llama.cpp builds and handles Q8_0 (and
  Q4_0) but not Q4_K, so Q8_0 prompt processing probably runs through it. A
  test trace showed it there (`tinyBLAS_Q0_AVX`).
- Flash attention runs on the CPU: llama-bench's default `-fa auto` turns it
  on (`ggml_compute_forward_flash_attn_ext_*` in the test traces).
- The model has its own output layer (`output.weight`), not the token
  embedding reused, so token generation reads every weight except the
  embedding once per token.
- On AVX2, Q4_K weights are repacked at load time into an 8 x 8 layout
  (`q4_K_8x8_q8_K`) when their row count is a multiple of 8. The setup's log
  shows 168 of the Q4_K_M model's 169 Q4_K tensors repacked; the other is the
  token embedding, which is looked up rather than multiplied. Its 29 Q6_K
  tensors are not repacked. Q8_0 has no x86 repack, and none of the Q8_0
  model's tensors were repacked.
- With BLAS on, ggml hands a matrix multiply to it only when all of its
  dimensions are at least 32, so token generation never uses it.

## Next step
Run the scripts, then read the results together (steps 3–8). Conditions, as
for every benchmark here: AC power, Low Power Mode off, other apps closed
(Docker Desktop too); the scripts keep the machine awake themselves, and
refuse to start while the screen saver is running. From the
repository root, with `LLAMA_CPP_DIR=~/src/llama.cpp MODELS_DIR=~/models`:

1. `uv run python projects/profile-llama-cpp/scripts/run_baseline.py`: both
   models, 1, 6 and 12 threads, 3 rounds (about 70 minutes; the 1-thread
   tests take most of it). Then `report_baseline.py` on its folder.
2. `run_baseline.py --models q8_0 --threads 1,2,3,4,6,8,10,12 --skip-ceilings
   --label scaling` (about 90 minutes), then `report_baseline.py` on its
   folder with `--ceilings` set to run 1's folder.
3. `run_baseline.py --build build-accelerate --threads 6 --skip-ceilings
   --label accelerate` (about 15 minutes).
4. `run_profile.py --model q8_0`, then `run_profile.py --model q4_k_m` (a few
   minutes each). Each writes summaries next to its `.trace` files, which open
   in Instruments.
5. `run_kernel_compare.py` (about 15 minutes).

All the scripts are in `projects/profile-llama-cpp/scripts/` and write to
`projects/profile-llama-cpp/results/`. The durations are estimates from the
smoke tests' speeds. Steps 1, 4 and 5 matter most if time is short; step 2
can run another day.

Plan: 1. settle the open questions; 2. setup script; 3. baseline timings;
4. roofline; 5. `xctrace` profiles of each phase; 6. ggml's Q8_0 `MUL_MAT`
against Quantized matmul's kernels; 7. thread scaling; 8. steps 3–5 for
Q4_K_M; 9. Linux `perf` (future work, below); 10. write-up and close.

## Future work
Everything in this project so far runs on the development MacBook Pro. Not
started, and not needed for steps 1–8:
- Step 9: repeat the key measurements on Linux x86 hardware with `perf`,
  building llama.cpp both with OpenMP and without (see Decisions). It waits
  for the Mac results, which show which measurements are worth repeating.
- On a CPU with AVX-512 or VNNI, the same measurements would also exercise
  ggml's kernels for those instruction sets.

## Open questions
- For the future work: which Linux x86 hardware, and when.

## Decisions
Record each as: date, decision, reason.

- 2026-11-14: Profile Qwen2.5-1.5B, in Q8_0 first and then Q4_K_M. Reason:
  it uses the mainstream Llama-style path (RMSNorm, SwiGLU, rotary embeddings,
  grouped-query attention) that llama.cpp is tuned for and that the inference
  runtime project will implement, so its hotspots are relevant upstream. It is
  Apache 2.0, and Qwen publishes official GGUFs (for the Instruct model, whose
  weight shapes are the same), so no conversion through PyTorch is needed.
  Q8_0 continues from Quantized matmul's format; Q4_K_M is the quantization
  people actually run. GPT-2 was rejected: its architecture is atypical, its
  small matrices and large lm_head distort the profile, and converting it
  needs PyTorch.
- 2026-11-14: Compare with Quantized matmul by timing ggml's Q8_0 `MUL_MAT`
  on GPT-2's shapes with `test-backend-ops` in perf mode, single-threaded.
  Reason: it gives the kernel-to-kernel comparison on identical shapes without
  profiling GPT-2 as a model. ggml stores Q8_0 scales in fp16, so its blocks
  are 34 bytes against Quantized matmul's 36.
- 2026-11-14: Develop on this machine. Reason: it has every tool needed to
  build, profile and test llama.cpp and to prepare a contribution (CMake,
  Ninja, Apple clang 17, `xctrace`, git, `gh`). It verifies the AVX2 code
  paths, so a contribution here is strongest on AVX2 or shared code. Repeating
  the key measurements on Linux hardware with `perf` is future work.
- 2026-11-14: Profile a 512-token prompt and 128 generated tokens. Reason:
  they are `llama-bench`'s defaults, so the results compare directly with the
  numbers llama.cpp's maintainers and users publish.
- 2026-11-14: Keep llama.cpp and the models outside the repository, at paths
  from `LLAMA_CPP_DIR` and `MODELS_DIR`: one blobless clone with a git
  worktree per pinned release beside it, and the GGUFs at a pinned revision
  checked by SHA-256. Reason: the same clone can carry a fork and branches for
  the upstream contribution project, the models can be reused by the
  inference runtime project as a reference, and nothing large is nested in the
  portfolio repository. The pins keep it reproducible anywhere.
- 2026-11-14: Pin llama.cpp release b12377 (commit 32bdc3a, 2026-11-14) and
  Qwen/Qwen2.5-1.5B-Instruct-GGUF revision 91cad51. Reason: the latest release
  when the project started.
- 2026-11-14: Build release binaries with debug information (`-g`, which
  doesn't change the generated code) and `GGML_NATIVE=ON`, in two
  configurations. `build-cpu`, for the main profiles: no Metal, so nothing
  runs on a GPU, even through llama.cpp's offloading of large-batch operations
  that `-ngl 0` alone doesn't prevent; and no BLAS, so every matrix multiply
  runs in ggml's own kernels, which a contribution could change, and the code
  matches a default Linux build. `build-accelerate`, macOS only: Accelerate as
  BLAS, macOS builds' default, to measure how much of prompt processing's
  speed comes from it. b12377 enables both Metal and BLAS by default on Apple.
- 2026-11-14: Also build `build-cpu` with `GGML_ACCELERATE=OFF`. Reason: the
  first setup run showed that this separate option, on by default and acting
  only on macOS, hands fp32 element-wise operations (adds, multiplies,
  scaling, the sums and maxima of softmax, the mean and variance of norms) to
  Accelerate's vDSP instead of ggml's SIMD code, against the two reasons
  above. `build-accelerate` keeps it, as macOS's full default.
- 2026-11-15: For the future Linux runs (step 9), build and measure both with
  OpenMP (GCC's default) and with `GGML_OPENMP=OFF`; this Mac keeps ggml's own
  thread pool, since
  Apple clang has no OpenMP. Reason: ggml synchronizes its threads with a
  barrier after every graph node, several hundred per generated token, through
  `#pragma omp barrier` with OpenMP and an atomic spin barrier in its own
  pool. The OpenMP build gives the numbers Linux users see; the thread-pool
  build runs the same code as the Mac, so differences can be attributed to the
  machine; the gap between the two measures the runtime itself. Costs a second
  build and a few more runs on Linux.
- 2026-11-15: Make step 9 (Linux `perf`) future work, after the Mac results.
  Reason: they show which measurements are worth repeating, which keeps the
  Linux runs focused.
- 2026-11-15: Time ggml's `MUL_MAT` through `test-backend-ops --test-file`,
  with GPT-2's shapes written by `run_kernel_compare.py`, and single-threaded
  through `TEST_ARG_HARDWARE_CONCURRENCY=2` (it runs half that many threads).
  Reason: it needs no edit to llama.cpp; its built-in cases don't include
  GPT-2's shapes, and it has no thread option.
- 2026-11-15: Measure the multi-core ceilings with new 6- and 12-thread
  variants of Quantized matmul's roofline benchmarks, before and after the
  rounds, taking the higher reading. Reason: llama.cpp runs multi-threaded,
  so one core's ceilings don't apply; a smoke test measured single-thread
  bandwidth at a third of its usual value right after a build, which one
  reading would have hidden. The FMA peak is reported as items per second,
  since Google Benchmark's rate counter overcounts across threads.
- 2026-11-15: Keep the `.trace` recordings out of git and commit their
  summaries. Reason: a recording runs to hundreds of MB.
- 2026-11-15: Share the run scripts' generic helpers with Quantized matmul
  through `common/benchkit`, a uv workspace package; `runkit.py` keeps only
  what's specific to llama.cpp. Reason: the first copy of them had already
  drifted from the original (its Low Power Mode parser accepted only "1").
