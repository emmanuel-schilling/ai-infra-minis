# Profile llama.cpp: status

## Current state
Started 2026-11-13. `scripts/setup_llama_cpp.py` sets up llama.cpp release
b12377 and the two Qwen2.5-1.5B models outside the repository (see Decisions),
builds them and runs a smoke test. It ran on 2026-11-14 into ~/src/llama.cpp
and ~/models: both builds succeeded, and both models generate coherent text.

Session A ran on 2026-11-16: `results/2026-11-16_i9-8950hk/` (baseline) and
`results/2026-11-17_i9-8950hk_accelerate/` (the folder dates are UTC), with
`report.md` in each. Verified complete and repeatable: rounds within 0.2%
(baseline) and 0.5% (Accelerate) of round 1, no slow rounds, no screen saver.
Every multi-threaded prompt-processing test (6 and 12 threads, both builds)
ran into a macOS CPU speed limit of 66–70% during the test, never before it:
sustained all-core compute exceeds the laptop's power and thermal limits
within seconds, so a longer cooldown wouldn't change it. Those tests still
agree across rounds within 0.4–1.5%, with 4–10% spread inside each test as
the limit sets in. Two tests are noisier: Q4_K_M token generation at 12
threads (11.6% across rounds) and Q8_0 token generation at 1 thread (round 1
8.8% faster). The ceilings agree before and after the run except the 6-thread
FMA peak (639 against 743 GFLOP/s); the report uses the higher, and the peak
is a short burst while the throttled tests are sustained, so prompt
processing's shares of it understate what the machine sustains. Sessions B
and C are still to run.

The scripts for steps 3–8 are written; session A's runs took about 45 and
12 minutes, about half the estimates below.
- `run_baseline.py` (steps 3, 7, 8): llama-bench rounds with cooldowns, and
  the ceilings from Quantized matmul's `qmm_bench`, measured before and after
  the rounds. `qmm_bench` gained 6- and 12-thread variants of its bandwidth
  and FMA-peak benchmarks for this.
- `report_baseline.py` (steps 4, 7): the roofline shares and thread scaling.
- `run_profile.py` (step 5): `xctrace` Time Profiler per phase, summarized by
  ggml operation, leaf function, stack function and thread. Waiting is split
  by what the stack proves: at a barrier, idle between graphs, or spinning in
  a worker's loop, where ggml inlines both the barrier and the polling for
  the next graph; those spins are listed by the source line they return to,
  resolved with `atos` (at b12377, ggml-cpu.c:3281–3282 is inside a graph).
  `--template` also records custom CPU Counters templates, for opening in
  Instruments.
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
for every benchmark here: AC power with the battery already charged (charging
adds heat), Low Power Mode off, other apps closed (Docker Desktop too), and a
machine that has been idle for a while; the scripts keep the machine awake
themselves, and refuse to start while the screen saver is running. A smoke
test with 1 s cooldowns, after hours of builds and with the battery
charging, saw macOS limit the CPU to 20% for several minutes; the per-test
records flagged it. From the
repository root, with `LLAMA_CPP_DIR=~/src/llama.cpp MODELS_DIR=~/models`:

The sessions are independent and can run in any order. Within session A,
Accelerate follows the baseline so the two are compared under the same
conditions.

Session A (about 2 hours):
1. `uv run python projects/profile-llama-cpp/scripts/run_baseline.py`: both
   models, 1, 6 and 12 threads, 3 rounds of 12 tests (about 100 minutes; the
   1-thread tests take most of it). Then `report_baseline.py` on its folder.
2. Right after it, `run_baseline.py --build build-accelerate --threads 6
   --skip-ceilings --label accelerate` (about 20 minutes), then
   `report_baseline.py` on its folder with `--ceilings` set to run 1's folder.

Session B (about 30 minutes):
3. `run_profile.py --model q8_0`, then `run_profile.py --model q4_k_m` (about
   6 minutes each). Each writes summaries next to its `.trace` files, which
   open in Instruments.
4. `run_kernel_compare.py` (about 16 minutes).

Session C (about 2 hours):
5. `run_baseline.py --models q8_0 --threads 1,2,3,4,6,8,10,12 --label
   scaling` (about 120 minutes, with its own ceilings), then
   `report_baseline.py` on its folder.

Later, once the two counter templates exist in Instruments (not yet built as
of 2026-11-16): `run_profile.py --model <m> --template "LLM Ports" --label
ports` and `--template "LLM Memory" --label memory` for each model (about
4 minutes each, 16 in all).

Between runs, check each `metadata.json`. A run is valid when
`slow_rounds`, `started_throttled_rounds` (a test or round that started
under a CPU speed limit) and `screen_saver_rounds` are empty. An invalid run
keeps its folder, with an `INVALID.md` giving the reason, and is rerun, with
a longer `--cooldown` if it started hot. `throttled_rounds` also lists
limits that set in during a test; those don't invalidate a run, but the
write-up states which tests ran under a limit, and how much. (Session A's
runs predate the `started_throttled_rounds` field; their per-test thermal
records show no test started under a limit.)

All the scripts are in `projects/profile-llama-cpp/scripts/` and write to
`projects/profile-llama-cpp/results/`. The durations are estimates from the
smoke tests' speeds, about ±30%.

Plan: 1. settle the open questions; 2. setup script; 3. baseline timings;
4. roofline; 5. `xctrace` profiles of each phase; 6. ggml's Q8_0 `MUL_MAT`
against Quantized matmul's kernels; 7. thread scaling; 8. steps 3–5 for
Q4_K_M; 9. Linux `perf` (future work, below, outside the project's
definition of done); 10. write-up and close.

## Write-up outline
Planned sections of RESULTS.md (step 10), filled in once the results are in.
You write the conclusions; Claude checks the figures.
1. Setup: llama.cpp b12377, the models, the builds and their flags, the
   machine and its measured ceilings.
2. Baseline: prompt processing and token generation for both models at 1, 6
   and 12 threads, on the roofline.
3. Prompt processing hotspots, per model: time by ggml operation and
   function, which kernels run (tinyBLAS for Q8_0, the repacked Q4_K), and
   the hardware counters behind them.
4. Token generation hotspots, per model, the same way, including time spent
   at the barrier between operations.
5. Thread scaling: where each phase stops scaling, and why.
6. ggml's matrix multiply against Quantized matmul's kernels on GPT-2's
   shapes, and what explains the difference.
7. Accelerate: how much of prompt processing's speed it adds on macOS.
8. Possible contribution targets, ranked by evidence and by what can be
   verified on AVX2.
9. Open questions and future work (the Linux `perf` runs).

## Future work
Everything in this project runs on the development MacBook Pro. The project
is done when the write-up (step 10), from measurements on this Mac,
identifies the hotspots per phase and the possible contribution targets. Not
started, and not part of that:
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
- 2026-11-15: Stage the baseline per test: one llama-bench run per model,
  phase and thread count, in a seeded shuffled order each round, each after
  a 30 s cooldown, with the thermal state recorded before and after every
  test. The profile and kernel-comparison scripts also record the thermal
  state, and the kernel comparison checks its rounds against round 1.
  Reason: llama-bench ran each model's tests back to back in a fixed order,
  so the multi-threaded tests always started on a warmer CPU, a bias the
  round checks can't see because it repeats in every round; and those two
  scripts checked nothing. Starting every test cool measures burst rather
  than sustained performance, which suits comparing kernels, models and
  thread counts; a warm steady state would need its own design. Costs about
  45 minutes over the whole set.
- 2026-11-15: Redefine done: the write-up, from measurements on this Mac,
  identifies the hotspots per phase and the possible contribution targets.
  Repeating the key measurements under Linux `perf` (step 9) is future work,
  outside that definition. Reason: the project's questions can be answered on
  this machine, and other hardware is not available on a set timeline.
- 2026-11-16: Session A's runs (`2026-11-16_i9-8950hk`,
  `2026-11-17_i9-8950hk_accelerate`) are valid, and the validity rule is
  refined: a run is invalid if a round is slow, a test starts under a CPU
  speed limit, or the screen saver runs; a limit that sets in during a test
  is recorded and reported, not disqualifying. The scripts now record
  `started_throttled_rounds` (and the tests or phases) for this. Reason:
  every multi-threaded prompt-processing test reached a 66–70% speed limit
  during the test, never before it, in every round: sustained all-core
  compute exceeds this laptop's power and thermal limits within seconds,
  which is its real behaviour, and those tests still agreed across rounds
  within 0.4–1.5%. The old rule, written for tests that start hot, would
  have rejected every possible multi-threaded run on this machine. The
  write-up states that multi-threaded prompt processing was measured under a
  66–70% limit, and that its shares of the short-burst FMA peak understate
  what the machine sustains.
