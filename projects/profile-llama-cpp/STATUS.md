# Profile llama.cpp: status

## Current state
Started 2026-11-13. Nothing is set up yet: no llama.cpp checkout, model,
scripts or profiles.

## Next step
Settle the open questions below, then script the setup: fetch llama.cpp at a
pinned commit and the chosen model, build them with recorded flags, and run a
first profile of prompt processing and of token generation under `xctrace`.

## Open questions
- Model and quantization to profile. A Q8_0 model would exercise the same
  format as Quantized matmul. GPT-2 would connect directly to its results,
  while Llama-family models are what llama.cpp is mostly tuned and profiled on.
- Which prompt and generation lengths to profile.
- Where the llama.cpp checkout and the model live, and how they are fetched
  and built: for example a script that fetches a pinned commit into a
  git-ignored folder, as Quantized matmul does for GPT-2's weights.
- Whether to time ggml's own Q8_0 matrix multiply on GPT-2's shapes with
  llama.cpp's operator benchmarks, single-threaded, for a direct comparison
  with Quantized matmul's kernels. ggml stores Q8_0 scales in fp16, so its
  blocks are 34 bytes against Quantized matmul's 36.
- Which rented Linux x86 machine to repeat the key measurements on with
  `perf`, and when (a cost decision).

## Decisions
Record each as: date, decision, reason.
