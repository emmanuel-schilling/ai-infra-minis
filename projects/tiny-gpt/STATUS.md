# Tiny GPT: status

## Current state
Started 2026-11-15, alongside Profile llama.cpp, which waits for its benchmark
runs and write-up. Nothing is built yet: no loader, fixtures or code.

## Next step
Settle the open questions below (step 2), then generate the reference
fixtures in Docker (step 3).

Plan: 1. start the project; 2. settle the open questions; 3. reference
fixtures; 4. weight loading; 5. the building blocks (embeddings, layer norm,
GELU, softmax, the MLP); 6. causal multi-head self-attention; 7. the forward
pass, tested layer by layer; 8. generation; 9. timing prefill and decode
separately, on the roofline; 10. write-up and close.

Who does what: the building blocks, attention, the forward pass and
generation (steps 5–8) are written by the project's owner with Claude
explaining and reviewing; Claude writes the loading, the fixtures, all tests
and the timing scripts (steps 3, 4, 9); the write-up is the owner's, with
Claude's help.

## Open questions
- Weight format to load, and how to convert it.
- Decoding for the reference match (greedy decoding keeps it deterministic).
- The fixed prompts and generation lengths.
- The tolerances for per-layer outputs and logits, and the reference's
  precision.
- Which fixtures the tests in CI can use, given that GPT-2's weights are not
  in the repository.

## Decisions
Record each as: date, decision, reason.

- 2026-11-15: Start now, alongside Profile llama.cpp. Reason: that project
  waits on benchmark runs and a write-up that need a free evening, and Tiny
  GPT doesn't depend on them.
