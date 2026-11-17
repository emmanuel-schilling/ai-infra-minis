# ggml's matrix multiply against Quantized matmul's kernels

Quantized matmul run: `2026-11-14_i9-8950hk`. Time ratio above 1: ggml is slower.

| Case | ggml (µs) | ggml GFLOP/s | Kernel | Kernel (µs) | Time ratio |
|---|---:|---:|---|---:|---:|
| q8_0/attn_qkv/tokens:1 | 84.4 | 41.9 | q8/avx2_v2 | 151.5 | 0.56 |
| q8_0/attn_qkv/tokens:128 | 6704.5 | 67.6 | q8/avx2_v2 | 6063.0 | 1.11 |
| q8_0/attn_qkv/tokens:1024 | 56523.7 | 64.1 | q8/avx2_v2 | 48021.6 | 1.18 |
| q8_0/attn_out/tokens:1 | 29.5 | 39.9 | q8/avx2_v2 | 52.0 | 0.57 |
| q8_0/attn_out/tokens:128 | 2420.6 | 62.4 | q8/avx2_v2 | 2096.7 | 1.15 |
| q8_0/attn_out/tokens:1024 | 19444.3 | 62.1 | q8/avx2_v2 | 16788.4 | 1.16 |
| q8_0/mlp_up/tokens:1 | 115.8 | 40.7 | q8/avx2_v2 | 201.9 | 0.57 |
| q8_0/mlp_up/tokens:128 | 9471.4 | 63.8 | q8/avx2_v2 | 8040.2 | 1.18 |
| q8_0/mlp_up/tokens:1024 | 78041.2 | 61.9 | q8/avx2_v2 | 63238.5 | 1.23 |
| q8_0/mlp_down/tokens:1 | 101.0 | 46.7 | q8/avx2_v2 | 201.5 | 0.50 |
| q8_0/mlp_down/tokens:128 | 9277.3 | 65.1 | q8/avx2_v2 | 8220.3 | 1.13 |
| q8_0/mlp_down/tokens:1024 | 72982.7 | 66.2 | q8/avx2_v2 | 65911.3 | 1.11 |
| q8_0/lm_head/tokens:1 | 3644.8 | 21.2 | q8/avx2_v2 | 3290.2 | 1.11 |
| q8_0/lm_head/tokens:128 | 155890.0 | 63.4 | q8/avx2_v2 | 129206.8 | 1.21 |
| f32/attn_qkv/tokens:1 | 150.0 | 23.6 | fp32/avx2_v4 | 398.3 | 0.38 |
| f32/attn_qkv/tokens:128 | 4156.7 | 109.0 | fp32/avx2_v4 | 4236.6 | 0.98 |
| f32/attn_qkv/tokens:1024 | 35094.7 | 103.3 | fp32/avx2_v4 | 32486.8 | 1.08 |
| f32/attn_out/tokens:1 | 44.2 | 26.7 | fp32/avx2_v4 | 132.3 | 0.33 |
| f32/attn_out/tokens:128 | 1361.0 | 110.9 | fp32/avx2_v4 | 1408.6 | 0.97 |
| f32/attn_out/tokens:1024 | 11197.8 | 107.9 | fp32/avx2_v4 | 10814.6 | 1.04 |
| f32/mlp_up/tokens:1 | 284.7 | 16.6 | fp32/avx2_v4 | 531.8 | 0.54 |
| f32/mlp_up/tokens:128 | 5783.9 | 104.4 | fp32/avx2_v4 | 5774.7 | 1.00 |
| f32/mlp_up/tokens:1024 | 47749.3 | 101.2 | fp32/avx2_v4 | 42900.5 | 1.11 |
| f32/mlp_down/tokens:1 | 288.5 | 16.4 | fp32/avx2_v4 | 549.9 | 0.52 |
| f32/mlp_down/tokens:128 | 7734.9 | 78.1 | fp32/avx2_v4 | 8259.0 | 0.94 |
| f32/mlp_down/tokens:1024 | 63570.8 | 76.0 | fp32/avx2_v4 | 65937.0 | 0.96 |
| f32/lm_head/tokens:1 | 9242.2 | 8.4 | fp32/avx2_v4 | 8864.3 | 1.04 |
| f32/lm_head/tokens:128 | 332035.9 | 29.8 | fp32/avx2_v4 | 98661.5 | 3.37 |
