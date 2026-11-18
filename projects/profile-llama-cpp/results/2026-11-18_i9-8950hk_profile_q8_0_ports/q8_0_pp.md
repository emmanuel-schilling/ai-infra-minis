# q8_0 pp, 6 threads

Samples in ggml's graphs: 115313 ms of thread time over 19292 ms, at 3.17 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 85 ms. Outside the graph computation, left out: 2795 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `UOPS_DISPATCHED_PORT.PORT_0` | 247,367,558,990 | 0.677 |  |
| `UOPS_DISPATCHED_PORT.PORT_1` | 245,688,662,302 | 0.672 |  |
| `UOPS_DISPATCHED_PORT.PORT_5` | 138,144,464,634 | 0.378 |  |
| `CORE_ACTIVE_CYCLE` | 365,350,865,392 |  |  |

## By ggml operation

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 90.9% | 0.718 | 0.713 | 0.400 | `mul_mat` |
| 3.4% | 0.171 | 0.170 | 0.104 | `(spinning in a worker's loop: barrier or next graph)` |
| 2.4% | 0.427 | 0.427 | 0.279 | `flash_attn_ext_tiled` |
| 1.4% | 0.138 | 0.136 | 0.067 | `swiglu_f32` |
| 0.7% | 0.436 | 0.433 | 0.263 | `(scheduler)` |
| 0.7% | 0.177 | 0.175 | 0.107 | `(barrier)` |
| 0.2% | 0.387 | 0.379 | 0.208 | `rms_norm_f32` |
| 0.2% | 0.193 | 0.184 | 0.066 | `mul_mat_one_chunk` |
| 0.0% | 0.489 | 0.483 | 0.273 | `rope_flt` |
| 0.0% | 0.084 | 0.033 | 0.046 | `get_rows_q` |
| 0.0% | 0.141 | 0.144 | 0.092 | `set_rows_impl` |
| 0.0% | 0.433 | 0.430 | 0.289 | `flash_attn_ext_f16` |
| 0.0% | 0.000 | 0.000 | 0.000 | `add_non_quantized` |

## By leaf function

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 45.0% | 0.726 | 0.721 | 0.404 | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` |
| 23.2% | 0.719 | 0.714 | 0.400 | `float vector[8] (anonymous namespace)::madd<float vector[8], float vector[8]>` |
| 21.4% | 0.722 | 0.717 | 0.401 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::updot` |
| 4.1% | 0.172 | 0.170 | 0.104 | `ggml_thread_cpu_relax` |
| 1.2% | 0.426 | 0.425 | 0.278 | `void simd_gemm_ukernel<6, 2>` |
| 0.9% | 0.211 | 0.209 | 0.129 | `quantize_row_q8_0` |
| 0.8% | 0.427 | 0.426 | 0.278 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 0.7% | 0.146 | 0.144 | 0.072 | `ggml_v_silu` |
| 0.5% | 0.123 | 0.122 | 0.059 | `ggml_vec_swiglu_f32` |
| 0.4% | 0.725 | 0.719 | 0.406 | `(anonymous namespace)::hsum` |
| 0.2% | 0.228 | 0.227 | 0.132 | `ggml_v_expf` |
| 0.2% | 0.387 | 0.379 | 0.208 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.2% | 0.418 | 0.418 | 0.275 | `void simd_gemm_ukernel<1, 2>` |
| 0.2% | 0.434 | 0.432 | 0.282 | `ggml_vec_max_f32` |
| 0.2% | 0.422 | 0.417 | 0.233 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.1% | 0.196 | 0.188 | 0.068 | `ggml_vec_dot_q8_0_q8_0` |
| 0.1% | 0.430 | 0.428 | 0.281 | `ggml_vec_soft_max_f32` |
| 0.1% | 0.187 | 0.179 | 0.063 | `mul_sum_i8_pairs_float` |
| 0.1% | 0.383 | 0.381 | 0.243 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 0.501 | 0.495 | 0.283 | `void rotate_pairs<float>` |
| 0.1% | 0.284 | 0.281 | 0.155 | `op_add` |
| 0.1% | 0.471 | 0.469 | 0.294 | `_platform_bzero$VARIANT$Haswell` |
| 0.0% | 0.535 | 0.531 | 0.303 | `0x7ff815c325e0` |
| 0.0% | 0.432 | 0.431 | 0.281 | `ggml_cpu_fp16_to_fp32` |
| 0.0% | 0.443 | 0.441 | 0.289 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.439 | 0.438 | 0.283 | `ggml_vec_scale_f32` |
| 0.0% | 0.401 | 0.402 | 0.264 | `simd_gemm` |
| 0.0% | 0.489 | 0.483 | 0.273 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 0.460 | 0.458 | 0.293 | `ggml_vec_add_f32` |
| 0.0% | 0.630 | 0.623 | 0.354 | `rope_yarn` |

## By thread

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.677 | 0.672 | 0.378 | `llama-bench 0x4e1605 (llama-bench, pid: 46756)` |
| 16.7% | 0.676 | 0.672 | 0.378 | `llama-bench 0x4e1606 (llama-bench, pid: 46756)` |
| 16.7% | 0.677 | 0.673 | 0.378 | `Main Thread 0x4e15dc (llama-bench, pid: 46756)` |
| 16.7% | 0.677 | 0.673 | 0.378 | `llama-bench 0x4e1607 (llama-bench, pid: 46756)` |
| 16.6% | 0.677 | 0.672 | 0.378 | `llama-bench 0x4e1608 (llama-bench, pid: 46756)` |
| 16.6% | 0.678 | 0.673 | 0.378 | `llama-bench 0x4e1609 (llama-bench, pid: 46756)` |
