# q8_0 pp, 6 threads

Samples in ggml's graphs: 116743 ms of thread time over 19549 ms, at 3.08 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 75 ms. Outside the graph computation, left out: 2556 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `CORE_ACTIVE_CYCLE` | 359,065,148,414 |  |  |
| `L2_RQSTS.MISS` | 13,584,005,326 | 0.038 | 44.5 |
| `LONGEST_LAT_CACHE.MISS` | 691,866,556 | 0.002 | 2.3 |
| `RESOURCE_STALLS.ANY` | 79,242,346,801 | 0.221 |  |

## By ggml operation

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 91.4% | 0.040 | 0.002 | 0.216 | `mul_mat` |
| 3.1% | 0.011 | 0.001 | 0.074 | `(spinning in a worker's loop: barrier or next graph)` |
| 2.4% | 0.030 | 0.002 | 0.315 | `flash_attn_ext_tiled` |
| 1.2% | 0.018 | 0.021 | 0.706 | `swiglu_f32` |
| 0.7% | 0.012 | 0.001 | 0.094 | `(barrier)` |
| 0.7% | 0.030 | 0.002 | 0.260 | `(scheduler)` |
| 0.3% | 0.026 | 0.008 | 0.268 | `rms_norm_f32` |
| 0.2% | 0.025 | 0.029 | 0.662 | `mul_mat_one_chunk` |
| 0.0% | 0.037 | 0.001 | 0.217 | `rope_flt` |
| 0.0% | 0.039 | 0.001 | 0.235 | `set_rows_impl` |
| 0.0% | 0.001 | 0.001 | 0.011 | `get_rows_q` |

## By leaf function

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 45.5% | 0.040 | 0.001 | 0.214 | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` |
| 23.4% | 0.040 | 0.001 | 0.212 | `float vector[8] (anonymous namespace)::madd<float vector[8], float vector[8]>` |
| 21.2% | 0.040 | 0.001 | 0.212 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::updot` |
| 3.8% | 0.011 | 0.001 | 0.077 | `ggml_thread_cpu_relax` |
| 1.2% | 0.030 | 0.002 | 0.316 | `void simd_gemm_ukernel<6, 2>` |
| 0.9% | 0.023 | 0.018 | 0.572 | `quantize_row_q8_0` |
| 0.8% | 0.030 | 0.002 | 0.315 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 0.6% | 0.018 | 0.021 | 0.701 | `ggml_v_silu` |
| 0.5% | 0.018 | 0.022 | 0.715 | `ggml_vec_swiglu_f32` |
| 0.4% | 0.039 | 0.002 | 0.218 | `(anonymous namespace)::hsum` |
| 0.3% | 0.026 | 0.008 | 0.268 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.2% | 0.022 | 0.017 | 0.605 | `ggml_v_expf` |
| 0.2% | 0.030 | 0.002 | 0.314 | `void simd_gemm_ukernel<1, 2>` |
| 0.2% | 0.030 | 0.002 | 0.319 | `ggml_vec_max_f32` |
| 0.1% | 0.027 | 0.003 | 0.182 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.1% | 0.030 | 0.001 | 0.317 | `ggml_vec_soft_max_f32` |
| 0.1% | 0.025 | 0.029 | 0.664 | `ggml_vec_dot_q8_0_q8_0` |
| 0.1% | 0.026 | 0.029 | 0.666 | `mul_sum_i8_pairs_float` |
| 0.1% | 0.029 | 0.007 | 0.391 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 0.033 | 0.003 | 0.221 | `void rotate_pairs<float>` |
| 0.1% | 0.030 | 0.002 | 0.314 | `ggml_cpu_fp16_to_fp32` |
| 0.1% | 0.033 | 0.002 | 0.289 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.036 | 0.002 | 0.211 | `0x7ff815c325e0` |
| 0.0% | 0.031 | 0.002 | 0.298 | `_platform_bzero$VARIANT$Haswell` |
| 0.0% | 0.022 | 0.003 | 0.175 | `op_add` |
| 0.0% | 0.030 | 0.002 | 0.322 | `ggml_vec_scale_f32` |
| 0.0% | 0.037 | 0.001 | 0.217 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 0.029 | 0.002 | 0.323 | `simd_gemm` |
| 0.0% | 0.019 | 0.007 | 0.266 | `ggml_compute_forward_mul_mat` |
| 0.0% | 0.025 | 0.031 | 0.664 | `sum_i16_pairs_float` |

## By thread

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.038 | 0.002 | 0.220 | `llama-bench 0x4e1d5b (llama-bench, pid: 46839)` |
| 16.7% | 0.038 | 0.002 | 0.225 | `llama-bench 0x4e1d5c (llama-bench, pid: 46839)` |
| 16.7% | 0.038 | 0.002 | 0.220 | `llama-bench 0x4e1d5d (llama-bench, pid: 46839)` |
| 16.7% | 0.038 | 0.002 | 0.220 | `Main Thread 0x4e1d4d (llama-bench, pid: 46839)` |
| 16.6% | 0.038 | 0.002 | 0.220 | `llama-bench 0x4e1d5e (llama-bench, pid: 46839)` |
| 16.6% | 0.038 | 0.002 | 0.219 | `llama-bench 0x4e1d5a (llama-bench, pid: 46839)` |
