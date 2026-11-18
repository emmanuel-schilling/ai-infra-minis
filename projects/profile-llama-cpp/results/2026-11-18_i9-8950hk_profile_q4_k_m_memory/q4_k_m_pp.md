# q4_k_m pp, 6 threads

Samples in ggml's graphs: 57181 ms of thread time over 9549 ms, at 3.45 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 85 ms. Outside the graph computation, left out: 1580 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `CORE_ACTIVE_CYCLE` | 196,989,252,012 |  |  |
| `L2_RQSTS.MISS` | 3,178,725,558 | 0.016 | 21.3 |
| `LONGEST_LAT_CACHE.MISS` | 582,091,992 | 0.003 | 3.9 |
| `RESOURCE_STALLS.ANY` | 46,124,875,340 | 0.234 |  |

## By ggml operation

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 71.7% | 0.015 | 0.002 | 0.199 | `mul_mat (repacked)` |
| 14.7% | 0.018 | 0.004 | 0.275 | `mul_mat` |
| 4.5% | 0.030 | 0.002 | 0.319 | `flash_attn_ext_tiled` |
| 2.8% | 0.015 | 0.019 | 0.717 | `swiglu_f32` |
| 2.7% | 0.010 | 0.002 | 0.143 | `(spinning in a worker's loop: barrier or next graph)` |
| 2.3% | 0.019 | 0.007 | 0.356 | `(scheduler)` |
| 0.5% | 0.011 | 0.001 | 0.138 | `(barrier)` |
| 0.4% | 0.016 | 0.008 | 0.331 | `rms_norm_f32` |
| 0.3% | 0.024 | 0.028 | 0.693 | `mul_mat_one_chunk` |
| 0.0% | 0.016 | 0.002 | 0.204 | `rope_flt` |
| 0.0% | 0.028 | 0.002 | 0.232 | `set_rows_impl` |
| 0.0% | 0.010 | 0.006 | 0.170 | `get_rows_q` |

## By leaf function

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 71.6% | 0.015 | 0.002 | 0.199 | `ggml_gemm_q4_K_8x8_q8_K` |
| 11.1% | 0.018 | 0.003 | 0.257 | `void tiled_run_microtile_avx2<16, false, 32, 1, true>` |
| 3.2% | 0.010 | 0.001 | 0.141 | `ggml_thread_cpu_relax` |
| 2.1% | 0.030 | 0.002 | 0.320 | `void simd_gemm_ukernel<6, 2>` |
| 1.6% | 0.030 | 0.002 | 0.318 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 1.2% | 0.015 | 0.019 | 0.707 | `ggml_v_silu` |
| 1.2% | 0.015 | 0.019 | 0.727 | `ggml_vec_swiglu_f32` |
| 1.0% | 0.018 | 0.017 | 0.492 | `quantize_row_q8_K_ref` |
| 0.9% | 0.017 | 0.003 | 0.263 | `tiled_unpack_src0` |
| 0.8% | 0.017 | 0.015 | 0.509 | `ggml_quantize_mat_q8_K_4x8` |
| 0.5% | 0.018 | 0.003 | 0.259 | `tiled_unpack_src1_q8_K` |
| 0.5% | 0.019 | 0.014 | 0.614 | `ggml_v_expf` |
| 0.4% | 0.016 | 0.008 | 0.331 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.4% | 0.014 | 0.003 | 0.218 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.4% | 0.018 | 0.003 | 0.263 | `void tiled_repack_src0<16>` |
| 0.3% | 0.029 | 0.002 | 0.321 | `void simd_gemm_ukernel<1, 2>` |
| 0.3% | 0.030 | 0.002 | 0.319 | `ggml_vec_max_f32` |
| 0.3% | 0.024 | 0.028 | 0.693 | `ggml_vec_dot_q6_K_q8_K` |
| 0.2% | 0.025 | 0.006 | 0.314 | `_platform_bzero$VARIANT$Haswell` |
| 0.2% | 0.030 | 0.002 | 0.317 | `ggml_vec_soft_max_f32` |
| 0.2% | 0.014 | 0.004 | 0.225 | `op_add` |
| 0.2% | 0.013 | 0.003 | 0.212 | `void rotate_pairs<float>` |
| 0.1% | 0.020 | 0.018 | 0.480 | `ggml_barrier` |
| 0.1% | 0.017 | 0.003 | 0.253 | `void tiled_unpk_or<0, 4, 3>` |
| 0.1% | 0.029 | 0.002 | 0.320 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 0.018 | 0.003 | 0.264 | `tiled_repack_src1` |
| 0.1% | 0.017 | 0.004 | 0.267 | `tiled_unpk_nib4` |
| 0.1% | 0.030 | 0.002 | 0.319 | `_platform_memmove$VARIANT$Haswell` |
| 0.1% | 0.019 | 0.002 | 0.194 | `0x7ff815c325e0` |
| 0.1% | 0.018 | 0.003 | 0.270 | `tiled_byte_sum_16` |

## By thread

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.016 | 0.003 | 0.234 | `llama-bench 0x4e2ba9 (llama-bench, pid: 46993)` |
| 16.7% | 0.016 | 0.003 | 0.236 | `llama-bench 0x4e2ba8 (llama-bench, pid: 46993)` |
| 16.7% | 0.016 | 0.003 | 0.234 | `llama-bench 0x4e2baa (llama-bench, pid: 46993)` |
| 16.6% | 0.016 | 0.003 | 0.233 | `llama-bench 0x4e2bac (llama-bench, pid: 46993)` |
| 16.6% | 0.016 | 0.003 | 0.234 | `llama-bench 0x4e2bab (llama-bench, pid: 46993)` |
| 16.6% | 0.016 | 0.003 | 0.233 | `Main Thread 0x4e2b90 (llama-bench, pid: 46993)` |
