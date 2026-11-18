# q4_k_m pp, 6 threads

Samples in ggml's graphs: 59500 ms of thread time over 9937 ms, at 3.39 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 70 ms. Outside the graph computation, left out: 1817 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `UOPS_DISPATCHED_PORT.PORT_0` | 103,008,161,174 | 0.511 |  |
| `UOPS_DISPATCHED_PORT.PORT_1` | 104,754,942,262 | 0.520 |  |
| `UOPS_DISPATCHED_PORT.PORT_5` | 127,798,285,732 | 0.634 |  |
| `CORE_ACTIVE_CYCLE` | 201,473,409,685 |  |  |

## By ggml operation

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 71.4% | 0.547 | 0.554 | 0.722 | `mul_mat (repacked)` |
| 14.6% | 0.552 | 0.576 | 0.568 | `mul_mat` |
| 4.4% | 0.401 | 0.402 | 0.292 | `flash_attn_ext_tiled` |
| 3.1% | 0.222 | 0.225 | 0.256 | `(spinning in a worker's loop: barrier or next graph)` |
| 2.7% | 0.146 | 0.146 | 0.124 | `swiglu_f32` |
| 2.4% | 0.327 | 0.330 | 0.333 | `(scheduler)` |
| 0.6% | 0.210 | 0.213 | 0.241 | `(barrier)` |
| 0.4% | 0.355 | 0.354 | 0.405 | `rms_norm_f32` |
| 0.3% | 0.187 | 0.188 | 0.224 | `mul_mat_one_chunk` |
| 0.0% | 0.399 | 0.405 | 0.507 | `set_rows_impl` |
| 0.0% | 0.356 | 0.357 | 0.442 | `rope_flt` |
| 0.0% | 0.122 | 0.078 | 0.087 | `get_rows_q` |
| 0.0% | 0.400 | 0.409 | 0.519 | `flash_attn_ext` |
| 0.0% | 0.367 | 0.371 | 0.470 | `flash_attn_ext_f16` |

## By leaf function

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 71.3% | 0.548 | 0.555 | 0.722 | `ggml_gemm_q4_K_8x8_q8_K` |
| 11.1% | 0.572 | 0.598 | 0.588 | `void tiled_run_microtile_avx2<16, false, 32, 1, true>` |
| 3.7% | 0.220 | 0.223 | 0.254 | `ggml_thread_cpu_relax` |
| 2.1% | 0.399 | 0.399 | 0.286 | `void simd_gemm_ukernel<6, 2>` |
| 1.5% | 0.401 | 0.402 | 0.294 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 1.3% | 0.149 | 0.148 | 0.127 | `ggml_v_silu` |
| 1.1% | 0.229 | 0.233 | 0.243 | `ggml_quantize_mat_q8_K_4x8` |
| 1.1% | 0.146 | 0.146 | 0.124 | `ggml_vec_swiglu_f32` |
| 1.0% | 0.307 | 0.312 | 0.319 | `quantize_row_q8_K_ref` |
| 0.9% | 0.561 | 0.586 | 0.577 | `tiled_unpack_src0` |
| 0.5% | 0.569 | 0.595 | 0.586 | `tiled_unpack_src1_q8_K` |
| 0.4% | 0.201 | 0.201 | 0.153 | `ggml_v_expf` |
| 0.4% | 0.355 | 0.354 | 0.405 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.4% | 0.426 | 0.431 | 0.530 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.3% | 0.411 | 0.411 | 0.295 | `void simd_gemm_ukernel<1, 2>` |
| 0.3% | 0.580 | 0.606 | 0.596 | `void tiled_repack_src0<16>` |
| 0.3% | 0.407 | 0.407 | 0.290 | `ggml_vec_max_f32` |
| 0.3% | 0.187 | 0.188 | 0.224 | `ggml_vec_dot_q6_K_q8_K` |
| 0.2% | 0.406 | 0.407 | 0.307 | `ggml_vec_soft_max_f32` |
| 0.1% | 0.375 | 0.386 | 0.420 | `op_add` |
| 0.1% | 0.416 | 0.416 | 0.311 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 0.437 | 0.446 | 0.403 | `_platform_bzero$VARIANT$Haswell` |
| 0.1% | 0.578 | 0.604 | 0.594 | `tiled_repack_src1` |
| 0.1% | 0.580 | 0.606 | 0.597 | `void tiled_unpk_or<0, 4, 3>` |
| 0.1% | 0.568 | 0.594 | 0.584 | `tiled_unpk_nib4` |
| 0.1% | 0.407 | 0.407 | 0.294 | `ggml_cpu_fp16_to_fp32` |
| 0.1% | 0.407 | 0.408 | 0.341 | `_platform_memmove$VARIANT$Haswell` |
| 0.1% | 0.564 | 0.590 | 0.579 | `tiled_byte_sum_16` |
| 0.1% | 0.458 | 0.460 | 0.575 | `void rotate_pairs<float>` |
| 0.1% | 0.400 | 0.403 | 0.506 | `0x7ff815c325e0` |

## By thread

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.511 | 0.520 | 0.634 | `llama-bench 0x4e2501 (llama-bench, pid: 46917)` |
| 16.7% | 0.510 | 0.519 | 0.634 | `llama-bench 0x4e2503 (llama-bench, pid: 46917)` |
| 16.7% | 0.513 | 0.522 | 0.637 | `Main Thread 0x4e24e6 (llama-bench, pid: 46917)` |
| 16.7% | 0.509 | 0.518 | 0.632 | `llama-bench 0x4e2502 (llama-bench, pid: 46917)` |
| 16.7% | 0.511 | 0.520 | 0.634 | `llama-bench 0x4e2500 (llama-bench, pid: 46917)` |
| 16.6% | 0.513 | 0.521 | 0.635 | `llama-bench 0x4e2504 (llama-bench, pid: 46917)` |
