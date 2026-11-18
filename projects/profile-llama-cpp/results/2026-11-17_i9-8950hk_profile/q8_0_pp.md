# q8_0 pp, 6 threads

CPU time in ggml's graphs: 113982 ms, the whole for the shares below. Worker threads idle between graphs: 80 ms. Outside the graph computation, left out: 219 ms.

## By ggml operation

| Share | ms | Name |
|---:|---:|---|
| 92.2% | 105078 | `mul_mat` |
| 2.4% | 2763 | `flash_attn_ext_tiled` |
| 2.4% | 2718 | `(spinning in a worker's loop: barrier or next graph)` |
| 1.3% | 1534 | `swiglu_f32` |
| 0.7% | 745 | `(scheduler)` |
| 0.5% | 557 | `(barrier)` |
| 0.3% | 289 | `rms_norm_f32` |
| 0.2% | 252 | `mul_mat_one_chunk` |
| 0.0% | 26 | `rope_flt` |
| 0.0% | 12 | `get_rows_q` |
| 0.0% | 8 | `set_rows_impl` |

## Self time (leaf function)

| Share | ms | Name |
|---:|---:|---|
| 45.8% | 52207 | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` |
| 23.4% | 26715 | `float vector[8] (anonymous namespace)::madd<float vector[8], float vector[8]>` |
| 21.7% | 24767 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::updot` |
| 2.9% | 3274 | `ggml_thread_cpu_relax` |
| 1.1% | 1290 | `void simd_gemm_ukernel<6, 2>` |
| 0.8% | 908 | `quantize_row_q8_0` |
| 0.8% | 908 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 0.6% | 739 | `ggml_v_silu` |
| 0.5% | 567 | `ggml_vec_swiglu_f32` |
| 0.4% | 472 | `(anonymous namespace)::hsum` |
| 0.3% | 292 | `ggml_v_expf` |
| 0.3% | 289 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.2% | 233 | `void simd_gemm_ukernel<1, 2>` |
| 0.2% | 217 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.2% | 173 | `ggml_vec_max_f32` |
| 0.1% | 134 | `ggml_vec_soft_max_f32` |
| 0.1% | 113 | `ggml_vec_dot_q8_0_q8_0` |
| 0.1% | 101 | `mul_sum_i8_pairs_float` |
| 0.1% | 79 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 67 | `ggml_cpu_fp16_to_fp32` |
| 0.1% | 58 | `_platform_bzero$VARIANT$Haswell` |
| 0.0% | 55 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 52 | `op_add` |
| 0.0% | 48 | `0x7ff815c325e0` |
| 0.0% | 40 | `void rotate_pairs<float>` |
| 0.0% | 27 | `ggml_vec_scale_f32` |
| 0.0% | 26 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 24 | `simd_gemm` |
| 0.0% | 13 | `ggml_vec_add_f32` |
| 0.0% | 13 | `sum_i16_pairs_float` |

## Total time (function anywhere on the stack)

| Share | ms | Name |
|---:|---:|---|
| 96.9% | 110457 | `ggml_compute_forward` |
| 92.2% | 105078 | `ggml_compute_forward_mul_mat` |
| 91.4% | 104161 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::mnpack` |
| 91.4% | 104161 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::matmul` |
| 83.3% | 94986 | `ggml_graph_compute_secondary_thread` |
| 83.3% | 94986 | `_pthread_start` |
| 83.3% | 94986 | `thread_start` |
| 45.8% | 52207 | `void (anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::gemm4xN<2>` |
| 23.4% | 26715 | `float vector[8] (anonymous namespace)::madd<float vector[8], float vector[8]>` |
| 21.7% | 24767 | `(anonymous namespace)::tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float>::updot` |
| 16.7% | 18996 | `ggml_graph_compute` |
| 16.7% | 18996 | `ggml_backend_cpu_graph_compute` |
| 16.7% | 18996 | `ggml_backend_graph_compute_async` |
| 16.7% | 18996 | `llama_context::graph_compute` |
| 16.7% | 18996 | `llama_context::process_ubatch` |
| 16.7% | 18996 | `llama_context::decode` |
| 16.7% | 18996 | `test_prompt` |
| 16.7% | 18996 | `llama_bench` |
| 16.7% | 18996 | `start` |
| 2.9% | 3274 | `ggml_thread_cpu_relax` |
| 2.4% | 2763 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 1.3% | 1534 | `ggml_compute_forward_swiglu_f32` |
| 1.1% | 1290 | `void simd_gemm_ukernel<6, 2>` |
| 0.8% | 908 | `quantize_row_q8_0` |
| 0.6% | 739 | `ggml_v_silu` |
| 0.5% | 567 | `ggml_vec_swiglu_f32` |
| 0.4% | 472 | `(anonymous namespace)::hsum` |
| 0.3% | 292 | `ggml_v_expf` |
| 0.3% | 289 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.3% | 289 | `ggml_cpu_try_fuse_ops` |

## By thread

| Share | ms | Name |
|---:|---:|---|
| 16.7% | 19003 | `llama-bench 0x3f8644 (llama-bench, pid: 71698)` |
| 16.7% | 18999 | `llama-bench 0x3f8647 (llama-bench, pid: 71698)` |
| 16.7% | 18998 | `llama-bench 0x3f8645 (llama-bench, pid: 71698)` |
| 16.7% | 18996 | `Main Thread 0x3f8424 (llama-bench, pid: 71698)` |
| 16.7% | 18993 | `llama-bench 0x3f8646 (llama-bench, pid: 71698)` |
| 16.7% | 18993 | `llama-bench 0x3f8648 (llama-bench, pid: 71698)` |

## Spinning in a worker's loop, by the source line it returns to

| Share | ms | Name |
|---:|---:|---|
| 2.4% | 2718 | `ggml_graph_compute_secondary_thread (ggml-cpu.c:3282)` |
