# q4_k_m pp, 6 threads

CPU time in ggml's graphs: 56714 ms, the whole for the shares below. Worker threads idle between graphs: 75 ms. Outside the graph computation, left out: 766 ms.

## By ggml operation

| Share | ms | Name |
|---:|---:|---|
| 72.2% | 40931 | `mul_mat (repacked)` |
| 14.9% | 8448 | `mul_mat` |
| 4.5% | 2547 | `flash_attn_ext_tiled` |
| 2.7% | 1527 | `swiglu_f32` |
| 2.5% | 1416 | `(scheduler)` |
| 2.0% | 1127 | `(spinning in a worker's loop: barrier or next graph)` |
| 0.5% | 268 | `rms_norm_f32` |
| 0.4% | 248 | `(barrier)` |
| 0.3% | 158 | `mul_mat_one_chunk` |
| 0.1% | 29 | `rope_flt` |
| 0.0% | 12 | `set_rows_impl` |
| 0.0% | 3 | `get_rows_q` |

## Self time (leaf function)

| Share | ms | Name |
|---:|---:|---|
| 72.1% | 40895 | `ggml_gemm_q4_K_8x8_q8_K` |
| 11.5% | 6548 | `void tiled_run_microtile_avx2<16, false, 32, 1, true>` |
| 2.4% | 1373 | `ggml_thread_cpu_relax` |
| 2.1% | 1178 | `void simd_gemm_ukernel<6, 2>` |
| 1.5% | 875 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 1.2% | 707 | `ggml_v_silu` |
| 1.1% | 637 | `ggml_vec_swiglu_f32` |
| 1.1% | 612 | `ggml_quantize_mat_q8_K_4x8` |
| 1.0% | 558 | `quantize_row_q8_K_ref` |
| 0.8% | 433 | `tiled_unpack_src0` |
| 0.5% | 295 | `tiled_unpack_src1_q8_K` |
| 0.5% | 268 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.4% | 241 | `ggml_v_expf` |
| 0.4% | 218 | `void simd_gemm_ukernel<1, 2>` |
| 0.4% | 215 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.4% | 201 | `ggml_vec_max_f32` |
| 0.3% | 163 | `void tiled_repack_src0<16>` |
| 0.3% | 157 | `ggml_vec_dot_q6_K_q8_K` |
| 0.2% | 109 | `ggml_vec_soft_max_f32` |
| 0.1% | 84 | `op_add` |
| 0.1% | 83 | `_platform_bzero$VARIANT$Haswell` |
| 0.1% | 69 | `tiled_repack_src1` |
| 0.1% | 68 | `void tiled_unpk_or<0, 4, 3>` |
| 0.1% | 59 | `ggml_lookup_fp16_to_fp32` |
| 0.1% | 57 | `void rotate_pairs<float>` |
| 0.1% | 57 | `ggml_cpu_fp16_to_fp32` |
| 0.1% | 55 | `_platform_memmove$VARIANT$Haswell` |
| 0.1% | 46 | `ggml_barrier` |
| 0.1% | 45 | `tiled_unpk_nib4` |
| 0.1% | 43 | `tiled_byte_sum_16` |

## Total time (function anywhere on the stack)

| Share | ms | Name |
|---:|---:|---|
| 97.1% | 55073 | `ggml_compute_forward` |
| 83.3% | 47266 | `ggml_graph_compute_secondary_thread` |
| 83.3% | 47266 | `_pthread_start` |
| 83.3% | 47266 | `thread_start` |
| 73.3% | 41550 | `ggml_cpu_extra_compute_forward` |
| 73.3% | 41544 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::compute_forward` |
| 72.2% | 40931 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat` |
| 72.1% | 40895 | `ggml_gemm_q4_K_8x8_q8_K` |
| 72.1% | 40895 | `void ggml::cpu::repack::gemm<block_q4_K, 8ll, 8ll, (ggml_type)15>` |
| 16.7% | 9448 | `ggml_graph_compute` |
| 16.7% | 9448 | `ggml_backend_cpu_graph_compute` |
| 16.7% | 9448 | `ggml_backend_graph_compute_async` |
| 16.7% | 9448 | `llama_context::graph_compute` |
| 16.7% | 9448 | `llama_context::process_ubatch` |
| 16.7% | 9448 | `llama_context::decode` |
| 16.7% | 9448 | `test_prompt` |
| 16.7% | 9448 | `llama_bench` |
| 16.7% | 9448 | `start` |
| 14.9% | 8448 | `ggml_compute_forward_mul_mat_tiled` |
| 14.9% | 8448 | `ggml_compute_forward_mul_mat` |
| 12.7% | 7229 | `void ggml_compute_forward_mul_mat_tiled_one_chunk<block_q6_K, 16, false, 32, true>` |
| 11.5% | 6548 | `void tiled_run_microtile_avx2<16, false, 32, 1, true>` |
| 4.5% | 2546 | `ggml_compute_forward_flash_attn_ext_tiled` |
| 2.7% | 1527 | `ggml_compute_forward_swiglu_f32` |
| 2.4% | 1373 | `ggml_thread_cpu_relax` |
| 2.1% | 1178 | `void simd_gemm_ukernel<6, 2>` |
| 1.2% | 707 | `ggml_v_silu` |
| 1.1% | 637 | `ggml_vec_swiglu_f32` |
| 1.1% | 613 | `void ggml_quantize_mat_t<8ll, (ggml_type)15>` |
| 1.1% | 612 | `ggml_quantize_mat_q8_K_4x8` |

## By thread

| Share | ms | Name |
|---:|---:|---|
| 16.7% | 9458 | `llama-bench 0x3f954a (llama-bench, pid: 71793)` |
| 16.7% | 9456 | `llama-bench 0x3f954d (llama-bench, pid: 71793)` |
| 16.7% | 9451 | `llama-bench 0x3f954b (llama-bench, pid: 71793)` |
| 16.7% | 9451 | `llama-bench 0x3f9549 (llama-bench, pid: 71793)` |
| 16.7% | 9450 | `llama-bench 0x3f954c (llama-bench, pid: 71793)` |
| 16.7% | 9448 | `Main Thread 0x3f92ed (llama-bench, pid: 71793)` |

## Spinning in a worker's loop, by the source line it returns to

| Share | ms | Name |
|---:|---:|---|
| 2.0% | 1127 | `ggml_graph_compute_secondary_thread (ggml-cpu.c:3282)` |
