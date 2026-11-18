# q4_k_m tg, 6 threads

CPU time in ggml's graphs: 84197 ms, the whole for the shares below. Worker threads idle between graphs: 270 ms. Outside the graph computation, left out: 897 ms.

## By ggml operation

| Share | ms | Name |
|---:|---:|---|
| 61.8% | 52039 | `mul_mat (repacked)` |
| 33.0% | 27773 | `mul_mat_one_chunk` |
| 2.3% | 1971 | `(spinning in a worker's loop: barrier or next graph)` |
| 0.9% | 780 | `(scheduler)` |
| 0.9% | 763 | `flash_attn_ext_f16` |
| 0.6% | 483 | `(barrier)` |
| 0.3% | 226 | `mul_mat` |
| 0.1% | 57 | `swiglu_f32` |
| 0.1% | 57 | `rms_norm_f32` |
| 0.0% | 32 | `rope_flt` |
| 0.0% | 5 | `flash_attn_ext` |
| 0.0% | 3 | `set_rows_impl` |
| 0.0% | 3 | `set_rows` |
| 0.0% | 2 | `glu` |
| 0.0% | 1 | `get_rows_f32` |
| 0.0% | 1 | `get_rows` |
| 0.0% | 1 | `flash_attn_ext_f16_one_chunk` |

## Self time (leaf function)

| Share | ms | Name |
|---:|---:|---|
| 59.1% | 49752 | `ggml_gemv_q4_K_8x8_q8_K` |
| 32.8% | 27624 | `ggml_vec_dot_q6_K_q8_K` |
| 2.7% | 2264 | `ggml_thread_cpu_relax` |
| 2.3% | 1962 | `ggml_barrier` |
| 0.6% | 471 | `quantize_row_q8_K_ref` |
| 0.4% | 370 | `hsum_float_8` |
| 0.3% | 255 | `ggml_vec_mad_f16_f32` |
| 0.3% | 250 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.2% | 197 | `ggml_vec_dot_f16_f32` |
| 0.2% | 134 | `expf` |
| 0.1% | 112 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 84 | `ggml_threadpool_chunk_add` |
| 0.1% | 83 | `ggml_compute_forward_mul_mat` |
| 0.1% | 57 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.1% | 46 | `ggml_graph_compute_thread` |
| 0.1% | 44 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat` |
| 0.0% | 39 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 38 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat_one_chunk` |
| 0.0% | 36 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 32 | `ggml_vec_swiglu_f32` |
| 0.0% | 28 | `0x7ff815c411c0` |
| 0.0% | 24 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 23 | `0x10bfc8168` |
| 0.0% | 22 | `void rotate_pairs<float>` |
| 0.0% | 20 | `ggml_v_expf` |
| 0.0% | 19 | `ggml_op_is_empty` |
| 0.0% | 14 | `ggml_cpu_extra_compute_forward` |
| 0.0% | 14 | `0x7ff815c325e0` |
| 0.0% | 14 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 14 | `nearest_int` |

## Total time (function anywhere on the stack)

| Share | ms | Name |
|---:|---:|---|
| 97.0% | 81702 | `ggml_compute_forward` |
| 83.3% | 70159 | `ggml_graph_compute_secondary_thread` |
| 83.3% | 70159 | `_pthread_start` |
| 83.3% | 70159 | `thread_start` |
| 61.9% | 52093 | `ggml_cpu_extra_compute_forward` |
| 61.8% | 52067 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::compute_forward` |
| 61.8% | 52039 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat` |
| 59.1% | 49766 | `void ggml::cpu::repack::gemv<block_q4_K, 8ll, 8ll, (ggml_type)15>` |
| 59.1% | 49752 | `ggml_gemv_q4_K_8x8_q8_K` |
| 33.0% | 27773 | `ggml_compute_forward_mul_mat_one_chunk` |
| 32.8% | 27624 | `ggml_vec_dot_q6_K_q8_K` |
| 16.7% | 14038 | `ggml_graph_compute` |
| 16.7% | 14038 | `ggml_backend_cpu_graph_compute` |
| 16.7% | 14038 | `ggml_backend_graph_compute_async` |
| 16.7% | 14038 | `llama_context::graph_compute` |
| 16.7% | 14038 | `llama_context::process_ubatch` |
| 16.7% | 14038 | `llama_context::decode` |
| 16.7% | 14038 | `test_gen` |
| 16.7% | 14038 | `llama_bench` |
| 16.7% | 14038 | `start` |
| 2.7% | 2264 | `ggml_thread_cpu_relax` |
| 2.3% | 1962 | `ggml_barrier` |
| 0.9% | 763 | `ggml_compute_forward_flash_attn_ext_f16` |
| 0.6% | 471 | `quantize_row_q8_K_ref` |
| 0.4% | 370 | `hsum_float_8` |
| 0.3% | 255 | `ggml_vec_mad_f16_f32` |
| 0.3% | 226 | `ggml_compute_forward_mul_mat` |
| 0.2% | 197 | `ggml_vec_dot_f16_f32` |
| 0.2% | 144 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.2% | 134 | `expf` |

## By thread

| Share | ms | Name |
|---:|---:|---|
| 16.7% | 14038 | `Main Thread 0x3f99a2 (llama-bench, pid: 71823)` |
| 16.7% | 14034 | `llama-bench 0x3f9bf5 (llama-bench, pid: 71823)` |
| 16.7% | 14034 | `llama-bench 0x3f9bf4 (llama-bench, pid: 71823)` |
| 16.7% | 14033 | `llama-bench 0x3f9bf7 (llama-bench, pid: 71823)` |
| 16.7% | 14031 | `llama-bench 0x3f9bf8 (llama-bench, pid: 71823)` |
| 16.7% | 14027 | `llama-bench 0x3f9bf6 (llama-bench, pid: 71823)` |

## Spinning in a worker's loop, by the source line it returns to

| Share | ms | Name |
|---:|---:|---|
| 2.3% | 1971 | `ggml_graph_compute_secondary_thread (ggml-cpu.c:3282)` |
