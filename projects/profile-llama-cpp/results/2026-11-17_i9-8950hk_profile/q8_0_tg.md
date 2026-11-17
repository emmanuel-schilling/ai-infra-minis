# q8_0 tg, 6 threads

CPU time in ggml's graphs: 135361 ms, the whole for the shares below. Worker threads idle between graphs: 275 ms. Outside the graph computation, left out: 270 ms.

## By ggml operation

| Share | ms | Name |
|---:|---:|---|
| 96.2% | 130284 | `mul_mat_one_chunk` |
| 1.7% | 2327 | `(spinning in a worker's loop: barrier or next graph)` |
| 0.6% | 867 | `flash_attn_ext_f16` |
| 0.5% | 642 | `mul_mat` |
| 0.5% | 641 | `(barrier)` |
| 0.3% | 410 | `(scheduler)` |
| 0.1% | 72 | `swiglu_f32` |
| 0.1% | 69 | `rms_norm_f32` |
| 0.0% | 31 | `rope_flt` |
| 0.0% | 5 | `rms_norm_mul_fused` |
| 0.0% | 4 | `add_non_quantized` |
| 0.0% | 3 | `glu` |
| 0.0% | 2 | `set_rows_impl` |
| 0.0% | 2 | `set_rows` |
| 0.0% | 1 | `get_rows_q` |
| 0.0% | 1 | `rope` |

## Self time (leaf function)

| Share | ms | Name |
|---:|---:|---|
| 48.3% | 65417 | `ggml_vec_dot_q8_0_q8_0` |
| 32.6% | 44186 | `mul_sum_i8_pairs_float` |
| 6.1% | 8197 | `sum_i16_pairs_float` |
| 6.0% | 8178 | `ggml_lookup_fp16_to_fp32` |
| 2.0% | 2743 | `ggml_thread_cpu_relax` |
| 1.4% | 1884 | `mul_sum_us8_pairs_float` |
| 0.9% | 1249 | `hsum_float_8` |
| 0.7% | 973 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.3% | 347 | `ggml_compute_forward_mul_mat` |
| 0.2% | 280 | `ggml_vec_mad_f16_f32` |
| 0.2% | 275 | `quantize_row_q8_0` |
| 0.2% | 254 | `ggml_barrier` |
| 0.1% | 192 | `ggml_vec_dot_f16_f32` |
| 0.1% | 160 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 141 | `expf` |
| 0.1% | 94 | `0x100956fd9` |
| 0.1% | 88 | `_platform_memmove$VARIANT$Haswell` |
| 0.1% | 74 | `ggml_graph_compute_thread` |
| 0.1% | 69 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.0% | 59 | `void rotate_pairs<float>` |
| 0.0% | 45 | `ggml_op_is_empty` |
| 0.0% | 41 | `0x7ff815c325e0` |
| 0.0% | 35 | `ggml_v_expf` |
| 0.0% | 33 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 32 | `ggml_vec_swiglu_f32` |
| 0.0% | 25 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 20 | `ggml_row_size` |
| 0.0% | 18 | `0x7ff815c411c0` |
| 0.0% | 17 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 16 | `ggml_is_contiguous_m_n` |

## Total time (function anywhere on the stack)

| Share | ms | Name |
|---:|---:|---|
| 97.8% | 132355 | `ggml_compute_forward` |
| 96.2% | 130284 | `ggml_compute_forward_mul_mat_one_chunk` |
| 83.3% | 112798 | `thread_start` |
| 83.3% | 112798 | `_pthread_start` |
| 83.3% | 112798 | `ggml_graph_compute_secondary_thread` |
| 48.3% | 65417 | `ggml_vec_dot_q8_0_q8_0` |
| 32.6% | 44186 | `mul_sum_i8_pairs_float` |
| 16.7% | 22563 | `ggml_backend_cpu_graph_compute` |
| 16.7% | 22563 | `test_gen` |
| 16.7% | 22563 | `llama_context::graph_compute` |
| 16.7% | 22563 | `llama_context::process_ubatch` |
| 16.7% | 22563 | `ggml_backend_graph_compute_async` |
| 16.7% | 22563 | `llama_context::decode` |
| 16.7% | 22563 | `ggml_graph_compute` |
| 16.7% | 22563 | `start` |
| 16.7% | 22563 | `llama_bench` |
| 6.1% | 8197 | `sum_i16_pairs_float` |
| 6.0% | 8178 | `ggml_lookup_fp16_to_fp32` |
| 2.0% | 2743 | `ggml_thread_cpu_relax` |
| 1.4% | 1884 | `mul_sum_us8_pairs_float` |
| 0.9% | 1249 | `hsum_float_8` |
| 0.6% | 867 | `ggml_compute_forward_flash_attn_ext_f16` |
| 0.5% | 642 | `ggml_compute_forward_mul_mat` |
| 0.2% | 280 | `ggml_vec_mad_f16_f32` |
| 0.2% | 275 | `quantize_row_q8_0` |
| 0.2% | 254 | `ggml_barrier` |
| 0.1% | 192 | `ggml_vec_dot_f16_f32` |
| 0.1% | 190 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 141 | `expf` |
| 0.1% | 94 | `0x100956fd9` |

## By thread

| Share | ms | Name |
|---:|---:|---|
| 16.7% | 22564 | `llama-bench 0x3f8d7f (llama-bench, pid: 71742)` |
| 16.7% | 22564 | `llama-bench 0x3f8d7d (llama-bench, pid: 71742)` |
| 16.7% | 22563 | `Main Thread 0x3f8b26 (llama-bench, pid: 71742)` |
| 16.7% | 22558 | `llama-bench 0x3f8d7e (llama-bench, pid: 71742)` |
| 16.7% | 22556 | `llama-bench 0x3f8d7c (llama-bench, pid: 71742)` |
| 16.7% | 22556 | `llama-bench 0x3f8d80 (llama-bench, pid: 71742)` |

## Spinning in a worker's loop, by the source line it returns to

| Share | ms | Name |
|---:|---:|---|
| 1.7% | 2327 | `ggml_graph_compute_secondary_thread (ggml-cpu.c:3282)` |
