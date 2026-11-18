# q4_k_m tg, 6 threads

Samples in ggml's graphs: 84492 ms of thread time over 14146 ms, at 2.82 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 264 ms. Outside the graph computation, left out: 1406 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `CORE_ACTIVE_CYCLE` | 237,877,604,386 |  |  |
| `L2_RQSTS.MISS` | 7,001,189,333 | 0.029 | 31.7 |
| `LONGEST_LAT_CACHE.MISS` | 8,015,788,378 | 0.034 | 36.3 |
| `RESOURCE_STALLS.ANY` | 153,756,872,903 | 0.646 |  |

## By ggml operation

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 61.8% | 0.029 | 0.033 | 0.629 | `mul_mat (repacked)` |
| 32.4% | 0.030 | 0.035 | 0.683 | `mul_mat_one_chunk` |
| 2.8% | 0.029 | 0.033 | 0.616 | `(spinning in a worker's loop: barrier or next graph)` |
| 1.0% | 0.030 | 0.033 | 0.640 | `flash_attn_ext_f16` |
| 0.9% | 0.029 | 0.035 | 0.681 | `(scheduler)` |
| 0.6% | 0.030 | 0.033 | 0.629 | `(barrier)` |
| 0.3% | 0.029 | 0.034 | 0.650 | `mul_mat` |
| 0.1% | 0.030 | 0.034 | 0.644 | `swiglu_f32` |
| 0.1% | 0.031 | 0.034 | 0.634 | `rms_norm_f32` |
| 0.0% | 0.029 | 0.033 | 0.637 | `rope_flt` |
| 0.0% | 0.030 | 0.033 | 0.634 | `set_rows_impl` |
| 0.0% | 0.029 | 0.033 | 0.611 | `flash_attn_ext` |
| 0.0% | 0.028 | 0.033 | 0.613 | `glu` |
| 0.0% | 0.029 | 0.033 | 0.625 | `flash_attn_ext_f16_one_chunk` |
| 0.0% | 0.029 | 0.033 | 0.602 | `rope` |
| 0.0% | 0.030 | 0.034 | 0.632 | `add_non_quantized` |
| 0.0% | 0.028 | 0.034 | 0.624 | `set_rows` |

## By leaf function

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 59.3% | 0.029 | 0.033 | 0.629 | `ggml_gemv_q4_K_8x8_q8_K` |
| 32.1% | 0.030 | 0.035 | 0.683 | `ggml_vec_dot_q6_K_q8_K` |
| 3.2% | 0.029 | 0.033 | 0.618 | `ggml_thread_cpu_relax` |
| 2.2% | 0.029 | 0.033 | 0.623 | `ggml_barrier` |
| 0.5% | 0.030 | 0.033 | 0.635 | `quantize_row_q8_K_ref` |
| 0.4% | 0.029 | 0.036 | 0.708 | `hsum_float_8` |
| 0.3% | 0.030 | 0.033 | 0.642 | `ggml_vec_mad_f16_f32` |
| 0.3% | 0.029 | 0.036 | 0.710 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.3% | 0.030 | 0.033 | 0.635 | `ggml_vec_dot_f16_f32` |
| 0.2% | 0.030 | 0.033 | 0.640 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.2% | 0.030 | 0.034 | 0.644 | `expf` |
| 0.1% | 0.029 | 0.033 | 0.637 | `ggml_threadpool_chunk_add` |
| 0.1% | 0.030 | 0.036 | 0.693 | `ggml_compute_forward_mul_mat` |
| 0.1% | 0.031 | 0.034 | 0.634 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.1% | 0.030 | 0.034 | 0.632 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat` |
| 0.1% | 0.029 | 0.033 | 0.630 | `ggml_graph_compute_thread` |
| 0.1% | 0.030 | 0.033 | 0.641 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat_one_chunk` |
| 0.0% | 0.030 | 0.035 | 0.667 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.030 | 0.034 | 0.640 | `ggml_v_expf` |
| 0.0% | 0.029 | 0.033 | 0.631 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 0.029 | 0.033 | 0.628 | `0x7ff815c325e0` |
| 0.0% | 0.030 | 0.034 | 0.650 | `ggml_vec_swiglu_f32` |
| 0.0% | 0.031 | 0.034 | 0.650 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 0.029 | 0.035 | 0.688 | `0x10c962168` |
| 0.0% | 0.030 | 0.033 | 0.634 | `ggml_op_is_empty` |
| 0.0% | 0.029 | 0.033 | 0.637 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 0.030 | 0.034 | 0.652 | `ggml_row_size` |
| 0.0% | 0.030 | 0.033 | 0.636 | `nearest_int` |
| 0.0% | 0.029 | 0.034 | 0.633 | `ggml_rope_cache_init` |
| 0.0% | 0.029 | 0.033 | 0.649 | `ggml_cpu_extra_compute_forward` |

## By thread

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.029 | 0.034 | 0.644 | `llama-bench 0x4e2ea1 (llama-bench, pid: 47023)` |
| 16.7% | 0.029 | 0.034 | 0.645 | `llama-bench 0x4e2e9d (llama-bench, pid: 47023)` |
| 16.7% | 0.029 | 0.034 | 0.643 | `llama-bench 0x4e2e9f (llama-bench, pid: 47023)` |
| 16.7% | 0.031 | 0.034 | 0.658 | `Main Thread 0x4e2e8d (llama-bench, pid: 47023)` |
| 16.6% | 0.029 | 0.034 | 0.644 | `llama-bench 0x4e2ea0 (llama-bench, pid: 47023)` |
| 16.6% | 0.029 | 0.034 | 0.644 | `llama-bench 0x4e2e9e (llama-bench, pid: 47023)` |
