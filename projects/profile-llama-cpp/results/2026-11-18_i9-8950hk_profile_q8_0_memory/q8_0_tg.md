# q8_0 tg, 6 threads

Samples in ggml's graphs: 135801 ms of thread time over 22721 ms, at 2.89 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 323 ms. Outside the graph computation, left out: 979 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `CORE_ACTIVE_CYCLE` | 392,106,086,082 |  |  |
| `L2_RQSTS.MISS` | 11,855,148,508 | 0.030 | 33.4 |
| `LONGEST_LAT_CACHE.MISS` | 13,865,348,416 | 0.035 | 39.1 |
| `RESOURCE_STALLS.ANY` | 284,722,317,536 | 0.726 |  |

## By ggml operation

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 96.3% | 0.030 | 0.035 | 0.726 | `mul_mat_one_chunk` |
| 1.8% | 0.030 | 0.035 | 0.723 | `(spinning in a worker's loop: barrier or next graph)` |
| 0.6% | 0.031 | 0.035 | 0.722 | `flash_attn_ext_f16` |
| 0.5% | 0.030 | 0.036 | 0.724 | `(barrier)` |
| 0.4% | 0.030 | 0.035 | 0.722 | `mul_mat` |
| 0.3% | 0.030 | 0.035 | 0.716 | `(scheduler)` |
| 0.1% | 0.029 | 0.036 | 0.738 | `swiglu_f32` |
| 0.0% | 0.031 | 0.035 | 0.725 | `rms_norm_f32` |
| 0.0% | 0.027 | 0.031 | 0.634 | `rope_flt` |
| 0.0% | 0.031 | 0.034 | 0.689 | `set_rows_impl` |
| 0.0% | 0.031 | 0.032 | 0.637 | `set_rows` |
| 0.0% | 0.030 | 0.037 | 0.738 | `glu` |
| 0.0% | 0.030 | 0.036 | 0.738 | `rope` |
| 0.0% | 0.030 | 0.035 | 0.735 | `flash_attn_ext` |
| 0.0% | 0.031 | 0.038 | 0.741 | `swiglu` |
| 0.0% | 0.031 | 0.037 | 0.743 | `add_non_quantized` |

## By leaf function

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 48.1% | 0.030 | 0.035 | 0.726 | `ggml_vec_dot_q8_0_q8_0` |
| 32.7% | 0.030 | 0.035 | 0.727 | `mul_sum_i8_pairs_float` |
| 6.1% | 0.030 | 0.035 | 0.726 | `sum_i16_pairs_float` |
| 6.1% | 0.030 | 0.035 | 0.727 | `ggml_lookup_fp16_to_fp32` |
| 2.2% | 0.030 | 0.035 | 0.723 | `ggml_thread_cpu_relax` |
| 1.4% | 0.030 | 0.035 | 0.726 | `mul_sum_us8_pairs_float` |
| 0.9% | 0.030 | 0.035 | 0.725 | `hsum_float_8` |
| 0.7% | 0.030 | 0.035 | 0.722 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.3% | 0.030 | 0.035 | 0.719 | `ggml_compute_forward_mul_mat` |
| 0.2% | 0.031 | 0.035 | 0.722 | `ggml_vec_mad_f16_f32` |
| 0.2% | 0.030 | 0.035 | 0.724 | `ggml_barrier` |
| 0.2% | 0.030 | 0.036 | 0.727 | `quantize_row_q8_0` |
| 0.2% | 0.031 | 0.035 | 0.723 | `ggml_vec_dot_f16_f32` |
| 0.1% | 0.031 | 0.035 | 0.719 | `expf` |
| 0.1% | 0.031 | 0.035 | 0.721 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 0.030 | 0.035 | 0.714 | `0x10e252fd9` |
| 0.1% | 0.030 | 0.035 | 0.723 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.030 | 0.036 | 0.731 | `void rotate_pairs<float>` |
| 0.0% | 0.031 | 0.035 | 0.725 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.0% | 0.030 | 0.036 | 0.725 | `0x7ff815c325e0` |
| 0.0% | 0.030 | 0.036 | 0.743 | `ggml_v_expf` |
| 0.0% | 0.030 | 0.036 | 0.732 | `ggml_graph_compute_thread` |
| 0.0% | 0.027 | 0.030 | 0.617 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 0.029 | 0.035 | 0.730 | `ggml_vec_swiglu_f32` |
| 0.0% | 0.031 | 0.036 | 0.730 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 0.030 | 0.036 | 0.730 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 0.029 | 0.035 | 0.701 | `0x7ff815c411c0` |
| 0.0% | 0.030 | 0.035 | 0.727 | `ggml_row_size` |
| 0.0% | 0.030 | 0.035 | 0.710 | `ggml_is_contiguous_m_n` |
| 0.0% | 0.030 | 0.036 | 0.740 | `ggml_op_is_empty` |

## By thread

| Share of cycles | `L2_RQSTS.MISS` per cycle | `LONGEST_LAT_CACHE.MISS` per cycle | `RESOURCE_STALLS.ANY` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.030 | 0.035 | 0.726 | `llama-bench 0x4e20da (llama-bench, pid: 46869)` |
| 16.7% | 0.030 | 0.035 | 0.724 | `llama-bench 0x4e20de (llama-bench, pid: 46869)` |
| 16.7% | 0.030 | 0.035 | 0.725 | `llama-bench 0x4e20dd (llama-bench, pid: 46869)` |
| 16.7% | 0.031 | 0.036 | 0.730 | `Main Thread 0x4e20c8 (llama-bench, pid: 46869)` |
| 16.6% | 0.030 | 0.035 | 0.725 | `llama-bench 0x4e20dc (llama-bench, pid: 46869)` |
| 16.6% | 0.030 | 0.035 | 0.726 | `llama-bench 0x4e20db (llama-bench, pid: 46869)` |
