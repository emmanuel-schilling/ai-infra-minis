# q8_0 tg, 6 threads

Samples in ggml's graphs: 135853 ms of thread time over 22717 ms, at 2.89 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 290 ms. Outside the graph computation, left out: 963 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `UOPS_DISPATCHED_PORT.PORT_0` | 82,723,021,110 | 0.211 |  |
| `UOPS_DISPATCHED_PORT.PORT_1` | 78,156,929,771 | 0.199 |  |
| `UOPS_DISPATCHED_PORT.PORT_5` | 23,686,553,385 | 0.060 |  |
| `CORE_ACTIVE_CYCLE` | 392,368,954,369 |  |  |

## By ggml operation

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 96.2% | 0.211 | 0.199 | 0.060 | `mul_mat_one_chunk` |
| 1.9% | 0.210 | 0.198 | 0.059 | `(spinning in a worker's loop: barrier or next graph)` |
| 0.6% | 0.209 | 0.197 | 0.058 | `flash_attn_ext_f16` |
| 0.5% | 0.213 | 0.201 | 0.059 | `(barrier)` |
| 0.4% | 0.210 | 0.198 | 0.060 | `mul_mat` |
| 0.3% | 0.210 | 0.197 | 0.058 | `(scheduler)` |
| 0.1% | 0.214 | 0.203 | 0.063 | `swiglu_f32` |
| 0.0% | 0.216 | 0.203 | 0.059 | `rms_norm_f32` |
| 0.0% | 0.206 | 0.195 | 0.059 | `rope_flt` |
| 0.0% | 0.216 | 0.203 | 0.059 | `set_rows_impl` |
| 0.0% | 0.185 | 0.174 | 0.052 | `rms_norm_mul_fused` |
| 0.0% | 0.217 | 0.206 | 0.063 | `glu` |
| 0.0% | 0.212 | 0.199 | 0.058 | `flash_attn_ext` |
| 0.0% | 0.196 | 0.187 | 0.062 | `rope` |
| 0.0% | 0.212 | 0.200 | 0.058 | `set_rows` |
| 0.0% | 0.209 | 0.196 | 0.056 | `add` |
| 0.0% | 0.194 | 0.186 | 0.064 | `get_rows` |
| 0.0% | 0.211 | 0.197 | 0.082 | `add_non_quantized` |

## By leaf function

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 48.2% | 0.211 | 0.199 | 0.060 | `ggml_vec_dot_q8_0_q8_0` |
| 32.7% | 0.211 | 0.199 | 0.060 | `mul_sum_i8_pairs_float` |
| 6.0% | 0.211 | 0.199 | 0.060 | `sum_i16_pairs_float` |
| 6.0% | 0.211 | 0.199 | 0.060 | `ggml_lookup_fp16_to_fp32` |
| 2.3% | 0.211 | 0.199 | 0.059 | `ggml_thread_cpu_relax` |
| 1.5% | 0.211 | 0.199 | 0.060 | `mul_sum_us8_pairs_float` |
| 0.9% | 0.211 | 0.199 | 0.061 | `hsum_float_8` |
| 0.7% | 0.210 | 0.198 | 0.060 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.3% | 0.211 | 0.199 | 0.060 | `ggml_compute_forward_mul_mat` |
| 0.2% | 0.210 | 0.198 | 0.058 | `ggml_barrier` |
| 0.2% | 0.209 | 0.197 | 0.058 | `ggml_vec_mad_f16_f32` |
| 0.2% | 0.209 | 0.197 | 0.059 | `quantize_row_q8_0` |
| 0.1% | 0.209 | 0.197 | 0.058 | `ggml_vec_dot_f16_f32` |
| 0.1% | 0.207 | 0.195 | 0.058 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 0.210 | 0.198 | 0.058 | `expf` |
| 0.1% | 0.209 | 0.198 | 0.060 | `0x10b9e3fd9` |
| 0.1% | 0.208 | 0.196 | 0.060 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.216 | 0.203 | 0.059 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.0% | 0.209 | 0.197 | 0.058 | `ggml_graph_compute_thread` |
| 0.0% | 0.210 | 0.197 | 0.057 | `void rotate_pairs<float>` |
| 0.0% | 0.212 | 0.202 | 0.063 | `ggml_vec_swiglu_f32` |
| 0.0% | 0.210 | 0.197 | 0.058 | `ggml_threadpool_chunk_add` |
| 0.0% | 0.211 | 0.198 | 0.059 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 0.209 | 0.197 | 0.057 | `0x7ff815c325e0` |
| 0.0% | 0.216 | 0.204 | 0.059 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 0.208 | 0.197 | 0.059 | `ggml_op_is_empty` |
| 0.0% | 0.216 | 0.205 | 0.063 | `ggml_v_expf` |
| 0.0% | 0.212 | 0.200 | 0.060 | `ggml_is_contiguous_m_n` |
| 0.0% | 0.211 | 0.200 | 0.061 | `ggml_row_size` |
| 0.0% | 0.205 | 0.195 | 0.060 | `void ggml_compute_forward_rope_flt<float>` |

## By thread

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.210 | 0.198 | 0.060 | `llama-bench 0x4e199c (llama-bench, pid: 46790)` |
| 16.7% | 0.210 | 0.198 | 0.060 | `llama-bench 0x4e199d (llama-bench, pid: 46790)` |
| 16.7% | 0.210 | 0.198 | 0.060 | `llama-bench 0x4e1999 (llama-bench, pid: 46790)` |
| 16.7% | 0.216 | 0.204 | 0.062 | `Main Thread 0x4e198e (llama-bench, pid: 46790)` |
| 16.6% | 0.210 | 0.198 | 0.060 | `llama-bench 0x4e199a (llama-bench, pid: 46790)` |
| 16.6% | 0.210 | 0.198 | 0.060 | `llama-bench 0x4e199b (llama-bench, pid: 46790)` |
