# q4_k_m tg, 6 threads

Samples in ggml's graphs: 85257 ms of thread time over 14286 ms, at 2.81 GHz on average; their cycles are the whole for the shares below. Worker threads idle between graphs: 315 ms. Outside the graph computation, left out: 1470 ms.

Each sample's counts cover the interval since its thread's previous sample, so a function or operation shorter than that shares its interval's counts with whatever else ran in it; see counters_profile.py.

## Totals

| Event | Count | Per cycle | GB/s (×64 B) |
|---|---:|---:|---:|
| `UOPS_DISPATCHED_PORT.PORT_0` | 51,289,673,532 | 0.214 |  |
| `UOPS_DISPATCHED_PORT.PORT_1` | 51,529,046,921 | 0.215 |  |
| `UOPS_DISPATCHED_PORT.PORT_5` | 67,069,115,838 | 0.280 |  |
| `CORE_ACTIVE_CYCLE` | 239,838,623,421 |  |  |

## By ggml operation

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 60.8% | 0.211 | 0.212 | 0.282 | `mul_mat (repacked)` |
| 32.6% | 0.221 | 0.221 | 0.276 | `mul_mat_one_chunk` |
| 3.5% | 0.205 | 0.206 | 0.276 | `(spinning in a worker's loop: barrier or next graph)` |
| 1.0% | 0.220 | 0.221 | 0.276 | `(scheduler)` |
| 0.9% | 0.210 | 0.211 | 0.280 | `flash_attn_ext_f16` |
| 0.8% | 0.217 | 0.218 | 0.286 | `(barrier)` |
| 0.3% | 0.212 | 0.214 | 0.280 | `mul_mat` |
| 0.1% | 0.223 | 0.224 | 0.294 | `swiglu_f32` |
| 0.1% | 0.227 | 0.228 | 0.295 | `rms_norm_f32` |
| 0.0% | 0.210 | 0.211 | 0.277 | `rope_flt` |
| 0.0% | 0.207 | 0.207 | 0.261 | `set_rows_impl` |
| 0.0% | 0.204 | 0.208 | 0.296 | `add` |
| 0.0% | 0.210 | 0.210 | 0.274 | `set_rows` |
| 0.0% | 0.233 | 0.233 | 0.300 | `glu` |
| 0.0% | 0.234 | 0.236 | 0.311 | `add_non_quantized` |
| 0.0% | 0.200 | 0.199 | 0.240 | `rope` |
| 0.0% | 0.196 | 0.197 | 0.274 | `mul_mat_one_chunk (repacked)` |
| 0.0% | 0.210 | 0.212 | 0.249 | `get_rows_q` |
| 0.0% | 0.065 | 0.065 | 0.081 | `flash_attn_ext` |

## By leaf function

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 58.1% | 0.211 | 0.212 | 0.282 | `ggml_gemv_q4_K_8x8_q8_K` |
| 32.3% | 0.221 | 0.221 | 0.276 | `ggml_vec_dot_q6_K_q8_K` |
| 4.1% | 0.207 | 0.208 | 0.277 | `ggml_thread_cpu_relax` |
| 2.3% | 0.206 | 0.207 | 0.277 | `ggml_barrier` |
| 0.5% | 0.222 | 0.223 | 0.292 | `quantize_row_q8_K_ref` |
| 0.4% | 0.225 | 0.226 | 0.274 | `hsum_float_8` |
| 0.3% | 0.225 | 0.225 | 0.273 | `ggml_compute_forward_mul_mat_one_chunk` |
| 0.3% | 0.211 | 0.212 | 0.281 | `ggml_vec_mad_f16_f32` |
| 0.2% | 0.208 | 0.209 | 0.277 | `ggml_vec_dot_f16_f32` |
| 0.2% | 0.211 | 0.212 | 0.281 | `expf` |
| 0.2% | 0.209 | 0.210 | 0.279 | `ggml_compute_forward_flash_attn_ext_f16_one_chunk` |
| 0.1% | 0.211 | 0.213 | 0.283 | `ggml_threadpool_chunk_add` |
| 0.1% | 0.223 | 0.224 | 0.275 | `ggml_compute_forward_mul_mat` |
| 0.1% | 0.207 | 0.208 | 0.271 | `ggml_graph_compute_thread` |
| 0.1% | 0.213 | 0.215 | 0.286 | `0x7ff815c325e0` |
| 0.1% | 0.227 | 0.228 | 0.295 | `void ggml_compute_forward_rms_norm_f32<(ggml_rms_norm_fuse_op)1>` |
| 0.0% | 0.215 | 0.216 | 0.269 | `_platform_memmove$VARIANT$Haswell` |
| 0.0% | 0.210 | 0.211 | 0.280 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat_one_chunk` |
| 0.0% | 0.211 | 0.212 | 0.277 | `ggml_op_is_empty` |
| 0.0% | 0.215 | 0.217 | 0.288 | `ggml::cpu::repack::tensor_traits<block_q4_K, 8ll, 8ll, (ggml_type)15>::forward_mul_mat` |
| 0.0% | 0.229 | 0.230 | 0.299 | `void vec_binary_op_contiguous<&op_add(float, float), float, float, float>` |
| 0.0% | 0.227 | 0.227 | 0.273 | `0x100f14168` |
| 0.0% | 0.213 | 0.214 | 0.286 | `void rotate_pairs<float>` |
| 0.0% | 0.221 | 0.222 | 0.291 | `ggml_v_expf` |
| 0.0% | 0.227 | 0.228 | 0.299 | `ggml_vec_swiglu_f32` |
| 0.0% | 0.211 | 0.212 | 0.276 | `void ggml_compute_forward_rope_flt<float>` |
| 0.0% | 0.214 | 0.215 | 0.286 | `nearest_int` |
| 0.0% | 0.208 | 0.209 | 0.280 | `ggml::cpu::repack::extra_buffer_type::get_tensor_traits` |
| 0.0% | 0.200 | 0.202 | 0.266 | `logf` |
| 0.0% | 0.197 | 0.198 | 0.261 | `ggml_cpu_extra_compute_forward` |

## By thread

| Share of cycles | `UOPS_DISPATCHED_PORT.PORT_0` per cycle | `UOPS_DISPATCHED_PORT.PORT_1` per cycle | `UOPS_DISPATCHED_PORT.PORT_5` per cycle | Name |
|---:|---:|---:|---:|---|
| 16.7% | 0.211 | 0.212 | 0.277 | `llama-bench 0x4e2803 (llama-bench, pid: 46946)` |
| 16.7% | 0.211 | 0.212 | 0.277 | `llama-bench 0x4e2805 (llama-bench, pid: 46946)` |
| 16.7% | 0.211 | 0.212 | 0.278 | `llama-bench 0x4e2804 (llama-bench, pid: 46946)` |
| 16.7% | 0.212 | 0.213 | 0.278 | `llama-bench 0x4e2801 (llama-bench, pid: 46946)` |
| 16.6% | 0.212 | 0.213 | 0.278 | `llama-bench 0x4e2802 (llama-bench, pid: 46946)` |
| 16.6% | 0.226 | 0.228 | 0.290 | `Main Thread 0x4e27e6 (llama-bench, pid: 46946)` |
