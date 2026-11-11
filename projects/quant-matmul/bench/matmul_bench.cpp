// Kernel benchmarks on GPT-2's weight shapes, for 1 token (GEMV, decode) and
// 128 or 1,024 tokens (GEMM, prefill). Kernels still pending in
// qmm/pending.hpp are left out.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <vector>

#include "qmm/avx2_steps.hpp"
#include "qmm/matmul.hpp"
#include "qmm/pending.hpp"
#include "random_floats.hpp"

namespace {

using qmm::BlockQ8;
using qmm::Shape;

using Fp32Kernel = void (*)(Shape, std::span<const float>, std::span<const float>,
                            std::span<float>);
using Q8Kernel = void (*)(Shape, std::span<const float>, std::span<const BlockQ8>,
                          std::span<float>);

struct WeightShape {
  const char* name;
  std::size_t n;
  std::size_t k;
};

// GPT-2 (124M): attention output projection, MLP up and down projections, and
// the output layer over the 50,257-token vocabulary.
constexpr WeightShape weight_shapes[] = {
    {"attn_out", 768, 768},
    {"mlp_up", 3072, 768},
    {"mlp_down", 768, 3072},
    {"lm_head", 50257, 768},
};

constexpr std::size_t token_counts[] = {1, 128, 1024};

// The output layer at 1,024 tokens is 79 GFLOP per call, which would dominate
// the run time; 128 tokens already measures it as a GEMM.
bool skipped(const WeightShape& ws, std::size_t m) { return ws.n == 50257 && m == 1024; }

// One token reads each weight once, so GEMV is memory-bound. Cycling through
// enough copies of W to exceed the 12 MB L3 makes the weights come from
// memory, as they do in a full model.
constexpr std::size_t gemv_working_set_bytes = std::size_t{64} << 20U;

std::size_t copies_for(std::size_t m, std::size_t weight_bytes) {
  if (m != 1) {
    return 1;
  }
  return std::max<std::size_t>(1, (gemv_working_set_bytes + weight_bytes - 1) / weight_bytes);
}

void set_counters(benchmark::State& state, Shape shape, std::size_t weight_bytes) {
  const double flops = 2.0 * static_cast<double>(shape.m * shape.n * shape.k);
  state.counters["FLOP/s"] =
      benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate);
  const std::size_t bytes =
      weight_bytes + ((shape.m * shape.k) + (shape.m * shape.n)) * sizeof(float);
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
}

void bench_fp32(benchmark::State& state, Fp32Kernel kernel, Shape shape) {
  const auto a = qmm::support::random_floats(shape.m * shape.k, 1);
  const auto w = qmm::support::random_floats(shape.n * shape.k, 2);
  const std::size_t weight_bytes = w.size() * sizeof(float);
  const std::vector<std::vector<float>> weights(copies_for(shape.m, weight_bytes), w);
  std::vector<float> c(shape.m * shape.n);

  std::size_t next = 0;
  for (auto _ : state) {
    kernel(shape, a, weights[next], c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
    next = (next + 1) % weights.size();
  }
  set_counters(state, shape, weight_bytes);
}

// Weights are quantized ahead of time, outside the timing; activations are
// quantized inside each call, so that cost is measured.
void bench_q8(benchmark::State& state, Q8Kernel kernel, Shape shape) {
  const auto a = qmm::support::random_floats(shape.m * shape.k, 1);
  const auto w_fp32 = qmm::support::random_floats(shape.n * shape.k, 2);
  std::vector<BlockQ8> w(w_fp32.size() / qmm::q8_block_size);
  qmm::quantize_q8(w_fp32, w);
  const std::size_t weight_bytes = w.size() * sizeof(BlockQ8);
  const std::vector<std::vector<BlockQ8>> weights(copies_for(shape.m, weight_bytes), w);
  std::vector<float> c(shape.m * shape.n);

  std::size_t next = 0;
  for (auto _ : state) {
    kernel(shape, a, weights[next], c);
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
    next = (next + 1) % weights.size();
  }
  set_counters(state, shape, weight_bytes);
}

template <typename Kernel, typename Bench>
void register_kernel(const std::string& family, const char* name, Kernel kernel, Bench bench) {
  for (const WeightShape& ws : weight_shapes) {
    for (const std::size_t m : token_counts) {
      if (skipped(ws, m)) {
        continue;
      }
      const Shape shape{m, ws.n, ws.k};
      const std::string label =
          family + "/" + name + "/" + ws.name + "/tokens:" + std::to_string(m);
      benchmark::RegisterBenchmark(label, [=](benchmark::State& state) {
        bench(state, kernel, shape);
      })->Unit(benchmark::kMillisecond);
    }
  }
}

void register_all() {
  if (!qmm::pending::scalar) {
    register_kernel("fp32", "scalar", qmm::matmul_scalar, bench_fp32);
  }
#if defined(QMM_HAVE_ACCELERATE)
  register_kernel("fp32", "accelerate", qmm::matmul_accelerate, bench_fp32);
#endif
#if defined(QMM_HAVE_OPENBLAS)
  register_kernel("fp32", "openblas", qmm::matmul_openblas, bench_fp32);
#endif
  if (!qmm::pending::q8_scalar && !qmm::pending::quantize) {
    register_kernel("q8", "scalar", qmm::matmul_q8_scalar, bench_q8);
  }
#if defined(QMM_HAVE_AVX2)
  if (qmm::cpu_has_avx2_fma()) {
    // The optimization steps, side by side. matmul_avx2 only calls one of
    // them, so timing it too would time the same code twice.
    namespace steps = qmm::avx2_steps;
    if (!qmm::pending::avx2) {
      register_kernel("fp32", "avx2_v1", steps::v1_one_accumulator, bench_fp32);
    }
    if (!qmm::pending::avx2_v2) {
      register_kernel("fp32", "avx2_v2", steps::v2_accumulators, bench_fp32);
    }
    if (!qmm::pending::avx2_v3) {
      register_kernel("fp32", "avx2_v3", steps::v3_register_blocking, bench_fp32);
    }
    if (!qmm::pending::avx2_v4) {
      register_kernel("fp32", "avx2_v4", steps::v4_cache_blocking, bench_fp32);
    }
    if (!qmm::pending::q8_avx2 && !qmm::pending::quantize) {
      register_kernel("q8", "avx2", qmm::matmul_q8_avx2, bench_q8);
    }
  }
#endif
#if defined(QMM_HAVE_NEON_DOTPROD)
  if (qmm::cpu_has_neon_dotprod() && !qmm::pending::q8_neon && !qmm::pending::quantize) {
    register_kernel("q8", "neon", qmm::matmul_q8_neon, bench_q8);
  }
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    register_all();
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
      return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qmm_bench: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "qmm_bench: unknown exception\n");
  }
  return 1;
}
