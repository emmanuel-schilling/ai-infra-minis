// The roofline's two ceilings: memory read bandwidth and peak fp32
// throughput. The kernels are single-threaded, so one core's ceilings are the
// ones that apply to them. The multi-threaded variants, on 6 and 12 threads,
// give the whole CPU's ceilings for multi-threaded engines such as llama.cpp
// (Profile llama.cpp's roofline).
#include "roofline.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <vector>

#include "qmm/matmul.hpp"

namespace {

// Far larger than the 12 MB L3, so the reads come from memory.
constexpr std::size_t bandwidth_bytes = std::size_t{512} << 20U;

// Integer addition is associative, so the compiler may vectorize this loop
// freely; the sum only keeps the reads from being optimized away. With several
// threads, each reads its own share of the total, so together they still read
// far more than L3 holds, and no thread finds another's data in the cache.
void read_bandwidth(benchmark::State& state) {
  const std::size_t bytes = bandwidth_bytes / static_cast<std::size_t>(state.threads());
  const std::vector<std::uint64_t> data(bytes / sizeof(std::uint64_t), 1);
  for (auto _ : state) {
    std::uint64_t sum = 0;
    for (const std::uint64_t value : data) {
      sum += value;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bytes));
}
BENCHMARK(read_bandwidth)->Name("roofline/read_bandwidth")->Unit(benchmark::kMillisecond);
// Real time, since each thread's CPU time would leave out the others'.
BENCHMARK(read_bandwidth)
    ->Name("roofline/read_bandwidth")
    ->Threads(6)
    ->Threads(12)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

#if defined(QMM_HAVE_AVX2)
void peak_fma(benchmark::State& state) {
  if (!qmm::cpu_has_avx2_fma()) {
    state.SkipWithMessage("this CPU lacks AVX2 or FMA");
    return;
  }
  constexpr std::int64_t iterations = 4'194'304;  // 2^22
  for (auto _ : state) {
    benchmark::DoNotOptimize(qmm::bench::peak_fma(iterations));
  }
  const double flops_per_call =
      static_cast<double>(iterations) * qmm::bench::peak_fma_flops_per_iteration;
  // FLOPs as items: like bytes, Google Benchmark sums them over the threads and
  // divides by the time, so items_per_second is the whole CPU's FLOP/s. The
  // counter is right for one thread only, and kept for Quantized matmul's
  // scripts.
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(flops_per_call));
  state.counters["FLOP/s"] =
      benchmark::Counter(flops_per_call, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(peak_fma)->Name("roofline/peak_fma")->Unit(benchmark::kMillisecond);
BENCHMARK(peak_fma)
    ->Name("roofline/peak_fma")
    ->Threads(6)
    ->Threads(12)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);
#endif

}  // namespace
