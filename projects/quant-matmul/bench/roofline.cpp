// The roofline's two ceilings for one core: memory read bandwidth and peak
// fp32 throughput. The kernels are single-threaded, so one core's ceilings are
// the ones that apply to them.
#include "roofline.hpp"

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "qmm/matmul.hpp"

namespace {

// Far larger than the 12 MB L3, so the reads come from memory.
constexpr std::size_t bandwidth_bytes = std::size_t{512} << 20U;

// Integer addition is associative, so the compiler may vectorize this loop
// freely; the sum only keeps the reads from being optimized away.
void read_bandwidth(benchmark::State& state) {
  const std::vector<std::uint64_t> data(bandwidth_bytes / sizeof(std::uint64_t), 1);
  for (auto _ : state) {
    std::uint64_t sum = 0;
    for (const std::uint64_t value : data) {
      sum += value;
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(state.iterations() * static_cast<std::int64_t>(bandwidth_bytes));
}
BENCHMARK(read_bandwidth)->Name("roofline/read_bandwidth")->Unit(benchmark::kMillisecond);

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
  state.counters["FLOP/s"] =
      benchmark::Counter(static_cast<double>(iterations) * qmm::bench::peak_fma_flops_per_iteration,
                         benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(peak_fma)->Name("roofline/peak_fma")->Unit(benchmark::kMillisecond);
#endif

}  // namespace
