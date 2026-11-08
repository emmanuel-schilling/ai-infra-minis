#pragma once

#include <cstdint>

namespace qmm::bench {

#if defined(QMM_HAVE_AVX2)
// Floating-point operations per loop iteration of peak_fma: 12 FMA chains of
// 8 floats, each FMA a multiply and an add.
inline constexpr double peak_fma_flops_per_iteration = 12.0 * 8.0 * 2.0;

// PLATFORM: x86-64 — AVX2 + FMA. Runs `iterations` rounds of 12 independent
// FMA chains and returns a value derived from them so the work cannot be
// optimized away. Call only when qmm::cpu_has_avx2_fma() is true.
float peak_fma(std::int64_t iterations);
#endif

}  // namespace qmm::bench
