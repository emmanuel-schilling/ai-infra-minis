// The AVX2 + FMA kernel's optimization steps, each kept as its own version so
// they can be tested and benchmarked side by side. Every version computes the
// same C = A * W^T, with the same layouts and preconditions as matmul_avx2 (see
// qmm/matmul.hpp); matmul_avx2 calls the fastest finished one.
#pragma once

#if defined(QMM_HAVE_AVX2)

#include <span>

#include "qmm/matmul.hpp"

// PLATFORM: x86-64 — AVX2 + FMA. Call only when cpu_has_avx2_fma() is true.
namespace qmm::avx2_steps {

// One 8-wide FMA accumulator per output.
void v1_one_accumulator(Shape shape, std::span<const float> a, std::span<const float> w,
                        std::span<float> c);

// Several independent accumulators per output, so each FMA doesn't wait for the
// one before it.
void v2_accumulators(Shape shape, std::span<const float> a, std::span<const float> w,
                     std::span<float> c);

// A tile of outputs at once, such as 3 rows of A by 4 rows of W, so each loaded
// vector feeds several FMAs.
void v3_register_blocking(Shape shape, std::span<const float> a, std::span<const float> w,
                          std::span<float> c);

// All of A against one block of W rows at a time, sized to stay in L2.
void v4_cache_blocking(Shape shape, std::span<const float> a, std::span<const float> w,
                       std::span<float> c);

}  // namespace qmm::avx2_steps

#endif  // QMM_HAVE_AVX2
