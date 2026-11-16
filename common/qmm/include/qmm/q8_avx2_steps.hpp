// The AVX2 Q8 kernel's optimization steps, each kept as its own version so they
// can be tested and benchmarked side by side. Every version computes the same
// C = A * W^T, with the same layouts and preconditions as matmul_q8_avx2 (see
// qmm/matmul.hpp); matmul_q8_avx2 calls the fastest finished one.
#pragma once

#if defined(QMM_HAVE_AVX2)

#include <span>

#include "qmm/matmul.hpp"

// PLATFORM: x86-64 — AVX2 + FMA. Call only when cpu_has_avx2_fma() is true.
namespace qmm::q8_avx2_steps {

// One output at a time: one 8-lane fp32 accumulator per output, one block pair
// per step.
void v1_one_accumulator(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                        std::span<float> c);

// A tile of 2 rows of A by 4 rows of W at once, so each loaded A block feeds 4
// dot products and each W block 2, within blocks of 64 rows of W that stay in
// L2.
void v2_register_blocking(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                          std::span<float> c);

}  // namespace qmm::q8_avx2_steps

#endif  // QMM_HAVE_AVX2
