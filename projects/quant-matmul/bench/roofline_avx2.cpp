// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own peak-throughput loop.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)

#include <immintrin.h>

#include <cstdint>

#include "roofline.hpp"

namespace qmm::bench {

// FMA has a latency of 4 cycles and two FMA units accept one each per cycle on
// Skylake-family cores, so at least 4 * 2 = 8 independent chains are needed to
// keep both units busy. 12 leaves a margin and fits in the 16 ymm registers
// with the two constant operands.
float peak_fma(std::int64_t iterations) {
  // acc = acc * x + y converges to y / (1 - x) = 0.1: no overflow, no denormals.
  const __m256 x = _mm256_set1_ps(0.999999F);
  const __m256 y = _mm256_set1_ps(1e-7F);
  __m256 acc0 = _mm256_set1_ps(0.00F);
  __m256 acc1 = _mm256_set1_ps(0.01F);
  __m256 acc2 = _mm256_set1_ps(0.02F);
  __m256 acc3 = _mm256_set1_ps(0.03F);
  __m256 acc4 = _mm256_set1_ps(0.04F);
  __m256 acc5 = _mm256_set1_ps(0.05F);
  __m256 acc6 = _mm256_set1_ps(0.06F);
  __m256 acc7 = _mm256_set1_ps(0.07F);
  __m256 acc8 = _mm256_set1_ps(0.08F);
  __m256 acc9 = _mm256_set1_ps(0.09F);
  __m256 acc10 = _mm256_set1_ps(0.10F);
  __m256 acc11 = _mm256_set1_ps(0.11F);
  for (std::int64_t i = 0; i < iterations; ++i) {
    acc0 = _mm256_fmadd_ps(acc0, x, y);
    acc1 = _mm256_fmadd_ps(acc1, x, y);
    acc2 = _mm256_fmadd_ps(acc2, x, y);
    acc3 = _mm256_fmadd_ps(acc3, x, y);
    acc4 = _mm256_fmadd_ps(acc4, x, y);
    acc5 = _mm256_fmadd_ps(acc5, x, y);
    acc6 = _mm256_fmadd_ps(acc6, x, y);
    acc7 = _mm256_fmadd_ps(acc7, x, y);
    acc8 = _mm256_fmadd_ps(acc8, x, y);
    acc9 = _mm256_fmadd_ps(acc9, x, y);
    acc10 = _mm256_fmadd_ps(acc10, x, y);
    acc11 = _mm256_fmadd_ps(acc11, x, y);
  }
  const __m256 sum = _mm256_add_ps(
      _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
                    _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7))),
      _mm256_add_ps(_mm256_add_ps(acc8, acc9), _mm256_add_ps(acc10, acc11)));
  return _mm256_cvtss_f32(sum);
}

}  // namespace qmm::bench

#endif  // QMM_HAVE_AVX2
