// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <span>

#include "qmm/matmul.hpp"

namespace qmm {

namespace {
// Helper to horizontally sum an 8-float AVX register
float hsum256_ps(__m256 v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  __m128 v128 = _mm_add_ps(vlow, vhigh);
  __m128 vshuf = _mm_movehl_ps(v128, v128);
  __m128 v128_2 = _mm_add_ps(v128, vshuf);
  __m128 vshuf2 = _mm_movehdup_ps(v128_2);
  return _mm_cvtss_f32(_mm_add_ss(v128_2, vshuf2));
}
}  // namespace

void matmul_avx2(Shape shape, std::span<const float> a, std::span<const float> w,
                 std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k);
  assert(c.size() == shape.m * shape.n);

  for (std::size_t i = 0; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      __m256 vsum = _mm256_setzero_ps();
      std::size_t p = 0;

      // Vectorized dot product along K
      for (; p + 8 <= shape.k; p += 8) {
        __m256 va = _mm256_loadu_ps(&a[i * shape.k + p]);
        __m256 vw = _mm256_loadu_ps(&w[j * shape.k + p]);
        vsum = _mm256_fmadd_ps(va, vw, vsum);
      }

      float sum = hsum256_ps(vsum);

      // Remainder scalar loop for k % 8 != 0
      for (; p < shape.k; ++p) {
        sum += a[i * shape.k + p] * w[j * shape.k + p];
      }

      c[i * shape.n + j] = sum;
    }
  }
}

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
