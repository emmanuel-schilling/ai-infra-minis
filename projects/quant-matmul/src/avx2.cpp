// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <span>

#include "qmm/avx2_steps.hpp"
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

// Points at the fastest finished version. Change it only after a benchmark run
// shows the newer version is faster.
void matmul_avx2(Shape shape, std::span<const float> a, std::span<const float> w,
                 std::span<float> c) {
  avx2_steps::v1_one_accumulator(shape, a, w, c);
}

namespace avx2_steps {

void v1_one_accumulator(Shape shape, std::span<const float> a, std::span<const float> w,
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

void v2_accumulators(Shape shape, std::span<const float> a, std::span<const float> w,
                     std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k);
  assert(c.size() == shape.m * shape.n);

  for (std::size_t i = 0; i < shape.m; ++i) {
    const float* a_row = a.data() + i * shape.k;

    for (std::size_t j = 0; j < shape.n; ++j) {
      const float* w_row = w.data() + j * shape.k;

      // 8 independent vector accumulators (each holds 8 partial sums)
      __m256 vsum0 = _mm256_setzero_ps();
      __m256 vsum1 = _mm256_setzero_ps();
      __m256 vsum2 = _mm256_setzero_ps();
      __m256 vsum3 = _mm256_setzero_ps();
      __m256 vsum4 = _mm256_setzero_ps();
      __m256 vsum5 = _mm256_setzero_ps();
      __m256 vsum6 = _mm256_setzero_ps();
      __m256 vsum7 = _mm256_setzero_ps();

      std::size_t p = 0;

      // Main loop: process 64 elements (8 vectors) per iteration
      for (; p + 64 <= shape.k; p += 64) {
        vsum0 =
            _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 0), _mm256_loadu_ps(w_row + p + 0), vsum0);
        vsum1 =
            _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 8), _mm256_loadu_ps(w_row + p + 8), vsum1);
        vsum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 16), _mm256_loadu_ps(w_row + p + 16),
                                vsum2);
        vsum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 24), _mm256_loadu_ps(w_row + p + 24),
                                vsum3);
        vsum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 32), _mm256_loadu_ps(w_row + p + 32),
                                vsum4);
        vsum5 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 40), _mm256_loadu_ps(w_row + p + 40),
                                vsum5);
        vsum6 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 48), _mm256_loadu_ps(w_row + p + 48),
                                vsum6);
        vsum7 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p + 56), _mm256_loadu_ps(w_row + p + 56),
                                vsum7);
      }

      // Vector remainder loop: process 8 elements at a time into vsum0
      for (; p + 8 <= shape.k; p += 8) {
        vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p), _mm256_loadu_ps(w_row + p), vsum0);
      }

      // Tree reduction of the 8 accumulators into a single __m256
      vsum0 = _mm256_add_ps(vsum0, vsum1);
      vsum2 = _mm256_add_ps(vsum2, vsum3);
      vsum4 = _mm256_add_ps(vsum4, vsum5);
      vsum6 = _mm256_add_ps(vsum6, vsum7);

      vsum0 = _mm256_add_ps(vsum0, vsum2);
      vsum4 = _mm256_add_ps(vsum4, vsum6);

      vsum0 = _mm256_add_ps(vsum0, vsum4);

      // Horizontal sum down to a single scalar float
      float sum = hsum256_ps(vsum0);

      // Scalar remainder loop: leftover 0 to 7 elements
      for (; p < shape.k; ++p) {
        sum += a_row[p] * w_row[p];
      }

      c[i * shape.n + j] = sum;
    }
  }
}

// To write: register blocking. Name the parameters when you write the body, and
// set pending::avx2_v3 to false.
void v3_register_blocking(Shape /*shape*/, std::span<const float> /*a*/,
                          std::span<const float> /*w*/, std::span<float> /*c*/) {}

// To write: cache blocking. Name the parameters when you write the body, and set
// pending::avx2_v4 to false.
void v4_cache_blocking(Shape /*shape*/, std::span<const float> /*a*/, std::span<const float> /*w*/,
                       std::span<float> /*c*/) {}

}  // namespace avx2_steps

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
