// Helpers shared by the AVX2 kernels in avx2.cpp and q8_avx2.cpp. Internal to
// the library: not part of the public interface in include/qmm.
//
// PLATFORM: x86-64 — AVX2 intrinsics; include only from files compiled with
// AVX2 enabled (see CMakeLists.txt). Other CPUs would need their own helpers.
#pragma once

#include <immintrin.h>

namespace qmm {

// The sum of an 8-float vector's lanes: the 128-bit halves are added, then
// pairs within the result, then the last two values.
inline float hsum256_ps(__m256 v) {
  const __m128 low = _mm256_castps256_ps128(v);
  const __m128 high = _mm256_extractf128_ps(v, 1);
  const __m128 sum4 = _mm_add_ps(low, high);
  const __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  return _mm_cvtss_f32(_mm_add_ss(sum2, _mm_movehdup_ps(sum2)));
}

}  // namespace qmm
