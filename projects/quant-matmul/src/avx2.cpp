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
  avx2_steps::v3_register_blocking(shape, a, w, c);
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
        __m256 va = _mm256_loadu_ps(a.data() + i * shape.k + p);
        __m256 vw = _mm256_loadu_ps(w.data() + j * shape.k + p);
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

void v3_register_blocking(Shape shape, std::span<const float> a, std::span<const float> w,
                          std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k);
  assert(c.size() == shape.m * shape.n);

  // Fallback helper for fringe elements (edges of M and N)
  auto compute_single = [&](std::size_t row, std::size_t col) {
    const float* a_row = a.data() + row * shape.k;
    const float* w_row = w.data() + col * shape.k;
    __m256 vsum = _mm256_setzero_ps();
    std::size_t p = 0;
    for (; p + 8 <= shape.k; p += 8) {
      vsum = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p), _mm256_loadu_ps(w_row + p), vsum);
    }
    float sum = hsum256_ps(vsum);
    for (; p < shape.k; ++p) {
      sum += a_row[p] * w_row[p];
    }
    c[row * shape.n + col] = sum;
  };

  std::size_t i = 0;
  // Step i by 3 (rows of A)
  for (; i + 3 <= shape.m; i += 3) {
    const float* a0 = a.data() + (i + 0) * shape.k;
    const float* a1 = a.data() + (i + 1) * shape.k;
    const float* a2 = a.data() + (i + 2) * shape.k;

    std::size_t j = 0;
    // Step j by 4 (rows of W / columns of C)
    for (; j + 4 <= shape.n; j += 4) {
      const float* w0 = w.data() + (j + 0) * shape.k;
      const float* w1 = w.data() + (j + 1) * shape.k;
      const float* w2 = w.data() + (j + 2) * shape.k;
      const float* w3 = w.data() + (j + 3) * shape.k;

      // 12 Accumulators: exactly 12 YMM registers
      __m256 c00 = _mm256_setzero_ps();
      __m256 c01 = _mm256_setzero_ps();
      __m256 c02 = _mm256_setzero_ps();
      __m256 c03 = _mm256_setzero_ps();

      __m256 c10 = _mm256_setzero_ps();
      __m256 c11 = _mm256_setzero_ps();
      __m256 c12 = _mm256_setzero_ps();
      __m256 c13 = _mm256_setzero_ps();

      __m256 c20 = _mm256_setzero_ps();
      __m256 c21 = _mm256_setzero_ps();
      __m256 c22 = _mm256_setzero_ps();
      __m256 c23 = _mm256_setzero_ps();

      std::size_t p = 0;
      // Main loop: 12 FMAs for 7 loads (p advances by 8 floats = 256 bits)
      for (; p + 8 <= shape.k; p += 8) {
        // 3 loads for A (3 YMM registers)
        __m256 va0 = _mm256_loadu_ps(a0 + p);
        __m256 va1 = _mm256_loadu_ps(a1 + p);
        __m256 va2 = _mm256_loadu_ps(a2 + p);

        // 1 load for W reused across 3 rows of A (1 temporary YMM register)
        __m256 vw;

        vw = _mm256_loadu_ps(w0 + p);
        c00 = _mm256_fmadd_ps(va0, vw, c00);
        c10 = _mm256_fmadd_ps(va1, vw, c10);
        c20 = _mm256_fmadd_ps(va2, vw, c20);

        vw = _mm256_loadu_ps(w1 + p);
        c01 = _mm256_fmadd_ps(va0, vw, c01);
        c11 = _mm256_fmadd_ps(va1, vw, c11);
        c21 = _mm256_fmadd_ps(va2, vw, c21);

        vw = _mm256_loadu_ps(w2 + p);
        c02 = _mm256_fmadd_ps(va0, vw, c02);
        c12 = _mm256_fmadd_ps(va1, vw, c12);
        c22 = _mm256_fmadd_ps(va2, vw, c22);

        vw = _mm256_loadu_ps(w3 + p);
        c03 = _mm256_fmadd_ps(va0, vw, c03);
        c13 = _mm256_fmadd_ps(va1, vw, c13);
        c23 = _mm256_fmadd_ps(va2, vw, c23);
      }

      // Horizontally sum each of the 12 accumulator vectors
      float s00 = hsum256_ps(c00), s01 = hsum256_ps(c01), s02 = hsum256_ps(c02),
            s03 = hsum256_ps(c03);
      float s10 = hsum256_ps(c10), s11 = hsum256_ps(c11), s12 = hsum256_ps(c12),
            s13 = hsum256_ps(c13);
      float s20 = hsum256_ps(c20), s21 = hsum256_ps(c21), s22 = hsum256_ps(c22),
            s23 = hsum256_ps(c23);

      // Scalar remainder along K (for remaining p < shape.k)
      for (std::size_t rem_p = p; rem_p < shape.k; ++rem_p) {
        float a0_val = a0[rem_p], a1_val = a1[rem_p], a2_val = a2[rem_p];
        float w0_val = w0[rem_p], w1_val = w1[rem_p], w2_val = w2[rem_p], w3_val = w3[rem_p];

        s00 += a0_val * w0_val;
        s01 += a0_val * w1_val;
        s02 += a0_val * w2_val;
        s03 += a0_val * w3_val;
        s10 += a1_val * w0_val;
        s11 += a1_val * w1_val;
        s12 += a1_val * w2_val;
        s13 += a1_val * w3_val;
        s20 += a2_val * w0_val;
        s21 += a2_val * w1_val;
        s22 += a2_val * w2_val;
        s23 += a2_val * w3_val;
      }

      // Write 3x4 tile to output matrix C
      c[(i + 0) * shape.n + (j + 0)] = s00;
      c[(i + 0) * shape.n + (j + 1)] = s01;
      c[(i + 0) * shape.n + (j + 2)] = s02;
      c[(i + 0) * shape.n + (j + 3)] = s03;

      c[(i + 1) * shape.n + (j + 0)] = s10;
      c[(i + 1) * shape.n + (j + 1)] = s11;
      c[(i + 1) * shape.n + (j + 2)] = s12;
      c[(i + 1) * shape.n + (j + 3)] = s13;

      c[(i + 2) * shape.n + (j + 0)] = s20;
      c[(i + 2) * shape.n + (j + 1)] = s21;
      c[(i + 2) * shape.n + (j + 2)] = s22;
      c[(i + 2) * shape.n + (j + 3)] = s23;
    }

    // Remainder columns for these 3 rows (if shape.n % 4 != 0)
    for (; j < shape.n; ++j) {
      compute_single(i + 0, j);
      compute_single(i + 1, j);
      compute_single(i + 2, j);
    }
  }

  // Remainder rows (if shape.m % 3 != 0)
  for (; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      compute_single(i, j);
    }
  }
}

void v4_cache_blocking(Shape shape, std::span<const float> a, std::span<const float> w,
                       std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k);
  assert(c.size() == shape.m * shape.n);

  // Fallback helper for fringe elements (edges of M, N, and K)
  auto compute_single = [&](std::size_t row, std::size_t col) {
    const float* a_row = a.data() + row * shape.k;
    const float* w_row = w.data() + col * shape.k;
    __m256 vsum = _mm256_setzero_ps();
    std::size_t p = 0;
    for (; p + 8 <= shape.k; p += 8) {
      vsum = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + p), _mm256_loadu_ps(w_row + p), vsum);
    }
    float sum = hsum256_ps(vsum);
    for (; p < shape.k; ++p) {
      sum += a_row[p] * w_row[p];
    }
    c[row * shape.n + col] = sum;
  };

  // 64 rows of W * 768 floats * 4 bytes = 192 KB (fits comfortably inside 256 KB L2)
  constexpr std::size_t BN = 64;

  // CACHE BLOCKING LEVEL (Outer loop over N chunks that fit into L2)
  for (std::size_t j_block = 0; j_block < shape.n; j_block += BN) {
    std::size_t j_end = std::min(j_block + BN, shape.n);

    // STREAMING LEVEL (Run all rows of A against the cached L2 block of W)
    std::size_t i = 0;
    for (; i + 3 <= shape.m; i += 3) {
      const float* a0 = a.data() + (i + 0) * shape.k;
      const float* a1 = a.data() + (i + 1) * shape.k;
      const float* a2 = a.data() + (i + 2) * shape.k;

      // REGISTER TILE LEVEL (Step through cached W rows in chunks of 4)
      std::size_t j = j_block;
      for (; j + 4 <= j_end; j += 4) {
        const float* w0 = w.data() + (j + 0) * shape.k;
        const float* w1 = w.data() + (j + 1) * shape.k;
        const float* w2 = w.data() + (j + 2) * shape.k;
        const float* w3 = w.data() + (j + 3) * shape.k;

        // 12 Accumulators kept in 12 YMM registers
        __m256 c00 = _mm256_setzero_ps();
        __m256 c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps();
        __m256 c03 = _mm256_setzero_ps();

        __m256 c10 = _mm256_setzero_ps();
        __m256 c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps();
        __m256 c13 = _mm256_setzero_ps();

        __m256 c20 = _mm256_setzero_ps();
        __m256 c21 = _mm256_setzero_ps();
        __m256 c22 = _mm256_setzero_ps();
        __m256 c23 = _mm256_setzero_ps();

        std::size_t p = 0;
        // Inner reduction: 7 loads for 12 FMAs
        for (; p + 8 <= shape.k; p += 8) {
          __m256 va0 = _mm256_loadu_ps(a0 + p);
          __m256 va1 = _mm256_loadu_ps(a1 + p);
          __m256 va2 = _mm256_loadu_ps(a2 + p);

          __m256 vw;

          vw = _mm256_loadu_ps(w0 + p);
          c00 = _mm256_fmadd_ps(va0, vw, c00);
          c10 = _mm256_fmadd_ps(va1, vw, c10);
          c20 = _mm256_fmadd_ps(va2, vw, c20);

          vw = _mm256_loadu_ps(w1 + p);
          c01 = _mm256_fmadd_ps(va0, vw, c01);
          c11 = _mm256_fmadd_ps(va1, vw, c11);
          c21 = _mm256_fmadd_ps(va2, vw, c21);

          vw = _mm256_loadu_ps(w2 + p);
          c02 = _mm256_fmadd_ps(va0, vw, c02);
          c12 = _mm256_fmadd_ps(va1, vw, c12);
          c22 = _mm256_fmadd_ps(va2, vw, c22);

          vw = _mm256_loadu_ps(w3 + p);
          c03 = _mm256_fmadd_ps(va0, vw, c03);
          c13 = _mm256_fmadd_ps(va1, vw, c13);
          c23 = _mm256_fmadd_ps(va2, vw, c23);
        }

        // Horizontal vector reductions
        float s00 = hsum256_ps(c00), s01 = hsum256_ps(c01), s02 = hsum256_ps(c02),
              s03 = hsum256_ps(c03);
        float s10 = hsum256_ps(c10), s11 = hsum256_ps(c11), s12 = hsum256_ps(c12),
              s13 = hsum256_ps(c13);
        float s20 = hsum256_ps(c20), s21 = hsum256_ps(c21), s22 = hsum256_ps(c22),
              s23 = hsum256_ps(c23);

        // Scalar remainder along K
        for (std::size_t rem_p = p; rem_p < shape.k; ++rem_p) {
          float a0_val = a0[rem_p], a1_val = a1[rem_p], a2_val = a2[rem_p];
          float w0_val = w0[rem_p], w1_val = w1[rem_p], w2_val = w2[rem_p], w3_val = w3[rem_p];

          s00 += a0_val * w0_val;
          s01 += a0_val * w1_val;
          s02 += a0_val * w2_val;
          s03 += a0_val * w3_val;
          s10 += a1_val * w0_val;
          s11 += a1_val * w1_val;
          s12 += a1_val * w2_val;
          s13 += a1_val * w3_val;
          s20 += a2_val * w0_val;
          s21 += a2_val * w1_val;
          s22 += a2_val * w2_val;
          s23 += a2_val * w3_val;
        }

        // Store outputs
        c[(i + 0) * shape.n + (j + 0)] = s00;
        c[(i + 0) * shape.n + (j + 1)] = s01;
        c[(i + 0) * shape.n + (j + 2)] = s02;
        c[(i + 0) * shape.n + (j + 3)] = s03;

        c[(i + 1) * shape.n + (j + 0)] = s10;
        c[(i + 1) * shape.n + (j + 1)] = s11;
        c[(i + 1) * shape.n + (j + 2)] = s12;
        c[(i + 1) * shape.n + (j + 3)] = s13;

        c[(i + 2) * shape.n + (j + 0)] = s20;
        c[(i + 2) * shape.n + (j + 1)] = s21;
        c[(i + 2) * shape.n + (j + 2)] = s22;
        c[(i + 2) * shape.n + (j + 3)] = s23;
      }

      // Remainder columns within this L2 block (if (j_end - j_block) % 4 != 0)
      for (; j < j_end; ++j) {
        compute_single(i + 0, j);
        compute_single(i + 1, j);
        compute_single(i + 2, j);
      }
    }

    // Remainder rows of A within this L2 block (if shape.m % 3 != 0)
    for (; i < shape.m; ++i) {
      for (std::size_t j = j_block; j < j_end; ++j) {
        compute_single(i, j);
      }
    }
  }
}

}  // namespace avx2_steps

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
