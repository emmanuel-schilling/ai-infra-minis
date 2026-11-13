// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)
#include <immintrin.h>

#include <cassert>
#include <vector>

#include "hsum_avx2.hpp"
#include "qmm/matmul.hpp"
#include "qmm/q8_avx2_steps.hpp"

namespace qmm {

// Points at the fastest finished version. Change it only after a benchmark run
// shows the newer version is faster.
void matmul_q8_avx2(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                    std::span<float> c) {
  q8_avx2_steps::v1_one_accumulator(shape, a, w, c);
}

namespace q8_avx2_steps {

// A is quantized once up front; each block pair's int8 dot product uses maddubs
// with the sign trick, is widened to int32 and converted to fp32, then scaled by
// both blocks' scales into one 8-lane accumulator per output.
void v1_one_accumulator(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                        std::span<float> c) {
  assert(shape.k % q8_block_size == 0);
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k / q8_block_size);
  assert(c.size() == shape.m * shape.n);

  const __m256i ones = _mm256_set1_epi16(1);
  const std::size_t blocks_per_row = shape.k / q8_block_size;
  std::vector<BlockQ8> a_q(shape.m * blocks_per_row);
  quantize_q8(a, a_q);

  for (std::size_t i = 0; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      const std::size_t a_index = i * blocks_per_row;
      const std::size_t w_index = j * blocks_per_row;
      __m256 accumulator = _mm256_setzero_ps();
      for (std::size_t b = 0; b < blocks_per_row; ++b) {
        const BlockQ8& a_blk = a_q[a_index + b];
        const BlockQ8& w_blk = w[w_index + b];
        const __m256i a_block =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk.q.data()));
        const __m256i w_block =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_blk.q.data()));

        const __m256i a_abs = _mm256_sign_epi8(a_block, a_block);
        const __m256i w_signed = _mm256_sign_epi8(w_block, a_block);
        const __m256i pairs16 = _mm256_maddubs_epi16(a_abs, w_signed);
        const __m256i pairs32 = _mm256_madd_epi16(pairs16, ones);
        const __m256 pairs_float = _mm256_cvtepi32_ps(pairs32);
        accumulator =
            _mm256_fmadd_ps(pairs_float, _mm256_set1_ps(a_blk.scale * w_blk.scale), accumulator);
      }
      c[(i * shape.n) + j] = hsum256_ps(accumulator);
    }
  }
}

// To write: the 2 x 4 tile within L2-sized blocks of W. Set
// pending::q8_avx2_v2 to false when you start.
void v2_register_blocking(Shape /*shape*/, std::span<const float> /*a*/,
                          std::span<const BlockQ8> /*w*/, std::span<float> /*c*/) {}

}  // namespace q8_avx2_steps

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
