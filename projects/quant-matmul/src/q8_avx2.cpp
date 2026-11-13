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

namespace {

// The dot product of one quantized row of A with one row of W, over their
// blocks: each block pair's int8 dot product uses maddubs with the sign trick,
// is widened to int32 and converted to fp32, then scaled by both blocks'
// scales into one 8-lane accumulator, summed horizontally at the end. The
// spans hold the same number of blocks. Used by v1 for every output and by v2
// for the outputs its tiles leave over.
float dot_q8_rows(std::span<const BlockQ8> a_row, std::span<const BlockQ8> w_row) {
  assert(a_row.size() == w_row.size());
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t b = 0; b < a_row.size(); ++b) {
    const BlockQ8& a_blk = a_row[b];
    const BlockQ8& w_blk = w_row[b];
    const __m256i a_block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk.q.data()));
    const __m256i w_block = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_blk.q.data()));

    const __m256i a_abs = _mm256_sign_epi8(a_block, a_block);
    const __m256i w_signed = _mm256_sign_epi8(w_block, a_block);
    const __m256i pairs16 = _mm256_maddubs_epi16(a_abs, w_signed);
    const __m256i pairs32 = _mm256_madd_epi16(pairs16, ones);
    const __m256 pairs_float = _mm256_cvtepi32_ps(pairs32);
    accumulator =
        _mm256_fmadd_ps(pairs_float, _mm256_set1_ps(a_blk.scale * w_blk.scale), accumulator);
  }
  return hsum256_ps(accumulator);
}

}  // namespace

// Points at the fastest finished version. Change it only after a benchmark run
// shows the newer version is faster.
void matmul_q8_avx2(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                    std::span<float> c) {
  q8_avx2_steps::v1_one_accumulator(shape, a, w, c);
}

namespace q8_avx2_steps {

// A is quantized once up front, then each output is one dot_q8_rows call.
void v1_one_accumulator(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                        std::span<float> c) {
  assert(shape.k % q8_block_size == 0);
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k / q8_block_size);
  assert(c.size() == shape.m * shape.n);

  const std::size_t blocks_per_row = shape.k / q8_block_size;
  std::vector<BlockQ8> a_q(shape.m * blocks_per_row);
  quantize_q8(a, a_q);
  const std::span<const BlockQ8> a_blocks(a_q);

  for (std::size_t i = 0; i < shape.m; ++i) {
    const auto a_row = a_blocks.subspan(i * blocks_per_row, blocks_per_row);
    for (std::size_t j = 0; j < shape.n; ++j) {
      const auto w_row = w.subspan(j * blocks_per_row, blocks_per_row);
      c[(i * shape.n) + j] = dot_q8_rows(a_row, w_row);
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
