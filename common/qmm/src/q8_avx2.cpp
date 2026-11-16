// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <vector>

#include "hsum_avx2.hpp"
#include "qmm/matmul.hpp"
#include "qmm/q8_avx2_steps.hpp"

namespace qmm {

namespace {
// Adds one block pair to acc: the int8 dot product of a_q and w_q, as 8 int32
// partial sums converted to fp32, times scale (the two blocks' scales
// multiplied). maddubs needs one unsigned operand, so a's signs are moved onto
// w and a_abs = |a_q| is used. The caller computes a_abs so that a tile can
// reuse it for every W row. ones is _mm256_set1_epi16(1), which widens the
// 16-bit pair sums to 32 bits.
__m256 accumulate_block(__m256 acc, __m256i a_q, __m256i a_abs, __m256i w_q, float scale,
                        __m256i ones) {
  const __m256i w_signed = _mm256_sign_epi8(w_q, a_q);
  const __m256i pairs16 = _mm256_maddubs_epi16(a_abs, w_signed);
  const __m256i pairs32 = _mm256_madd_epi16(pairs16, ones);
  const __m256 pairs_float = _mm256_cvtepi32_ps(pairs32);
  return _mm256_fmadd_ps(pairs_float, _mm256_set1_ps(scale), acc);
}

// The dot product of one quantized row of A with one row of W, which hold the
// same number of blocks: one accumulate_block per block pair into a single
// accumulator, summed horizontally at the end. Used by v1 for every output and
// by v2 for the outputs its tiles leave over.
float dot_q8_rows(std::span<const BlockQ8> a_row, std::span<const BlockQ8> w_row) {
  assert(a_row.size() == w_row.size());
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 accumulator = _mm256_setzero_ps();
  for (std::size_t b = 0; b < a_row.size(); ++b) {
    const BlockQ8& a_blk = a_row[b];
    const BlockQ8& w_blk = w_row[b];
    const __m256i a_q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_blk.q.data()));
    const __m256i w_q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w_blk.q.data()));
    const __m256i a_abs = _mm256_sign_epi8(a_q, a_q);
    accumulator = accumulate_block(accumulator, a_q, a_abs, w_q, a_blk.scale * w_blk.scale, ones);
  }
  return hsum256_ps(accumulator);
}

}  // namespace

// Points at the fastest finished version. Change it only after a benchmark run
// shows the newer version is faster.
void matmul_q8_avx2(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                    std::span<float> c) {
  q8_avx2_steps::v2_register_blocking(shape, a, w, c);
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

// A is quantized once up front, as in v1. W is processed in blocks of BN rows
// that fit in L2, and every pair of A rows reads a block while it is cached.
// Within a block, a 2 x 4 tile computes 8 outputs at once in 8 accumulators,
// so each A block loaded is used 4 times and each W block twice. Outputs the
// tiles leave over (the last row of A when m is odd, or the last 1 to 3 W rows
// of a block) are one dot_q8_rows call each.
void v2_register_blocking(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                          std::span<float> c) {
  assert(shape.k % q8_block_size == 0);
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k / q8_block_size);
  assert(c.size() == shape.m * shape.n);

  // 64 rows of W at k / 32 blocks of 36 bytes: 54 KB at k = 768, 216 KB at
  // k = 3,072, both within the 256 KB L2.
  constexpr std::size_t BN = 64;

  const __m256i ones = _mm256_set1_epi16(1);

  const std::size_t blocks_per_row = shape.k / q8_block_size;
  std::vector<BlockQ8> a_q(shape.m * blocks_per_row);
  quantize_q8(a, a_q);
  const std::span<const BlockQ8> a_blocks(a_q);

  for (std::size_t j_block = 0; j_block < shape.n; j_block += BN) {
    const std::size_t j_end = std::min(j_block + BN, shape.n);
    std::size_t i = 0;
    for (; i + 2 <= shape.m; i += 2) {
      const auto a0_row = a_blocks.subspan(i * blocks_per_row, blocks_per_row);
      const auto a1_row = a_blocks.subspan((i + 1) * blocks_per_row, blocks_per_row);
      std::size_t j = j_block;
      for (; j + 4 <= j_end; j += 4) {
        __m256 acc_a0_w0 = _mm256_setzero_ps();
        __m256 acc_a0_w1 = _mm256_setzero_ps();
        __m256 acc_a0_w2 = _mm256_setzero_ps();
        __m256 acc_a0_w3 = _mm256_setzero_ps();
        __m256 acc_a1_w0 = _mm256_setzero_ps();
        __m256 acc_a1_w1 = _mm256_setzero_ps();
        __m256 acc_a1_w2 = _mm256_setzero_ps();
        __m256 acc_a1_w3 = _mm256_setzero_ps();

        const auto w0_row = w.subspan(j * blocks_per_row, blocks_per_row);
        const auto w1_row = w.subspan((j + 1) * blocks_per_row, blocks_per_row);
        const auto w2_row = w.subspan((j + 2) * blocks_per_row, blocks_per_row);
        const auto w3_row = w.subspan((j + 3) * blocks_per_row, blocks_per_row);

        for (std::size_t b = 0; b < blocks_per_row; ++b) {
          const BlockQ8& a0_blk = a0_row[b];
          const __m256i a0_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a0_blk.q.data()));
          const __m256i a0_abs = _mm256_sign_epi8(a0_q, a0_q);

          const BlockQ8& a1_blk = a1_row[b];
          const __m256i a1_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a1_blk.q.data()));
          const __m256i a1_abs = _mm256_sign_epi8(a1_q, a1_q);

          const BlockQ8& w0_blk = w0_row[b];
          const __m256i w0_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0_blk.q.data()));
          acc_a0_w0 =
              accumulate_block(acc_a0_w0, a0_q, a0_abs, w0_q, a0_blk.scale * w0_blk.scale, ones);
          acc_a1_w0 =
              accumulate_block(acc_a1_w0, a1_q, a1_abs, w0_q, a1_blk.scale * w0_blk.scale, ones);

          const BlockQ8& w1_blk = w1_row[b];
          const __m256i w1_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1_blk.q.data()));
          acc_a0_w1 =
              accumulate_block(acc_a0_w1, a0_q, a0_abs, w1_q, a0_blk.scale * w1_blk.scale, ones);
          acc_a1_w1 =
              accumulate_block(acc_a1_w1, a1_q, a1_abs, w1_q, a1_blk.scale * w1_blk.scale, ones);

          const BlockQ8& w2_blk = w2_row[b];
          const __m256i w2_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2_blk.q.data()));
          acc_a0_w2 =
              accumulate_block(acc_a0_w2, a0_q, a0_abs, w2_q, a0_blk.scale * w2_blk.scale, ones);
          acc_a1_w2 =
              accumulate_block(acc_a1_w2, a1_q, a1_abs, w2_q, a1_blk.scale * w2_blk.scale, ones);

          const BlockQ8& w3_blk = w3_row[b];
          const __m256i w3_q =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3_blk.q.data()));
          acc_a0_w3 =
              accumulate_block(acc_a0_w3, a0_q, a0_abs, w3_q, a0_blk.scale * w3_blk.scale, ones);
          acc_a1_w3 =
              accumulate_block(acc_a1_w3, a1_q, a1_abs, w3_q, a1_blk.scale * w3_blk.scale, ones);
        }
        c[(i * shape.n) + j] = hsum256_ps(acc_a0_w0);
        c[(i * shape.n) + j + 1] = hsum256_ps(acc_a0_w1);
        c[(i * shape.n) + j + 2] = hsum256_ps(acc_a0_w2);
        c[(i * shape.n) + j + 3] = hsum256_ps(acc_a0_w3);
        c[((i + 1) * shape.n) + j] = hsum256_ps(acc_a1_w0);
        c[((i + 1) * shape.n) + j + 1] = hsum256_ps(acc_a1_w1);
        c[((i + 1) * shape.n) + j + 2] = hsum256_ps(acc_a1_w2);
        c[((i + 1) * shape.n) + j + 3] = hsum256_ps(acc_a1_w3);
      }
      for (; j < j_end; ++j) {
        const auto w_row = w.subspan(j * blocks_per_row, blocks_per_row);
        c[(i * shape.n) + j] = dot_q8_rows(a0_row, w_row);
        c[((i + 1) * shape.n) + j] = dot_q8_rows(a1_row, w_row);
      }
    }
    if (i < shape.m) {
      const auto a_row = a_blocks.subspan(i * blocks_per_row, blocks_per_row);
      for (std::size_t jj = j_block; jj < j_end; ++jj) {
        const auto w_row = w.subspan(jj * blocks_per_row, blocks_per_row);
        c[(i * shape.n) + jj] = dot_q8_rows(a_row, w_row);
      }
    }
  }
}

}  // namespace q8_avx2_steps

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
