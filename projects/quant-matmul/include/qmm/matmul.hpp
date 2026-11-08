// Matrix multiply kernels: one interface, several implementations.
//
// Every kernel computes C = A * W^T with row-major matrices:
//   A is m x k (activations, one row per token),
//   W is n x k (weights, one row per output, as in PyTorch's Linear layers),
//   C is m x n.
// Each output is a dot product along k, the axis that Q8 blocks run along.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace qmm {

struct Shape {
  std::size_t m;
  std::size_t n;
  std::size_t k;
};

// fp32 kernels. Preconditions: a.size() == m * k, w.size() == n * k and
// c.size() == m * n. They overwrite c.
void matmul_scalar(Shape shape, std::span<const float> a, std::span<const float> w,
                   std::span<float> c);

#if defined(QMM_HAVE_AVX2)
// PLATFORM: x86-64 — AVX2 + FMA. Call only when cpu_has_avx2_fma() is true.
void matmul_avx2(Shape shape, std::span<const float> a, std::span<const float> w,
                 std::span<float> c);
#endif

#if defined(QMM_HAVE_ACCELERATE)
// PLATFORM: macOS — Apple's Accelerate BLAS (cblas_sgemm), the baseline.
void matmul_accelerate(Shape shape, std::span<const float> a, std::span<const float> w,
                       std::span<float> c);
#endif

#if defined(QMM_HAVE_OPENBLAS)
// PLATFORM: Linux and Windows — OpenBLAS (cblas_sgemm), the baseline where
// Accelerate is absent. Built single-threaded.
void matmul_openblas(Shape shape, std::span<const float> a, std::span<const float> w,
                     std::span<float> c);
#endif

// Q8: symmetric int8 values in [-127, 127] with one scale per block of 32
// consecutive values, like llama.cpp's Q8_0. A value is scale * q.
inline constexpr std::size_t q8_block_size = 32;

struct BlockQ8 {
  float scale;  // Open question: fp32 or fp16 (see STATUS.md).
  std::array<std::int8_t, q8_block_size> q;
};

// Precondition: x.size() == out.size() * q8_block_size.
void quantize_q8(std::span<const float> x, std::span<BlockQ8> out);

// Precondition: out.size() == x.size() * q8_block_size.
void dequantize_q8(std::span<const BlockQ8> x, std::span<float> out);

// Q8 kernels. W is pre-quantized: n rows of k / 32 blocks each. Each row of A
// is quantized with quantize_q8 inside the call. Each block's dot product is
// computed exactly in integers, then multiplied by the two blocks' scales.
// Preconditions: k % 32 == 0, a.size() == m * k, w.size() == n * k / 32 and
// c.size() == m * n. They overwrite c.
void matmul_q8_scalar(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                      std::span<float> c);

#if defined(QMM_HAVE_AVX2)
// PLATFORM: x86-64 — AVX2 integer dot products. Call only when
// cpu_has_avx2_fma() is true.
void matmul_q8_avx2(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                    std::span<float> c);
#endif

#if defined(QMM_HAVE_NEON_DOTPROD)
// PLATFORM: 64-bit Arm — NEON with the dot-product extension (SDOT); a stretch
// goal. Call only when cpu_has_neon_dotprod() is true.
void matmul_q8_neon(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                    std::span<float> c);
#endif

// True when this CPU and operating system support AVX2 and FMA. Always false
// off x86-64.
bool cpu_has_avx2_fma();

// True when this CPU supports Arm's dot-product extension. Always false off
// 64-bit Arm.
bool cpu_has_neon_dotprod();

}  // namespace qmm
