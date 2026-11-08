// PLATFORM: macOS — Apple's Accelerate framework provides the baseline BLAS;
// Linux and Windows would need another BLAS, such as OpenBLAS, behind the same
// function.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

#include "qmm/matmul.hpp"

namespace qmm {

namespace {

using BlasInt = __LAPACK_int;

BlasInt to_blas_int(std::size_t value) {
  if (value > static_cast<std::size_t>(std::numeric_limits<BlasInt>::max())) {
    throw std::length_error("matrix dimension too large for Accelerate");
  }
  return static_cast<BlasInt>(value);
}

}  // namespace

void matmul_accelerate(Shape shape, std::span<const float> a, std::span<const float> w,
                       std::span<float> c) {
  const BlasInt m = to_blas_int(shape.m);
  const BlasInt n = to_blas_int(shape.n);
  const BlasInt k = to_blas_int(shape.k);
  // C = 1 * A * W^T + 0 * C, all row-major; the leading dimensions are row lengths.
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0F, a.data(), k, w.data(), k,
              0.0F, c.data(), n);
}

}  // namespace qmm
