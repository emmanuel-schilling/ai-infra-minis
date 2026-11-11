// PLATFORM: macOS — Apple's Accelerate framework provides the baseline BLAS;
// Linux and Windows would need another BLAS, such as OpenBLAS, behind the same
// function.
// The build defines QMM_HAVE_ACCELERATE only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_ACCELERATE)

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
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
  // BLAS rejects a leading dimension below 1, which an empty C (m or n = 0) or
  // k = 0 would pass, so those cases are handled here: k = 0 makes every
  // output an empty sum, 0.
  if (c.empty()) {
    return;
  }
  if (shape.k == 0) {
    std::fill(c.begin(), c.end(), 0.0F);
    return;
  }
  const BlasInt m = to_blas_int(shape.m);
  const BlasInt n = to_blas_int(shape.n);
  const BlasInt k = to_blas_int(shape.k);
  // C = 1 * A * W^T + 0 * C, all row-major; the leading dimensions are row lengths.
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0F, a.data(), k, w.data(), k,
              0.0F, c.data(), n);
}

}  // namespace qmm

#endif  // QMM_HAVE_ACCELERATE
