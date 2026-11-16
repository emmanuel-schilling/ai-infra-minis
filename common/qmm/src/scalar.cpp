// The reference kernel: a plain triple loop, deliberately unoptimized, that
// every other kernel is tested against. It accumulates in fp32 along k in
// order; the tests allow for that rounding (see tests/reference.hpp).
#include <cassert>

#include "qmm/matmul.hpp"

namespace qmm {

void matmul_scalar(Shape shape, std::span<const float> a, std::span<const float> w,
                   std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k);
  assert(c.size() == shape.m * shape.n);

  for (std::size_t i = 0; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      float sum = 0.0F;
      for (std::size_t p = 0; p < shape.k; ++p) {
        sum += a[i * shape.k + p] * w[j * shape.k + p];
      }
      c[i * shape.n + j] = sum;
    }
  }
}

}  // namespace qmm
