#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "qmm/matmul.hpp"

namespace qmm::test {

// C = A * W^T in double precision, with, for each element of C, the sum of
// |a_i * w_i| that scales its fp32 error bound. Products of two floats are
// exact in double, so the reference's own error is negligible.
struct Reference {
  std::vector<double> c;
  std::vector<double> abs_sum;
};

inline Reference reference_matmul(Shape shape, std::span<const float> a, std::span<const float> w) {
  Reference ref{std::vector<double>(shape.m * shape.n), std::vector<double>(shape.m * shape.n)};
  for (std::size_t i = 0; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      double sum = 0.0;
      double abs_sum = 0.0;
      for (std::size_t p = 0; p < shape.k; ++p) {
        const double product =
            static_cast<double>(a[(i * shape.k) + p]) * static_cast<double>(w[(j * shape.k) + p]);
        sum += product;
        abs_sum += std::abs(product);
      }
      ref.c[(i * shape.n) + j] = sum;
      ref.abs_sum[(i * shape.n) + j] = abs_sum;
    }
  }
  return ref;
}

// gamma(k) = k * u / (1 - k * u), where u = 2^-24 is fp32's unit roundoff.
// Any fp32 evaluation of a k-term dot product, in any order and with or
// without FMA, is within gamma(k) * sum |a_i * w_i| of the exact result
// (Higham, Accuracy and Stability of Numerical Algorithms, section 3.1).
inline double gamma(std::size_t k) {
  const double ku = static_cast<double>(k) * 0x1.0p-24;
  return ku / (1.0 - ku);
}

}  // namespace qmm::test
