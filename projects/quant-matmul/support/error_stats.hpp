// How far a kernel's output is from the double-precision reference.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

namespace qmm::support {

struct ErrorStats {
  double relative;  // ||C - C_ref|| / ||C_ref||, Frobenius norms
  double max_abs;   // max |C - C_ref| over all elements
};

// Precondition: c.size() == ref.size(). A zero reference with zero error gives
// a relative error of 0.
inline ErrorStats error_stats(std::span<const float> c, std::span<const double> ref) {
  assert(c.size() == ref.size());
  double error_sq = 0.0;
  double ref_sq = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const double error = static_cast<double>(c[i]) - ref[i];
    error_sq += error * error;
    ref_sq += ref[i] * ref[i];
    max_abs = std::max(max_abs, std::abs(error));
  }
  const double relative = (error_sq == 0.0) ? 0.0 : std::sqrt(error_sq / ref_sq);
  return {relative, max_abs};
}

}  // namespace qmm::support
