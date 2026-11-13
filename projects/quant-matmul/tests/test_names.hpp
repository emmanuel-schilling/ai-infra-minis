// How the tests name shapes: in GoogleTest's failure messages, and in the
// names of parameterized tests.
#pragma once

#include <ostream>
#include <string>

#include "qmm/matmul.hpp"

namespace qmm {

// Lets GoogleTest print shapes in failure messages, as m x n x k. GoogleTest
// finds it by argument-dependent lookup, so it lives in Shape's namespace.
inline void PrintTo(const Shape& shape, std::ostream* os) {
  *os << shape.m << "x" << shape.n << "x" << shape.k;
}

}  // namespace qmm

namespace qmm::test {

// A shape as part of a test name, such as m3_n5_k7.
inline std::string shape_name(const Shape& s) {
  return "m" + std::to_string(s.m) + "_n" + std::to_string(s.n) + "_k" + std::to_string(s.k);
}

}  // namespace qmm::test
