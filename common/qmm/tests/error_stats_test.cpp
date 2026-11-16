// The error metric the kernels are judged by against the double reference.
#include "error_stats.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

TEST(ErrorStats, ExactResultHasNoError) {
  const std::vector<float> c{1.0F, -2.0F, 3.0F};
  const std::vector<double> ref{1.0, -2.0, 3.0};
  const auto errors = qmm::support::error_stats(c, ref);
  EXPECT_EQ(errors.relative, 0.0);
  EXPECT_EQ(errors.max_abs, 0.0);
}

TEST(ErrorStats, MatchesHandComputedValues) {
  // Errors 0, 0.5 and -1 against a reference of norm sqrt(1 + 4 + 4) = 3.
  const std::vector<float> c{1.0F, 2.5F, 1.0F};
  const std::vector<double> ref{1.0, 2.0, 2.0};
  const auto errors = qmm::support::error_stats(c, ref);
  EXPECT_DOUBLE_EQ(errors.relative, std::sqrt(1.25) / 3.0);
  EXPECT_DOUBLE_EQ(errors.max_abs, 1.0);
}

TEST(ErrorStats, AllZeroIsExact) {
  const std::vector<float> c(4, 0.0F);
  const std::vector<double> ref(4, 0.0);
  EXPECT_EQ(qmm::support::error_stats(c, ref).relative, 0.0);
}

}  // namespace
