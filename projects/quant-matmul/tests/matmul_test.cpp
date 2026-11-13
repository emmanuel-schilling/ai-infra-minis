// fp32 kernels against a double-precision reference.
#include "qmm/matmul.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "qmm/avx2_steps.hpp"
#include "qmm/pending.hpp"
#include "random_floats.hpp"
#include "reference.hpp"
#include "tdd.hpp"
#include "test_names.hpp"

namespace {

using qmm::Fp32Kernel;
using qmm::Shape;

struct Fp32Case {
  const char* name;
  Fp32Kernel kernel;
  bool pending;
};

void PrintTo(const Fp32Case& c, std::ostream* os) { *os << c.name; }

// The kernels this build and CPU can run.
std::vector<Fp32Case> fp32_kernels() {
  std::vector<Fp32Case> kernels{{"scalar", qmm::matmul_scalar, qmm::pending::scalar}};
#if defined(QMM_HAVE_AVX2)
  if (qmm::cpu_has_avx2_fma()) {
    namespace steps = qmm::avx2_steps;
    namespace pending = qmm::pending;
    kernels.push_back({"avx2", qmm::matmul_avx2, pending::avx2});
    kernels.push_back({"avx2_v1", steps::v1_one_accumulator, pending::avx2});
    kernels.push_back({"avx2_v2", steps::v2_accumulators, pending::avx2_v2});
    kernels.push_back({"avx2_v3", steps::v3_register_blocking, pending::avx2_v3});
    kernels.push_back({"avx2_v4", steps::v4_cache_blocking, pending::avx2_v4});
  }
#endif
#if defined(QMM_HAVE_ACCELERATE)
  kernels.push_back({"accelerate", qmm::matmul_accelerate, false});
#endif
#if defined(QMM_HAVE_OPENBLAS)
  kernels.push_back({"openblas", qmm::matmul_openblas, false});
#endif
  return kernels;
}

// Odd sizes that catch tail handling, then GPT-2's weight shapes with a few
// tokens (the benchmarks use the full token counts). {7, 131, 72} exercises the
// blocked AVX2 versions' edges together: leftover rows of A after 3-row tiles,
// more than two 64-row blocks of W with leftover rows not a multiple of 4, and
// a k that 8 accumulators of 8 floats don't divide. The first three are empty:
// with k = 0 every output is an empty sum, exactly 0, and with m = 0 or n = 0
// there is no output to write.
std::vector<Shape> shapes() {
  return {{3, 5, 0},      {0, 5, 7},     {3, 0, 7},      {1, 1, 1},      {1, 7, 9},
          {3, 5, 7},      {7, 9, 33},    {33, 7, 1},     {2, 33, 64},    {7, 131, 72},
          {4, 2304, 768}, {4, 768, 768}, {4, 3072, 768}, {4, 768, 3072}, {1, 50257, 768}};
}

class Fp32Accuracy : public testing::TestWithParam<std::tuple<Fp32Case, Shape>> {};

TEST_P(Fp32Accuracy, MatchesDoublePrecisionReference) {
  const auto& [kernel_case, shape] = GetParam();
  QMM_PASS_WHILE_PENDING(kernel_case.pending);

  const auto a = qmm::support::random_floats(shape.m * shape.k, 1);
  const auto w = qmm::support::random_floats(shape.n * shape.k, 2);
  // NaN everywhere, so any output the kernel leaves unwritten fails.
  std::vector<float> c(shape.m * shape.n, std::nanf(""));
  kernel_case.kernel(shape, a, w, c);

  const auto ref = qmm::test::reference_matmul(shape, a, w);
  const double g = qmm::test::gamma(shape.k);
  std::size_t failures = 0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const double error = std::abs(static_cast<double>(c[i]) - ref.c[i]);
    const double bound = g * ref.abs_sum[i];
    if (!(error <= bound) && ++failures <= 5) {
      ADD_FAILURE() << "C[" << i / shape.n << "][" << i % shape.n << "] = " << c[i]
                    << ", reference " << ref.c[i] << ", error " << error << " > bound " << bound;
    }
  }
  EXPECT_EQ(failures, 0U) << "elements outside the fp32 error bound";
}

INSTANTIATE_TEST_SUITE_P(Kernels, Fp32Accuracy,
                         testing::Combine(testing::ValuesIn(fp32_kernels()),
                                          testing::ValuesIn(shapes())),
                         [](const testing::TestParamInfo<Fp32Accuracy::ParamType>& info) {
                           return std::string(std::get<0>(info.param).name) + "_" +
                                  qmm::test::shape_name(std::get<1>(info.param));
                         });

class Fp32Exact : public testing::TestWithParam<Fp32Case> {};

// Multiplying by an identity matrix involves only products with 1 and sums
// with 0, so every kernel must reproduce A exactly.
TEST_P(Fp32Exact, IdentityWeightsReturnA) {
  QMM_PASS_WHILE_PENDING(GetParam().pending);
  const Shape shape{5, 9, 9};
  const auto a = qmm::support::random_floats(shape.m * shape.k, 3);
  std::vector<float> w(shape.n * shape.k, 0.0F);
  for (std::size_t i = 0; i < shape.n; ++i) {
    w[(i * shape.k) + i] = 1.0F;
  }
  std::vector<float> c(shape.m * shape.n, std::nanf(""));
  GetParam().kernel(shape, a, w, c);
  for (std::size_t i = 0; i < c.size(); ++i) {
    EXPECT_EQ(c[i], a[i]) << "at " << i;
  }
}

TEST_P(Fp32Exact, ZeroWeightsGiveZero) {
  QMM_PASS_WHILE_PENDING(GetParam().pending);
  const Shape shape{3, 17, 40};
  const auto a = qmm::support::random_floats(shape.m * shape.k, 4);
  const std::vector<float> w(shape.n * shape.k, 0.0F);
  std::vector<float> c(shape.m * shape.n, std::nanf(""));
  GetParam().kernel(shape, a, w, c);
  for (std::size_t i = 0; i < c.size(); ++i) {
    EXPECT_EQ(c[i], 0.0F) << "at " << i;
  }
}

INSTANTIATE_TEST_SUITE_P(Kernels, Fp32Exact, testing::ValuesIn(fp32_kernels()),
                         [](const testing::TestParamInfo<Fp32Case>& info) {
                           return std::string(info.param.name);
                         });

}  // namespace
