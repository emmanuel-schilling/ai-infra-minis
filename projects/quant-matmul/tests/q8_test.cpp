// Q8 quantization and the Q8 kernels.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "qmm/matmul.hpp"
#include "qmm/pending.hpp"
#include "random_floats.hpp"
#include "reference.hpp"
#include "tdd.hpp"

namespace {

using qmm::BlockQ8;
using qmm::q8_block_size;
using qmm::Shape;

std::vector<BlockQ8> quantize(const std::vector<float>& x) {
  std::vector<BlockQ8> blocks(x.size() / q8_block_size);
  qmm::quantize_q8(x, blocks);
  return blocks;
}

// --- quantize_q8 and dequantize_q8 ------------------------------------------

TEST(QuantizeQ8, ValuesAreWithinHalfAStep) {
  QMM_PASS_WHILE_PENDING(qmm::pending::quantize);
  auto x = qmm::support::random_floats(64 * q8_block_size, 5);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::size_t block = i / q8_block_size;
    x[i] *= static_cast<float>(block + 1);  // a different range per block
  }
  const auto blocks = quantize(x);

  for (std::size_t b = 0; b < blocks.size(); ++b) {
    const auto first = x.begin() + static_cast<std::ptrdiff_t>(b * q8_block_size);
    const float amax = std::abs(*std::max_element(
        first, first + q8_block_size, [](float l, float r) { return std::abs(l) < std::abs(r); }));
    const float scale = blocks[b].scale;
    // An fp16 scale would need a looser tolerance here.
    EXPECT_NEAR(scale, amax / 127.0F, 1e-6F * amax) << "block " << b;

    bool reaches_127 = false;
    for (std::size_t i = 0; i < q8_block_size; ++i) {
      const std::int8_t q = blocks[b].q[i];
      EXPECT_LE(std::abs(q), 127) << "block " << b << " value " << i;
      reaches_127 = reaches_127 || std::abs(q) == 127;
      const float error = std::abs(x[(b * q8_block_size) + i] - (scale * static_cast<float>(q)));
      EXPECT_LE(error, 0.5F * scale * (1.0F + 1e-5F)) << "block " << b << " value " << i;
    }
    EXPECT_TRUE(reaches_127) << "block " << b << ": its largest value must map to +/-127";
  }
}

TEST(QuantizeQ8, ZeroBlockHasZeroScale) {
  QMM_PASS_WHILE_PENDING(qmm::pending::quantize);
  const auto blocks = quantize(std::vector<float>(q8_block_size, 0.0F));
  EXPECT_EQ(blocks[0].scale, 0.0F);
  for (const std::int8_t q : blocks[0].q) {
    EXPECT_EQ(q, 0);
  }
  std::vector<float> back(q8_block_size, std::nanf(""));
  qmm::dequantize_q8(blocks, back);
  for (const float v : back) {
    EXPECT_EQ(v, 0.0F);  // not NaN from dividing by a zero scale
  }
}

TEST(QuantizeQ8, DequantizeMultipliesByScale) {
  QMM_PASS_WHILE_PENDING(qmm::pending::quantize);
  BlockQ8 block{};
  block.scale = 0.5F;
  for (std::size_t i = 0; i < q8_block_size; ++i) {
    block.q[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
  }
  std::vector<float> out(q8_block_size, std::nanf(""));
  qmm::dequantize_q8(std::span<const BlockQ8>(&block, 1), out);
  for (std::size_t i = 0; i < q8_block_size; ++i) {
    EXPECT_EQ(out[i], 0.5F * static_cast<float>(static_cast<int>(i) - 16)) << "at " << i;
  }
}

// --- Q8 kernels ---------------------------------------------------------------

using Q8Kernel = void (*)(Shape, std::span<const float>, std::span<const BlockQ8>,
                          std::span<float>);

struct Q8Case {
  const char* name;
  Q8Kernel kernel;
  bool pending;
};

void PrintTo(const Q8Case& c, std::ostream* os) { *os << c.name; }

// The kernels this build and CPU can run. Each also depends on quantize_q8.
std::vector<Q8Case> q8_kernels() {
  std::vector<Q8Case> kernels{
      {"scalar", qmm::matmul_q8_scalar, qmm::pending::q8_scalar || qmm::pending::quantize}};
#if defined(QMM_HAVE_AVX2)
  if (qmm::cpu_has_avx2_fma()) {
    kernels.push_back(
        {"avx2", qmm::matmul_q8_avx2, qmm::pending::q8_avx2 || qmm::pending::quantize});
  }
#endif
#if defined(QMM_HAVE_NEON_DOTPROD)
  if (qmm::cpu_has_neon_dotprod()) {
    kernels.push_back(
        {"neon", qmm::matmul_q8_neon, qmm::pending::q8_neon || qmm::pending::quantize});
  }
#endif
  return kernels;
}

// k is a multiple of 32 throughout.
std::vector<Shape> q8_shapes() {
  return {{1, 1, 32},    {3, 7, 64},     {5, 9, 96},     {4, 2304, 768},
          {4, 768, 768}, {4, 3072, 768}, {4, 768, 3072}, {1, 50257, 768}};
}

std::string shape_name(const Shape& s) {
  return "m" + std::to_string(s.m) + "_n" + std::to_string(s.n) + "_k" + std::to_string(s.k);
}

// The exact result of the Q8 computation: sum over blocks of
// scale_a * scale_w * (integer dot product), in double, plus the sum of the
// magnitudes of those terms for the error bound.
qmm::test::Reference q8_reference(Shape shape, const std::vector<BlockQ8>& a,
                                  const std::vector<BlockQ8>& w) {
  const std::size_t blocks_per_row = shape.k / q8_block_size;
  qmm::test::Reference ref{std::vector<double>(shape.m * shape.n),
                           std::vector<double>(shape.m * shape.n)};
  for (std::size_t i = 0; i < shape.m; ++i) {
    for (std::size_t j = 0; j < shape.n; ++j) {
      double sum = 0.0;
      double abs_sum = 0.0;
      for (std::size_t b = 0; b < blocks_per_row; ++b) {
        const BlockQ8& ab = a[(i * blocks_per_row) + b];
        const BlockQ8& wb = w[(j * blocks_per_row) + b];
        std::int32_t dot = 0;
        for (std::size_t p = 0; p < q8_block_size; ++p) {
          dot += static_cast<std::int32_t>(ab.q[p]) * static_cast<std::int32_t>(wb.q[p]);
        }
        const double term = static_cast<double>(ab.scale) * static_cast<double>(wb.scale) *
                            static_cast<double>(dot);
        sum += term;
        abs_sum += std::abs(term);
      }
      ref.c[(i * shape.n) + j] = sum;
      ref.abs_sum[(i * shape.n) + j] = abs_sum;
    }
  }
  return ref;
}

std::vector<float> run(const Q8Case& kernel_case, Shape shape, const std::vector<float>& a,
                       const std::vector<BlockQ8>& w) {
  std::vector<float> c(shape.m * shape.n, std::nanf(""));
  kernel_case.kernel(shape, a, w, c);
  return c;
}

class Q8Kernels : public testing::TestWithParam<std::tuple<Q8Case, Shape>> {};

// Given the same quantized inputs, a kernel's only error is fp32 rounding in
// scaling and summing the blocks: one multiply for scale_a * scale_w, one for
// the integer dot product (exact in fp32, since |dot| <= 32 * 127^2 < 2^24),
// and k / 32 additions.
TEST_P(Q8Kernels, MatchesExactQ8Result) {
  const auto& [kernel_case, shape] = GetParam();
  QMM_PASS_WHILE_PENDING(kernel_case.pending);

  const auto a = qmm::support::random_floats(shape.m * shape.k, 6);
  const auto w = quantize(qmm::support::random_floats(shape.n * shape.k, 7));
  const auto c = run(kernel_case, shape, a, w);

  const auto ref = q8_reference(shape, quantize(a), w);
  const double g = qmm::test::gamma((shape.k / q8_block_size) + 2);
  std::size_t failures = 0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const double error = std::abs(static_cast<double>(c[i]) - ref.c[i]);
    const double bound = g * ref.abs_sum[i];
    if (!(error <= bound) && ++failures <= 5) {
      ADD_FAILURE() << "C[" << i / shape.n << "][" << i % shape.n << "] = " << c[i]
                    << ", exact Q8 result " << ref.c[i] << ", error " << error << " > bound "
                    << bound;
    }
  }
  EXPECT_EQ(failures, 0U) << "elements outside the rounding bound";
}

// Quantization error against the fp32 product, which is the cost of Q8. The
// measured values are recorded in the test report.
TEST_P(Q8Kernels, StaysCloseToFp32) {
  const auto& [kernel_case, shape] = GetParam();
  QMM_PASS_WHILE_PENDING(kernel_case.pending);

  const auto a = qmm::support::random_floats(shape.m * shape.k, 8);
  const auto w_fp32 = qmm::support::random_floats(shape.n * shape.k, 9);
  const auto c = run(kernel_case, shape, a, quantize(w_fp32));
  const auto ref = qmm::test::reference_matmul(shape, a, w_fp32);

  double error_sq = 0.0;
  double ref_sq = 0.0;
  double max_abs_error = 0.0;
  for (std::size_t i = 0; i < c.size(); ++i) {
    const double error = static_cast<double>(c[i]) - ref.c[i];
    error_sq += error * error;
    ref_sq += ref.c[i] * ref.c[i];
    max_abs_error = std::max(max_abs_error, std::abs(error));
  }
  const double relative_error = std::sqrt(error_sq / ref_sq);
  RecordProperty("relative_error", std::to_string(relative_error));
  RecordProperty("max_abs_error", std::to_string(max_abs_error));
  EXPECT_LE(relative_error, 2e-2);
}

INSTANTIATE_TEST_SUITE_P(Kernels, Q8Kernels,
                         testing::Combine(testing::ValuesIn(q8_kernels()),
                                          testing::ValuesIn(q8_shapes())),
                         [](const testing::TestParamInfo<Q8Kernels::ParamType>& info) {
                           return std::string(std::get<0>(info.param).name) + "_" +
                                  shape_name(std::get<1>(info.param));
                         });

class Q8Signs : public testing::TestWithParam<Q8Case> {};

// Every value is +/-1, so every q is +/-127 and each block's integer dot
// product reaches the largest magnitude, 32 * 127^2. This catches the AVX2
// pitfalls: _mm256_maddubs_epi16 treats one operand as unsigned and saturates
// pairs of products at 16 bits.
TEST_P(Q8Signs, ExtremeValuesKeepTheirSigns) {
  QMM_PASS_WHILE_PENDING(GetParam().pending);
  const Shape shape{1, 1, 64};
  const auto filled = [&](auto value_at) {
    std::vector<float> v(shape.k);
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] = value_at(i);
    }
    return v;
  };
  const auto plus = filled([](std::size_t) { return 1.0F; });
  const auto minus = filled([](std::size_t) { return -1.0F; });
  const auto alternating = filled([](std::size_t i) { return i % 2 == 0 ? 1.0F : -1.0F; });
  const auto expected_k = static_cast<float>(shape.k);

  struct Case {
    const char* name;
    const std::vector<float>* a;
    const std::vector<float>* w;
    float expected;
  };
  const Case cases[] = {
      {"+ * +", &plus, &plus, expected_k},
      {"+ * -", &plus, &minus, -expected_k},
      {"- * +", &minus, &plus, -expected_k},
      {"- * -", &minus, &minus, expected_k},
      {"alternating * alternating", &alternating, &alternating, expected_k},
      {"alternating * +", &alternating, &plus, 0.0F},
  };
  for (const Case& c : cases) {
    const auto out = run(GetParam(), shape, *c.a, quantize(*c.w));
    EXPECT_NEAR(out[0], c.expected, 1e-4F * expected_k) << c.name;
  }
}

INSTANTIATE_TEST_SUITE_P(Kernels, Q8Signs, testing::ValuesIn(q8_kernels()),
                         [](const testing::TestParamInfo<Q8Case>& info) {
                           return std::string(info.param.name);
                         });

}  // namespace
