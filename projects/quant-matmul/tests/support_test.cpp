// The accuracy tool's helpers: the error metric and the weight file reader.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "error_stats.hpp"
#include "weight_files.hpp"

namespace {

namespace fs = std::filesystem;

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

// A fresh folder for one test, removed afterwards.
class WeightFiles : public testing::Test {
 protected:
  void SetUp() override {
    const auto* info = testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::temp_directory_path() / (std::string("qmm_weight_files_") + info->name());
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  void write_text(const std::string& name, const std::string& text) const {
    std::ofstream(dir_ / name) << text;
  }
  void write_floats(const std::string& name, const std::vector<float>& values) const {
    std::ofstream out(dir_ / name, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  }

  fs::path dir_;
};

TEST_F(WeightFiles, ReadsIndexAndMatrix) {
  write_text("index.txt", "# revision abc\n\nh0 attn_out 2 3 h0_attn_out.f32\n");
  write_floats("h0_attn_out.f32", {1, 2, 3, 4, 5, 6});

  const auto files = qmm::support::read_weight_index(dir_);
  ASSERT_EQ(files.size(), 1U);
  EXPECT_EQ(files[0].source, "h0");
  EXPECT_EQ(files[0].shape, "attn_out");
  EXPECT_EQ(files[0].n, 2U);
  EXPECT_EQ(files[0].k, 3U);
  EXPECT_EQ(qmm::support::read_weights(files[0]), (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST_F(WeightFiles, RejectsWrongSize) {
  write_text("index.txt", "h0 attn_out 2 3 w.f32\n");
  write_floats("w.f32", {1, 2, 3, 4, 5});
  const auto files = qmm::support::read_weight_index(dir_);
  EXPECT_THROW(qmm::support::read_weights(files.at(0)), std::runtime_error);
}

TEST_F(WeightFiles, RejectsMalformedLine) {
  write_text("index.txt", "h0 attn_out two 3 w.f32\n");
  EXPECT_THROW(qmm::support::read_weight_index(dir_), std::runtime_error);
}

TEST_F(WeightFiles, RejectsMissingIndex) {
  EXPECT_THROW(qmm::support::read_weight_index(dir_), std::runtime_error);
}

}  // namespace
