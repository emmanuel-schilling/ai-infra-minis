// The accuracy of every kernel on GPT-2's weight shapes: the relative and
// maximum absolute error of C against a double-precision product of the same
// fp32 A and W. Q8 kernels quantize W first, so their error includes
// quantization's. Writes one JSON object per kernel and matrix.
//
// Two kinds of W: uniform random values in [-1, 1), as in the benchmarks, and,
// with --weights, GPT-2's own matrices saved by scripts/fetch_gpt2.py. A is
// uniform random in both cases: real activations need the model running.
//
// Usage: qmm_accuracy [--tokens N] [--weights DIR] [--out FILE]
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "error_stats.hpp"
#include "gpt2_shapes.hpp"
#include "kernel_list.hpp"
#include "qmm/matmul.hpp"
#include "random_floats.hpp"
#include "reference.hpp"
#include "weight_files.hpp"

namespace {

using qmm::BlockQ8;
using qmm::Shape;

// The same seeds as the benchmarks, so the uniform W matches theirs.
constexpr std::uint32_t a_seed = 1;
constexpr std::uint32_t w_seed = 2;

struct Options {
  std::size_t tokens = 128;
  std::optional<std::filesystem::path> weights;
  std::optional<std::filesystem::path> out;
};

Options parse_options(std::span<char*> args) {
  Options options;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (i + 1 >= args.size()) {
      throw std::invalid_argument("missing value after " + std::string(arg));
    }
    const std::string value = args[++i];
    if (arg == "--tokens") {
      options.tokens = std::stoul(value);
    } else if (arg == "--weights") {
      options.weights = value;
    } else if (arg == "--out") {
      options.out = value;
    } else {
      throw std::invalid_argument("unknown option " + std::string(arg));
    }
  }
  return options;
}

// One matrix of W to measure every kernel on.
struct Matrix {
  std::string data;    // uniform or gpt2
  std::string source;  // random, or where in GPT-2 it comes from
  std::string shape;
  Shape dims;
  std::vector<float> w;
};

double max_abs(std::span<const double> values) {
  double result = 0.0;
  for (const double v : values) {
    result = std::max(result, std::abs(v));
  }
  return result;
}

void write_row(std::ostream& out, bool& first, const Matrix& matrix, const std::string& kernel,
               qmm::support::ErrorStats errors, double ref_max_abs) {
  out << (first ? "[\n" : ",\n");
  first = false;
  out << R"(  {"data": ")" << matrix.data << R"(", "source": ")" << matrix.source
      << R"(", "shape": ")" << matrix.shape << R"(", "m": )" << matrix.dims.m << R"(, "n": )"
      << matrix.dims.n << R"(, "k": )" << matrix.dims.k << R"(, "kernel": ")" << kernel
      << R"(", "relative_error": )" << errors.relative << R"(, "max_abs_error": )" << errors.max_abs
      << R"(, "ref_max_abs": )" << ref_max_abs << "}";
}

void measure(std::ostream& out, bool& first, const Matrix& matrix) {
  const Shape shape = matrix.dims;
  const auto a = qmm::support::random_floats(shape.m * shape.k, a_seed);
  const auto ref = qmm::support::reference_matmul(shape, a, matrix.w);
  const double ref_max_abs = max_abs(ref.c);
  std::vector<float> c(shape.m * shape.n);

  for (const auto& kernel_case : qmm::support::fp32_kernels()) {
    if (kernel_case.pending) {
      continue;
    }
    kernel_case.kernel(shape, a, matrix.w, c);
    write_row(out, first, matrix, std::string("fp32/") + kernel_case.name,
              qmm::support::error_stats(c, ref.c), ref_max_abs);
  }

  std::vector<BlockQ8> w_q(matrix.w.size() / qmm::q8_block_size);
  qmm::quantize_q8(matrix.w, w_q);
  for (const auto& kernel_case : qmm::support::q8_kernels()) {
    if (kernel_case.pending) {
      continue;
    }
    kernel_case.kernel(shape, a, w_q, c);
    write_row(out, first, matrix, std::string("q8/") + kernel_case.name,
              qmm::support::error_stats(c, ref.c), ref_max_abs);
  }
  out.flush();
  std::fprintf(stderr, "qmm_accuracy: %s %s %s done\n", matrix.data.c_str(), matrix.source.c_str(),
               matrix.shape.c_str());
}

void run(const Options& options, std::ostream& out) {
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  bool first = true;
  for (const auto& ws : qmm::support::gpt2_weight_shapes) {
    const Shape shape{options.tokens, ws.n, ws.k};
    measure(
        out, first,
        {"uniform", "random", ws.name, shape, qmm::support::random_floats(ws.n * ws.k, w_seed)});
  }
  if (options.weights) {
    for (const auto& file : qmm::support::read_weight_index(*options.weights)) {
      const Shape shape{options.tokens, file.n, file.k};
      measure(out, first,
              {"gpt2", file.source, file.shape, shape, qmm::support::read_weights(file)});
    }
  }
  out << (first ? "[]\n" : "\n]\n");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(std::span(argv, static_cast<std::size_t>(argc)));
    if (options.out) {
      std::ofstream out(*options.out);
      if (!out) {
        throw std::runtime_error("cannot write " + options.out->string());
      }
      run(options, out);
    } else {
      run(options, std::cout);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "qmm_accuracy: %s\n", e.what());
  } catch (...) {
    std::fprintf(stderr, "qmm_accuracy: unknown exception\n");
  }
  return 1;
}
