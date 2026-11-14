// Weight matrices saved by scripts/fetch_gpt2.py: one raw fp32 file per
// matrix, already in the kernels' n x k layout, listed in index.txt.
//
// Each index line is `<source> <shape> <n> <k> <file>`, for example
// `h0 attn_qkv 2304 768 h0_attn_qkv.f32`; lines starting with # are comments.
#pragma once

#include <bit>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmm::support {

// The files are written little-endian, the byte order of every platform this
// project builds on.
static_assert(std::endian::native == std::endian::little);

struct WeightFile {
  std::string source;  // where in the model it comes from, such as h0 or wte
  std::string shape;   // the benchmark shape it fills, such as attn_qkv
  std::size_t n;
  std::size_t k;
  std::filesystem::path path;
};

inline std::vector<WeightFile> read_weight_index(const std::filesystem::path& dir) {
  std::ifstream in(dir / "index.txt");
  if (!in) {
    throw std::runtime_error("cannot open " + (dir / "index.txt").string());
  }
  std::vector<WeightFile> files;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line.front() == '#') {
      continue;
    }
    std::istringstream fields(line);
    WeightFile file;
    std::string name;
    if (!(fields >> file.source >> file.shape >> file.n >> file.k >> name)) {
      throw std::runtime_error("malformed line in index.txt: " + line);
    }
    file.path = dir / name;
    files.push_back(file);
  }
  return files;
}

// Reads n * k floats, checking the file holds exactly that many.
inline std::vector<float> read_weights(const WeightFile& file) {
  const std::size_t count = file.n * file.k;
  if (std::filesystem::file_size(file.path) != count * sizeof(float)) {
    throw std::runtime_error(file.path.string() + " doesn't hold " + std::to_string(count) +
                             " floats");
  }
  std::vector<float> values(count);
  std::ifstream in(file.path, std::ios::binary);
  in.read(reinterpret_cast<char*>(values.data()),
          static_cast<std::streamsize>(count * sizeof(float)));
  if (!in) {
    throw std::runtime_error("cannot read " + file.path.string());
  }
  return values;
}

}  // namespace qmm::support
