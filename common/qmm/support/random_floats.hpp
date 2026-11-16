#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace qmm::support {

// Uniform floats in [-1, 1), the same on every platform: they are built from
// std::mt19937's raw output, which the C++ standard specifies exactly, while
// std::uniform_real_distribution may differ between standard libraries.
inline std::vector<float> random_floats(std::size_t count, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::vector<float> values(count);
  for (float& value : values) {
    // The top 24 bits give an integer below 2^24, exactly representable.
    value = static_cast<float>(generator() >> 8U) * 0x1.0p-23F - 1.0F;
  }
  return values;
}

}  // namespace qmm::support
