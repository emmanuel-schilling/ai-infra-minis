// Q8 quantization: fp32 values to blocks of 32 int8 values in [-127, 127] with
// one fp32 scale each (amax / 127, or 0 for a block that would divide by
// zero), and back by multiplying each value by its block's scale.
#include <algorithm>
#include <cassert>
#include <cmath>

#include "qmm/matmul.hpp"

namespace qmm {

void quantize_q8(std::span<const float> x, std::span<BlockQ8> out) {
  assert(x.size() == out.size() * q8_block_size);

  const std::size_t num_blocks = x.size() / q8_block_size;

  for (std::size_t b = 0; b < num_blocks; ++b) {
    const float* __restrict block_x = x.data() + b * q8_block_size;
    BlockQ8& block_out = out[b];

    float amax = 0.0F;
    for (std::size_t i = 0; i < q8_block_size; ++i) {
      amax = std::max(amax, std::abs(block_x[i]));
    }

    const float scale = amax / 127.0F;
    block_out.scale = scale;

    for (std::size_t i = 0; i < q8_block_size; ++i) {
      const float q = (scale == 0.0F) ? 0.0F : std::nearbyint(block_x[i] / scale);
      block_out.q[i] = static_cast<std::int8_t>(std::clamp(q, -127.0F, 127.0F));
    }
  }
}

void dequantize_q8(std::span<const BlockQ8> x, std::span<float> out) {
  assert(out.size() == x.size() * q8_block_size);

  const std::size_t num_blocks = x.size();

  for (std::size_t b = 0; b < num_blocks; ++b) {
    const BlockQ8& block_in = x[b];
    const float scale = block_in.scale;
    float* __restrict block_out = out.data() + b * q8_block_size;

    for (std::size_t i = 0; i < q8_block_size; ++i) {
      block_out[i] = static_cast<float>(block_in.q[i]) * scale;
    }
  }
}

}  // namespace qmm
