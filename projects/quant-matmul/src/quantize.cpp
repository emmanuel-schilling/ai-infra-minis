#include "qmm/matmul.hpp"

namespace qmm {

// To write: fp32 to Q8 blocks and back. Set pending::quantize to false when
// you start.
void quantize_q8(std::span<const float> /*x*/, std::span<BlockQ8> /*out*/) {}

void dequantize_q8(std::span<const BlockQ8> /*x*/, std::span<float> /*out*/) {}

}  // namespace qmm
