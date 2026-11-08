#include "qmm/matmul.hpp"

namespace qmm {

// To write: the scalar Q8 kernel, the reference for the AVX2 one. Set
// pending::q8_scalar to false when you start.
void matmul_q8_scalar(Shape /*shape*/, std::span<const float> /*a*/, std::span<const BlockQ8> /*w*/,
                      std::span<float> /*c*/) {}

}  // namespace qmm
