#include "qmm/matmul.hpp"

namespace qmm {

// To write: the fp32 reference kernel. Name the parameters when you write the
// body, and set pending::scalar to false.
void matmul_scalar(Shape /*shape*/, std::span<const float> /*a*/, std::span<const float> /*w*/,
                   std::span<float> /*c*/) {}

}  // namespace qmm
