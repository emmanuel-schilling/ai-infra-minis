// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
#include "qmm/matmul.hpp"

namespace qmm {

// To write: the AVX2 Q8 kernel. Set pending::q8_avx2 to false when you start.
void matmul_q8_avx2(Shape /*shape*/, std::span<const float> /*a*/, std::span<const BlockQ8> /*w*/,
                    std::span<float> /*c*/) {}

}  // namespace qmm
