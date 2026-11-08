// PLATFORM: 64-bit Arm — NEON with the dot-product extension (SDOT), compiled
// with that extension enabled (see CMakeLists.txt); a stretch goal, tested in
// CI on Arm runners. x86-64 uses src/q8_avx2.cpp instead.
// The build defines QMM_HAVE_NEON_DOTPROD only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_NEON_DOTPROD)

#include "qmm/matmul.hpp"

namespace qmm {

// To write (stretch goal): the NEON Q8 kernel. vdotq_s32 multiplies signed
// int8 by signed int8 without saturating, which removes AVX2's maddubs
// workaround. Set pending::q8_neon to false when you start.
void matmul_q8_neon(Shape /*shape*/, std::span<const float> /*a*/, std::span<const BlockQ8> /*w*/,
                    std::span<float> /*c*/) {}

}  // namespace qmm

#endif  // QMM_HAVE_NEON_DOTPROD
