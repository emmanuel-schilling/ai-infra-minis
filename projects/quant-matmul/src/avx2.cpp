// PLATFORM: x86-64 — compiled with AVX2 and FMA enabled (see CMakeLists.txt);
// other CPUs would need their own SIMD file.
// The build defines QMM_HAVE_AVX2 only where it compiles this file. Elsewhere the file
// is empty, so tools that read every file, such as clang-tidy, skip it.
#if defined(QMM_HAVE_AVX2)

#include "qmm/matmul.hpp"

namespace qmm {

// To write: the AVX2 + FMA kernel. Set pending::avx2 to false when you start.
void matmul_avx2(Shape /*shape*/, std::span<const float> /*a*/, std::span<const float> /*w*/,
                 std::span<float> /*c*/) {}

}  // namespace qmm

#endif  // QMM_HAVE_AVX2
