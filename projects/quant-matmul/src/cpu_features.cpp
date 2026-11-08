#include "qmm/matmul.hpp"

// PLATFORM: x86-64 with MSVC, gcc or clang — MSVC reads CPUID and XCR0
// itself, gcc and clang use their built-ins; other CPUs report false and would
// need their own feature checks (getauxval on Arm Linux, sysctl on Arm macOS).
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>

namespace qmm {

bool cpu_has_avx2_fma() {
  constexpr int fma_bit = 1 << 12;               // CPUID leaf 1, ECX
  constexpr int osxsave_bit = 1 << 27;           // CPUID leaf 1, ECX: the OS uses XSAVE
  constexpr int avx_bit = 1 << 28;               // CPUID leaf 1, ECX
  constexpr int avx2_bit = 1 << 5;               // CPUID leaf 7, EBX
  constexpr unsigned long long ymm_state = 0x6;  // XCR0: the OS saves SSE and AVX state

  int regs[4] = {};
  __cpuid(regs, 0);
  if (regs[0] < 7) {
    return false;
  }
  __cpuid(regs, 1);
  const int needed = fma_bit | osxsave_bit | avx_bit;
  if ((regs[2] & needed) != needed || (_xgetbv(0) & ymm_state) != ymm_state) {
    return false;
  }
  __cpuidex(regs, 7, 0);
  return (regs[1] & avx2_bit) != 0;
}

}  // namespace qmm

#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

namespace qmm {

bool cpu_has_avx2_fma() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
}

}  // namespace qmm

#else

namespace qmm {

bool cpu_has_avx2_fma() { return false; }

}  // namespace qmm

#endif
