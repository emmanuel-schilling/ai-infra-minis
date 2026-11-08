// Test-driven development switches, one per kernel.
//
// While a kernel's switch is true, its tests pass without checking anything
// and the benchmarks leave it out. To start a kernel:
//   1. Set its switch to false. Its tests now fail.
//   2. Write the kernel until they pass.
#pragma once

namespace qmm::pending {

inline constexpr bool scalar = true;
inline constexpr bool avx2 = true;
inline constexpr bool quantize = true;  // quantize_q8 and dequantize_q8
inline constexpr bool q8_scalar = true;
inline constexpr bool q8_avx2 = true;
inline constexpr bool q8_neon = true;  // stretch goal

}  // namespace qmm::pending
