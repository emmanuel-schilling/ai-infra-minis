# Compiler flags and CPU checks shared by the qmm library and the projects
# that build SIMD code next to it (Quantized matmul's benchmarks). Included by
# each, so both compile with the same settings.

if(MSVC)
  set(qmm_warnings /W4)
  set(qmm_avx2_flags /arch:AVX2)
else()
  set(qmm_warnings -Wall -Wextra -Wpedantic)
  set(qmm_avx2_flags -mavx2 -mfma)
endif()

# PLATFORM: x86-64 — the AVX2 + FMA files are built only here, with AVX2
# enabled for those files alone, so everything else runs on any x86-64 CPU.
# Other CPUs build without them and would need their own SIMD files (NEON on
# Arm, for example). Inline functions from headers compiled into these files
# may use AVX2 too, so this split protects the scalar code only on CPUs that
# have AVX2; every x86-64 machine this repo targets does.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(qmm_x86_64 ON)
else()
  set(qmm_x86_64 OFF)
endif()

# PLATFORM: 64-bit Arm — the NEON Q8 kernel uses the dot-product extension
# (SDOT), so its file is built with that extension enabled; Apple's arm64
# target already includes it. Windows on Arm is untested.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
  set(qmm_arm64 ON)
else()
  set(qmm_arm64 OFF)
endif()
