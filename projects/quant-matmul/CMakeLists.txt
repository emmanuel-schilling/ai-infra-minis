find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

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
# have AVX2; every x86-64 machine this project targets does.
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

add_library(qmm
  src/cpu_features.cpp
  src/q8_scalar.cpp
  src/quantize.cpp
  src/scalar.cpp
)
target_include_directories(qmm PUBLIC include)
target_compile_features(qmm PUBLIC cxx_std_20)
target_compile_options(qmm PRIVATE ${qmm_warnings})

if(qmm_x86_64)
  target_sources(qmm PRIVATE src/avx2.cpp src/q8_avx2.cpp)
  set_source_files_properties(src/avx2.cpp src/q8_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "${qmm_avx2_flags}")
  target_compile_definitions(qmm PUBLIC QMM_HAVE_AVX2=1)
endif()

if(qmm_arm64)
  target_sources(qmm PRIVATE src/q8_neon.cpp)
  if(NOT APPLE AND NOT MSVC)
    set_source_files_properties(src/q8_neon.cpp PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  endif()
  target_compile_definitions(qmm PUBLIC QMM_HAVE_NEON_DOTPROD=1)
endif()

# PLATFORM: macOS — Apple's Accelerate framework is the baseline there.
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_sources(qmm PRIVATE src/accelerate.cpp)
  target_link_libraries(qmm PRIVATE ${ACCELERATE_FRAMEWORK})
  target_compile_definitions(qmm PUBLIC QMM_HAVE_ACCELERATE=1)
endif()

# PLATFORM: Linux and Windows — OpenBLAS is the baseline where Accelerate is
# absent. macOS uses Accelerate alone: both define cblas_sgemm, and linking
# OpenBLAS statically would silently replace Accelerate's.
if(NOT APPLE)
  find_package(OpenBLAS CONFIG REQUIRED)
  target_sources(qmm PRIVATE src/openblas.cpp)
  target_link_libraries(qmm PRIVATE OpenBLAS::OpenBLAS)
  target_compile_definitions(qmm PUBLIC QMM_HAVE_OPENBLAS=1)
endif()

add_executable(qmm_tests
  tests/matmul_test.cpp
  tests/q8_test.cpp
  tests/support_test.cpp
)
target_include_directories(qmm_tests PRIVATE support tests)
target_compile_options(qmm_tests PRIVATE ${qmm_warnings})
target_link_libraries(qmm_tests PRIVATE qmm GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(qmm_tests DISCOVERY_MODE PRE_TEST)

add_executable(qmm_bench
  bench/matmul_bench.cpp
  bench/roofline.cpp
)
if(qmm_x86_64)
  target_sources(qmm_bench PRIVATE bench/roofline_avx2.cpp)
  set_source_files_properties(bench/roofline_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "${qmm_avx2_flags}")
endif()
target_include_directories(qmm_bench PRIVATE support)
target_compile_options(qmm_bench PRIVATE ${qmm_warnings})
target_link_libraries(qmm_bench PRIVATE qmm benchmark::benchmark)

# The accuracy of every kernel against a double-precision reference (see
# bench/accuracy.cpp). The test only checks that it runs, on one token.
add_executable(qmm_accuracy bench/accuracy.cpp)
target_include_directories(qmm_accuracy PRIVATE support)
target_compile_options(qmm_accuracy PRIVATE ${qmm_warnings})
target_link_libraries(qmm_accuracy PRIVATE qmm)
add_test(NAME qmm_accuracy.RunsOnOneToken
  COMMAND qmm_accuracy --tokens 1 --out ${CMAKE_CURRENT_BINARY_DIR}/accuracy_smoke.json)
