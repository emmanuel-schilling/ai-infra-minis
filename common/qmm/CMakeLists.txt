# qmm: the matrix-multiply kernels (scalar, AVX2 + FMA, 8-bit quantized, and
# the vendor BLAS baselines), shared by the projects that multiply matrices.
# Built and benchmarked first in projects/quant-matmul.

find_package(GTest CONFIG REQUIRED)

include(QmmPlatform.cmake)

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

# Helpers for testing and measuring the kernels: the double-precision
# reference, the error metric, seeded random inputs and the list of kernels.
add_library(qmm_support INTERFACE)
target_include_directories(qmm_support INTERFACE support)
target_link_libraries(qmm_support INTERFACE qmm)

add_executable(qmm_tests
  tests/error_stats_test.cpp
  tests/matmul_test.cpp
  tests/q8_test.cpp
)
target_include_directories(qmm_tests PRIVATE tests)
target_compile_options(qmm_tests PRIVATE ${qmm_warnings})
target_link_libraries(qmm_tests PRIVATE qmm_support GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(qmm_tests DISCOVERY_MODE PRE_TEST)
