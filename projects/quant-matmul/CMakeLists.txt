# Quantized matmul's benchmarks and accuracy tool. The kernels themselves, their
# tests and the generic test helpers are the qmm library in common/qmm.

find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

include(${PROJECT_SOURCE_DIR}/common/qmm/QmmPlatform.cmake)

# GPT-2's weight file reader, used by the accuracy tool.
add_executable(quant_matmul_tests tests/weight_files_test.cpp)
target_include_directories(quant_matmul_tests PRIVATE support)
target_compile_options(quant_matmul_tests PRIVATE ${qmm_warnings})
target_link_libraries(quant_matmul_tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(quant_matmul_tests DISCOVERY_MODE PRE_TEST)

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
target_link_libraries(qmm_bench PRIVATE qmm_support benchmark::benchmark)

# The accuracy of every kernel against a double-precision reference (see
# bench/accuracy.cpp). The test only checks that it runs, on one token.
add_executable(qmm_accuracy bench/accuracy.cpp)
target_include_directories(qmm_accuracy PRIVATE support)
target_compile_options(qmm_accuracy PRIVATE ${qmm_warnings})
target_link_libraries(qmm_accuracy PRIVATE qmm_support)
add_test(NAME qmm_accuracy.RunsOnOneToken
  COMMAND qmm_accuracy --tokens 1 --out ${CMAKE_CURRENT_BINARY_DIR}/accuracy_smoke.json)
