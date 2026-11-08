cmake_minimum_required(VERSION 4.0)
project(ai_infra_minis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# No C++20 modules here. Without this, CMake adds module-map arguments to the
# compile commands that exist only after a build, which breaks clang-tidy runs
# that only configure.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# MSVC reads source files in the system code page unless told otherwise.
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

option(AIM_SANITIZERS "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(AIM_SANITIZERS)
  # PLATFORM: gcc and clang (Linux, macOS) — ASan and UBSan through
  # -fsanitize; MSVC has ASan only (/fsanitize=address, with its runtime DLL
  # on PATH when tests run) and no UBSan, so it builds without sanitizers.
  if(MSVC)
    message(STATUS "AIM_SANITIZERS: not supported with MSVC; building without sanitizers")
  else()
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                        -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
  endif()
endif()

enable_testing()

add_subdirectory(projects/quant-matmul)
