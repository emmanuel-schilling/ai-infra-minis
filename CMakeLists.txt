cmake_minimum_required(VERSION 3.25)
project(ai_infra_minis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Each C++ project is added here with add_subdirectory once it has code.
