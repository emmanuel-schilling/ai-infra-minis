// The kernels this build and CPU can run, with their names and pending
// switches. Shared by the tests and the accuracy tool so both cover the same
// kernels.
#pragma once

#include <ostream>
#include <vector>

#include "qmm/avx2_steps.hpp"
#include "qmm/matmul.hpp"
#include "qmm/pending.hpp"
#include "qmm/q8_avx2_steps.hpp"

namespace qmm::support {

struct Fp32Case {
  const char* name;
  Fp32Kernel kernel;
  bool pending;
};

struct Q8Case {
  const char* name;
  Q8Kernel kernel;
  bool pending;
};

// Lets GoogleTest print a case by its name, found by argument-dependent lookup.
inline void PrintTo(const Fp32Case& c, std::ostream* os) { *os << c.name; }
inline void PrintTo(const Q8Case& c, std::ostream* os) { *os << c.name; }

inline std::vector<Fp32Case> fp32_kernels() {
  std::vector<Fp32Case> kernels{{"scalar", matmul_scalar, pending::scalar}};
#if defined(QMM_HAVE_AVX2)
  if (cpu_has_avx2_fma()) {
    namespace steps = avx2_steps;
    kernels.push_back({"avx2", matmul_avx2, pending::avx2});
    kernels.push_back({"avx2_v1", steps::v1_one_accumulator, pending::avx2});
    kernels.push_back({"avx2_v2", steps::v2_accumulators, pending::avx2_v2});
    kernels.push_back({"avx2_v3", steps::v3_register_blocking, pending::avx2_v3});
    kernels.push_back({"avx2_v4", steps::v4_cache_blocking, pending::avx2_v4});
  }
#endif
#if defined(QMM_HAVE_ACCELERATE)
  kernels.push_back({"accelerate", matmul_accelerate, false});
#endif
#if defined(QMM_HAVE_OPENBLAS)
  kernels.push_back({"openblas", matmul_openblas, false});
#endif
  return kernels;
}

// Each also depends on quantize_q8.
inline std::vector<Q8Case> q8_kernels() {
  std::vector<Q8Case> kernels{
      {"scalar", matmul_q8_scalar, pending::q8_scalar || pending::quantize}};
#if defined(QMM_HAVE_AVX2)
  if (cpu_has_avx2_fma()) {
    namespace steps = q8_avx2_steps;
    kernels.push_back({"avx2", matmul_q8_avx2, pending::q8_avx2 || pending::quantize});
    kernels.push_back(
        {"avx2_v1", steps::v1_one_accumulator, pending::q8_avx2 || pending::quantize});
    kernels.push_back(
        {"avx2_v2", steps::v2_register_blocking, pending::q8_avx2_v2 || pending::quantize});
  }
#endif
#if defined(QMM_HAVE_NEON_DOTPROD)
  if (cpu_has_neon_dotprod()) {
    kernels.push_back({"neon", matmul_q8_neon, pending::q8_neon || pending::quantize});
  }
#endif
  return kernels;
}

}  // namespace qmm::support
