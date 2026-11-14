// GPT-2 (124M)'s weight shapes, shared by the benchmarks and the accuracy tool.
#pragma once

#include <cstddef>

namespace qmm::support {

struct WeightShape {
  const char* name;
  std::size_t n;
  std::size_t k;
};

// The attention input projection (query, key and value together), the
// attention output projection, the MLP up and down projections, and the output
// layer over the 50,257-token vocabulary.
inline constexpr WeightShape gpt2_weight_shapes[] = {
    {"attn_qkv", 2304, 768}, {"attn_out", 768, 768},  {"mlp_up", 3072, 768},
    {"mlp_down", 768, 3072}, {"lm_head", 50257, 768},
};

}  // namespace qmm::support
