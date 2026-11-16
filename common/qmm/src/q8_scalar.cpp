// The scalar Q8 kernel, the reference the other Q8 kernels are tested against.
// Each row of A is quantized once into a reused buffer of k / 32 blocks; each
// block pair's dot product is exact in int32, then scaled by both blocks'
// scales and summed across blocks in fp32.
#include <cassert>
#include <vector>

#include "qmm/matmul.hpp"

namespace qmm {

void matmul_q8_scalar(Shape shape, std::span<const float> a, std::span<const BlockQ8> w,
                      std::span<float> c) {
  assert(a.size() == shape.m * shape.k);
  assert(w.size() == shape.n * shape.k / q8_block_size);
  assert(c.size() == shape.m * shape.n);
  assert(shape.k % q8_block_size == 0);

  const std::size_t blocks_per_row = shape.k / q8_block_size;
  std::vector<BlockQ8> a_row(blocks_per_row);

  for (std::size_t i = 0; i < shape.m; ++i) {
    quantize_q8(a.subspan(i * shape.k, shape.k), a_row);
    for (std::size_t j = 0; j < shape.n; ++j) {
      float sum = 0.0F;
      for (std::size_t b = 0; b < blocks_per_row; ++b) {
        std::int32_t dot = 0;
        for (std::size_t p = 0; p < q8_block_size; ++p) {
          dot += static_cast<std::int32_t>(a_row[b].q[p]) *
                 static_cast<std::int32_t>(w[(j * blocks_per_row) + b].q[p]);
        }
        sum += (a_row[b].scale * w[j * (blocks_per_row) + b].scale) * static_cast<float>(dot);
      }
      c[i * shape.n + j] = sum;
    }
  }
}

}  // namespace qmm
