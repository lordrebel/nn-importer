

#pragma once

#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;

namespace im {

typedef struct {
  uint64_t axis;
  std::vector<memory::dims> src_shapes;
  memory::dims dst_shape;
  uint64_t num_src;
} concat_attr_t;

class Concat {
public:
  Concat();
  void setup(std::vector<float *> inputs, float *output, concat_attr_t &attr);
  void run();
  ~Concat() = default;

private:
  engine eng;
  stream eng_stream;
  primitive concat_prim;
  std::unordered_map<int, memory> concat_args;
  std::vector<float *> p_inputs;
  float *p_output;
  concat_attr_t attr_;
};

} // namespace im
