

#include "Support/Dnnl/LRN.h"

using namespace dnnl;

namespace im {
LRN::LRN() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void LRN::setup() {
  // define a primitive
  auto pd = lrn_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm_, src_mem.get_desc(),
      dst_mem.get_desc(), size_, alpha_, beta_, bias_);
  lrn_prim = lrn_forward(pd);
}

void LRN::run() {
  lrn_prim.execute(engine_stream,
                   {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  engine_stream.wait();
}

} // namespace im
