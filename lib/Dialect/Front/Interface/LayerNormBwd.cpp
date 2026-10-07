

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::LayerNormBwdOp::getFLOPs() { return 0; }

LogicalResult front::LayerNormBwdOp::init(InferenceParameter &p) {
  return success();
}
void front::LayerNormBwdOp::deinit(InferenceParameter &p) {}

LogicalResult front::LayerNormBwdOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::LayerNormBwdOp::shape_inference() {}
