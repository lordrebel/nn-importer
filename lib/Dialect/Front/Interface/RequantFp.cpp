

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::RequantFpOp::getFLOPs() { return 0; }
LogicalResult front::RequantFpOp::init(InferenceParameter &p) {
  return success();
}
void front::RequantFpOp::deinit(InferenceParameter &p) {}

LogicalResult front::RequantFpOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::RequantFpOp::shape_inference() {
  common_shape_inference(getOperation());
}
