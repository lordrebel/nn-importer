

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::RandnLikeOp::getFLOPs() { return 0; }

LogicalResult front::RandnLikeOp::init(InferenceParameter &p) {
  return success();
}
void front::RandnLikeOp::deinit(InferenceParameter &p) {}

LogicalResult front::RandnLikeOp::inference(InferenceParameter &p) {
  llvm_unreachable("Should be convert to other ops in it's canonicalize pass.");
  return success();
}

void front::RandnLikeOp::shape_inference() {
  common_shape_inference(getOperation());
}
