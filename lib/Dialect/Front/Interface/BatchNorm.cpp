

#include "Support/Module.h"

int64_t front::BatchNormOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::BatchNormOp::init(InferenceParameter &p) {
  return success();
}
void front::BatchNormOp::deinit(InferenceParameter &p) {}

LogicalResult front::BatchNormOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::BatchNormOp::shape_inference() {
  common_shape_inference(getOperation());
}
