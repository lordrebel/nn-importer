

#include "Support/Module.h"

int64_t front::NormalizeOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::NormalizeOp::init(InferenceParameter &p) {
  return success();
}
void front::NormalizeOp::deinit(InferenceParameter &p) {}
LogicalResult front::NormalizeOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::NormalizeOp::shape_inference() {
  common_shape_inference(getOperation());
}
