

#include "Support/Module.h"

int64_t front::SoftmaxBwdOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::SoftmaxBwdOp::init(InferenceParameter &p) {
  return success();
}
void front::SoftmaxBwdOp::deinit(InferenceParameter &p) {}

LogicalResult front::SoftmaxBwdOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::SoftmaxBwdOp::shape_inference() {}
