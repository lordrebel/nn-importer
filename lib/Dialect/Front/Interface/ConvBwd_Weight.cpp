#include "Dialect/Front/IR/Front.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

int64_t front::ConvBwdWeightOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::ConvBwdWeightOp::init(InferenceParameter &p) {
  return success();
}
void front::ConvBwdWeightOp::deinit(InferenceParameter &p) {}

LogicalResult front::ConvBwdWeightOp::inference(InferenceParameter &p) {
  // UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::ConvBwdWeightOp::shape_inference() {}
