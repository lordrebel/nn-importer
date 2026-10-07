

#include "Support/Module.h"

int64_t front::ScaleLutOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::ScaleLutOp::init(InferenceParameter &p) {
  return success();
}
void front::ScaleLutOp::deinit(InferenceParameter &p) {}

LogicalResult front::ScaleLutOp::inference(InferenceParameter &p) {
  // front::ScaleLutOp no need to inference
  llvm_unreachable("front::ScaleLutOp no need to inference");
  return failure();
}

void front::ScaleLutOp::shape_inference() {
  common_shape_inference(getOperation());
}
