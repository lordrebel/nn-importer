

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::PreprocessOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::PreprocessOp::init(InferenceParameter &p) {
  return success();
}
void front::PreprocessOp::deinit(InferenceParameter &p) {}

LogicalResult front::PreprocessOp::inference(InferenceParameter &p) {
  // front::PreprocessOp no need to inference
  llvm_unreachable("front::PreprocessOp no need to inference");
  return failure();
}

void front::PreprocessOp::shape_inference() {
  common_shape_inference(getOperation());
}
