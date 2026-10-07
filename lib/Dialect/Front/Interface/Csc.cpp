

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::CscOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::CscOp::init(InferenceParameter &p) { return success(); }
void front::CscOp::deinit(InferenceParameter &p) {}

LogicalResult front::CscOp::inference(InferenceParameter &p) {
  // front::CscOp no need to inference
  llvm_unreachable("front::CscOp no need to inference");
  return failure();
}

void front::CscOp::shape_inference() {}
