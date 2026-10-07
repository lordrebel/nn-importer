#include "Dialect/Front/IR/Front.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

int64_t front::LogicalAndOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::LogicalAndOp::init(InferenceParameter &p) {
  return success();
}
void front::LogicalAndOp::deinit(InferenceParameter &p) {}

LogicalResult front::LogicalAndOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}

void front::LogicalAndOp::shape_inference() {}
