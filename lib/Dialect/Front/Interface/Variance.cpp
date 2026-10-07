#include "Dialect/Front/IR/Front.h"
#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

int64_t front::VarianceOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::VarianceOp::init(InferenceParameter &p) {
  return success();
}
void front::VarianceOp::deinit(InferenceParameter &p) {}

LogicalResult front::VarianceOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::VarianceOp::shape_inference() {}
