

#include "Support/Module.h"

int64_t front::WeightReorderOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::WeightReorderOp::init(InferenceParameter &p) {
  return success();
}
void front::WeightReorderOp::deinit(InferenceParameter &p) {}

LogicalResult front::WeightReorderOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::WeightReorderOp::shape_inference() {}
