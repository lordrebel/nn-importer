

#include "Support/Module.h"

int64_t front::BinaryConstShiftOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::BinaryConstShiftOp::init(InferenceParameter &p) {
  return success();
}
void front::BinaryConstShiftOp::deinit(InferenceParameter &p) {}

LogicalResult front::BinaryConstShiftOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::BinaryConstShiftOp::shape_inference() {
  common_shape_inference(getOperation());
}
