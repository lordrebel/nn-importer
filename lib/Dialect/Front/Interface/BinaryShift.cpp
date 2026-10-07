

#include "Support/Module.h"

int64_t front::BinaryShiftOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::BinaryShiftOp::init(InferenceParameter &p) {
  return success();
}
void front::BinaryShiftOp::deinit(InferenceParameter &p) {}

LogicalResult front::BinaryShiftOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::BinaryShiftOp::shape_inference() {
  broadcast_shape_inference(getOperation());
  broadcast_tensor_reshape(getOutput(), getInput1());
  broadcast_tensor_reshape(getOutput(), getInput2());
}
