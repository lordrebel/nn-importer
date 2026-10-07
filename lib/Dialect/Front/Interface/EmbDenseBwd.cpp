

#include "Support/Module.h"

int64_t front::EmbDenseBwdOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult front::EmbDenseBwdOp::init(InferenceParameter &p) {
  return success();
}
void front::EmbDenseBwdOp::deinit(InferenceParameter &p) {}

LogicalResult front::EmbDenseBwdOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void front::EmbDenseBwdOp::shape_inference() {}
