

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::ReciprocalOp::getFLOPs() {
  return module::getNumElements(getOutput()) * (1 + (getDoRelu() ? 1 : 0));
}

LogicalResult front::ReciprocalOp::init(InferenceParameter &p) {
  return success();
}
void front::ReciprocalOp::deinit(InferenceParameter &p) {}

LogicalResult front::ReciprocalOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  int64_t num_elem = module::getNumElements(getOutput());
  float const_s = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = const_s / p.inputs[0][i];
  }
  if (getDoRelu()) {
    auto limit = getReluLimit().convertToDouble();
    function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
  }
  return success();
}

void front::ReciprocalOp::shape_inference() {
  common_shape_inference(getOperation());
}
