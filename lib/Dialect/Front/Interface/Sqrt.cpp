

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::SqrtOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::SqrtOp::init(InferenceParameter &p) { return success(); }
void front::SqrtOp::deinit(InferenceParameter &p) {}

LogicalResult front::SqrtOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::sqrt(val);
  }
  return success();
}

void front::SqrtOp::shape_inference() {
  common_shape_inference(getOperation());
}
