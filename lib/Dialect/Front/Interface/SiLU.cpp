

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::SiLUOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult front::SiLUOp::init(InferenceParameter &p) { return success(); }
void front::SiLUOp::deinit(InferenceParameter &p) {}

LogicalResult front::SiLUOp::inference(InferenceParameter &p) {
  auto input_shape = module::getShape(getInput());
  module::setShape(getOutput(), input_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = val / (1 + std::exp(-val));
  }
  return success();
}

void front::SiLUOp::shape_inference() {
  common_shape_inference(getOperation());
}
