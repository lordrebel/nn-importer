

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::SinOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::SinOp::init(InferenceParameter &p) { return success(); }
void front::SinOp::deinit(InferenceParameter &p) {}

LogicalResult front::SinOp::inference(InferenceParameter &p) {
  auto input_shape = module::getShape(getInput());
  module::setShape(getOutput(), input_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::sin(val);
  }
  return success();
}

void front::SinOp::shape_inference() { common_shape_inference(getOperation()); }
