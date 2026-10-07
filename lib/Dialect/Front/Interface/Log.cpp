

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::LogOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::LogOp::init(InferenceParameter &p) { return success(); }
void front::LogOp::deinit(InferenceParameter &p) {}

LogicalResult front::LogOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::log(val);
  }
  return success();
}

void front::LogOp::shape_inference() { common_shape_inference(getOperation()); }
