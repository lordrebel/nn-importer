

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::ExpOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::ExpOp::init(InferenceParameter &p) { return success(); }
void front::ExpOp::deinit(InferenceParameter &p) {}

LogicalResult front::ExpOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::exp(val);
  }
  return success();
}

void front::ExpOp::shape_inference() { common_shape_inference(getOperation()); }
