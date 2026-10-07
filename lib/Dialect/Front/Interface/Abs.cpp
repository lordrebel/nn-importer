

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::AbsOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::AbsOp::init(InferenceParameter &p) { return success(); }
void front::AbsOp::deinit(InferenceParameter &p) {}

LogicalResult front::AbsOp::inference(InferenceParameter &p) {
  auto input_shape = module::getShape(getInput());
  module::setShape(getOutput(), input_shape);
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::abs(val);
  }
  return success();
}

void front::AbsOp::shape_inference() { common_shape_inference(getOperation()); }
