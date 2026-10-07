

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::CosOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 4;
}

LogicalResult front::CosOp::init(InferenceParameter &p) { return success(); }
void front::CosOp::deinit(InferenceParameter &p) {}

LogicalResult front::CosOp::inference(InferenceParameter &p) {
  auto input_shape = module::getShape(getInput());
  module::setShape(getOutput(), input_shape);
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = std::cos(val);
  }
  return success();
}

void front::CosOp::shape_inference() { common_shape_inference(getOperation()); }
