

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::SigmoidOp::getFLOPs() {
  return module::getNumElements(getInput()) * 4;
}

LogicalResult front::SigmoidOp::init(InferenceParameter &p) {
  return success();
}
void front::SigmoidOp::deinit(InferenceParameter &p) {}

LogicalResult front::SigmoidOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto num_element = module::getNumElements(getInput());
  bool log = getLog();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] =
        log ? std::log(1 / (1 + std::exp(-val))) : 1 / (1 + std::exp(-val));
  }
  return success();
}

void front::SigmoidOp::shape_inference() {
  common_shape_inference(getOperation());
}
