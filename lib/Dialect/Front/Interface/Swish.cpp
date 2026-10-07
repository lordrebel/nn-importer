

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::SwishOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 5;
}

LogicalResult front::SwishOp::init(InferenceParameter &p) { return success(); }
void front::SwishOp::deinit(InferenceParameter &p) {}

LogicalResult front::SwishOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
  auto beta = getBeta().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    p.outputs[0][i] = val / (1 + std::exp(-val * beta));
  }
  return success();
}

void front::SwishOp::shape_inference() {
  common_shape_inference(getOperation());
}
