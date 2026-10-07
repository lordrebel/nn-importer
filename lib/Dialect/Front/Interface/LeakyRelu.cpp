

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::LeakyReluOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::LeakyReluOp::init(InferenceParameter &p) {
  return success();
}
void front::LeakyReluOp::deinit(InferenceParameter &p) {}

LogicalResult front::LeakyReluOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  float *dst = p.outputs[0];
  int64_t num_elements = module::getNumElements(getInput());
  float alpha = static_cast<float>(getAlpha().convertToDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
  for (int64_t i = 0; i < num_elements; ++i) {
    dst[i] = src[i] > 0 ? src[i] : (alpha * src[i]);
  }
  return success();
}

void front::LeakyReluOp::shape_inference() {
  common_shape_inference(getOperation());
}
