

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::EluOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::EluOp::init(InferenceParameter &p) { return success(); }
void front::EluOp::deinit(InferenceParameter &p) {}

LogicalResult front::EluOp::inference(InferenceParameter &p) {
  const float *src = p.inputs[0];
  float *dst = p.outputs[0];
  int64_t num_elements = module::getNumElements(getInput());
  float alpha = static_cast<float>(getAlpha().convertToDouble());
#pragma omp parallel for schedule(static, omp_schedule(num_elements))
  for (int64_t i = 0; i < num_elements; ++i) {
    dst[i] = src[i] > 0 ? src[i] : alpha * (std::exp(src[i]) - 1);
  }
  return success();
}

void front::EluOp::shape_inference() { common_shape_inference(getOperation()); }
