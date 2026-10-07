

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::Pow2Op::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::Pow2Op::init(InferenceParameter &p) { return success(); }
void front::Pow2Op::deinit(InferenceParameter &p) {}

LogicalResult front::Pow2Op::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  auto val = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto ex = p.inputs[0][i];
    p.outputs[0][i] = std::pow(val, ex);
  }
  return success();
}

void front::Pow2Op::shape_inference() {
  common_shape_inference(getOperation());
}
