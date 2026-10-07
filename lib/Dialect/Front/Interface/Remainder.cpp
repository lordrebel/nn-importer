

#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

int64_t front::RemainderOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::RemainderOp::init(InferenceParameter &p) {
  return success();
}
void front::RemainderOp::deinit(InferenceParameter &p) {}

LogicalResult front::RemainderOp::inference(InferenceParameter &p) {
  int64_t num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int64_t i = 0; i < num_element; ++i) {

    double quo = p.inputs[0][i] / p.inputs[1][i];
    auto quo_floor = std::floor(quo);
    p.outputs[0][i] = p.inputs[0][i] - p.inputs[1][i] * quo_floor;
  }
  return success();
}

void front::RemainderOp::shape_inference() {
  common_shape_inference(getOperation());
}
