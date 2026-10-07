

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::LutOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult front::LutOp::init(InferenceParameter &p) { return success(); }
void front::LutOp::deinit(InferenceParameter &p) {}

LogicalResult front::LutOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getInput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    int offset = p.inputs[0][i];
    if (offset < 0) {
      offset += 256;
    }
    ASSERT_THIS(offset >= 0 && offset <= 255);
    p.outputs[0][i] = p.inputs[1][offset];
  }
  return success();
}

void front::LutOp::shape_inference() { common_shape_inference(getOperation()); }
