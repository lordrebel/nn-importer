

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::MinConstOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::MinConstOp::init(InferenceParameter &p) {
  return success();
}
void front::MinConstOp::deinit(InferenceParameter &p) {}

LogicalResult front::MinConstOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  const int64_t num_elem = module::getNumElements(getOutput());
  const float const_val_ = getConstVal().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p.outputs[0][i] = std::min(p.inputs[0][i], const_val_);
  }
  return success();
}

void front::MinConstOp::shape_inference() {
  common_shape_inference(getOperation());
  if (module::isShape(getInput())) {
    auto input_v = module::getShapeTensorValue(getInput());
    auto out_shape = module::getShape(getOutput());
    auto output_shape_v =
        module::commonShapeValInfer(getOperation(), {input_v}, out_shape);
    module::bindShapeTensorValue(getOutput(), output_shape_v);
  }
}
