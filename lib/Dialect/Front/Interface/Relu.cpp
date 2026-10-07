

#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::ReluOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::ReluOp::init(InferenceParameter &p) { return success(); }
void front::ReluOp::deinit(InferenceParameter &p) {}

LogicalResult front::ReluOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);
  auto limit = getReluLimit().convertToDouble();
  function_relu(p.inputs[0], p.outputs[0], module::getNumElements(getInput()),
                limit);
  return success();
}

void front::ReluOp::shape_inference() {
  common_shape_inference(getOperation());
}
