

#include "Support/Float16.h"
#include "Support/MathUtil.h"
#include "Support/Module.h"

int64_t front::DtypeCastOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult front::DtypeCastOp::init(InferenceParameter &p) {
  return success();
}
void front::DtypeCastOp::deinit(InferenceParameter &p) {}

LogicalResult front::DtypeCastOp::inference(InferenceParameter &p) {
  // llvm_unreachable("Not Implemented");
  auto in_shape = module::getShape(getInput());
  module::setShape(getOutput(), in_shape);

  auto in_type = module::getStorageType(getInput());
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());

  if (in_type.isF32() && out_type.isF16()) {
    F16(p.inputs[0], p.outputs[0], num_elem);
  };

  return success();
}

void front::DtypeCastOp::shape_inference() {
  common_shape_inference(getOperation());
}
