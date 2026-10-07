

#include "Support/MathUtil.h"
#include "Support/Module.h"
int64_t front::ListOp::getFLOPs() { return 0; }

LogicalResult front::ListOp::init(InferenceParameter &p) { return success(); }

void front::ListOp::deinit(InferenceParameter &p) {}

LogicalResult front::ListOp::inference(InferenceParameter &p) {
  int64_t offset = 0;
  int64_t num_inputs = getInputs().size();
  for (int i = 0; i < num_inputs; i++) {
    if (module::isNone(getInputs()[i])) {
      continue;
    }
    auto num = module::getNumElements(getInputs()[i]);
    memcpy(p.outputs[0] + offset, p.inputs[i], num * sizeof(float));
    offset += num;
  }
  return success();
}

// ListOp is special, will convert to WeightOp
void front::ListOp::shape_inference() {
  int64_t num_outputs = 0;
  for (auto in : getInputs()) {
    if (module::isNone(in)) {
      continue;
    }
    num_outputs += module::getNumElements(in);
  }
  std::vector<int64_t> new_shape = {num_outputs};
  module::setShapeOrVerify(getOutput(), new_shape);
}
