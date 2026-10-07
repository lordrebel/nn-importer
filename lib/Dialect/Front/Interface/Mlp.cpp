#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"

mlp_attr_t front::MlpOp::parseParam() {
  mlp_attr_t p = {0};
  return p;
}

int64_t front::MlpOp::getFLOPs() { return 0; }

LogicalResult front::MlpOp::init(InferenceParameter &p) { return success(); }

void front::MlpOp::deinit(InferenceParameter &p) { return; }

LogicalResult front::MlpOp::inference(InferenceParameter &p) {
  return success();
}

void front::MlpOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  ASSERT_THIS(input_shape.size() == 3);
  if (getIsExpert()) {
    auto num_expert_per_tok = getNumExpertPerTok();
    std::vector<int64_t> out_shape = {input_shape[0] * input_shape[1],
                                      static_cast<int64_t>(num_expert_per_tok),
                                      input_shape[2]};
    module::setShapeOrVerify(getOutput(), out_shape);
  } else {
    module::setShapeOrVerify(getOutput(), input_shape);
  }
}
