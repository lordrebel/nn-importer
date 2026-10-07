

#include "Support/MathUtil.h"
#include "Support/Module.h"
void front::InputOp::shape_inference() {
  if (!getShapeTensor().has_value())
    return;
  std::vector<int64_t> shape_tensor =
      *(module::getI64Array(getShapeTensor().value()));
  if (shape_tensor.size() > 0) {
    module::bindShapeTensorValue(getOutput(), shape_tensor);
  }
  // removeShapeTensorAttr();
}
