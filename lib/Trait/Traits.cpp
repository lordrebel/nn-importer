#include "Support/Module.h"

namespace im {

namespace trait {
namespace impl {

static LogicalResult check_type(Value v) {
  auto type = v.getType();
  if (isa<NoneType>(type)) {
    return success();
  }

  if (auto tensor_type = mlir::dyn_cast<RankedTensorType>(type)) {
    auto etype = tensor_type.getElementType();
    if (etype.isIntOrFloat()) {
      return success();
    }
    if (isa<quant::UniformQuantizedType, quant::CalibratedQuantizedType>(
            etype)) {
      return success();
    }
  }
  return failure();
}

LogicalResult verifyTpuTypeRestrictTrait(Operation *op) {
  for (auto out : op->getResults()) {
    if (failed(check_type(out))) {
      return op->emitError("expected tpu supported type");
    }
  }
  return mlir::success();
}

LogicalResult verifyInOutSameShapeTrait(Operation *op) {
  if (op->hasAttr("indexing_map") || op->hasAttr("indexing_map_s2l") ||
      op->hasAttr("indexing_map_l2s")) {
    return mlir::success();
  }
  auto in_shape =
     mlir::cast<RankedTensorType>( op->getOperand(0).getType()).getShape();
  auto out_shape =
      mlir::cast<RankedTensorType>(op->getResult(0).getType()).getShape();
  if (in_shape != out_shape) {
    return op->emitError("expected input and output with same shape");
  }
  return mlir::success();
}

} // namespace impl
} // namespace trait

} // namespace im
