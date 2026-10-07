

#include "Support/MathUtil.h"
#include "Support/Module.h"
#include "mlir/IR/TypeUtilities.h"
int64_t front::LoopOp::getFLOPs() { return 0; }

LogicalResult front::LoopOp::init(InferenceParameter &p) { return success(); }
void front::LoopOp::deinit(InferenceParameter &p) {}

LogicalResult front::LoopOp::inference(InferenceParameter &p) {
  return success();
}

Operation::result_range front::LoopOp::v_final() {
  auto results = getResults();
  return llvm::make_range(results.begin(),
                          results.begin() + getVInitial().size());
}

Operation::result_range front::LoopOp::scan_outputs() {
  auto results = getResults();
  return llvm::make_range(results.begin() + getVInitial().size(),
                          results.end());
}

static inline bool containSubgraph(Operation &op) {
  return isa<front::LoopOp>(op);
}

static inline bool isUsedByReturnOp(Operation &op) {
  return std::any_of(op.getUsers().begin(), op.getUsers().end(),
                     [](Operation *user) {
                       return isa<func::ReturnOp, front::YieldOp>(user);
                     });
}

static inline bool returnsDynamicOrUnknownShape(Operation &op) {
  return std::any_of(
      op.getResultTypes().begin(), op.getResultTypes().end(),
      [](Type result_type) {
        if (mlir::isa<RankedTensorType>(result_type))
          return std::any_of(
              mlir::dyn_cast<RankedTensorType>(result_type).getShape().begin(),
              mlir::dyn_cast<RankedTensorType>(result_type).getShape().end(),
              [](int64_t dim) { return dim < 0; });
        else
          return !mlir::isa<NoneType>(result_type);
      });
}

static inline LogicalResult runShapeInferenceOnRegion(Region &r) {
  for (Operation &op : r.getOps()) {
    if (!isa<front::InputOp>(op) && !containSubgraph(op) &&
        !isUsedByReturnOp(op) && !returnsDynamicOrUnknownShape(op))
      continue;
    if (auto shape_op = llvm::dyn_cast<ShapeInterface>(op)) {
      std::optional<RegisteredOperationName> registeredInfo =
          op.getName().getRegisteredInfo();
      if (registeredInfo && failed(registeredInfo->verifyInvariants(&op)))
        return op.emitError("verification failed");

      // Attempt to infer the shape of the produced output(s).
      shape_op.shape_inference();
    } else if (isa<front::InputOp>(op)) {
      auto inputOp = cast<front::InputOp>(op);
      inputOp.getOutput().setType(inputOp.getInput().getType());
    } else if (!isa<front::WeightOp, front::NoneOp>(op))
      return op.emitError("unable to infer shape of operation without shape "
                          "inference interface");
  }
  return success();
}

static inline void updateType(Value val, ArrayRef<int64_t> shape,
                              Type elementType, Attribute encoding) {
  SmallVector<int64_t, 4> inferredShape;
  for (size_t i = 0; i < shape.size(); ++i)
    inferredShape.emplace_back(shape[i] != -1 ? shape[i]
                                              : ShapedType::kDynamic);

  if (!elementType)
    elementType = mlir::getElementTypeOrSelf(val.getType());

  if (auto valType = mlir::dyn_cast<RankedTensorType>(val.getType())) {
    if (!encoding)
      encoding = valType.getEncoding();
  }

  RankedTensorType resType;
  if (encoding)
    resType = RankedTensorType::get(inferredShape, elementType, encoding);
  else
    resType = RankedTensorType::get(inferredShape, elementType);
  val.setType(resType);
}

void front::LoopOp::shape_inference() {
  auto &loopBody = getRegion();
  ASSERT_THIS(loopBody.getNumArguments() >= 2 &&
              "Loop body must take at least 2 inputs.");
  loopBody.getArgument(0).setType(getM().getType());
  loopBody.getArgument(1).setType(getCond().getType());

  auto bodyInputs = loopBody.getArguments();
  auto bodyVRange = llvm::make_range(bodyInputs.begin() + 2, bodyInputs.end());
  for (auto opVToBodyVTy : llvm::zip(getVInitial(), bodyVRange)) {
    auto opVTy = std::get<0>(opVToBodyVTy).getType();
    std::get<1>(opVToBodyVTy).setType(opVTy);
  }

  std::function<void(Region &)> doShapeInference = [](Region &region) {
    (void)runShapeInferenceOnRegion(region);
  };

  doShapeInference(loopBody);

  auto bodyResultTys = loopBody.back().getTerminator()->getOperandTypes();
  auto scanStartItr =
      std::next(bodyResultTys.begin(), 1 + getVInitial().size());
  auto bodyResVFinalTys =
      llvm::make_range(std::next(bodyResultTys.begin(), 1), scanStartItr);
  auto bodyResScanTys = llvm::make_range(scanStartItr, bodyResultTys.end());
  for (auto vFinalValToTy : llvm::zip(v_final(), bodyResVFinalTys)) {
    std::get<0>(vFinalValToTy).setType(std::get<1>(vFinalValToTy));
  }

  for (auto vScanOutputValToTy : llvm::zip(scan_outputs(), bodyResScanTys)) {
    auto rankedScanTy =
        mlir::cast<RankedTensorType>(std::get<1>(vScanOutputValToTy));
    auto shape = rankedScanTy.getShape();
    SmallVector<int64_t, 4> unsqueezedShape(shape.begin(), shape.end());
    // unsqueezedShape.insert(unsqueezedShape.begin(), ShapedType::kDynamic);
    updateType(std::get<0>(vScanOutputValToTy), unsqueezedShape,
               rankedScanTy.getElementType(), nullptr);
  }

  return;
}
