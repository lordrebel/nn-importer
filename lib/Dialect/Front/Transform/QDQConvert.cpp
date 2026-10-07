
#include "Dialect/Front/Transform/Pass.h"

#include "Support/OpRewriterPatternEx.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace im {
namespace front {

// A pattern to convert QuantizeLinearOp return element type to quant.calibrated
class QuantizeLinearCastTypePattern : public OpRewriterPatternEx3 {
public:
  QuantizeLinearCastTypePattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearCastTypePattern", benefit,
                             front::QuantizeLinearOp::getOperationName()) {}
  QuantizeLinearCastTypePattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearCastTypePattern", 1,
                             front::QuantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    auto return_value = op->getResult(0);
    auto operand_type =
        mlir::cast<RankedTensorType>(op->getOperand(0).getType());
    assert(mlir::dyn_cast<RankedTensorType>(return_value.getType()) &&
           "return type of QuantizeLinear check failure.");
    assert(operand_type && "operand type of QuantizeLinear check failure.");
    if (mlir::dyn_cast<quant::CalibratedQuantizedType>(
            mlir::dyn_cast<RankedTensorType>(return_value.getType())
                .getElementType())) {
      return failure(); // This op is calibrated.
    }
    auto y_scale =
        *module::getF64Array(mlir::dyn_cast<ArrayAttr>(op->getAttr("y_scale")));
    auto y_zero_point = *module::getI32Array(
        mlir::dyn_cast<ArrayAttr>(op->getAttr("y_zero_point")));
    assert(y_scale.size() == y_zero_point.size() &&
           "y_scale.size() & y_zero_point.size() must be the same.");
    assert(y_scale.size() == 1 &&
           "Cannot support per chanel quant for activation tensor now.");
    assert(y_scale[0] > 0 && "Scale should be positive.");
    // TO-Do : support asymmetric
    float min = (std::numeric_limits<int8_t>::min() - y_zero_point[0]) *
                y_scale[0],
          max = (std::numeric_limits<int8_t>::max() - y_zero_point[0]) *
                y_scale[0];
    auto quant_type = quant::CalibratedQuantizedType::get(
        operand_type.getElementType(), min, max);
    auto new_type = RankedTensorType::get(operand_type.getShape(), quant_type);
    return_value.setType(new_type);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

// A pattern to fuse quantizelinear with the former ops.
class QuantizeLinearFusePattern : public OpRewriterPatternEx3 {
public:
  QuantizeLinearFusePattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearFusePattern", benefit,
                             front::QuantizeLinearOp::getOperationName()) {}
  QuantizeLinearFusePattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "QuantizeLinearFusePattern", 1,
                             front::QuantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    Value oprand = op->getOperand(0);
    auto quantized_type =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!quantized_type) {
      return failure();
    }
    if (!mlir::dyn_cast<quant::CalibratedQuantizedType>(
            quantized_type.getElementType())) {
      return failure(); // Not quantized.
    }
    oprand.setType(quantized_type);
    while (!op->use_empty()) {
      op->getUses().begin()->set(oprand);
    }
    rewriter.eraseOp(op);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class RemoveDequantizeLinearPattern : public OpRewriterPatternEx3 {
public:
  RemoveDequantizeLinearPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "RemoveDequantizeLinearPattern", benefit,
                             front::DequantizeLinearOp::getOperationName()) {}
  RemoveDequantizeLinearPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "RemoveDequantizeLinearPattern", 1,
                             front::DequantizeLinearOp::getOperationName()) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    Value operand = op->getOperand(0);
    auto formerOp = op->getOperand(0).getDefiningOp();
    if (auto weightOp = dyn_cast<WeightOp>(formerOp)) {
      auto scale =
          *module::getF64Array(dyn_cast<DequantizeLinearOp>(op).getXScale());
      weightOp->setAttr("scale",
                        rewriter.getF64ArrayAttr(ArrayRef<double>(scale)));
      auto element_type = rewriter.getIntegerType(8, true);
      auto operand_type = mlir::cast<RankedTensorType>(operand.getType());
      auto new_type =
          RankedTensorType::get(operand_type.getShape(), element_type);
      weightOp.getResult().setType(new_type);
    } else if (auto quantOp = dyn_cast<QuantizeLinearOp>(formerOp)) {
      // To-Do: Check Type and scale.
    } else if (auto result_type = mlir::dyn_cast<RankedTensorType>(
                   formerOp->getResult(0).getType())) {
      if (!mlir::dyn_cast<quant::CalibratedQuantizedType>(
              result_type.getElementType())) {
        llvm_unreachable("Cannot handle this case.");
      }
    } else {
      llvm_unreachable("Cannot handle this case.");
    }
    while (!op->use_empty()) {
      op->getUses().begin()->set(operand);
    }
    rewriter.eraseOp(op);
    return success();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

// Not a graceful pattern. We make the quantized type transparent for Ops like
// reshape, permute, etc...
class CalibratedTypeTransparentPattern : public OpRewriterPatternEx3 {
public:
  CalibratedTypeTransparentPattern(PatternBenefit benefit, MLIRContext *context)
      : OpRewriterPatternEx3(context, "CalibratedTypeTransparentPattern",
                             benefit) {}
  CalibratedTypeTransparentPattern(MLIRContext *context)
      : OpRewriterPatternEx3(context, "CalibratedTypeTransparentPattern", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override {
    if (op->getResults().empty() ||
        isa<front::WeightOp, ReturnOp, front::NoneOp>(op)) {
      return failure();
    }
    if (!mlir::isa_and_nonnull<RankedTensorType>(op->getResult(0).getType())) {
      return failure();
    }
    auto result_tensor_type =
        mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto result_quant_type = result_tensor_type.getElementType();
    if (isa<quant::CalibratedQuantizedType, quant::UniformQuantizedType>(
            result_quant_type)) {
      return failure();
    }
    auto succeedingOps = op->getUsers();
    while (!succeedingOps.empty()) {
      auto succeedingOp =
          *succeedingOps.begin(); // We only use the first user of this op. Not
                                  // a good choice.
      if (!mlir::isa_and_nonnull<RankedTensorType>(
              succeedingOp->getResult(0).getType())) {
        return failure();
      }
      auto result_tensor_type = mlir::dyn_cast<RankedTensorType>(
          succeedingOp->getResult(0).getType());
      auto result_quant_type = result_tensor_type.getElementType();
      if (isa<quant::CalibratedQuantizedType, quant::UniformQuantizedType>(
              result_quant_type)) {
        op->getResult(0).setType(result_tensor_type);
        return success();
      }
      succeedingOps = succeedingOp->getUsers();
    }
    return failure();
  }
  bool shouldPrint(Operation *op) const override { return false; }
};

class QDQConvertPass : public QDQConvertBase<QDQConvertPass> {
public:
  QDQConvertPass() {}
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    ConversionTarget target(*context);

    target.addIllegalOp<QuantizeLinearOp, DequantizeLinearOp>();

    RewritePatternSet patterns(context), b_patterns(context);
    patterns.add<QuantizeLinearCastTypePattern>(context);
    patterns.add<QuantizeLinearFusePattern>(context);
    patterns.add<RemoveDequantizeLinearPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    b_patterns.add<CalibratedTypeTransparentPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(b_patterns));
    module::updateModuleTypes();
    module::setState(module::State::FRONT_CALIBRATED);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createQDQConvertPass() {
  return std::make_unique<QDQConvertPass>();
}

} // namespace front
} // namespace im
