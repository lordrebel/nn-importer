
#include "Support/Module.h"
#include "Support/OpRewriterPatternEx.h"

using namespace im::front;

// unsqueeze + squeeze && in == out
struct FrontFuseSqueeze : public OpRewriterPatternEx<SqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  FrontFuseSqueeze(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SqueezeOp>(context, "FrontFuseSqueeze") {}

  LogicalResult matchAndRewriteImpl(SqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_op = op.getInput().getDefiningOp();
    if (in_op->hasOneUse() && isa<UnsqueezeOp>(in_op)) {
      auto former_op = dyn_cast<UnsqueezeOp>(in_op);
      auto shape0 = module::getShape(op.getOutput());
      auto shape1 = module::getShape(former_op.getInput());
      if (shape0 != shape1) {
        return failure();
      }
      op.getOutput().replaceAllUsesWith(former_op.getInput());
      rewriter.eraseOp(op);
      rewriter.eraseOp(former_op);
      return success();
    }
    return failure();
  }
};

// squeeze scalar [1] -> [1]
struct FrontSqueezeErase : public OpRewriterPatternEx<SqueezeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  FrontSqueezeErase(mlir::MLIRContext *context)
      : OpRewriterPatternEx<SqueezeOp>(context, "FrontSqueezeErase") {}

  LogicalResult matchAndRewriteImpl(SqueezeOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op.getIsScalar()) {
      return failure();
    }
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

void SqueezeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<FrontFuseSqueeze, FrontSqueezeErase>(context);
}
