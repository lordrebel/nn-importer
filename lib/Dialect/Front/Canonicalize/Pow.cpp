
#include "Support/Module.h"
#include "Support/OpRewriterPatternEx.h"

using namespace im::front;
using namespace im::trait;

struct PowToBinary : public OpRewriterPatternEx<PowOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PowToBinary(mlir::MLIRContext *context)
      : OpRewriterPatternEx<PowOp>(context, "PowToBinary") {}

  LogicalResult matchAndRewriteImpl(PowOp op,
                                    PatternRewriter &rewriter) const override {
    auto exp = op.getExponent().convertToDouble();
    std::vector<NamedAttribute> attrs;
    if (exp == 2) {
      rewriter.replaceOpWithNewOp<MulOp>(
          op, op.getOutput().getType(),
          ValueRange{op.getInput(), op.getInput()}, attrs);
      return success();
    } else if (exp == 0.5) {
      rewriter.replaceOpWithNewOp<SqrtOp>(op, op.getOutput().getType(),
                                          ValueRange{op.getInput()}, attrs);
      return success();
    }
    return failure();
  }
};

void PowOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                        MLIRContext *context) {
  results.insert<PowToBinary>(context);
}
