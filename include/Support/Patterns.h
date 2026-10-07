

#pragma once

#include "Dialect/Front/IR/Front.h"
#include "Support/Module.h"
#include "Support/OpRewriterPatternEx.h"
#include "mlir/IR/PatternMatch.h"

// Common Patterns
namespace im {
namespace patterns {

struct FuseSameOp : public OpRewriterPatternEx3 {
  FuseSameOp(MLIRContext *context)
      : OpRewriterPatternEx3(context, "FuseSameOp", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override;
};

struct InputReshape : public OpRewriterPatternEx3 {
  InputReshape(MLIRContext *context)
      : OpRewriterPatternEx3(context, "InputReshape", 1) {}
  LogicalResult matchAndRewriteImpl(Operation *op,
                                    PatternRewriter &rewriter) const override;
};

// if op =  op + op, fuse to one op. such as front::Reshape
template <typename OpTy>
struct FuseRepeatPattern : public OpRewriterPatternEx<OpTy> {
public:
  FuseRepeatPattern(MLIRContext *context, int benifit = 1)
      : OpRewriterPatternEx<OpTy>(context, "FuseRepeatPattern", benifit) {}
  LogicalResult matchAndRewriteImpl(OpTy op, PatternRewriter &rewriter) const {
    auto in_op = op.getInput().getDefiningOp();
    if (nullptr == in_op || in_op->hasOneUse() == false) {
      return failure();
    }
    if (!isa<OpTy>(in_op)) {
      return failure();
    }
    op->setOperand(0, in_op->getOperand(0));
    rewriter.eraseOp(in_op);
    return success();
  }
};

// convert op a to op b, not care attributes. such as front::Sequence to
// front::Reshape
template <typename SourceOp, typename TargetOp>
struct GeneralPattern : public OpRewriterPatternEx<SourceOp> {
  GeneralPattern(MLIRContext *context, const std::string &patternName,
                 int benefit = 1)
      : OpRewriterPatternEx<SourceOp>(context, patternName, benefit) {}

  LogicalResult matchAndRewriteImpl(SourceOp op,
                                    PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, op.getOutput().getType(),
                                          op->getOperands(),
                                          std::vector<NamedAttribute>());
    return success();
  }
};

struct SqueezeToReshapePattern
    : public GeneralPattern<front::SqueezeOp, front::ReshapeOp> {
  SqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<front::SqueezeOp, front::ReshapeOp>(
            context, "SqueezeToReshapePattern", benefit) {}
};

struct UnsqueezeToReshapePattern
    : public GeneralPattern<front::UnsqueezeOp, front::ReshapeOp> {
  UnsqueezeToReshapePattern(MLIRContext *context, int benefit = 1)
      : GeneralPattern<front::UnsqueezeOp, front::ReshapeOp>(
            context, "UnsqueezeToReshapePattern", benefit) {}
};

} // namespace patterns
} // namespace im
