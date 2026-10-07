
#include "Support/Module.h"
#include "Support/OpRewriterPatternEx.h"

// upack => slice + reshape
struct Unpack2Split : public OpRewriterPatternEx<front::UnpackOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  Unpack2Split(mlir::MLIRContext *context)
      : OpRewriterPatternEx<front::UnpackOp>(context, "Unpack2Split") {}

  LogicalResult matchAndRewriteImpl(front::UnpackOp op,
                                    PatternRewriter &rewriter) const override {
    std::vector<int64_t> in_shape = module::getShape(op.getInput());
    auto axis = op.getAxis();
    auto num = in_shape[axis];
    auto out_shape = in_shape;
    out_shape[axis] = 1;
    std::vector<Type> newTypes(num);
    for (int i = 0; i < num; i++) {
      auto etype = module::getElementType(op.getOutputs()[i]);
      auto newType = RankedTensorType::get(out_shape, etype);
      newTypes[i] = newType;
    }

    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("axis", op.getAxisAttr()));
    attrs.push_back(
        rewriter.getNamedAttr("num", rewriter.getI64IntegerAttr(num)));
    rewriter.setInsertionPointAfter(op);
    auto splitOp = rewriter.create<front::SplitOp>(
        op.getLoc(), newTypes, ValueRange{op.getInput()}, attrs);

    std::vector<int64_t> oshape(module::getShape(op.getOutputs()[0]));
    std::vector<Value> operands;
    for (int i = 0; i < num; ++i) {
      auto etype = module::getElementType(op.getOutputs()[i]);
      auto newType = RankedTensorType::get(oshape, etype);
      auto out = splitOp.getResult(i);
      std::vector<NamedAttribute> attrs = {};
      auto new_name = module::getName(out).str() + "_r_reshape";
      auto name_loc = NameLoc::get(rewriter.getStringAttr(new_name));
      auto newOp = rewriter.create<front::ReshapeOp>(name_loc, newType,
                                                     ValueRange{out}, attrs);
      operands.push_back(newOp.getOutput());
    }
    rewriter.replaceOp(op, operands);
    return success();
  }
};

void front::UnpackOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.insert<Unpack2Split>(context);
}
