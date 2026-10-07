

#include "Support/Module.h"
#include "Support/OpRewriterPatternEx.h"

using namespace im::front;
using namespace im::trait;

struct FrontGroupNormReshape : public OpRewriterPatternEx<GroupNormOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  FrontGroupNormReshape(mlir::MLIRContext *context)
      : OpRewriterPatternEx<GroupNormOp>(context, "FrontGroupNormReshape") {}

  LogicalResult matchAndRewriteImpl(GroupNormOp op,
                                    PatternRewriter &rewriter) const override {
    auto in_shape = module::getShape(op.getInput());
    auto num_dims = in_shape.size();
    LogicalResult res = failure();

    auto weight = op.getWeight();
    if (module::isWeight(weight)) {
      auto weight_shape = module::getShape(weight);
      if (num_dims != weight_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        new_shape[1] = weight_shape[0];
        module::setShape(weight, new_shape);
        res = success();
      }
    }

    auto bias = op.getBias();
    if (module::isWeight(bias)) {
      auto bias_shape = module::getShape(bias);
      if (num_dims != bias_shape.size()) {
        std::vector<int64_t> new_shape(num_dims, 1);
        new_shape[1] = bias_shape[0];
        module::setShape(bias, new_shape);
        res = success();
      }
    }

    return res;
  }
};

void GroupNormOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<FrontGroupNormReshape>(context);
}
