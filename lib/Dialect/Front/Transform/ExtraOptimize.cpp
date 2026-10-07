#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/Front/Transform/Pass.h"
#include "Support/Patterns.h"

using namespace llvm;

namespace im {
namespace front {
template <typename OpTy>
struct RemoveUnuseOutput : public OpRewriterPatternEx<OpTy> {
public:
  RemoveUnuseOutput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context) {}

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override {
    for (Value out : op.getResults()) {
      if (out.getUsers().empty() && !isa<front::TopKOp>(op)) {
        out.setType(mlir::NoneType::get(rewriter.getContext()));
      }
    }
    return success();
  }
};

class ExtraOptimizePass : public ExtraOptimizeBase<ExtraOptimizePass> {
public:
  ExtraOptimizePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    MLIRContext *ctx = &getContext();
    // remove unuse output
    RewritePatternSet patterns(ctx);
    patterns.add<RemoveUnuseOutput<front::LSTMOp>, RemoveUnuseOutput<front::GRUOp>,
                 patterns::FuseSameOp, patterns::InputReshape>(ctx);
    (void)applyPatternsAndFoldGreedily(mOp, std::move(patterns));
    // mark flops
    int64_t flops = 0;
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](FlopsInterface op) { flops += op.getFLOPs(); });
    }
    // weight fold after top optimized
    for (auto func : mOp.getOps<FuncOp>()) {
      func.walk([&](InferenceInterface op) { WeightFolder(op); });
    }
    module::setFLOPs(flops);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createExtraOptimizePass() {
  return std::make_unique<ExtraOptimizePass>();
}
} // namespace front
} // namespace im
