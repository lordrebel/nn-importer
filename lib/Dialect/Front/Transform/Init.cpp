
#include "Dialect/Front/Transform/Pass.h"
#include "Support/Module.h"

using namespace llvm;

namespace im {
int32_t cur_log_level = 0;
namespace front {

class InitPass : public InitBase<InitPass> {
public:
  InitPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    module::init(mOp);
    module::init_loglevel(this->level);
    module::setWeightInMemFlag(weight_in_mem);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createInitPass() {
  return std::make_unique<InitPass>();
}
} // namespace front
} // namespace im
