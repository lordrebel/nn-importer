#include "Dialect/Front/Transform/Pass.h"
#include "Support/Module.h"
using namespace llvm;

namespace im {
namespace front {

class DeinitPass : public DeinitBase<DeinitPass> {
public:
  DeinitPass() {}
  void runOnOperation() override {
    auto state = module::getState();
    if (state >= module::State::TOSA_F32) {
      return;
    }
    module::removeUnusedOp();
    if (!no_save_weight)
      module::saveWeight();
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createDeinitPass() {
  return std::make_unique<DeinitPass>();
}
} // namespace front
} // namespace im
