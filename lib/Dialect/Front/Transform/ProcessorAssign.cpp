
#include "Dialect/Front/Transform/Pass.h"
#include "Support/Module.h"
using namespace llvm;

namespace im {
namespace front {

class ProcessorAssignPass : public ProcessorAssignBase<ProcessorAssignPass> {
public:
  ProcessorAssignPass() {}
  void runOnOperation() override {
    auto chip_ = StringRef(chip).lower();
    auto chip = module::symbolizeChip(chip_);
    assert(chip.has_value());
    module::setChip(chip.value());
    
    auto mode_ = StringRef(mode).upper();
    auto quant_mode = module::symbolizeMode(mode_);
    assert(quant_mode.has_value());
    module::setMode(quant_mode.value());
    assert(num_device > 0);
    module::setDeviceNum(num_device);
    assert(num_core > 0);
    module::setCoreNum(num_core);
    auto mode = module::AddrMode::BASIC;
    if (addr_mode != "auto") {
      mode = module::symbolizeAddrMode(addr_mode).value_or(
          module::AddrMode::BASIC);
    }
    module::setAddrMode(mode);
    module::setHighPrecision(high_precision);
    module::updateModuleTypes();
  }

private:
  void input_type_process(ModuleOp mOp) {
    auto mainFunc = module::getMainFuncOp(mOp);
    mainFunc.walk([&](Operation *op) {
      if (isa<front::InputOp>(op)) {
        auto output_value = op->getResult(0);
        auto storage_type = module::getStorageType(output_value);
        if (storage_type.isIntOrIndex()) {
          auto new_type = RankedTensorType::get(module::getShape(output_value),
                                                Builder(op).getF32Type());
          output_value.setType(new_type);
        }
      }
    });
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createProcessorAssignPass() {
  return std::make_unique<ProcessorAssignPass>();
}
} // namespace front
} // namespace im
