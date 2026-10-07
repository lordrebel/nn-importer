

#include "Support/Dnnl/Dnnl.h"
#include "Support/Module.h"
using namespace dnnl;

namespace im {
memory::data_type getDnnlType(mlir::Value v) {
  auto type = module::getStorageType(v);
  if (type.isF32()) {
    return memory::data_type::f32;
  }
  if (type.isSignedInteger(8) || type.isSignlessInteger(8)) {
    return memory::data_type::s8;
  }
  if (type.isUnsignedInteger(8)) {
    return memory::data_type::u8;
  }
  if (type.isInteger(16) || type.isInteger(32)) {
    return memory::data_type::s32;
  }
  llvm::errs() << "Unsupport type: ";
  type.dump();
  return memory::data_type::f32;
}
} // namespace im
