#include "capi/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "nn_importer Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
  m.def("register_llvm_translations", [](MlirContext) {});
}
