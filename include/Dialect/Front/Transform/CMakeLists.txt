
set(LLVM_TARGET_DEFINITIONS Pass.td)
mlir_tablegen(Pass.h.inc -gen-pass-decls -name Front)
add_public_tablegen_target(MLIRFrontPassIncGen)
