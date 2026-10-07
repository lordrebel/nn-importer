set(LLVM_TARGET_DEFINITIONS Pass.td)
mlir_tablegen(Pass.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(NNImporterConversionPassIncGen)
