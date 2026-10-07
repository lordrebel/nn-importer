
add_mlir_dialect(Front front)

set(LLVM_TARGET_DEFINITIONS Front.td)
mlir_tablegen(FrontAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(FrontAttr.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRFrontAttrIncGen)
