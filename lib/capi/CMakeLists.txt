add_mlir_public_c_api_library(NNImporterCAPI
  Dialect.cpp
  RegisterEverything.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  NNImporterFront
  MLIRFuncDialect
  MLIRQuantDialect
)
