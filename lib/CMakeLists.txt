add_subdirectory(Interface)
add_subdirectory(Trait)
add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(capi)

add_mlir_library(NNImporterInitAll
  InitAll.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRTosaDialect
  MLIRLinalgDialect
  MLIRInferenceInterface
  NNImporterFrontToTosa
  NNImporterFrontToLinalg
)
