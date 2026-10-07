file(GLOB _sources
          IR/*.cpp
          Interface/*.cpp
          Canonicalize/*.cpp
          Transform/*.cpp
    )

add_mlir_library(NNImporterFront
  ${_sources}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/nn_importer/Dialect/Front

  DEPENDS
  MLIRFrontAttrIncGen
  MLIRFrontIncGen
  MLIRFrontPassIncGen
  MLIRInferenceInterfaceIncGen
  MLIRFlopsInterfaceIncGen
  MLIRShapeInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRParser
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRRewrite
  MLIRTransforms
  MLIRInferenceInterface
  MLIRFlopsInterface
  MLIRShapeInterface
  NNImporterSupport
  NNImporterTrait
)

