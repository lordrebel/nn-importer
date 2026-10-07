file(GLOB _sources
          *.cpp
    )

add_mlir_conversion_library(NNImporterFrontToLinalg
      ${_sources}

      ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/include/Dialect/Front
      DEPENDS
      NNImporterConversionPassIncGen

      LINK_LIBS PUBLIC
      NNImporterFront
      MLIRLinalgDialect
      )
