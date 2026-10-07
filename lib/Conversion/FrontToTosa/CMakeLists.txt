file(GLOB _sources
          *.cpp
    )

add_mlir_conversion_library(NNImporterFrontToTosa
      ${_sources}

      ADDITIONAL_HEADER_DIRS
      ${PROJECT_SOURCE_DIR}/include/Dialect/Front
      DEPENDS
      NNImporterConversionPassIncGen

      LINK_LIBS PUBLIC
      NNImporterFront
      MLIRTosaDialect
      )
