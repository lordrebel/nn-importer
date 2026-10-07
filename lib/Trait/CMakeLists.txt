file(GLOB _sources *.cpp)

add_llvm_library(NNImporterTrait
  ${_sources}
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRQuantDialect
)
