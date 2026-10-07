# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror -Wno-pedantic -fno-strict-aliasing -Wno-maybe-uninitialized")
file(GLOB _sources *.cpp Dnnl/*.cpp)

add_llvm_library(NNImporterSupport
  ${_sources}

  DEPENDS
  ModuleAttrGen
  MLIRFrontIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRQuantDialect
  MLIRQuantUtils
  MLIRInferenceInterface
  cnpy
  DNNL::dnnl
  z
)
