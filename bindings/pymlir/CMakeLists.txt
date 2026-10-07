cmake_minimum_required(VERSION 3.20)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pymlir)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB _sources pymodule.cpp pymlir.cpp host/*.cpp)



pybind11_add_module(pymlir ${_sources})

target_link_libraries(pymlir PRIVATE
  NNImporterInitAll
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)


install(TARGETS pymlir DESTINATION python)
