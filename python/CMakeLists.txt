include(AddMLIRPython)

include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

set(NN_IMPORTER_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nn_importer")
set(NN_IMPORTER_PYTHON_PACKAGES_DIR "${CMAKE_BINARY_DIR}/python")

################################################################################
# Structural groupings.
################################################################################
declare_mlir_python_sources(NNImporterPythonSources)
declare_mlir_python_sources(NNImporterPythonSources.Dialects
  ROOT_DIR "${NN_IMPORTER_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT NNImporterPythonSources
)
################################################################################
# Dialect bindings
################################################################################
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT NNImporterPythonSources.Dialects
  ROOT_DIR "${NN_IMPORTER_PYTHON_ROOT_DIR}"
  TD_FILE dialects/Front.td
  SOURCES dialects/front.py
  DIALECT_NAME front)


declare_mlir_python_extension(NNImporterPythonExtension.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ROOT_DIR "${NN_IMPORTER_PYTHON_ROOT_DIR}"
  SOURCES
    RegisterEverything.cpp
  EMBED_CAPI_LINK_LIBS
    NNImporterCAPI
)

################################################################################
# Common CAPI dependency DSO.
# All python extensions must link through one DSO which exports the CAPI, and
# this must have a globally unique name amongst all embeddors of the python
# library since it will effectively have global scope.
################################################################################

add_mlir_python_common_capi_library(NNImporterPythonCAPI
  INSTALL_COMPONENT TPUMLIRPythonModules
  INSTALL_DESTINATION python/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    NNImporterPythonSources
    NNImporterPythonExtension.RegisterEverything
    MLIRPythonSources.Core
  EMBED_LIBS
    MLIRCAPIQuant
    # 上游 _mlir 的 Rewrite.cpp 需要 MLIRCAPITransforms 里的
    # mlirRewritePatternSetFromPDLPatternModule 等符号, 上游是由
    # MLIRPythonExtension.RegisterEverything 链进来的; 这里用的是自建的
    # RegisterEverything, 所以要显式补上, 否则 import mlir 会报 undefined symbol.
    MLIRCAPITransforms
)

################################################################################
# The fully assembled package of modules.
################################################################################

add_mlir_python_modules(NNImporterPythonModules
  ROOT_PREFIX "${CMAKE_BINARY_DIR}/python/mlir"
  INSTALL_PREFIX "python/mlir/"
  DECLARED_SOURCES
    NNImporterPythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.quant
    MLIRPythonSources.Dialects.tosa
    MLIRPythonSources.Dialects.func
    NNImporterPythonExtension.RegisterEverything
  COMMON_CAPI_LINK_LIBS
    NNImporterPythonCAPI
)

install(
  DIRECTORY ./ DESTINATION python FILES_MATCHING
  USE_SOURCE_PERMISSIONS
  PATTERN "*.py"
  PATTERN "*.js"
  PATTERN "*.html"
  PATTERN "xlsx_to_py.py" EXCLUDE
  PATTERN "test_*" EXCLUDE
  PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE
  PATTERN "nn_importer" EXCLUDE
  PATTERN "tools/soc_infer" EXCLUDE
)

install(CODE "
  execute_process(COMMAND find \"\${CMAKE_INSTALL_PREFIX}/python\" -type d -empty -delete)
")
