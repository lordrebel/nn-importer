
set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

set(LIBS
  MLIRTosaDialect
  MLIRFuncDialect
  MLIROptLib
  NNImporterInitAll
  )

add_llvm_executable(importer-opt
  importer-opt.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(importer-opt PRIVATE ${LIBS})
llvm_update_compile_flags(importer-opt)

mlir_check_all_link_libraries(importer-opt)

install(TARGETS importer-opt DESTINATION bin)
