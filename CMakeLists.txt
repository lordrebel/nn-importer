cmake_minimum_required(VERSION 3.20)
if (POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

if (POLICY CMP0148)
  cmake_policy(SET CMP0148 OLD)
endif()

project(nn-importer LANGUAGES CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
else()
    find_program(SCCACHE_PROGRAM sccache)
    if(SCCACHE_PROGRAM)
        set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
        message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
    endif()
endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
# reference https://github.com/llvm/circt/blob/main/cmake/modules/AddCIRCT.cmake
set(MLIR_BINARY_DIR ${CMAKE_INSTALL_PREFIX})

find_package(MLIR REQUIRED CONFIG
  HINTS ${PROJECT_SOURCE_DIR}/../llvm_release ${PROJECT_SOURCE_DIR}/../llvm_release/lib/cmake ${PROJECT_SOURCE_DIR}/../llvm_release/lib/cmake/mlir
  NO_DEFAULT_PATH )
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "Using LLVM_INCLUDE_DIRS in: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")
#-------------------------------------------------------------------------------
# oneDNN Configuration
#-------------------------------------------------------------------------------
find_package(DNNL REQUIRED 
 HINTS ${PROJECT_SOURCE_DIR}/../onednn_release ${PROJECT_SOURCE_DIR}/../onednn_release/lib/cmake ${PROJECT_SOURCE_DIR}/../onednn_release/lib/cmake/mlir
  NO_DEFAULT_PATH
)
include_directories(${DNNL_INCLUDE_DIRS})
link_libraries(${DNNL_LIBS})

#-------------------------------------------------------------------------------
# Bundle oneDNN runtime library into the install tree
#-------------------------------------------------------------------------------
# 运行期 importer-opt / python 扩展都还需要 libdnnl.so.3, 而它不在系统默认
# 搜索路径里(系统装的通常是别的主版本, 如 libdnnl.so.3.6), 所以必须随安装树分发.
# install(IMPORTED_RUNTIME_ARTIFACTS) 会连同 SONAME 软链接一起安装到 <prefix>/lib.
install(IMPORTED_RUNTIME_ARTIFACTS DNNL::dnnl LIBRARY DESTINATION lib)

# python 扩展(_mlir*.so / pymlir.so)的 RUNPATH 是 $ORIGIN, 看不见 <prefix>/lib,
# 因此在对应目录里补一个指向 lib/ 的相对软链接(避免重复拷贝几十 MB).
install(CODE [[
  file(GLOB _dnnl_libs "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libdnnl.so*")
  foreach(_reldir "" "mlir/_mlir_libs")
    set(_dest "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/python/${_reldir}")
    file(MAKE_DIRECTORY "${_dest}")
    foreach(_lib ${_dnnl_libs})
      get_filename_component(_name "${_lib}" NAME)
      file(RELATIVE_PATH _tgt "${CMAKE_INSTALL_PREFIX}/python/${_reldir}"
                              "${CMAKE_INSTALL_PREFIX}/lib/${_name}")
      file(REMOVE "${_dest}/${_name}")
      file(CREATE_LINK "${_tgt}" "${_dest}/${_name}" SYMBOLIC)
    endforeach()
  endforeach()
]])

#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
set(CNPY_PATH ${PROJECT_SOURCE_DIR}/3rdparty/cnpy)
include_directories(${CNPY_PATH})

#-------------------------------------------------------------------------------
# cuDNN Configuration
#-------------------------------------------------------------------------------
option(NN_IMPORTER_USE_CUDA "SUPPORT to use cuda to do inference")

if(NN_IMPORTER_USE_CUDA)
  include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/include)
  link_directories(${CUDA_LIBRARY_DIRS} /usr/local/cuda/lib64)
  add_definitions(-DUSE_CUDA)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
#-------------------------------------------------------------------------------
OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

#-------------------------------------------------------------------------------
function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()
if(NN_IMPORTER_USE_LLD)
  append("-fuse-ld=lld"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()
#-------------------------------------------------------------------------------

# generate version
execute_process(
  COMMAND git describe --tags --always
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_EXEC_RESULT
  OUTPUT_VARIABLE GIT_SHORT_HASH)

if (GIT_SHORT_HASH)
  string(STRIP ${GIT_SHORT_HASH} GIT_SHORT_HASH)
  string(FIND ${GIT_SHORT_HASH} "v" v_index)
  string(FIND ${GIT_SHORT_HASH} "V" V_index)
  if(v_index GREATER_EQUAL 0 OR V_index GREATER_EQUAL 0)
    string(FIND ${GIT_SHORT_HASH} "-" dash_index)
    if (dash_index GREATER_EQUAL 0)
      # ex: v1.12-beta.0-xx-xxxx
      string(SUBSTRING ${GIT_SHORT_HASH} 0 ${dash_index} MAIN_VERSION)
      math(EXPR dash_index "${dash_index} + 1")
      string(SUBSTRING ${GIT_SHORT_HASH} ${dash_index} -1 PATCH_VERSION)
      if (NN_IMPORTER_USE_CUDA)
        set(MAIN_VERSION "${MAIN_VERSION}+cuda")
      endif()
      if ("${MAIN_VERSION}" STREQUAL "${PATCH_VERSION}")
        set(GIT_SHORT_HASH "${MAIN_VERSION}")
      else()
        set(GIT_SHORT_HASH "${MAIN_VERSION}.${PATCH_VERSION}")
      endif()
    else()
      # ex: v1.12
      # set(GIT_SHORT_HASH "${GIT_SHORT_HASH}")
      if (NN_IMPORTER_USE_CUDA)
        set(GIT_SHORT_HASH "${GIT_SHORT_HASH}+cuda")
      endif()
    endif()
  else()
    # not find tag, cmt_id only, and set standard version
    if (NN_IMPORTER_USE_CUDA)
      set(GIT_SHORT_HASH "v1.0.0.dev+cuda-${GIT_SHORT_HASH}")
    else()
      set(GIT_SHORT_HASH "v1.0.0.dev-${GIT_SHORT_HASH}")
    endif()
  endif()
else()
  set(GIT_SHORT_HASH "UNKNOWN.UNKNOWN")
  message(STATUS "Cannot get version from git info, use default: ${GIT_SHORT_HASH}")
endif()

string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(MLIR_VERSION "${GIT_SHORT_HASH}-${BUILD_TIME}" CACHE STRING "nn-importer version" FORCE)
set(BUILD_TIME "${BUILD_TIME}" CACHE STRING "Build time" FORCE)
message(STATUS "nn-importer version: ${MLIR_VERSION}")
add_definitions(-DMLIR_VERSION="${MLIR_VERSION}")
#-------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Werror -Wno-unused-result -Wreturn-type -Wunused-variable")

# ThinLTO for release builds: parallelizes link-time optimization across cores
set(CMAKE_CXX_LINK_OPTIONS "${CMAKE_CXX_LINK_OPTIONS} -Wl,-plugin-opt=-import-instr-limit=5")

# Debug builds: use split DWARF to reduce object size and link time
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gsplit-dwarf")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gsplit-dwarf")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
add_subdirectory(include)
add_subdirectory(3rdparty)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(bindings)
add_subdirectory(python)

#-------------------------------------------------------------------------------
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

#-------------------------------------------------------------------------------
