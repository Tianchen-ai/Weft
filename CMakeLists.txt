cmake_minimum_required(VERSION 3.20)

project(Weft LANGUAGES C CXX)

if(SKBUILD)
  set(WEFT_INSTALL_ROOT "weft/_native")
else()
  set(WEFT_INSTALL_ROOT ".")
endif()

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

set_target_properties(weft-compile weft-opt PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
  INSTALL_RPATH_USE_LINK_PATH FALSE)
install(TARGETS weft-compile weft-opt
  RUNTIME DESTINATION "${WEFT_INSTALL_ROOT}/bin" COMPONENT WeftRuntime)

install(TARGETS WeftRuntime EXPORT WeftTargets
  LIBRARY DESTINATION "${WEFT_INSTALL_ROOT}/lib" COMPONENT WeftRuntime)
install(FILES include/weft/runtime.h include/weft/runtime.hpp
  DESTINATION "${WEFT_INSTALL_ROOT}/include/weft" COMPONENT WeftRuntime)
install(EXPORT WeftTargets NAMESPACE Weft::
  DESTINATION "${WEFT_INSTALL_ROOT}/lib/cmake/Weft" COMPONENT WeftRuntime)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/WeftConfig.cmake"
  "include(\"\${CMAKE_CURRENT_LIST_DIR}/WeftTargets.cmake\")\nset(Weft_LLAMA_CPP_DIR \"\${CMAKE_CURRENT_LIST_DIR}/../../../share/weft/llama.cpp\")\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/WeftConfig.cmake"
  DESTINATION "${WEFT_INSTALL_ROOT}/lib/cmake/Weft" COMPONENT WeftRuntime)
install(DIRECTORY integrations/llama.cpp/
  DESTINATION "${WEFT_INSTALL_ROOT}/share/weft/llama.cpp" COMPONENT WeftRuntime)
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md
  DESTINATION "${WEFT_INSTALL_ROOT}/share/weft/licenses" COMPONENT WeftRuntime)
install(DIRECTORY licenses/
  DESTINATION "${WEFT_INSTALL_ROOT}/share/weft/licenses/licenses" COMPONENT WeftRuntime)

# Distribution LLVM packages may provide a shared LLVM target; source SDKs
# commonly link LLVM statically. Only the actual shared dependency is installed.
if(TARGET LLVM)
  get_target_property(WEFT_LLVM_TYPE LLVM TYPE)
  if(WEFT_LLVM_TYPE STREQUAL "SHARED_LIBRARY")
    install(CODE "
      file(REAL_PATH \"$<TARGET_FILE:LLVM>\" WEFT_LLVM_REAL)
      file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${WEFT_INSTALL_ROOT}/lib\"
        TYPE FILE FILES \"\${WEFT_LLVM_REAL}\"
        RENAME \"$<TARGET_SONAME_FILE_NAME:LLVM>\")
      " COMPONENT WeftRuntime)
  endif()
endif()
