add_library(WeftRuntime SHARED Metadata.cpp Library.cpp CallContract.cpp Layout.cpp Binding.cpp Resources.cpp API.cpp)
add_library(Weft::Runtime ALIAS WeftRuntime)
target_link_libraries(WeftRuntime PRIVATE WeftRISCVPlatform ${CMAKE_DL_LIBS})
target_include_directories(WeftRuntime PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${WEFT_INSTALL_ROOT}/include>)
target_include_directories(WeftRuntime SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_options(WeftRuntime PRIVATE -fexceptions -fvisibility=hidden -Wall -Wextra -Werror)
target_link_options(WeftRuntime PRIVATE "LINKER:--exclude-libs,ALL")
set_target_properties(WeftRuntime PROPERTIES OUTPUT_NAME weft_runtime
  EXPORT_NAME Runtime INSTALL_RPATH "$ORIGIN" INSTALL_RPATH_USE_LINK_PATH FALSE)
