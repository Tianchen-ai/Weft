add_subdirectory(weft-opt)
add_subdirectory(weft-compile)

# Weft's command-line compiler and IR driver are ordinary project artifacts.
# LLVM's add_mlir_tool() excludes tools from the default build when the
# embedding project does not enable LLVM_BUILD_TOOLS; override that inherited
# policy locally instead of requiring callers to name both targets explicitly.
set_property(TARGET weft-opt PROPERTY EXCLUDE_FROM_ALL OFF)
set_property(TARGET weft-compile PROPERTY EXCLUDE_FROM_ALL OFF)
