add_mlir_dialect_library(WeftKernelDialect
  KernelDialect.cpp
  KernelInterface.cpp

  DEPENDS
  WeftKernelIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
)
