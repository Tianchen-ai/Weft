add_mlir_dialect_library(WeftRISCVDialect
  RISCVDialect.cpp
  FragmentOps.cpp
  StreamOps.cpp
  VectorTile.cpp
  DenseSegments.cpp
  DenseBounds.cpp
  DenseRelations.cpp
  Contraction.cpp
  FMAIssue.cpp
  ScalarPacket.cpp
  PanelTransfer.cpp
  ContractionSelection.cpp
  LocalTransferOps.cpp
  Layout.cpp
  Coordinates.cpp
  Memory.cpp
  PointerMemory.cpp
  Numerics.cpp
  Predicate.cpp
  Verification.cpp
  FieldView.cpp

  DEPENDS
  WeftRISCVIncGen

  LINK_LIBS PUBLIC
  WeftRISCVContracts
  WeftRISCVPlatform
  WeftKernelDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRIR
)
