#ifndef WEFT_DIALECT_RISCV_IR_VECTORTILE_H
#define WEFT_DIALECT_RISCV_IR_VECTORTILE_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace weft::riscv {

// One selected lane axis with independently resident replicas, including
// replicas along that same axis. Every part names its complete logical start.
// This is not a reduction stream: no part is implicitly summed with another.
struct VectorTileGeometry {
  int64_t axis = 0;
  unsigned lanePosition = 0;
  int64_t lanes = 0;
  llvm::SmallVector<llvm::SmallVector<int64_t>> starts;

  static std::optional<VectorTileGeometry> get(ValueType type, int64_t axis);
  int64_t parts() const { return starts.size(); }
};

bool tileMemoryElementMatches(MemDescType memory, mlir::Type element);
bool tileExtentsMatch(mlir::Value memory, ValueType value, mlir::ValueRange extents);
bool tileIsCompleteMemory(TargetAttr target, mlir::Value memory, ValueType value);
bool tileHasCompleteFields(mlir::Operation *at, mlir::Value memory,
                           ValueType value, int64_t fieldAxis);
llvm::StringRef tileContractInstruction(llvm::StringRef instruction);

struct VectorTileSegments {
  VectorTileGeometry tile;
  int64_t fieldAxis;
  int64_t fields;
  size_t fieldPosition;
  llvm::SmallVector<llvm::SmallVector<int64_t>> groups;
  static std::optional<VectorTileSegments> get(ValueType type, int64_t laneAxis,
                                              int64_t fieldAxis, int64_t fields);
};

struct VectorTileProjection {
  llvm::SmallVector<int64_t> sourceParts;
  llvm::SmallVector<int64_t> laneOffsets;
  static std::optional<VectorTileProjection> get(ValueType source, ValueType result,
      int64_t axis, llvm::ArrayRef<int64_t> offsets);
};

} // namespace weft::riscv
#endif
