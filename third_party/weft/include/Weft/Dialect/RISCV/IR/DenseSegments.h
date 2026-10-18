#ifndef WEFT_DIALECT_RISCV_IR_DENSESEGMENTS_H
#define WEFT_DIALECT_RISCV_IR_DENSESEGMENTS_H

#include "Weft/Dialect/RISCV/IR/VectorTile.h"

namespace weft::riscv {

// Independently named numerical values whose dense descriptors select adjacent
// fields of one source slice. The query proves coordinates and storage only;
// moving the original memory events remains the physical pass's responsibility.
struct DenseSegmentFields {
  mlir::Value base;
  int64_t fieldAxis;
  unsigned fieldPosition;
  int64_t firstField;
  int64_t fields;
  VectorTileGeometry tile;

  static std::optional<DenseSegmentFields> get(mlir::ValueRange descriptors,
                                               ValueType type, int64_t laneAxis);
  bool hasCompleteFields(mlir::Operation *at) const;
  bool hasReadStrides(mlir::Operation *at) const;
  // Stores also require distinct lane records: merging independent field stores
  // must not change the last writer when an arbitrary runtime stride overlaps.
  bool hasStoreStrides(mlir::Operation *at) const;
  bool canStore() const;
};

} // namespace weft::riscv
#endif
