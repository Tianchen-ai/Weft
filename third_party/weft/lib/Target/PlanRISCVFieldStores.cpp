#include "RISCVPhysicalSupport.h"
#include "FieldStores.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <numeric>

using namespace weft;

mlir::LogicalResult riscv_internal::materializeFieldStoreView(
    mlir::IRRewriter &rewriter, riscv::StoreOp operation) {
  llvm::SmallVector<mlir::Operation *> projections;
  mlir::Value root = operation.getRegion();
  while (mlir::isa_and_nonnull<riscv::SliceOp, riscv::SubviewOp>(root.getDefiningOp())) {
    projections.push_back(root.getDefiningOp());
    root = root.getDefiningOp()->getOperand(0);
  }
  auto field = root.getDefiningOp<riscv::FieldOp>();
  if (!field || projections.empty()) return mlir::success();
  auto owner = mlir::dyn_cast<riscv::MemDescType>(field.getOwner().getType());
  auto type = mlir::cast<riscv::MemDescType>(field.getType());
  auto facts = fieldFacts(field);
  if (!owner || owner.getShape().size() != 1 || type.getShape().size() != 1 ||
      owner.getShape()[0] != owner.getElements() || type.getShape()[0] <= 0 ||
      facts.mapping != "natural" || facts.bitOffset % 8 ||
      owner.getInterleaveRows() || type.getStorageBits() % 8 ||
      type.getStorageBits() < 8 || type.getStorageBits() > 64 ||
      facts.alignment < type.getStorageBits() / 8 || owner.getElements() % type.getShape()[0])
    return operation.emitError("projected field store requires an aligned natural record-local field view");
  // Early memory planning precedes the lowering of Level point coordinates.
  mlir::Value recordBase;
  mlir::Value cursor = field.getOwner();
  while (auto slice = cursor.getDefiningOp<riscv::SliceOp>()) {
    size_t index = 0;
    for (auto [axis, selector] : llvm::zip(slice.getBase().getType().getAxisIds().asArrayRef(), slice.getSelectors())) {
      auto kind = mlir::cast<mlir::StringAttr>(selector).getValue();
      if (kind == "all") continue;
      mlir::Value selected = slice.getIndices()[index++];
      if (axis == type.getAxisIds()[0] && kind == "domain") {
        auto point = selected.getDefiningOp<riscv::PhysicalPointOp>();
        if (!point) return mlir::success();
        recordBase = point.getBase();
        break;
      }
    }
    if (recordBase) break;
    cursor = slice.getBase();
  }
  for (mlir::Value base = field.getOwner();;) {
    if (base.getDefiningOp<riscv::SubviewOp>())
      return operation.emitError("projected field store does not support a subview of the encoded record owner");
    auto slice = base.getDefiningOp<riscv::SliceOp>();
    if (!slice) break;
    base = slice.getBase();
  }
  rewriter.setInsertionPoint(operation);
  if (!field.getOwner().getDefiningOp<riscv::SliceOp>()) {
    auto sliceType = riscv::MemDescType::get(rewriter.getContext(), owner.getEncoding(), owner.getShape(),
        owner.getAxisIds(), owner.getStrides(), owner.getOrigins(), owner.getAlignment(), "slice",
        owner.getAccess(), owner.getAliasSet(), owner.getLayoutIdentity(), owner.getStorageBits(),
        owner.getElements(), owner.getInterleaveRows());
    auto record = riscv::SliceOp::create(rewriter, operation.getLoc(), sliceType, field.getOwner(),
        mlir::ValueRange{}, rewriter.getArrayAttr({rewriter.getStringAttr("all")}));
    auto projected = riscv::FieldOp::create(rewriter, field.getLoc(), type, record.getResult(),
        field.getName(), field.getAccess(), field.getLeaf());
    copyOrigin(field, projected);
    root = projected.getResult();
  }
  mlir::Value origin = recordBase;
  if (!origin) origin = mlir::arith::ConstantIndexOp::create(rewriter, operation.getLoc(), 0);
  else if (owner.getElements() != type.getShape()[0]) {
    auto divisor = mlir::arith::ConstantIndexOp::create(rewriter, operation.getLoc(), owner.getElements() / type.getShape()[0]);
    origin = mlir::arith::DivUIOp::create(rewriter, operation.getLoc(), origin, divisor);
  }
  int64_t alignment = std::min(facts.alignment, type.getStorageBits() / 8);
  auto denseType = riscv::MemDescType::get(rewriter.getContext(), type.getEncoding(), type.getShape(),
      type.getAxisIds(), rewriter.getDenseI64ArrayAttr({1}), rewriter.getDenseI64ArrayAttr({-1}),
      alignment, "slice", type.getAccess(), type.getAliasSet(), type.getLayoutIdentity(),
      type.getStorageBits(), 1, 0);
  mlir::Value region = riscv::FieldViewOp::create(rewriter, operation.getLoc(), denseType, root, origin);
  copyOrigin(operation, region.getDefiningOp());
  for (auto projection : llvm::reverse(projections)) {
    auto prior = mlir::cast<riscv::MemDescType>(projection->getResult(0).getType());
    auto result = riscv::MemDescType::get(rewriter.getContext(), prior.getEncoding(), prior.getShape(),
        prior.getAxisIds(), rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(prior.getShape().size(), 1)),
        rewriter.getDenseI64ArrayAttr(llvm::SmallVector<int64_t>(prior.getShape().size(), -1)),
        alignment, "slice", prior.getAccess(), prior.getAliasSet(), prior.getLayoutIdentity(),
        prior.getStorageBits(), 1, 0);
    if (auto slice = mlir::dyn_cast<riscv::SliceOp>(projection))
      region = riscv::SliceOp::create(rewriter, slice.getLoc(), result, region,
          slice.getIndices(), slice.getSelectors());
    else {
      auto subview = mlir::cast<riscv::SubviewOp>(projection);
      region = riscv::SubviewOp::create(rewriter, subview.getLoc(), result, region,
          subview.getDynamicBounds(), subview.getOffsetsAttr(), subview.getExtentsAttr());
    }
    copyOrigin(projection, region.getDefiningOp());
  }
  operation.getRegionMutable().assign(region);
  return mlir::success();
}

mlir::LogicalResult riscv_internal::planFieldStore(
    mlir::IRRewriter &rewriter, riscv::StoreOp operation) {
  auto field = operation.getRegion().getDefiningOp<riscv::FieldOp>();
  auto facts = fieldFacts(field);
  auto owner = mlir::dyn_cast<kernel::EncodingType>(logicalElement(field.getOwner().getType()));
  if (!owner || facts.bitOffset % 8 || facts.storageBits % 8)
    return operation.emitError("encoded field store requires a byte-addressable storage field");
  const int64_t rows = interleaveRows(field, owner);
  const int64_t bits = logicalBitWidth(operation.getValue().getType());
  auto value = mlir::dyn_cast<riscv::ValueType>(operation.getValue().getType());
  auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
  llvm::StringRef kind;
  int64_t storeBits = bits, lanes = 1, sourcesPerStore = 1, temporaries = 0;
  llvm::SmallVector<int64_t> offsets, sources, shifts;
  const bool scalar = !value || value.getLayout().getCarrier() == "scalar";
  if (scalar) {
    if (!facts.scalarPerRecord || facts.mapping != "natural" ||
        (bits != 8 && bits != 16 && bits != 32))
      return operation.emitError("scalar field store requires one complete natural byte, half, or word");
    kind = "scalar";
    if (rows > 1 && facts.chunkBytes < bits / 8) {
      kind = "scalar_bytes";
      storeBits = 8;
      for (int64_t byte = 0; byte < bits / 8; ++byte) {
        offsets.push_back(byte);
        sources.push_back(0);
        shifts.push_back(byte * 8);
      }
    } else {
      offsets.push_back(0);
      sources.push_back(0);
      shifts.push_back(0);
    }
  } else {
    if (value.getLayout().getCarrier() != "rvv" || value.getShape().size() != 1 ||
        value.getShape()[0] <= 0 || facts.logicalRank != 1 ||
        value.getShape()[0] * bits != facts.storageBits)
      return operation.emitError("vector field store requires one complete record-local field axis");
    const int64_t count = value.getShape()[0];
    if (facts.mapping == "natural" && (bits == 8 || bits == 16 || bits == 32)) {
      if (rows > 1 && facts.chunkBytes < bits / 8)
        return operation.emitError("vector field store has no byte-distributed multi-byte element relation");
      kind = "natural";
      lanes = rows > 1 ? std::gcd(count, facts.chunkBytes / (bits / 8))
                       : std::min(count, value.getLayout().getVl());
    } else if (facts.mapping == "grouped_layered" &&
               (bits == 1 || bits == 2 || bits == 4) && facts.layer > 0 &&
               facts.group == facts.layer * (8 / bits) && count % facts.group == 0) {
      kind = "layered";
      storeBits = 8;
      sourcesPerStore = 8 / bits;
      lanes = rows > 1 ? std::gcd(facts.layer, facts.chunkBytes) : facts.layer;
    } else {
      return operation.emitError("field store has no closed natural or whole-byte layered packing relation");
    }
    const int64_t sew = value.getLayout().getSew();
    int64_t lmul = 0;
    while (lanes > 0 && !lmul) {
      for (int64_t candidate : target.getLegalLMULEighths().asArrayRef())
        if (candidate * target.getVlenBits() >= lanes * sew * 8 &&
            (!lmul || candidate < lmul))
          lmul = candidate;
      if (!lmul)
        lanes /= 2;
    }
    if (!lanes || count % lanes || (kind == "layered" && facts.layer % lanes))
      return operation.emitError("field store lane partition has no legal RVV carrier");
    const int64_t groups = (lmul + 7) / 8;
    auto layout = riscv::LayoutAttr::get(
        rewriter.getContext(), "rvv", value.getAxisIds(),
        rewriter.getDenseI64ArrayAttr({count / lanes}),
        rewriter.getDenseI64ArrayAttr({lanes}),
        rewriter.getDenseI64ArrayAttr({1}), rewriter.getDenseI64ArrayAttr({1}),
        rewriter.getDenseI64ArrayAttr({1}), sew, lmul, lanes, groups,
        value.getLayout().getValidity());
    if (!riscv::supportsRVVLayout(target, layout))
      return operation.emitError("field store selected an unsupported RVV layout");
    auto selected = riscv::ValueType::get(rewriter.getContext(), value.getElementType(),
                                         value.getShape(), value.getAxisIds(), layout);
    if (selected != value) {
      rewriter.setInsertionPoint(operation);
      auto conversionOp = riscv::ConvertLayoutOp::create(rewriter,
          operation.getLoc(), selected, operation.getValue(),
          layoutConversion(rewriter, value.getLayout(), layout),
          riscv::AccessAttr(), unselectedLeaf(rewriter));
      copyOrigin(operation, conversionOp);
      operation.getValueMutable().assign(conversionOp.getResult());
      value = selected;
    }
    if (kind == "natural") {
      for (int64_t part = 0; part < count / lanes; ++part) {
        offsets.push_back(part * lanes * (bits / 8));
        sources.push_back(part);
        shifts.push_back(0);
      }
    } else {
      temporaries = 2 * groups;
      for (int64_t group = 0; group < count / facts.group; ++group)
        for (int64_t stripe = 0; stripe < facts.layer; stripe += lanes) {
          offsets.push_back(group * facts.layer + stripe);
          for (int64_t layer = 0; layer < sourcesPerStore; ++layer) {
            sources.push_back((group * facts.group + layer * facts.layer + stripe) / lanes);
            shifts.push_back((facts.order == "lo_first" ? layer
                                                       : sourcesPerStore - 1 - layer) * bits);
          }
        }
    }
  }
  auto access = riscv::AccessAttr::get(
      rewriter.getContext(), "unit", facts.mapping, facts.alignment, 0, 0,
      facts.group, facts.layer, facts.joinFields, facts.joinLowBits, facts.joinRole,
      facts.bitOffset, facts.storageBits, facts.order, facts.chunkBytes);
  operation.setAccessAttr(access);
  operation.setFieldStorePlanAttr(riscv::FieldStorePlanAttr::get(
      rewriter.getContext(), kind, bits, storeBits, lanes, sourcesPerStore,
      rewriter.getDenseI64ArrayAttr(offsets), rewriter.getDenseI64ArrayAttr(sources),
      rewriter.getDenseI64ArrayAttr(shifts)));
  operation.setLeafAttr(leaf(
      rewriter, scalar ? "scalar" : "rvv", "encoded-field-store",
      scalar ? "scalar.field-store" : "rvv.field-store",
      scalar ? "scalar.field-store" : "rvv.field-store",
      0, 0, temporaries, 0, "none", "exact"));
  return mlir::success();
}
