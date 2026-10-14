#include "Weft/Dialect/RISCV/IR/PanelTransfer.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "Weft/Dialect/RISCV/IR/VectorTile.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

using namespace mlir;
using namespace weft;
using namespace weft::riscv;

namespace {
bool powerOfTwo(int64_t value, int64_t maximum) {
  return value > 0 && value <= maximum && !(value & (value - 1));
}

bool denseMemory(MemDescType memory) {
  auto encoding = dyn_cast<kernel::EncodingType>(memory.getEncoding());
  return encoding && encoding.getKind() == "dense" && memory.getElements() == 1 &&
         !memory.getInterleaveRows() && memory.getAxisIds().size() == 2;
}
}

LogicalResult PanelTransferTileAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t axis,
    int64_t fields, int64_t lanes, int64_t lmulEighths, int64_t registers,
    int64_t loads, int64_t stores, int64_t work) {
  if (axis <= 0 || !powerOfTwo(fields, 8) || !powerOfTwo(lmulEighths, 64) ||
      lanes <= 1 || !powerOfTwo(lanes, 65536))
    return error() << "panel transfer requires bounded lane, field and LMUL geometry";
  int64_t resident = fields * ((lmulEighths + 7) / 8);
  if (fields * lmulEighths > 64 || registers != resident || registers > 8)
    return error() << "panel transfer tuple must fit its exact eight-register local issue";
  if (loads != fields || stores != 1 || work != loads + stores + 2 * resident)
    return error() << "panel transfer work must count its loads, store and transferred register groups";
  return success();
}

LogicalResult PanelTransferCaseAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error,
    PanelTransferTileAttr main, PanelTransferTileAttr tail) {
  if (main.getAxis() != tail.getAxis() || tail.getFields() != 1)
    return error() << "panel transfer case requires one orientation and a one-field remainder";
  if (main.getFields() == 1 && main != tail)
    return error() << "single-field panel transfer cannot hide an unused remainder choice";
  return success();
}

LogicalResult PanelTransferPlanAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t unitAxis,
    int64_t nonunitAxis, ArrayAttr cases) {
  if (unitAxis < 0 || nonunitAxis < 0 || (nonunitAxis && (!unitAxis || nonunitAxis == unitAxis)))
    return error() << "panel transfer stride predicates require distinct positive axes";
  if (cases.size() != (unitAxis ? 2u : 1u))
    return error() << "panel transfer must cover both outcomes of its selected stride predicate";
  for (auto attribute : cases)
    if (!mlir::isa<PanelTransferCaseAttr>(attribute))
      return error() << "panel transfer cases must contain typed transfer choices";
  if (unitAxis && (mlir::cast<PanelTransferCaseAttr>(cases[0]).getMain().getAxis() != unitAxis ||
                  mlir::cast<PanelTransferCaseAttr>(cases[1]).getMain().getAxis() == unitAxis))
    return error() << "panel transfer branches disagree with the selected contiguous orientation";
  return success();
}

bool weft::riscv::samePanelExtent(Value first, unsigned firstAxis, Value second, unsigned secondAxis) {
  auto a = denseDescriptorExtent(first, firstAxis);
  auto b = denseDescriptorExtent(second, secondAxis);
  if (!a || !b) return false;
  if (a == b) return true;
  auto ac = getConstantIntValue(a), bc = getConstantIntValue(b);
  if (ac && bc) return *ac == *bc;
  auto av = dyn_cast<Value>(a), bv = dyn_cast<Value>(b);
  return av && bv && sameDenseIndex(av, bv);
}

std::optional<PanelTransferGeometry> PanelTransferGeometry::get(DensePanelCopyOp operation) {
  auto from = operation.getSource().getType();
  auto to = operation.getDestination().getType();
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel || !denseMemory(from) || !denseMemory(to) || from.getEncoding() != to.getEncoding())
    return std::nullopt;
  for (auto [position, axis] : llvm::enumerate(from.getAxisIds().asArrayRef())) {
    auto destination = llvm::find(to.getAxisIds().asArrayRef(), axis);
    if (axis <= 0 || destination == to.getAxisIds().asArrayRef().end() ||
        !samePanelExtent(operation.getSource(), position, operation.getDestination(),
                         destination - to.getAxisIds().asArrayRef().begin())) return std::nullopt;
  }
  if (from.getAxisIds()[0] == from.getAxisIds()[1]) return std::nullopt;
  Builder b(operation.getContext());
  auto encoding = cast<weft::kernel::EncodingType>(from.getEncoding());
  Type element;
  if (encoding.getFamily() == "f16") element = b.getF16Type();
  else if (encoding.getFamily() == "bf16") element = b.getBF16Type();
  else if (encoding.getFamily() == "f32") element = b.getF32Type();
  else if (encoding.getFamily() == "f64") element = b.getF64Type();
  if (!element || !tileMemoryElementMatches(from, element) || !tileMemoryElementMatches(to, element))
    return std::nullopt;
  return PanelTransferGeometry{element,
      SmallVector<int64_t, 2>(from.getAxisIds().asArrayRef()), kernel.getTarget()};
}

int64_t PanelTransferGeometry::fieldAxis(int64_t copyAxis) const {
  if (axes[0] == copyAxis) return axes[1];
  if (axes[1] == copyAxis) return axes[0];
  return 0;
}

ValueType PanelTransferGeometry::numeric(Builder &b, PanelTransferTileAttr tile) const {
  SmallVector<int64_t> shape, lanes, replicas;
  for (int64_t axis : axes) {
    bool lane = axis == tile.getAxis();
    shape.push_back(lane ? tile.getLanes() : tile.getFields());
    lanes.push_back(lane ? tile.getLanes() : 1);
    replicas.push_back(lane ? 1 : tile.getFields());
  }
  auto one = b.getDenseI64ArrayAttr({1, 1});
  auto ids = b.getDenseI64ArrayAttr(axes);
  auto layout = LayoutAttr::get(b.getContext(), "rvv", ids, one,
      b.getDenseI64ArrayAttr(lanes), b.getDenseI64ArrayAttr(replicas), one, one,
      element.getIntOrFloatBitWidth(), tile.getLmulEighths(), tile.getLanes(),
      tile.getRegisters(), "tail");
  return ValueType::get(b.getContext(), element, b.getDenseI64ArrayAttr(shape), ids, layout);
}

bool PanelTransferGeometry::supports(PanelTransferTileAttr tile) const {
  if (!fieldAxis(tile.getAxis()) || tile.getLanes() * element.getIntOrFloatBitWidth() * 8 !=
      target.getVlenBits() * tile.getLmulEighths()) return false;
  Builder b(element.getContext());
  auto type = numeric(b, tile);
  if (!supportsRVVLayout(target, type.getLayout()) || !VectorTileGeometry::get(type, tile.getAxis())) return false;
  return tile.getFields() == 1 || (target.getHasSegmentMemory() &&
      VectorTileSegments::get(type, tile.getAxis(), fieldAxis(tile.getAxis()), tile.getFields()).has_value());
}

LogicalResult weft::riscv::verifyPanelTransferPlan(DensePanelCopyOp operation) {
  auto plan = operation.getPlan();
  if (!plan) return operation.emitOpError("dense panel transfer has no selected physical plan");
  auto geometry = PanelTransferGeometry::get(operation);
  if (!geometry) return operation.emitOpError("panel transfer lost its exact source and destination geometry");
  if (plan->getUnitAxis()) {
    if (!denseArgumentStride(operation.getSource(), plan->getUnitAxis()) ||
        (plan->getNonunitAxis() && !denseArgumentStride(operation.getSource(), plan->getNonunitAxis())))
      return operation.emitOpError("panel transfer stride predicate must refer to exact ABI stride operands");
  }
  for (Attribute attribute : plan->getCases()) {
    auto selected = cast<PanelTransferCaseAttr>(attribute);
    if (operation->getParentOfType<KernelOp>().getPanelTransferStrategy() == "single-field" &&
        selected.getMain().getFields() != 1)
      return operation.emitOpError("panel transfer violates the explicit single-field structure binding");
    for (auto tile : {selected.getMain(), selected.getTail()}) {
      if (tile.getLmulEighths() > operation.getMaximumLmul() || tile.getFields() > operation.getMaximumFields() ||
          !geometry->supports(tile))
        return operation.emitOpError("panel transfer selection exceeds its target or instantiated finite domain");
      if (tile.getFields() > 1 && !hasUnitStrideProof(operation, operation.getDestination(),
                                                    geometry->fieldAxis(tile.getAxis())))
        return operation.emitOpError("segmented panel store requires consecutive destination field coordinates");
    }
  }
  return success();
}

LogicalResult DensePanelCopyOp::verify() {
  auto geometry = PanelTransferGeometry::get(*this);
  if (!geometry || !geometry->fieldAxis(getCopyAxis()))
    return emitOpError("dense panel copy requires identical rank-two floating logical windows");
  if (getSource().getType().getAccess() == "write" || getDestination().getType().getAccess() == "read")
    return emitOpError("dense panel copy requires readable source and writable destination");
  if (!powerOfTwo(getMaximumLmul(), 64) || !powerOfTwo(getMaximumFields(), 8))
    return emitOpError("dense panel copy has an invalid finite transfer binding");
  if (getMaximumFields() != (*this)->getParentOfType<KernelOp>().getPanelTransferFields())
    return emitOpError("dense panel copy field domain differs from the kernel binding");
  if (mayAliasStorage(getSource(), getDestination()))
    return emitOpError("dense panel representation copy requires disjoint storage");
  return getPlan() ? verifyPanelTransferPlan(*this) : success();
}
