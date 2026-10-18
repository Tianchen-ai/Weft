#include "Weft/Target/RISCVPasses.h"
#include "DensePanelTransfer.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVPipeline.h"
#include "Weft/Dialect/RISCV/IR/PanelTransfer.h"
#include "Weft/Dialect/RISCV/IR/Memory.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace weft;

namespace {
// One finite target family: individual row transfers or a legal segmented
// store. The ranking counts issued memory operations and register-width work.
// It is not a latency prediction and makes no claim about cache hit rates.
FailureOr<riscv::PanelTransferCaseAttr> selectCase(riscv::DensePanelCopyOp operation,
    const riscv::PanelTransferGeometry &geometry, int64_t axis) {
  Builder b(operation.getContext());
  bool segment = geometry.target.getHasSegmentMemory() &&
      riscv::hasUnitStrideProof(operation, operation.getDestination(), geometry.fieldAxis(axis));
  auto selected = riscv_internal::selectPanelTransferCase(b, geometry, axis,
      operation.getMaximumLmul(), operation.getMaximumFields(), segment,
      operation->getParentOfType<riscv::KernelOp>().getPanelTransferStrategy());
  if (failed(selected)) return operation.emitError("no legal panel transfer in the exact finite target domain"), failure();
  return selected;
}

LogicalResult select(riscv::DensePanelCopyOp operation) {
  if (operation.getPlan()) return riscv::verifyPanelTransferPlan(operation);
  auto geometry = riscv::PanelTransferGeometry::get(operation);
  if (!geometry) return operation.emitOpError("cannot select a transfer without exact dense window geometry");
  int64_t primary = operation.getCopyAxis();
  int64_t alternate = geometry->fieldAxis(primary);
  int64_t unitAxis = 0, nonunitAxis = 0;
  SmallVector<int64_t> axes;
  if (riscv::hasUnitStrideProof(operation, operation.getSource(), primary)) axes.push_back(primary);
  else if (riscv::hasUnitStrideProof(operation, operation.getSource(), alternate)) axes.push_back(alternate);
  else if (riscv::denseArgumentStride(operation.getSource(), alternate)) {
    unitAxis = alternate;
    if (riscv::denseArgumentStride(operation.getSource(), primary)) nonunitAxis = primary;
    axes = {alternate, primary};
  } else axes.push_back(primary);
  SmallVector<Attribute> cases;
  for (int64_t axis : axes) {
    auto selected = selectCase(operation, *geometry, axis);
    if (failed(selected)) return failure();
    cases.push_back(*selected);
  }
  operation.setPlanAttr(riscv::PanelTransferPlanAttr::get(operation.getContext(), unitAxis, nonunitAxis,
      ArrayAttr::get(operation.getContext(), cases)));
  return riscv::verifyPanelTransferPlan(operation);
}

class SelectRISCVPanelTransfersPass
    : public PassWrapper<SelectRISCVPanelTransfersPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVPanelTransfersPass)
  StringRef getArgument() const override { return "weft-riscv-select-panel-transfers"; }
  StringRef getDescription() const override { return "Select bounded panel transfer orientation, memory issues and tail coverage"; }
  void runOnOperation() override {
    bool failedSelection = false;
    getOperation().walk([&](riscv::DensePanelCopyOp operation) {
      if (operation->getParentOfType<riscv::KernelOp>().getResourcesMaterialized()) {
        operation.emitError("panel transfer selection requires open physical resources");
        failedSelection = true;
      } else failedSelection |= failed(select(operation));
    });
    if (failedSelection) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> weft::createSelectRISCVPanelTransfersPass() {
  return std::make_unique<SelectRISCVPanelTransfersPass>();
}

FailureOr<std::string> weft::inspectRISCVPanelTransfers(ModuleOp module) {
  if (failed(readRISCVPipelineInput(module, RISCVCheckpoint::PanelTransfers))) return failure();
  auto array = [](ArrayRef<int64_t> values) {
    llvm::json::Array result;
    for (int64_t value : values) result.push_back(value);
    return result;
  };
  auto tile = [](riscv::PanelTransferTileAttr value) {
    return llvm::json::Object{{"axis", value.getAxis()}, {"fields", value.getFields()},
        {"lanes", value.getLanes()}, {"lmul_eighths", value.getLmulEighths()},
        {"register_groups", value.getRegisters()}, {"loads", value.getLoads()},
        {"stores", value.getStores()}, {"work", value.getWork()}};
  };
  llvm::json::Array kernels;
  bool invalid = false;
  module.walk([&](riscv::KernelOp kernel) {
    llvm::json::Array transfers;
    kernel.walk([&](riscv::DensePanelCopyOp operation) {
      if (failed(operation.verify())) { invalid = true; return; }
      auto plan = operation.getPlan();
      if (!plan) { invalid = true; return; }
      auto geometry = *riscv::PanelTransferGeometry::get(operation);
      llvm::json::Array cases;
      for (Attribute attribute : plan->getCases()) {
        auto selected = cast<riscv::PanelTransferCaseAttr>(attribute);
        cases.push_back(llvm::json::Object{{"main", tile(selected.getMain())}, {"tail", tile(selected.getTail())}});
      }
      std::string element;
      llvm::raw_string_ostream printed(element);
      geometry.element.print(printed);
      transfers.push_back(llvm::json::Object{{"axes", array(geometry.axes)}, {"element", element},
          {"strategy", kernel.getPanelTransferStrategy()},
          {"source_shape", array(operation.getSource().getType().getShape().asArrayRef())},
          {"destination_axes", array(operation.getDestination().getType().getAxisIds().asArrayRef())},
          {"destination_shape", array(operation.getDestination().getType().getShape().asArrayRef())},
          {"maximum_lmul_eighths", operation.getMaximumLmul()}, {"maximum_fields", operation.getMaximumFields()},
          {"unit_axis", plan->getUnitAxis()}, {"nonunit_axis", plan->getNonunitAxis()}, {"cases", std::move(cases)}});
    });
    kernels.push_back(llvm::json::Object{{"name", kernel.getSymName()}, {"vlen_bits", kernel.getTarget().getVlenBits()},
                                      {"transfers", std::move(transfers)}});
  });
  if (invalid) return failure();
  return llvm::formatv("{0:2}\n", llvm::json::Value(llvm::json::Object{
      {"kind", "weft-panel-transfer-selections"}, {"scope", "static-physical-choices"},
      {"kernels", std::move(kernels)}})).str();
}
