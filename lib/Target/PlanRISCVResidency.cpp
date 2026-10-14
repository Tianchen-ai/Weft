#include "RISCVResidencyPlanning.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "RISCVPhysicalSupport.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>

namespace {

using namespace weft;

bool isDenseLoadBacked(riscv::MaterializeOp materialize) {
  auto load = riscv_internal::sourceLoad(materialize.getInput());
  if (!load)
    return false;
  auto encoding =
      mlir::dyn_cast<kernel::EncodingType>(load.getRegion().getType().getEncoding());
  return encoding && encoding.getKind() == "dense";
}

bool requiresAddressableReload(riscv::MaterializeOp materialize,
                               riscv::TargetAttr target) {
  auto value = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  if (!value || value.getLayout().getCarrier() != "rvv" ||
      !isDenseLoadBacked(materialize))
    return false;

  bool hasExtract = false;
  bool allExtracts = true;
  bool registerProjectionUnsupported = false;
  for (mlir::Operation *user : materialize.getResult().getUsers()) {
    auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
    if (!extract || extract.getInput() != materialize.getResult()) {
      allExtracts = false;
      continue;
    }
    hasExtract = true;
    int64_t selectedAxes = 0;
    for (mlir::Attribute selector : extract.getSelectors())
      if (mlir::cast<mlir::StringAttr>(selector).getValue() != "all")
        ++selectedAxes;
    registerProjectionUnsupported |= selectedAxes != 1;
  }
  if (!hasExtract || !allExtracts)
    return false;

  auto timeParts = riscv_internal::staticProduct(
      value.getLayout().getTimeFactors().asArrayRef());
  if (!timeParts)
    return true;
  const int64_t residentGroups =
      *timeParts * value.getLayout().getRegisterGroups();
  return registerProjectionUnsupported ||
         residentGroups > target.getVectorRegisters();
}

std::optional<riscv::LocalPackPlanAttr>
planEncodedLocalPack(mlir::Builder &builder,
                     riscv::MaterializeOp materialize) {
  auto input = mlir::dyn_cast<riscv::ValueType>(materialize.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  auto encoding = result
                      ? mlir::dyn_cast<kernel::EncodingType>(result.getElementType())
                      : kernel::EncodingType();
  auto load = riscv_internal::sourceLoad(materialize.getInput());
  if (!input || !result || !encoding || encoding.getKind() != "base" || !load ||
      result.getLayout().getCarrier() != "local" ||
      result.getShape().size() != 2 || result.getAxisIds().size() != 2) {
    return std::nullopt;
  }

  auto declaration =
      riscv_internal::findEncoding(materialize, encoding.getFamily());
  if (!declaration || declaration.getStorageBits() <= 0 ||
      declaration.getStorageBits() % 8 || declaration.getElements() <= 0) {
    return std::nullopt;
  }

  const int64_t rowAxis = result.getAxisIds()[0];
  const int64_t recordAxis = result.getAxisIds()[1];
  int64_t interleaveRows = 0;
  bool hasConsumer = false;
  llvm::SmallVector<mlir::Value> worklist{materialize.getResult()};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (mlir::Operation *user : current.getUsers()) {
      if (auto convert = mlir::dyn_cast<riscv::ConvertLayoutOp>(user);
          convert && convert.getInput() == current &&
          convert.getConversion().getEffect() == "pure") {
        worklist.push_back(convert.getResult());
        continue;
      }
      auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
      auto extracted =
          extract && extract.getInput() == current
              ? mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType())
              : riscv::ValueType();
      if (!extracted) {
        return std::nullopt;
      }
      auto row = llvm::find(extracted.getAxisIds().asArrayRef(), rowAxis);
      auto record = llvm::find(extracted.getAxisIds().asArrayRef(), recordAxis);
      if (row == extracted.getAxisIds().asArrayRef().end() ||
          record == extracted.getAxisIds().asArrayRef().end()) {
        return std::nullopt;
      }
      const int64_t rows = extracted.getShape()[static_cast<size_t>(
          row - extracted.getAxisIds().asArrayRef().begin())];
      if (rows <= 1 || (interleaveRows != 0 && interleaveRows != rows)) {
        return std::nullopt;
      }
      interleaveRows = rows;
      hasConsumer = true;
    }
  }
  if (!hasConsumer || interleaveRows <= 1) {
    return std::nullopt;
  }

  auto representedExtent = [&](size_t axis) -> std::optional<int64_t> {
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             result.getLayout().getTimeFactors(),
             result.getLayout().getLaneFactors(),
             result.getLayout().getReplicaFactors(),
             result.getLayout().getFragmentFactors(),
             result.getLayout().getLocalFactors()}) {
      const int64_t factor = factors[axis];
      if (factor <= 0 || extent > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      extent *= factor;
    }
    return extent;
  };
  auto localRows = representedExtent(0);
  auto localElements = representedExtent(1);
  const int64_t recordElements = declaration.getElements();
  const int64_t recordBytes = declaration.getStorageBits() / 8;
  if (!localRows || !localElements || *localRows % interleaveRows ||
      *localElements % recordElements) {
    return std::nullopt;
  }
  const int64_t records = *localRows * (*localElements / recordElements);
  if (records > std::numeric_limits<int64_t>::max() / recordBytes)
    return std::nullopt;
  const int64_t localBytes = records * recordBytes;
  auto kernel = materialize->getParentOfType<riscv::KernelOp>();
  if (!kernel || localBytes > kernel.getTarget().getMaxPrivateStackBytes()) {
    return std::nullopt;
  }

  return riscv::LocalPackPlanAttr::get(
      builder.getContext(), "interleave", rowAxis, recordAxis,
      interleaveRows, recordBytes, recordElements, localBytes);
}

} // namespace

mlir::LogicalResult weft::planRISCVResidency(mlir::ModuleOp module) {
  mlir::Builder builder(module.getContext());
  bool invalid = false;
  module.walk([&](riscv::MaterializeOp materialize) {
    if (materialize.getPlacement() != "shared") {
      materialize.emitError(
          "residency planning requires one unresolved shared materialization");
      invalid = true;
      return;
    }
    materialize.setLocalPackPlanAttr({});
    if (auto plan = planEncodedLocalPack(builder, materialize)) {
      materialize.setPlacementAttr(builder.getStringAttr("local"));
      materialize.setSchemaAttr(builder.getStringAttr("encoded-interleave"));
      materialize.setLocalPackPlanAttr(*plan);
      return;
    }
    if (auto value =
            mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
        value && value.getLayout().getCarrier() == "local" &&
        !mlir::isa<kernel::EncodingType>(value.getElementType())) {
      materialize.setPlacementAttr(builder.getStringAttr("local"));
      return;
    }
    auto kernel = materialize->getParentOfType<riscv::KernelOp>();
    if (kernel && requiresAddressableReload(materialize, kernel.getTarget()))
      materialize.setPlacementAttr(builder.getStringAttr("reload"));
  });
  return mlir::failure(invalid);
}
