#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace weft::riscv;

namespace {
bool positiveProduct(ArrayRef<int64_t> shape, int64_t limit) {
  int64_t product = 1;
  for (int64_t size : shape) {
    if (size < 1 || size > limit / product) return false;
    product *= size;
  }
  return true;
}

bool uniqueAxes(ArrayRef<int64_t> axes) {
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t axis : axes)
    if (axis <= 0 || !seen.insert(axis).second) return false;
  return true;
}
}

LogicalResult ContractionTileAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, DenseI64ArrayAttr shape,
    int64_t laneAxis, int64_t laneInput, int64_t lanes, int64_t lmulEighths,
    int64_t scalarSteps, int64_t registers, int64_t loads, int64_t conversions) {
  if (shape.size() > 4 || !positiveProduct(shape.asArrayRef(), 65536))
    return error() << "contraction tile requires a bounded positive output shape";
  if (laneAxis <= 0 || (laneInput != 0 && laneInput != 1))
    return error() << "contraction tile requires an exact lane axis and operand";
  if (lanes <= 1 || lanes > 65536 || (lanes & (lanes - 1)) ||
      lmulEighths < 1 || lmulEighths > 64 || (lmulEighths & (lmulEighths - 1)))
    return error() << "contraction tile requires power-of-two lanes and legal bounded LMUL";
  if (!llvm::is_contained(ArrayRef<int64_t>({1, 2, 4}), scalarSteps))
    return error() << "contraction tile requires a finite scalar supply packet";
  if (registers < 1 || registers > 32 || loads < 0 || conversions < 0)
    return error() << "contraction tile has an invalid selected resource or supply count";
  return success();
}

LogicalResult ContractionEpilogueAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, StringRef mode,
    DenseI64ArrayAttr shape, int64_t laneAxis, int64_t lanes,
    int64_t lmulEighths, int64_t registers) {
  if (shape.size() > 4 || !positiveProduct(shape.asArrayRef(), 65536) || registers < 0 || registers > 32)
    return error() << "contraction epilogue requires bounded shape and resources";
  if (mode == "scalar") {
    if (laneAxis || lanes != 1 || lmulEighths || registers)
      return error() << "scalar contraction epilogue cannot contain a vector placement";
    return success();
  }
  if (mode != "direct" && mode != "vector")
    return error() << "unknown contraction epilogue representation";
  if (shape.empty() || laneAxis <= 0 || lanes <= 1 || (lanes & (lanes - 1)) ||
      lmulEighths < 1 || lmulEighths > 64 || (lmulEighths & (lmulEighths - 1)))
    return error() << "vector contraction epilogue requires exact lane and LMUL placement";
  return success();
}

LogicalResult ContractionCaseAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, int64_t singletonAxis,
    ContractionTileAttr tile, DenseI64ArrayAttr panelShape, int64_t panelK,
    ContractionTileAttr updateTile, ContractionTileAttr outputTile,
    ContractionEpilogueAttr directEpilogue, ContractionEpilogueAttr epilogue) {
  if (singletonAxis < 0)
    return error() << "contraction case uses zero for general or an existing positive singleton axis";
  if (epilogue.getShape().size() != tile.getShape().size() ||
      directEpilogue.getShape().size() != tile.getShape().size())
    return error() << "contraction epilogue must retain the selected output rank";
  if (panelShape.empty()) {
    if (panelK || updateTile != tile || outputTile != tile || directEpilogue != epilogue)
      return error() << "direct contraction case cannot carry panel-only decisions";
    return success();
  }
  if (singletonAxis || panelShape.size() != tile.getShape().size() ||
      panelShape.size() < 2 || !positiveProduct(panelShape.asArrayRef(), 4096) ||
      !llvm::is_contained(ArrayRef<int64_t>({32, 64, 128, 256}), panelK))
    return error() << "contraction panel requires a bounded general output domain and finite K span";
  if (updateTile.getShape().size() != panelShape.size() ||
      outputTile.getShape().size() != panelShape.size() ||
      updateTile.getLaneAxis() != tile.getLaneAxis() ||
      updateTile.getLaneInput() != tile.getLaneInput())
    return error() << "panel update must retain its selected input representation";
  return success();
}

LogicalResult ContractionPlanAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> error, DenseI64ArrayAttr outputAxes,
    DenseI64ArrayAttr reductionAxes, ArrayAttr cases, int64_t maximumLMUL,
    int64_t unroll, int64_t pipelineDepth, bool streamedStore) {
  if (outputAxes.size() > 4 || reductionAxes.empty() || reductionAxes.size() > 8 ||
      !uniqueAxes(outputAxes.asArrayRef()) || !uniqueAxes(reductionAxes.asArrayRef()))
    return error() << "contraction plan requires bounded unique numerical axes";
  for (int64_t axis : outputAxes.asArrayRef())
    if (llvm::is_contained(reductionAxes.asArrayRef(), axis))
      return error() << "contraction plan output and reduction coordinates must be disjoint";
  if (maximumLMUL < 1 || maximumLMUL > 64 || (maximumLMUL & (maximumLMUL - 1)) ||
      unroll < 1 || unroll > 16 || (pipelineDepth != 1 && pipelineDepth != 2))
    return error() << "contraction plan has an invalid target schedule binding";
  if (cases.empty() || cases.size() > outputAxes.size() + 1)
    return error() << "contraction plan must end in one general case";
  llvm::SmallDenseSet<int64_t> singletons;
  for (auto [index, attribute] : llvm::enumerate(cases)) {
    auto selected = mlir::dyn_cast<ContractionCaseAttr>(attribute);
    if (!selected) return error() << "contraction plan cases must be typed selections";
    int64_t singleton = selected.getSingletonAxis();
    if ((index + 1 == cases.size()) != (singleton == 0))
      return error() << "only the last contraction case may be general";
    if (singleton && (!streamedStore || !llvm::is_contained(outputAxes.asArrayRef(), singleton) ||
                      !singletons.insert(singleton).second))
      return error() << "singleton cases require distinct existing streamed output coordinates";
    if (!selected.getPanelShape().empty() && !streamedStore)
      return error() << "panel storage requires a selected streamed destination";
    for (auto epilogue : {selected.getDirectEpilogue(), selected.getEpilogue()}) {
      if (epilogue.getMode() == "scalar") continue;
      auto position = llvm::find(outputAxes.asArrayRef(), epilogue.getLaneAxis());
      if (position == outputAxes.asArrayRef().end() ||
          epilogue.getShape()[position - outputAxes.asArrayRef().begin()] % epilogue.getLanes())
        return error() << "contraction epilogue lane does not divide its selected output issue";
    }
    for (auto tile : {selected.getTile(), selected.getUpdateTile(), selected.getOutputTile()}) {
      if (tile.getShape().size() != outputAxes.size())
        return error() << "selected tile rank differs from the closed output domain";
      int64_t lane = tile.getLaneAxis();
      bool reduction = lane == reductionAxes.asArrayRef().back();
      if (!reduction && !llvm::is_contained(outputAxes.asArrayRef(), lane))
        return error() << "tile lane must be an output coordinate or the innermost reduction coordinate";
      if (reduction && tile.getScalarSteps() != 1)
        return error() << "reduction lanes cannot carry a scalar K packet";
    }
  }
  return success();
}
