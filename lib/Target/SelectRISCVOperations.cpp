#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVCompilation.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Contraction.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "Weft/Dialect/RISCV/IR/Numerics.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <memory>
using namespace weft;

namespace {

bool isShaped(mlir::Type type) {
  return mlir::isa<riscv::ValueType>(type);
}

bool isTrivialShaped(mlir::Type type) {
  auto value = mlir::dyn_cast<riscv::ValueType>(type);
  return value && llvm::all_of(value.getShape().asArrayRef(),
                               [](int64_t extent) { return extent == 1; });
}

unsigned width(mlir::Type type) {
  return riscv_internal::logicalBitWidth(type);
}

bool isFloat(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(riscv_internal::logicalElement(type));
}

bool isUnsigned(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(type));
  return integer && integer.isUnsigned();
}

bool isSigned(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(type));
  return integer && integer.isSigned();
}

bool supportsRVVType(riscv::TargetAttr target, mlir::Type type) {
  auto value = mlir::dyn_cast<riscv::ValueType>(type);
  if (!value)
    return true;
  mlir::Type element = value.getElementType();
  if (mlir::isa<kernel::EncodingType>(element))
    return true;
  const unsigned sew = std::max<unsigned>(8, width(type));
  if (!target.getHasRVV() || target.getVlenBits() <= 0 ||
      target.getVectorRegisters() <= 0 ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(), sew))
    return false;
  auto layout = value.getLayout();
  if (layout.getCarrier() == "rvv" &&
      (layout.getSew() != sew ||
       !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                           layout.getLmulEighths())))
    return false;
  return !element.isF16() || target.getHasVectorF16Conversion();
}

bool supportsRVVOperation(mlir::Operation *operation) {
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  if (!kernel)
    return false;
  if (!kernel.getTarget().getHasVectorF16() &&
      riscv_internal::requiresFullVectorF16(operation))
    return false;
  for (mlir::Type type : operation->getOperandTypes())
    if (!supportsRVVType(kernel.getTarget(), type))
      return false;
  for (mlir::Type type : operation->getResultTypes())
    if (!supportsRVVType(kernel.getTarget(), type))
      return false;
  return true;
}

bool hasEncodedFieldOrigin(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (!visited.insert(definition).second)
      return false;
    if (mlir::isa<riscv::FieldOp>(definition))
      return true;
    if (!mlir::isa<riscv::Fold2Op, riscv::CastOp, riscv::NarrowOp,
                   riscv::WidenOp, riscv::ConvertLayoutOp,
                   riscv::MaterializeOp, riscv::RegisterMaterializeOp>(
            definition) ||
        definition->getNumOperands() == 0)
      return false;
    value = definition->getOperand(0);
  }
  return false;
}

riscv::FragmentCapabilityAttr matchingFragment(mlir::Operation *operation,
                                               mlir::Value lhs,
                                               mlir::Value rhs,
                                               mlir::Type result) {
  auto signednessMatches = [](mlir::Type type, llvm::StringRef expected) {
    auto integer =
        mlir::dyn_cast<mlir::IntegerType>(riscv_internal::logicalElement(type));
    return integer && !integer.isSignless() &&
           ((expected == "signed" && integer.isSigned()) ||
            (expected == "unsigned" && integer.isUnsigned()));
  };
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
  if (!over || over.size() != 1)
    return {};
  const int64_t reduction = over[0];
  riscv::FragmentCapabilityAttr selected;
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(attribute);
    const auto *contract = findRISCVFragmentCapability(fragment.getInstruction());
    auto lhsAxes = riscv_internal::logicalAxes(lhs.getType());
    auto rhsAxes = riscv_internal::logicalAxes(rhs.getType());
    auto resultAxes = riscv_internal::logicalAxes(result);
    if (fragment.getLhsBits() == width(lhs.getType()) &&
        fragment.getRhsBits() == width(rhs.getType()) &&
        signednessMatches(lhs.getType(), fragment.getLhsSignedness()) &&
        signednessMatches(rhs.getType(), fragment.getRhsSignedness()) &&
        riscv::fragmentAccumulatorTypeMatches(fragment, riscv_internal::logicalElement(result)) &&
        lhsAxes.size() == 2 && rhsAxes.size() == 2 && resultAxes.size() == 2 &&
        llvm::is_contained(lhsAxes, resultAxes[0]) && llvm::is_contained(lhsAxes, reduction) &&
        llvm::is_contained(rhsAxes, resultAxes[1]) && llvm::is_contained(rhsAxes, reduction) &&
        reduction != resultAxes[0] && reduction != resultAxes[1] &&
        contract && supportsRISCVFragmentTile(*contract,
            riscv_internal::physicalExtent(lhs, resultAxes[0]),
            riscv_internal::physicalExtent(rhs, resultAxes[1]),
            riscv_internal::physicalExtent(lhs, reduction)) &&
        riscv_internal::physicalExtent(lhs, reduction) == fragment.getKFactor() &&
        riscv_internal::physicalExtent(rhs, reduction) == fragment.getKFactor() &&
        riscv_internal::physicalExtent(operation->getResult(0), resultAxes[0]) ==
            riscv_internal::physicalExtent(lhs, resultAxes[0]) &&
        riscv_internal::physicalExtent(operation->getResult(0), resultAxes[1]) ==
            riscv_internal::physicalExtent(rhs, resultAxes[1]))
      if (!selected || fragment.getActiveMFactor() * fragment.getNFactor() <
                           selected.getActiveMFactor() * selected.getNFactor())
        selected = fragment;
  }
  return selected;
}

riscv::ImplementationAttr scalarImplementation(mlir::Builder &builder,
                                               llvm::StringRef family,
                                               llvm::StringRef operation) {
  return riscv_internal::implementation(builder, "scalar", family, operation);
}

riscv::ImplementationAttr rvvImplementation(
    mlir::Builder &builder, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters = {}) {
  return riscv_internal::implementation(builder, "rvv", family, operation,
                                        parameters);
}

void selectLoopCarriedWidenProducts(mlir::ModuleOp module,
                                    mlir::IRRewriter &rewriter) {
  module.walk([&](mlir::scf::ForOp loop) {
    auto kernel = loop->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      return;
    auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
      auto type = mlir::dyn_cast<riscv::ValueType>(argument.getType());
      auto element = type ? mlir::dyn_cast<mlir::IntegerType>(type.getElementType())
                          : mlir::IntegerType();
      if (!type || !element || !element.isSigned() || element.getWidth() != 16 ||
          type.getShape().size() != 1 || type.getShape()[0] <= 1)
        continue;
      auto layout = type.getLayout();
      if (layout.getCarrier() != "rvv" || layout.getValidity() != "full" ||
          layout.getTimeFactors()[0] != 1 ||
          layout.getLaneFactors()[0] != type.getShape()[0] ||
          layout.getReplicaFactors()[0] != 1 ||
          layout.getFragmentFactors()[0] != 1 ||
          layout.getLocalFactors()[0] != 1)
        continue;

      llvm::SmallVector<std::pair<riscv::BinaryOp, riscv::RVVWidenMultiplyOp>> chain;
      mlir::Value current = argument;
      while (current.hasOneUse()) {
        mlir::Operation *user = *current.getUsers().begin();
        if (user == yield.getOperation())
          break;
        auto add = mlir::dyn_cast<riscv::BinaryOp>(user);
        if (!add || add.getKind() != "add" ||
            add->getBlock() != loop.getBody() || add.getResult().getType() != type)
          break;
        mlir::Value other = add.getLhs() == current ? add.getRhs() : add.getLhs();
        auto product = other.getDefiningOp<riscv::RVVWidenMultiplyOp>();
        if (!product || !other.hasOneUse() || other.getType() != type ||
            product->getBlock() != loop.getBody())
          break;
        auto lhs = product.getLhs().getType();
        auto rhs = product.getRhs().getType();
        if (width(lhs) > 8 || width(rhs) > 8 ||
            (!isSigned(lhs) && !isSigned(rhs)) ||
            lhs.getLayout().getValidity() != "full" ||
            rhs.getLayout().getValidity() != "full")
          break;
        chain.emplace_back(add, product);
        current = add.getResult();
      }
      // Select the entire carried chain together; a prefix would not satisfy
      // the existing partial operation's closed-chain contract.
      if (chain.empty() || !current.hasOneUse() ||
          *current.getUsers().begin() != yield.getOperation() ||
          yield.getOperand(index) != current)
        continue;
      for (auto [add, product] : chain) {
        mlir::Value accumulator = add.getLhs() == product.getResult()
                                      ? add.getRhs() : add.getLhs();
        auto lhs = product.getLhs().getType().getLayout();
        auto rhs = product.getRhs().getType().getLayout();
        rewriter.setInsertionPoint(add);
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            add.getLoc(), type, product.getLhs(), product.getRhs(), accumulator,
            type.getAxisIds(),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial", lhs.getRegisterGroups() +
                    rhs.getRegisterGroups() + layout.getRegisterGroups(),
                layout.getRegisterGroups(), 0, 0, "none", "agnostic"));
        riscv_internal::copyOrigin(add, accumulate);
        rewriter.replaceOp(add, accumulate.getResult());
        rewriter.eraseOp(product);
      }
    }
  });
}

void selectZeroSeedProducts(mlir::ModuleOp module,
                            mlir::IRRewriter &rewriter) {
  // Seed a full byte-product chain without keeping a zero vector live.
  // Other partial representations retain their selected accumulator form.
  llvm::SmallVector<riscv::RVVWidenAccumulateOp> accumulates;
  module.walk([&](riscv::RVVWidenAccumulateOp accumulate) {
    accumulates.push_back(accumulate);
  });
  for (riscv::RVVWidenAccumulateOp accumulate : accumulates) {
    auto kernel = accumulate->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      continue;
    auto seed = accumulate.getAccumulator().getDefiningOp<riscv::RVVSplatOp>();
    auto constant = seed ? seed.getScalar().getDefiningOp<riscv::ConstantOp>()
                         : riscv::ConstantOp();
    auto zero = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
    if (!zero || !zero.getValue().isZero())
      continue;
    auto lhs = accumulate.getLhs().getType();
    auto rhs = accumulate.getRhs().getType();
    auto result = accumulate.getResult().getType();
    auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
    auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
    auto narrow = lhs.getLayout();
    auto wide = result.getLayout();
    auto next = accumulate.getResult().hasOneUse()
        ? mlir::dyn_cast<riscv::RVVWidenAccumulateOp>(
              *accumulate.getResult().getUsers().begin())
        : riscv::RVVWidenAccumulateOp();
    if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
        rhsElement.isSignless() ||
        (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
        lhsElement.getWidth() > 8 || rhsElement.getWidth() > 8 ||
        narrow.getSew() != 8 || wide.getSew() != 16 ||
        narrow.getValidity() != "full" || wide.getValidity() != "full" ||
        !next || next.getAccumulator() != accumulate.getResult() ||
        next.getReductionAxes() != accumulate.getReductionAxes() ||
        next->getBlock() != accumulate->getBlock() ||
        lhs.getShape() != rhs.getShape() ||
        lhs.getShape() != result.getShape() ||
        lhs.getAxisIds() != rhs.getAxisIds() ||
        lhs.getAxisIds() != result.getAxisIds() || narrow != rhs.getLayout() ||
        narrow.getTimeFactors() != wide.getTimeFactors() ||
        narrow.getLaneFactors() != wide.getLaneFactors() ||
        narrow.getReplicaFactors() != wide.getReplicaFactors() ||
        narrow.getFragmentFactors() != wide.getFragmentFactors() ||
        narrow.getLocalFactors() != wide.getLocalFactors())
      continue;
    llvm::StringRef instruction = lhsElement.isSigned() && rhsElement.isSigned()
        ? "rvv.vwmul.vv"
        : lhsElement.isSigned() ? "rvv.vwmulsu.vv"
                                : "rvv.vwmulsu.vv.swap";
    rewriter.setInsertionPoint(accumulate);
    auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
        accumulate.getLoc(), result, accumulate.getLhs(), accumulate.getRhs(),
        riscv_internal::leaf(
            rewriter, "rvv", "widen-multiply", instruction, instruction,
            narrow.getRegisterGroups() + rhs.getLayout().getRegisterGroups(),
            wide.getRegisterGroups(), 0, 0, "none", "exact"));
    riscv_internal::copyOrigin(accumulate, product);
    accumulate.getResult().replaceAllUsesWith(product.getResult());
    rewriter.eraseOp(accumulate);
    if (seed.getResult().use_empty())
      rewriter.eraseOp(seed);
  }
}

void selectWidenContractSteps(mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  module.walk([&](riscv::RVVContractStepOp step) {
    auto kernel = step->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() || !kernel.getTarget().getHasWideningInteger() ||
        step.getLeaf().getInstruction() != "rvv.vmacc.vx") return;
    auto lhs = step.getLhs().getDefiningOp<riscv::WidenOp>();
    auto rhs = step.getRhs().getDefiningOp<riscv::WidenOp>();
    if (!lhs || !rhs || lhs == rhs || !lhs->hasOneUse() || !rhs->hasOneUse()) return;
    auto geometry = riscv::ContractStepGeometry::get(lhs.getInput().getType(), rhs.getInput().getType(),
        step.getResult().getType(), step.getReductionAxis(), step.getLaneOperand(), "rvv.vwmacc.vx");
    if (!geometry) return;
    step.getLhsMutable().assign(lhs.getInput());
    step.getRhsMutable().assign(rhs.getInput());
    step.setLeafAttr(riscv_internal::leaf(rewriter, "rvv", "contract-step", "rvv.vwmacc.vx", "rvv.vwmacc.vx",
        0, 0, 0, 0, "none", step.getActive() ? "undisturbed" : "agnostic",
        {static_cast<int64_t>(step.getReductionAxis())}));
    rewriter.eraseOp(lhs);
    rewriter.eraseOp(rhs);
  });
}

void selectWidenBinaries(mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  module.walk([&](riscv::BinaryOp binary) {
    const bool multiply = binary.getKind() == "mul";
    if (binary.getKind() != "add" && !multiply)
      return;
    auto lhs = binary.getLhs().getDefiningOp<riscv::WidenOp>();
    auto rhs = binary.getRhs().getDefiningOp<riscv::WidenOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(binary.getResult().getType());
    auto kernel = binary->getParentOfType<riscv::KernelOp>();
    if (!lhs || !rhs || !result || !kernel ||
        !lhs.getResult().hasOneUse() || !rhs.getResult().hasOneUse() ||
        lhs.getResult().getType() != result || rhs.getResult().getType() != result)
      return;
    auto left = mlir::dyn_cast<riscv::ValueType>(lhs.getInput().getType());
    auto right = mlir::dyn_cast<riscv::ValueType>(rhs.getInput().getType());
    if (kernel.getResourcesMaterialized() ||
        !riscv::supportsRVVWidenBinary(kernel.getTarget(), left, right, result))
      return;
    auto element = mlir::cast<mlir::IntegerType>(left.getElementType());
    if (multiply && (element.getWidth() > 16 ||
                     result.getLayout().getValidity() != "full"))
      return;
    const llvm::StringRef instruction =
        multiply ? (element.isUnsigned() ? "rvv.vwmulu.vv" : "rvv.vwmul.vv")
                 : (element.isUnsigned() ? "rvv.vwaddu.vv" : "rvv.vwadd.vv");
    rewriter.setInsertionPoint(binary);
    auto leaf = riscv_internal::leaf(
            rewriter, "rvv", multiply ? "widen-multiply" : "widen-add", instruction, instruction,
            left.getLayout().getRegisterGroups() +
                right.getLayout().getRegisterGroups(),
            result.getLayout().getRegisterGroups(), 0, 0, "none",
            result.getLayout().getValidity() == "tail" ? "agnostic" : "exact");
    mlir::Operation *selected;
    if (multiply)
      selected = rewriter.create<riscv::RVVWidenMultiplyOp>(
          binary.getLoc(), result, lhs.getInput(), rhs.getInput(), leaf);
    else
      selected = rewriter.create<riscv::RVVWidenAddOp>(
          binary.getLoc(), result, lhs.getInput(), rhs.getInput(), leaf);
    riscv_internal::copyOrigin(binary, selected);
    rewriter.replaceOp(binary, selected->getResults());
    rewriter.eraseOp(lhs);
    rewriter.eraseOp(rhs);
  });
}

void coalescePackedScaleSlices(mlir::ModuleOp module,
                               mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::RVVPartialPackedScaleOp> operations;
  module.walk([&](riscv::RVVPartialPackedScaleOp op) { operations.push_back(op); });
  const auto completeVector = [](riscv::ValueType type) {
    auto layout = type.getLayout();
    return type.getShape().size() == 1 && type.getShape()[0] > 0 &&
           layout.getCarrier() == "rvv" && layout.getValidity() == "full" &&
           layout.getTimeFactors()[0] == 1 &&
           layout.getLaneFactors()[0] == type.getShape()[0] &&
           layout.getReplicaFactors()[0] == 1 &&
           layout.getFragmentFactors()[0] == 1 &&
           layout.getLocalFactors()[0] == 1 && layout.getVl() == type.getShape()[0];
  };
  for (auto operation : operations) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() ||
        operation.getScales().size() < 2 || operation.getScales().size() > 8)
      continue;
    mlir::Value source;
    int64_t base = 0, lanes = 0;
    bool complete = true;
    for (mlir::Value scale : operation.getScales()) {
      auto type = mlir::cast<riscv::ValueType>(scale.getType());
      if (!completeVector(type) || type.getShape()[0] > 8 - lanes) {
        complete = false;
        break;
      }
      if (auto conversion = scale.getDefiningOp<riscv::ConvertLayoutOp>()) {
        auto input = conversion.getInput().getType();
        if (conversion.getConversion().getEffect() != "pure" ||
            conversion.getSourceAccess() || !completeVector(input) ||
            input.getShape() != type.getShape() || input.getAxisIds() != type.getAxisIds() ||
            input.getElementType() != type.getElementType() ||
            input.getLayout().getSew() != type.getLayout().getSew()) {
          complete = false;
          break;
        }
        scale = conversion.getInput();
      }
      auto slice = scale.getDefiningOp<riscv::RVVIssueSliceOp>();
      if (!slice || slice.getSourceParts().size() != 1 ||
          slice.getSourceParts()[0] != 0 || slice.getLaneOffsets().size() != 1 ||
          !completeVector(slice.getInput().getType())) {
        complete = false;
        break;
      }
      if (!source) {
        source = slice.getInput();
        base = slice.getLaneOffsets()[0];
      }
      if (slice.getInput() != source || base > INT64_MAX - lanes ||
          slice.getLaneOffsets()[0] != base + lanes) {
        complete = false;
        break;
      }
      lanes += type.getShape()[0];
    }
    if (!complete || !source || lanes <= 0 || base % lanes)
      continue;
    auto sourceType = mlir::cast<riscv::ValueType>(source.getType());
    if (sourceType.getShape()[0] % lanes ||
        base > sourceType.getShape()[0] - lanes)
      continue;
    auto one = rewriter.getDenseI64ArrayAttr({1});
    auto shape = rewriter.getDenseI64ArrayAttr({lanes});
    auto layout = riscv::LayoutAttr::get(
        rewriter.getContext(), "rvv", sourceType.getAxisIds(), one, shape,
        one, one, one, 32, 8, lanes, 1, "full");
    auto scaleType = riscv::ValueType::get(
        rewriter.getContext(), sourceType.getElementType(), shape,
        sourceType.getAxisIds(), layout);
    llvm::SmallVector<riscv::PartialSetType> inputs;
    int64_t inputGroups = 0;
    for (mlir::Value input : operation.getInputs()) {
      auto type = mlir::cast<riscv::PartialSetType>(input.getType());
      inputs.push_back(type);
      inputGroups += type.getResourceGroups();
    }
    if (sourceType.getLayout().getLmulEighths() < 8 ||
        !riscv::supportsRVVPartialPackedScale(
            kernel.getTarget(), inputs, {scaleType}, operation.getResult().getType()))
      continue;
    rewriter.setInsertionPoint(operation);
    auto combined = rewriter.create<riscv::RVVIssueSliceOp>(
        operation.getLoc(), scaleType, source, rewriter.getDenseI64ArrayAttr({0}),
        rewriter.getDenseI64ArrayAttr({base}),
        riscv_internal::leaf(
            rewriter, "transfer", "issue-slice", "rvv.issue-slice", "rvv.issue-slice",
            sourceType.getLayout().getRegisterGroups(), 1,
            sourceType.getLayout().getRegisterGroups(), 0, "none", "exact"));
    riscv_internal::copyOrigin(operation, combined);
    const int64_t eliminatedPacks = operation.getScales().size() - 1;
    operation.getScalesMutable().assign(mlir::ValueRange{combined.getResult()});
    operation.setLeafAttr(riscv_internal::leaf(
        rewriter, "rvv", "partial-packed-scale", "rvv.partial-packed-scale",
        "rvv.partial-packed-scale", inputGroups + 1, 1, 3, 0, "none", "exact"));
    if (auto costs = operation->getAttrOfType<mlir::DenseI64ArrayAttr>(
            "weft.riscv.packed_scale_cost"); costs && costs.size() == 2)
      operation->setAttr("weft.riscv.packed_scale_cost",
                        rewriter.getDenseI64ArrayAttr(
                            {costs[0] - eliminatedPacks, costs[1]}));
  }
}

void selectContiguousRegisterExtracts(mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::ExtractOp> extracts;
  module.walk([&](riscv::ExtractOp op) { extracts.push_back(op); });
  for (auto op : extracts) {
    auto kernel = op->getParentOfType<riscv::KernelOp>();
    auto input = op.getInput().getType();
    auto result = mlir::dyn_cast<riscv::ValueType>(op.getResult().getType());
    if (!kernel || kernel.getResourcesMaterialized() || !result ||
        op.getAccess().getMapping() != "dense" || hasEncodedFieldOrigin(op.getInput()) ||
        input.getShape().size() != 1 || result.getShape().size() != 1 ||
        input.getAxisIds() != result.getAxisIds() ||
        input.getElementType() != result.getElementType() ||
        op.getSelectors().size() != 1 || op.getIndices().size() != 1 ||
        mlir::cast<mlir::StringAttr>(op.getSelectors()[0]).getValue() != "gather")
      continue;
    auto a = input.getLayout(), b = result.getLayout();
    if (a.getCarrier() != "rvv" || b.getCarrier() != "rvv" ||
        a.getValidity() != "full" || b.getValidity() != "full" ||
        a.getReplicaGroup() != 1 || b.getReplicaGroup() != 1 ||
        a.getReplicaFactors()[0] != 1 || b.getReplicaFactors()[0] != 1 ||
        a.getFragmentFactors()[0] != 1 || b.getFragmentFactors()[0] != 1 ||
        a.getLocalFactors()[0] != 1 || b.getLocalFactors()[0] != 1 ||
        b.getTimeFactors()[0] != 1 || result.getShape()[0] != b.getLaneFactors()[0] ||
        result.getShape()[0] <= 0 || input.getShape()[0] < result.getShape()[0] ||
        a.getLaneFactors()[0] % result.getShape()[0] ||
        (a.getTimeFactors()[0] != 1 && a.getLaneFactors()[0] != result.getShape()[0]) ||
        a.getLmulEighths() < b.getLmulEighths() ||
        a.getLmulEighths() % b.getLmulEighths())
      continue;
    auto base = riscv_internal::constantContiguousIndexBase(op.getIndices()[0]);
    if (!base || *base > input.getShape()[0] - result.getShape()[0] ||
        *base % result.getShape()[0])
      continue;
    rewriter.setInsertionPoint(op);
    auto slice = rewriter.create<riscv::RVVIssueSliceOp>(
        op.getLoc(), result, op.getInput(),
        rewriter.getDenseI64ArrayAttr({*base / a.getLaneFactors()[0]}),
        rewriter.getDenseI64ArrayAttr({*base % a.getLaneFactors()[0]}),
        riscv_internal::leaf(rewriter, "transfer", "issue-slice", "rvv.issue-slice", "rvv.issue-slice",
            a.getRegisterGroups(), b.getRegisterGroups(),
            (a.getLmulEighths() + 7) / 8, 0, "none", "exact"));
    riscv_internal::copyOrigin(op, slice);
    rewriter.replaceOp(op, slice.getResult());
  }
}

void selectRegisterSlices(mlir::ModuleOp module, mlir::Builder &builder) {
  module.walk([&](riscv::RVVIssueSliceOp slice) {
    auto kernel = slice->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() ||
        !riscv::supportsRVVRegisterSlice(kernel.getTarget(), slice.getInput().getType(),
                                        slice.getResult().getType(), slice.getLaneOffsets()))
      return;
    slice.setLeafAttr(riscv_internal::leaf(
        builder, "transfer", "issue-slice", "rvv.issue-slice.split", "rvv.issue-slice.split",
        slice.getInput().getType().getLayout().getRegisterGroups(),
        slice.getResult().getType().getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
  });
}

class SelectRISCVOperationsPass
    : public mlir::PassWrapper<SelectRISCVOperationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-select-operations";
  }
  llvm::StringRef getDescription() const override {
    return "Select target-local operation families from typed numerical facts";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    selectLoopCarriedWidenProducts(getOperation(), rewriter);
    selectZeroSeedProducts(getOperation(), rewriter);
    selectWidenContractSteps(getOperation(), rewriter);
    selectWidenBinaries(getOperation(), rewriter);
    coalescePackedScaleSlices(getOperation(), rewriter);
    selectContiguousRegisterExtracts(getOperation(), rewriter);
    selectRegisterSlices(getOperation(), rewriter);
    mlir::Builder builder(&getContext());
    bool failed = false;
    getOperation().walk([&](mlir::Operation *operation) {
      auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
      if (failed || (!leaf && !mlir::isa<riscv::NewOp>(operation)))
        return;
      // Memory/layout owners still have to decide whether an f16 value is a
      // storage projection, a load, or a register movement. Do not reject their
      // unselected placeholders as floating arithmetic.
      if (leaf && leaf.getEngine() == "unselected" &&
          mlir::isa<riscv::LoadOp, riscv::StoreOp, riscv::FieldOp,
                    riscv::ExtractOp, riscv::LookupOp,
                    riscv::RVVIndexedEntryLoadOp,
                    riscv::RVVUnitEntryWindowLoadOp,
                    riscv::RVVBitmaskWindowLoadOp,
                    riscv::ConvertLayoutOp>(operation))
        return;
      auto kernel = operation->getParentOfType<riscv::KernelOp>();
      if (kernel && !kernel.getTarget().getHasVectorF16() &&
          riscv_internal::requiresFullVectorF16(operation)) {
        rejectRISCVCandidate(operation, RISCVFailureCategory::Unsupported,
            "selected half-vector operation requires full Zvfh; Zvfhmin supports only half storage and f16/f32 conversion");
        failed = true;
        return;
      }
      if (!leaf || leaf.getEngine() != "unselected") return;
      // Memory forms and their terminal transfer leaves have one producer:
      // PlanRISCVMemory, after layouts are known.  Selection establishes only
      // numerical/structured operation anchors.
      if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation);
          update && update.getInput().getType().getLayout().getCarrier() == "local")
        return;
      riscv::ImplementationAttr selected = select(operation, builder);
      if (!selected) {
        auto diagnostic = operation->emitError()
                          << "SelectRISCVOperations has no legal target-local operation for '"
                          << operation->getName() << "'";
        if (operation->getNumResults() == 1)
          diagnostic << " with result " << operation->getResult(0).getType();
        failed = true;
        return;
      }
      operation->setAttr("implementation", selected);
    });
    if (failed)
      signalPassFailure();
  }

private:
  riscv::ImplementationAttr select(mlir::Operation *operation,
                                   mlir::Builder &builder) {
    auto chooseScalarOrRVV = [&](llvm::StringRef family,
                                 llvm::StringRef scalarInstruction,
                                 llvm::StringRef vectorInstruction,
                                 mlir::Type result) {
      auto value = mlir::dyn_cast<riscv::ValueType>(result);
      if (!value || isTrivialShaped(result) ||
          value.getLayout().getCarrier() == "scalar")
        return scalarImplementation(builder, family, scalarInstruction);
      if (value.getLayout().getCarrier() != "unassigned" &&
          value.getLayout().getCarrier() != "rvv")
        return riscv::ImplementationAttr();
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, family, vectorInstruction)
                 : riscv::ImplementationAttr();
    };

    if (auto op = mlir::dyn_cast<riscv::IotaOp>(operation)) {
      auto result = mlir::cast<riscv::ValueType>(op.getResult().getType());
      if (isTrivialShaped(result) || result.getLayout().getCarrier() == "scalar")
        return scalarImplementation(builder, "iota", "register.iota");
      if (result.getLayout().getCarrier() != "unassigned" &&
          result.getLayout().getCarrier() != "rvv")
        return {};
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, "iota", "rvv.vid")
                 : riscv::ImplementationAttr();
    }
    if (auto op = mlir::dyn_cast<riscv::UpdateOp>(operation))
      return chooseScalarOrRVV("state-update", "scalar.update", "rvv.merge",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
      if (op.getKind() == "log") {
        auto element = riscv_internal::logicalElement(op.getResult().getType());
        auto value = mlir::dyn_cast<riscv::ValueType>(op.getResult().getType());
        if (element.isF32() && value && value.getLayout().getCarrier() != "scalar" && !isTrivialShaped(value))
          return rvvImplementation(builder, "log-approx", riscv::vectorMathInstruction(op));
        if ((!element.isF32() && !element.isF64()) ||
            (value && !isTrivialShaped(value) &&
             value.getLayout().getCarrier() != "scalar" &&
             !(value.getLayout().getCarrier() == "unassigned" &&
               riscv_internal::staticProduct(value.getShape().asArrayRef()))))
          return {};
        return scalarImplementation(builder, "unary", element.isF32() ? "scalar.log-approx-f32" : "scalar.log");
      }
      if (op.getKind() == "rsqrt" || op.getKind() == "sqrt") {
        auto element = riscv_internal::logicalElement(op.getResult().getType());
        if (!element.isF32() && !element.isF64())
          return {};
      }
      std::string scalar = ("scalar." + op.getKind()).str();
      std::string vector = ("rvv.v" + op.getKind()).str();
      auto result = mlir::dyn_cast<riscv::ValueType>(op.getResult().getType());
      if (op.getKind() == "exp2" && result &&
          result.getLayout().getCarrier() != "scalar" && !isTrivialShaped(result)) {
        auto approximate = op->getAttrOfType<mlir::BoolAttr>("approximate");
        if (!approximate || !approximate.getValue() ||
            !riscv_internal::logicalElement(op.getResult().getType()).isF32())
          return {};
        return rvvImplementation(builder, "exp2-approx", "rvv.exp2-approx-f32");
      }
      if (op.getKind() == "exp" && result &&
          result.getLayout().getCarrier() != "scalar" &&
          riscv_internal::logicalElement(op.getResult().getType()).isF32())
        return rvvImplementation(builder, "exp-approx",
                                 riscv::vectorMathInstruction(op));
      if (op.getKind() == "exp" && riscv_internal::logicalElement(op.getResult().getType()).isF32())
        return scalarImplementation(builder, "exp-approx", "scalar.exp-approx-f32");
      return chooseScalarOrRVV("unary", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
      std::string scalar = ("scalar." + op.getKind()).str();
      std::string vector = ("rvv.v" + op.getKind()).str();
      return chooseScalarOrRVV("pointwise", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::CompareOp>(operation)) {
      std::string scalar = ("scalar.cmp." + op.getPredicate()).str();
      std::string vector = ("rvv.vm" + op.getPredicate()).str();
      return chooseScalarOrRVV("compare", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::SelectOp>(operation))
      return chooseScalarOrRVV("select", "scalar.select", "rvv.select", op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::CastOp>(operation))
      return chooseScalarOrRVV("cast", "scalar.cast", "rvv.convert",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::BitcastOp>(operation))
      return chooseScalarOrRVV("bitcast", "scalar.bitcast", "rvv.bitcast",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::NarrowOp>(operation))
      return chooseScalarOrRVV("narrow", "scalar.narrow", "rvv.vnclip",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::WidenOp>(operation))
      return chooseScalarOrRVV("widen", "scalar.widen", "rvv.widen",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::MacGroupsOp>(operation)) {
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      if (!supportsRVVOperation(operation))
        return {};
      const bool mixedUnsignedSigned =
          (isUnsigned(op.getLhs().getType()) &&
           isSigned(op.getRhs().getType())) ||
          (isSigned(op.getLhs().getType()) &&
           isUnsigned(op.getRhs().getType()));
      if (mixedUnsignedSigned && width(op.getLhs().getType()) <= 8 &&
          width(op.getRhs().getType()) <= 8 &&
          target.getHasWideningInteger())
        return rvvImplementation(
            builder, "grouped-mac", "rvv.vwmaccsu.typed",
            {static_cast<int64_t>(op.getGroup())});
      return {};
    }
    if (auto op = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
      auto input = mlir::dyn_cast<riscv::ValueType>(op.getInput().getType());
      if (!input)
        return {};
      if (input.getLayout().getCarrier() == "scalar") {
        std::string instruction =
            ("scalar.register-reduce." + op.getKind()).str();
        return riscv_internal::implementation(
            builder, "scalar", "reduce", instruction,
            {static_cast<int64_t>(op.getAxis())});
      }
      if (!supportsRVVOperation(operation))
        return {};
      std::string instruction =
          isFloat(op.getResult().getType())
              ? ("rvv.vfred" + op.getKind()).str()
              : ("rvv.vred" + op.getKind()).str();
      return rvvImplementation(builder, "reduce", instruction,
                               {static_cast<int64_t>(op.getAxis())});
    }
    if (auto op = mlir::dyn_cast<riscv::Fold2Op>(operation))
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, "fold2", "rvv.pair-fold-add")
                 : riscv::ImplementationAttr();
    auto selectContract = [&](mlir::Value lhs, mlir::Value rhs,
                              mlir::Type result, bool permitIME,
                              bool permitWidenDot,
                              llvm::ArrayRef<int64_t> over)
        -> riscv::ImplementationAttr {
      if (permitIME)
        if (auto fragment = matchingFragment(operation, lhs, rhs, result))
          return riscv_internal::implementation(
              builder, "ime", "fragment-mma", fragment.getInstruction(),
              {fragment.getMFactor(), fragment.getNFactor(),
               fragment.getKFactor(),
               riscv_internal::physicalExtent(lhs, riscv_internal::logicalAxes(result)[0]) /
                   fragment.getActiveMFactor(),
               riscv_internal::physicalExtent(rhs, riscv_internal::logicalAxes(result)[1]) /
                   fragment.getNFactor()});
      if (!supportsRVVOperation(operation))
        return {};
      auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(lhs.getType()));
      auto rhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(rhs.getType()));
      auto resultInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(result));
      auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(lhs.getType()));
      auto rhsFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(rhs.getType()));
      auto resultFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(result));
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      auto wideningDotFitsRegisterMicrotile = [&]() {
        if (!mlir::isa<riscv::OuterContractOp>(operation) &&
            (riscv_internal::logicalAxes(result).empty() || over.size() != 1))
          return true;
        auto value = mlir::dyn_cast<riscv::ValueType>(result);
        if (!value)
          return false;
        int64_t replicas = 1;
        for (int64_t axis : value.getAxisIds().asArrayRef()) {
          int64_t extent = riscv_internal::physicalExtent(
              operation->getResult(0), axis);
          if (extent <= 0 ||
              replicas > target.getVectorRegisters() / extent)
            return false;
          replicas *= extent;
        }
        // An 8/16-bit widening product needs at least two register groups per
        // accumulator at LMUL=m1, plus one group for each operand.  This is the
        // minimum resource proof available before layout propagation; the
        // exact selected LMUL is checked again by LowerRISCVComposites.
        return replicas > 0 &&
               replicas * 2 + 2 <= target.getVectorRegisters();
      };
      // The widening-dot family follows from the typed numerical relation and
      // target capability.  It must be selected before layout propagation so
      // its reduction axis can anchor the operand layouts.  The concrete
      // doubled-LMUL legality is checked after those layouts exist, when the
      // composite is lowered.
      if (permitWidenDot && lhsInteger && rhsInteger && resultInteger &&
          lhsInteger.getWidth() <= 16 && rhsInteger.getWidth() <= 16 &&
          std::max<unsigned>(8, lhsInteger.getWidth()) ==
              std::max<unsigned>(8, rhsInteger.getWidth()) &&
          (lhsInteger.isSigned() || rhsInteger.isSigned()) &&
          resultInteger.isSigned() && resultInteger.getWidth() == 32 &&
          !over.empty() && target.getHasWideningInteger() &&
          wideningDotFitsRegisterMicrotile())
        return rvvImplementation(builder, "widen-dot",
                                 "rvv.vwmul-vwredsum", over);
      if (lhsFloat && rhsFloat && resultFloat && lhsFloat.isF16() &&
          rhsFloat.isF16() && resultFloat.isF32() &&
          target.getHasVectorF16())
        return rvvImplementation(builder, "widen-float-contract",
                                 "rvv.vfwmacc", over);
      llvm::StringRef instruction = isFloat(result) ? "rvv.vfmacc"
                                                    : "rvv.vmacc";
      return rvvImplementation(builder, "contract", instruction, over);
    };
    if (auto op = mlir::dyn_cast<riscv::DotOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            false, true, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::ContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, true, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::OuterContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, true, op.getOver());
    return {};
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVOperationsPass() {
  return std::make_unique<SelectRISCVOperationsPass>();
}
