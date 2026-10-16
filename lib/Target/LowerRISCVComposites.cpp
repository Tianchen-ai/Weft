#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "ContractionSupply.h"
#include "DenseValuePrograms.h"
#include "LocalMemory.h"
#include "DenseContractions.h"
#include "IME/FragmentMaterialization.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <numeric>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>

using namespace weft;

namespace {

using riscv_internal::accessOf;
using riscv_internal::sourceField;
using riscv_internal::sourceLoad;
using riscv_internal::projectedReductionType;
using riscv_internal::projectContractionRead;
using riscv_internal::queryContractionSupply;
using riscv_internal::materializeContractionReadView;

int64_t product(mlir::DenseI64ArrayAttr values) {
  return riscv_internal::staticProduct(values.asArrayRef()).value_or(-1);
}

int64_t product(std::initializer_list<int64_t> values) {
  return riscv_internal::staticProduct(values).value_or(-1);
}

int64_t sum(int64_t lhs, int64_t rhs) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs)
    return -1;
  return lhs + rhs;
}

int64_t resultParts(riscv::LayoutAttr layout) {
  return product(
      {product(layout.getTimeFactors()), product(layout.getReplicaFactors())});
}

std::optional<size_t> axisPosition(riscv::ValueType value, int64_t axis) {
  auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
  if (found == value.getAxisIds().asArrayRef().end())
    return std::nullopt;
  return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
}

using WidenDotLaneSlicePlan = riscv_internal::WidenDotLaneSlicePlan;

std::optional<WidenDotLaneSlicePlan>
planWidenDotLaneSlices(riscv::ValueType operand, riscv::ValueType result,
                       llvm::ArrayRef<int64_t> reductionAxes,
                       riscv::TargetAttr target) {
  return riscv_internal::planWidenDotLaneSlices(operand, result, reductionAxes,
                                                 target);
}

std::optional<int64_t> constantInteger(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

void copyIdentity(mlir::Operation *source, mlir::Operation *target) {
  if (auto origin = source->getAttr("source_origin"))
    target->setAttr("source_origin", origin);
  if (auto canonical = source->getAttr("canonical_op"))
    target->setAttr("canonical_op", canonical);
}

class CompositeLowering {
public:
  explicit CompositeLowering(mlir::ModuleOp module) : module(module) {}

  mlir::LogicalResult prepare() {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    lowerMaterializations(rewriter, failed);
    lowerLocalStates(rewriter, failed);
    lowerLoops(rewriter, failed);
    lowerLocalGroupExtracts(rewriter, failed);
    closeOrderedLoops(failed);
    lowerPartitionedWidenReduceStores(rewriter, failed);
    lowerWidenReductions(rewriter, failed);
    lowerReciprocalSquareRoots(rewriter, failed);
    lowerPropagatingExtrema(rewriter, failed);
    return mlir::failure(failed);
  }

  mlir::LogicalResult finish(int64_t maximumLMUL) {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    if (mlir::failed(riscv_internal::materializeDenseContractions(module, rewriter, maximumLMUL)) ||
        mlir::failed(riscv_internal::materializeDenseValuePrograms(module)))
      return mlir::failure();
    lowerGroupedReductions(rewriter, failed);
    lowerLoopCarriedWideningPartials(rewriter, failed);
    lowerContracts(rewriter, failed);
    eraseDeadComposites(rewriter);
    return mlir::failure(failed);
  }

private:
  mlir::ModuleOp module;
  mlir::ModuleOp getOperation() { return module; }
  mlir::MLIRContext &getContext() { return *module.getContext(); }
  static int64_t ownerDomain(mlir::Operation *operation) {
    for (mlir::Operation *parent = operation; parent;
         parent = parent->getParentOp()) {
      if (auto level =
              parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
        return level.getDomainId();
    }
    return 0;
  }

  int64_t nextPartialBirthId = 0;

  void lowerPropagatingExtrema(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::BinaryOp> operations;
    getOperation().walk([&](riscv::BinaryOp operation) {
      if (operation.getKind() == "maximum" || operation.getKind() == "minimum")
        operations.push_back(operation);
    });
    for (auto operation : operations) {
      rewriter.setInsertionPoint(operation);
      auto type = operation.getResult().getType();
      auto element = riscv_internal::logicalElement(type);
      auto numeric = rewriter.create<riscv::BinaryOp>(
          operation.getLoc(), type, operation.getLhs(), operation.getRhs(),
          operation.getKind() == "maximum" ? "max" : "min",
          riscv_internal::unselectedLeaf(rewriter));
      copyIdentity(operation, numeric);
      auto nan = rewriter.create<riscv::ConstantOp>(
          operation.getLoc(), element,
          mlir::FloatAttr::get(element, llvm::APFloat::getQNaN(
              mlir::cast<mlir::FloatType>(element).getFloatSemantics())));
      copyIdentity(operation, nan);
      mlir::Value result = numeric.getResult();
      for (mlir::Value operand : operation->getOperands()) {
        mlir::Type predicateType = rewriter.getI1Type();
        if (auto value = mlir::dyn_cast<riscv::ValueType>(operand.getType())) {
          auto predicate = riscv::ValueType::get(
              &getContext(), rewriter.getI1Type(), value.getShape(),
              value.getAxisIds(), value.getLayout());
          auto layout = riscv_internal::projectLayout(
              rewriter, predicate, value.getLayout(),
              operation->getParentOfType<riscv::KernelOp>().getTarget());
          if (!layout) {
            operation.emitError("NaN predicate has no legal physical layout");
            failed = true;
            break;
          }
          predicateType = riscv_internal::withLayout(predicate, layout);
        }
        auto isNaN = rewriter.create<riscv::CompareOp>(
            operation.getLoc(), predicateType, operand, operand, "ne",
            riscv_internal::unselectedLeaf(rewriter));
        auto selected = rewriter.create<riscv::SelectOp>(
            operation.getLoc(), type, isNaN.getResult(), nan.getResult(), result,
            riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, isNaN);
        copyIdentity(operation, selected);
        result = selected.getResult();
      }
      rewriter.replaceOp(operation, result);
    }
  }

  void lowerReciprocalSquareRoots(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::UnaryOp> operations;
    getOperation().walk([&](riscv::UnaryOp operation) {
      if (operation.getKind() == "rsqrt")
        operations.push_back(operation);
    });
    for (auto operation : operations) {
      mlir::Type resultType = operation.getResult().getType();
      mlir::Type element = riscv_internal::logicalElement(resultType);
      auto shaped = mlir::dyn_cast<riscv::ValueType>(resultType);
      llvm::StringRef carrier =
          shaped ? shaped.getLayout().getCarrier() : llvm::StringRef("scalar");
      auto implementation = operation->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if ((!element.isF32() && !element.isF64()) || !implementation ||
          (carrier != "scalar" && carrier != "rvv") ||
          implementation.getEngine() != carrier) {
        operation.emitError(
            "rsqrt requires a selected f32/f64 scalar or RVV representation");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(operation);
      auto squareRoot = rewriter.create<riscv::UnaryOp>(
          operation.getLoc(), resultType, operation.getInput(), "sqrt",
          riscv_internal::unselectedLeaf(rewriter));
      auto constant = rewriter.create<riscv::ConstantOp>(
          operation.getLoc(), element, rewriter.getFloatAttr(element, 1.0));
      mlir::Value one = constant.getResult();
      copyIdentity(operation, squareRoot);
      copyIdentity(operation, constant);
      if (carrier == "rvv") {
        auto splat = rewriter.create<riscv::RVVSplatOp>(
            operation.getLoc(), shaped, one,
            riscv_internal::leaf(
                rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                shaped.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
        copyIdentity(operation, splat);
        one = splat.getResult();
      }
      auto quotient = rewriter.create<riscv::BinaryOp>(
          operation.getLoc(), resultType, one, squareRoot.getResult(), "div",
          riscv_internal::unselectedLeaf(rewriter));
      copyIdentity(operation, quotient);
      rewriter.replaceOp(operation, quotient.getResult());
    }
  }

  void lowerPartitionedWidenReduceStores(mlir::IRRewriter &rewriter,
                                         bool &failed) {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.level"))
        loops.push_back(loop);
    });
    for (mlir::scf::ForOp loop : loops) {
      if (!loop.getInitArgs().empty() || loop.getNumResults() != 0)
        continue;
      llvm::SmallVector<riscv::StoreOp> stores;
      loop.getBody()->walk([&](riscv::StoreOp store) { stores.push_back(store); });
      if (stores.size() != 1)
        continue;
      riscv::StoreOp store = stores.front();
      auto reduce = riscv_internal::stripRepresentationConversions(store.getValue())
                        .getDefiningOp<riscv::ReduceOp>();
      auto widen =
          reduce ? riscv_internal::stripRepresentationConversions(reduce.getInput())
                       .getDefiningOp<riscv::WidenOp>()
                 : riscv::WidenOp();
      auto extract =
          widen ? riscv_internal::stripRepresentationConversions(widen.getInput())
                      .getDefiningOp<riscv::ExtractOp>()
                : riscv::ExtractOp();
      auto point = extract && extract.getIndices().size() == 1
                       ? riscv_internal::stripRepresentationConversions(
                             extract.getIndices().front())
                             .getDefiningOp<riscv::PhysicalPointOp>()
                       : riscv::PhysicalPointOp();
      auto destinationSlice = store.getRegion().getDefiningOp<riscv::SliceOp>();
      auto destinationField = destinationSlice
                                  ? destinationSlice.getBase().getDefiningOp<
                                        riscv::FieldOp>()
                                  : riscv::FieldOp();
      auto destinationOwner = destinationField
                                  ? destinationField.getOwner().getDefiningOp<
                                        riscv::SliceOp>()
                                  : riscv::SliceOp();
      auto outerPoint = point
                            ? point.getParent().getDefiningOp<
                                  riscv::PhysicalPointOp>()
                            : riscv::PhysicalPointOp();
      auto partitionConstant =
          point ? point.getPartition().getDefiningOp<
                      mlir::arith::ConstantIndexOp>()
                : mlir::arith::ConstantIndexOp();
      auto input = extract
                       ? mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType())
                       : riscv::ValueType();
      auto destination = destinationField
                             ? mlir::dyn_cast<riscv::MemDescType>(
                                   destinationField.getResult().getType())
                             : riscv::MemDescType();
      auto owner = destinationField
                       ? mlir::dyn_cast<riscv::MemDescType>(
                             destinationField.getOwner().getType())
                       : riscv::MemDescType();
      const int64_t partitionValue =
          partitionConstant ? partitionConstant.value() : 0;
      const int64_t count =
          input && partitionValue > 0 && input.getShape().size() == 1
              ? input.getShape()[0] / partitionValue
              : -1;
      const int64_t inputParts = input ? resultParts(input.getLayout()) : -1;
      auto target = loop->getParentOfType<riscv::KernelOp>().getTarget();
      if (!reduce || !widen || !extract || !point || !outerPoint ||
          !destinationSlice || !destinationField || !destinationOwner ||
          !partitionConstant || !input || !destination || !owner ||
          reduce.getKind() != "add" ||
          extract.getSelectors().size() != 1 ||
          mlir::cast<mlir::StringAttr>(extract.getSelectors()[0]).getValue() !=
              "domain" ||
          input.getShape().size() != 1 || partitionConstant.value() <= 0 ||
          input.getShape()[0] % partitionConstant.value() ||
          count <= 1 || inputParts <= 0 || count % inputParts ||
          destination.getAxisIds().size() != 1 || owner.getElements() <= 0 ||
          !target.getHasRVV() || !target.getHasWideningInteger() ||
          !riscv::supportsRVVLayout(target, input.getLayout()) ||
          !loop->isProperAncestor(point) ||
          loop->isProperAncestor(extract.getInput().getDefiningOp()))
        continue;

      bool extraEffects = false;
      loop.getBody()->walk([&](mlir::Operation *nested) {
        if (nested == store.getOperation())
          return;
        if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(nested))
          if (!effects.hasNoEffect())
            extraEffects = true;
      });
      if (extraEffects)
        continue;

      rewriter.setInsertionPoint(loop);
      destinationOwner->moveBefore(loop);
      destinationField->moveBefore(loop);
      mlir::Value elements = rewriter.create<mlir::arith::ConstantIndexOp>(
          loop.getLoc(), owner.getElements());
      mlir::Value partition = rewriter.create<mlir::arith::ConstantIndexOp>(
          loop.getLoc(), partitionConstant.value());
      mlir::Value within = rewriter.create<mlir::arith::RemUIOp>(
          loop.getLoc(), outerPoint.getBase(), elements);
      mlir::Value baseIndex = rewriter.create<mlir::arith::DivUIOp>(
          loop.getLoc(), within, partition);
      auto cluster =
          rewriter.create<riscv::RVVPartitionedWidenReduceStoreOp>(
              loop.getLoc(), extract.getInput(), destinationField.getResult(),
              baseIndex, partitionConstant.value(), count,
              destination.getAxisIds()[0], store.getAccess(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partitioned-widen-reduce-store",
                  "rvv.partitioned-vwredsum-store",
                  "rvv.partitioned-vwredsum-store",
                  input.getLayout().getRegisterGroups(), 0, 1));
      copyIdentity(reduce, cluster);
      if (auto parentLoop = loop->getParentOfType<mlir::scf::ForOp>())
        parentLoop->setAttr("weft.riscv.system_unroll",
                            rewriter.getStringAttr("disable"));
      rewriter.eraseOp(loop);
    }
  }

  void lowerWidenReductions(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
      llvm::SmallVector<riscv::ConvertLayoutOp> bridges;
      mlir::Value reduced = riscv_internal::stripRepresentationConversions(
          reduce.getInput(), &bridges);
      auto widen = reduced.getDefiningOp<riscv::WidenOp>();
      auto input = widen
                       ? mlir::dyn_cast<riscv::ValueType>(widen.getInput().getType())
                       : riscv::ValueType();
      auto inputElement = input
                              ? mlir::dyn_cast<mlir::IntegerType>(
                                    input.getElementType())
                              : mlir::IntegerType();
      auto resultElement =
          mlir::dyn_cast<mlir::IntegerType>(reduce.getResult().getType());
      auto target = reduce->getParentOfType<riscv::KernelOp>().getTarget();
      if (!widen || !input || !inputElement || inputElement.isSignless() ||
          !resultElement || resultElement.isSignless() ||
          reduce.getKind() != "add" ||
          resultElement.getWidth() != inputElement.getWidth() * 2 ||
          mlir::isa<riscv::ValueType>(reduce.getResult().getType()) ||
          input.getLayout().getCarrier() != "rvv" ||
          resultParts(input.getLayout()) != 1 || !target.getHasRVV() ||
          !target.getHasWideningInteger() ||
          !riscv::supportsRVVLayout(target, input.getLayout()))
        continue;
      rewriter.setInsertionPoint(reduce);
      auto folded = rewriter.create<riscv::RVVWidenReduceOp>(
          reduce.getLoc(), reduce.getResult().getType(), widen.getInput(),
          reduce.getKindAttr(), reduce.getAxisAttr(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-reduce", "rvv.vwredsum",
              "rvv.vwredsum", input.getLayout().getRegisterGroups(), 0, 1));
      copyIdentity(reduce, folded);
      reduce.getResult().replaceAllUsesWith(folded.getResult());
      rewriter.eraseOp(reduce);
      for (riscv::ConvertLayoutOp bridge : llvm::reverse(bridges))
        if (bridge.getResult().use_empty())
          rewriter.eraseOp(bridge);
      if (widen.getResult().use_empty())
        rewriter.eraseOp(widen);
    }
  }

  void lowerLocalStates(mlir::IRRewriter &rewriter, bool &failed) {
    int64_t identity = riscv_internal::lastLocalIdentity(getOperation());
    llvm::SmallVector<riscv::EmptyOp> emptyStates;
    getOperation().walk([&](riscv::EmptyOp empty) { emptyStates.push_back(empty); });
    for (auto empty : emptyStates) {
      auto type = empty.getResult().getType();
      if (type.getLayout().getCarrier() != "local") {
        empty.emitError("empty storage has no addressable local representation"); failed = true; continue;
      }
      rewriter.setInsertionPoint(empty);
      auto binding = riscv_internal::allocateLocalValue(rewriter, type, empty, identity, true);
      if (mlir::failed(binding)) { failed = true; continue; }
      rewriter.replaceOp(empty, *binding);
    }
    llvm::SmallVector<riscv::NewOp> states;
    getOperation().walk([&](riscv::NewOp state) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType());
          value && value.getLayout().getCarrier() == "local")
        states.push_back(state);
    });
    for (riscv::NewOp state : states) {
      auto value = mlir::cast<riscv::ValueType>(state.getResult().getType());
      rewriter.setInsertionPoint(state);
      auto binding = riscv_internal::allocateLocalValue(rewriter, value, state, identity, true);
      if (mlir::failed(binding)) { failed = true; continue; }
      if (state.getInitialized() && mlir::failed(riscv_internal::initializeLocalValue(
              rewriter, *binding, state.getInitial(), state))) { failed = true; continue; }
      rewriter.replaceOp(state, *binding);
    }
  }

  void eraseDeadComposites(mlir::IRRewriter &rewriter) {
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<mlir::Operation *> dead;
      getOperation().walk<mlir::WalkOrder::PostOrder>(
          [&](mlir::Operation *operation) {
            if (mlir::isa<riscv::Fold2Op, riscv::MacGroupsOp,
                          riscv::DotOp, riscv::ContractOp,
                          riscv::OuterContractOp,
                          riscv::ConvertLayoutOp, riscv::LoadOp>(operation) &&
                llvm::all_of(operation->getResults(),
                             [](mlir::Value value) { return value.use_empty(); }))
              dead.push_back(operation);
          });
      for (mlir::Operation *operation : dead) {
        rewriter.eraseOp(operation);
        changed = true;
      }
    }
  }

  mlir::Value findOperandPoint(mlir::Value value, int64_t axis) const {
    llvm::SmallVector<mlir::Value> worklist{value};
    llvm::SmallPtrSet<mlir::Operation *, 8> visited;
    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();
      if (auto point = mlir::dyn_cast<riscv::PointType>(current.getType());
          point && point.getDomain().getAxisId() == axis)
        return current;
      mlir::Operation *definition = current.getDefiningOp();
      if (!definition || !visited.insert(definition).second)
        continue;
      for (mlir::Value operand : definition->getOperands())
        worklist.push_back(operand);
    }
    return {};
  }

  mlir::Value shapeExtent(mlir::Operation *operation, int64_t symbolId) const {
    return riscv_internal::shapeExtentAt(operation, symbolId);
  }

  mlir::Value fullAxisExtent(mlir::Operation *operation, int64_t axis) const {
    for (mlir::Value operand : operation->getOperands()) {
      auto axes = riscv_internal::logicalAxes(operand.getType());
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) continue;
      auto shape = riscv_internal::logicalShape(operand.getType());
      int64_t extent = shape[found - axes.begin()];
      if (extent < 0) return shapeExtent(operation, -extent);
      mlir::OpBuilder builder(operation);
      return builder.create<mlir::arith::ConstantIndexOp>(operation->getLoc(), extent);
    }
    return {};
  }

  void lowerLoops(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::LoopOp> loops;
    getOperation().walk<mlir::WalkOrder::PreOrder>(
        [&](riscv::LoopOp loop) { loops.push_back(loop); });
    for (riscv::LoopOp loop : loops) {
      auto domain = loop.getDomain().getDefiningOp<riscv::DomainOp>();
      if (!domain || loop.getBody().empty()) {
        loop.emitError("physical Level has no explicit domain definition/body");
        failed = true;
        continue;
      }
      auto domainType = loop.getDomain().getType();
      mlir::Value parentPoint = loop.getParentPoint();
      mlir::Value origin;
      mlir::Value total = domain.getExtent();
      rewriter.setInsertionPoint(loop);
      origin = rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);
      mlir::Value ancestor = parentPoint;
      while (true) {
        auto ancestorType =
            mlir::cast<riscv::PointType>(ancestor.getType()).getDomain();
        if (ancestorType.getRelation() == "root")
          break;
        auto point = ancestor.getDefiningOp<riscv::PhysicalPointOp>();
        if (!point) {
          loop.emitError(
              "physical Level ancestry lost an explicit point entity");
          failed = true;
          break;
        }
        if (ancestorType.getAxisId() == domainType.getAxisId()) {
          origin = point.getBase();
          total = point.getActive();
          break;
        }
        ancestor = point.getParent();
      }
      if (failed)
        continue;
      mlir::Value lower =
          rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 0);
      mlir::Value upper = total;
      mlir::Value step = domain.getPartition();
      const bool ordinalIteration = domainType.getTail() == "exact";
      if (ordinalIteration) {
        upper = rewriter.create<mlir::arith::DivUIOp>(
            loop.getLoc(), total, domain.getPartition());
        step = rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 1);
      }
      auto physical = rewriter.create<mlir::scf::ForOp>(
          loop.getLoc(), lower, upper, step, loop.getCarried());
      if (auto source = loop->getAttr("source_origin"))
        physical->setAttr("source_origin", source);
      physical->setAttr(
          "weft.riscv.level",
          riscv::LevelAttr::get(
              rewriter.getContext(), domainType.getDomainId(),
              domainType.getAxisId(), domainType.getRelation(),
              loop.getStateBirthCount(), loop.getStagedBirthCount(),
              loop.getHandoffCount(), "ascending"));
      if (loop.getSchedule().getUnroll() > 1)
        physical->setAttr(
            "weft.riscv.unroll_factor",
            rewriter.getI64IntegerAttr(loop.getSchedule().getUnroll()));
      if (loop.getSchedule().getPipelineDepth() > 1)
        physical->setAttr("weft.riscv.schedule", loop.getSchedule());

      mlir::Block &sourceBody = loop.getBody().front();
      mlir::Block *targetBody = physical.getBody();
      rewriter.setInsertionPointToStart(targetBody);
      mlir::Value logicalOffset = physical.getInductionVar();
      if (ordinalIteration)
        logicalOffset = rewriter.create<mlir::arith::MulIOp>(
            loop.getLoc(), physical.getInductionVar(), domain.getPartition());
      mlir::Value base = rewriter.create<mlir::arith::AddIOp>(
          loop.getLoc(), origin, logicalOffset);
      mlir::Value remaining = rewriter.create<mlir::arith::SubIOp>(
          loop.getLoc(), total, logicalOffset);
      mlir::Value active = domain.getPartition();
      if (domainType.getTail() != "exact")
        active = rewriter.create<mlir::arith::MinUIOp>(
            loop.getLoc(), remaining, domain.getPartition());
      auto point = rewriter.create<riscv::PhysicalPointOp>(
          loop.getLoc(), sourceBody.getArgument(0).getType(), parentPoint, base,
          active, domain.getPartition());
      sourceBody.getArgument(0).replaceAllUsesWith(point.getResult());
      if (sourceBody.getNumArguments() !=
          physical.getRegionIterArgs().size() + 1) {
        loop.emitError("physical Level birth/handoff arity was not preserved");
        failed = true;
        continue;
      }
      for (auto [source, target] : llvm::zip(
               sourceBody.getArguments().drop_front(),
               physical.getRegionIterArgs()))
        source.replaceAllUsesWith(target);
      auto sourceYield = mlir::cast<riscv::YieldOp>(sourceBody.getTerminator());
      llvm::SmallVector<mlir::Value> yielded(sourceYield.getValues());
      if (!targetBody->empty())
        if (auto existing =
                mlir::dyn_cast<mlir::scf::YieldOp>(targetBody->back()))
          rewriter.eraseOp(existing);
      for (mlir::Operation &nested : llvm::make_early_inc_range(sourceBody)) {
        if (&nested == sourceYield.getOperation())
          continue;
        nested.moveBefore(targetBody, targetBody->end());
      }
      rewriter.setInsertionPointToEnd(targetBody);
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), yielded);
      rewriter.eraseOp(sourceYield);
      loop.getResults().replaceAllUsesWith(physical.getResults());
      rewriter.eraseOp(loop);
    }
  }

  void closeOrderedLoops(bool &failed) {
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.direction") ||
          loop->hasAttr("weft.riscv.level"))
        return;
      auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (!step || step.value() == 0) {
        loop.emitError(
            "ordered scalar loop requires one non-zero compile-time direction");
        failed = true;
        return;
      }
      loop->setAttr("weft.riscv.direction",
                    mlir::StringAttr::get(loop.getContext(),
                                          step.value() > 0 ? "ascending"
                                                           : "descending"));
    });
  }

  void lowerMaterializations(mlir::IRRewriter &rewriter, bool &failed) {
    int64_t identity = riscv_internal::lastLocalIdentity(getOperation());
    llvm::SmallVector<riscv::MaterializeOp> operations;
    getOperation().walk(
        [&](riscv::MaterializeOp operation) { operations.push_back(operation); });
    for (riscv::MaterializeOp operation : operations) {
      if (operation.getPlacement() == "local") {
        auto input =
            mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
        auto result =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
        if (auto plan = operation.getLocalPackPlanAttr()) {
          auto load = riscv_internal::sourceLoad(operation.getInput());
          auto slice = load ? load.getRegion().getDefiningOp<riscv::SliceOp>()
                            : riscv::SliceOp();
          mlir::Value rowPoint;
          mlir::Value recordPoint;
          size_t cursor = 0;
          if (slice) {
            auto descriptor = slice.getResult().getType();
            for (auto [dimension, selectorAttribute] :
                 llvm::enumerate(slice.getSelectors())) {
              llvm::StringRef selector =
                  mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
              if (selector == "all")
                continue;
              if (cursor >= slice.getIndices().size())
                break;
              mlir::Value coordinate = slice.getIndices()[cursor++];
              if (selector != "domain")
                continue;
              const int64_t axis = descriptor.getAxisIds()[dimension];
              if (axis == plan.getRowAxis())
                rowPoint = coordinate;
              if (axis == plan.getRecordAxis())
                recordPoint = coordinate;
            }
          }
          if (!input || !result || !load || !slice || !rowPoint || !recordPoint) {
            operation.emitError(
                "encoded local pack requires one load-backed two-axis materialization");
            failed = true;
            continue;
          }
          rewriter.setInsertionPoint(operation);
          mlir::Value allocationBytes =
              rewriter.create<mlir::arith::ConstantIndexOp>(
                  operation.getLoc(), plan.getLocalBytes());
          auto storageType = riscv::LocalType::get(
              rewriter.getContext(), result.getElementType(), result.getShape(),
              result.getAxisIds(), plan.getLocalBytes(), 8,
              operation.getBirthId(), "pack", operation.getOwnerDomainId(),
              operation.getBirthId(), operation.getLifetimeEndDomainId(),
              "encoded-interleave");
          auto allocation = rewriter.create<riscv::LocalAllocOp>(
              operation.getLoc(), storageType, allocationBytes);
          auto kernel = operation->getParentOfType<riscv::KernelOp>();
          const int64_t transferLmulEighths = 8;
          const int64_t transferSew = 8;
          const int64_t transferLanes = plan.getInterleaveRows();
          auto transferLayout =
              kernel
                  ? riscv::LayoutAttr::get(
                        rewriter.getContext(), "rvv",
                        riscv_internal::integers(rewriter, {plan.getRowAxis()}),
                        riscv_internal::integers(rewriter, {1}),
                        riscv_internal::integers(rewriter, {transferLanes}),
                        riscv_internal::integers(rewriter, {1}),
                        riscv_internal::integers(rewriter, {1}),
                        riscv_internal::integers(rewriter, {1}), transferSew,
                        transferLmulEighths, transferLanes, 1, "tail")
                  : riscv::LayoutAttr();
          if (!kernel || transferLanes <= 0 ||
              transferLanes > kernel.getTarget().getVlenBits() / transferSew ||
              !riscv::supportsRVVLayout(kernel.getTarget(), transferLayout)) {
            operation.emitError(
                "encoded local pack has no legal one-vector row transfer");
            failed = true;
            continue;
          }
          auto packed = rewriter.create<riscv::EncodedLocalPackOp>(
              operation.getLoc(), result, operation.getInput(),
              allocation.getResult(), rowPoint, recordPoint, plan,
              transferLayout,
              riscv_internal::leaf(
                  rewriter, "transfer", "local-pack",
                  "rvv.local-pack.interleave",
                  "rvv.local-pack.interleave", 0, 0,
                  transferLayout.getRegisterGroups()));
          copyIdentity(operation, allocation);
          copyIdentity(operation, packed);
          operation.getResult().replaceAllUsesWith(packed.getResult());
          rewriter.eraseOp(operation);
          continue;
        }
        rewriter.setInsertionPoint(operation);
        if (!input || !result || result.getLayout().getCarrier() != "local" ||
            !mlir::isa<mlir::IntegerType, mlir::FloatType>(result.getElementType())) {
          operation.emitError("local materialize requires a complete numeric value");
          failed = true; continue;
        }
        auto binding = riscv_internal::allocateLocalValue(rewriter, result, operation, identity, true);
        if (mlir::failed(binding) || mlir::failed(riscv_internal::initializeLocalValue(
                rewriter, *binding, operation.getInput(), operation))) { failed = true; continue; }
        rewriter.replaceOp(operation, *binding);
        continue;
      }
      if (operation.getPlacement() == "reload") {
        if (auto load = sourceLoad(operation.getInput())) {
          rewriter.setInsertionPoint(operation);
          auto staged = rewriter.create<riscv::StagedViewOp>(
              operation.getLoc(), load.getRegion().getType(), load.getRegion(),
              operation.getSchema(),
              operation.getOwnerDomainId(), operation.getBirthId(),
              operation.getLifetimeEndDomainId());
          copyIdentity(operation, staged);
          llvm::SmallVector<riscv::ExtractOp> extracts;
          for (mlir::Operation *user : operation.getResult().getUsers())
            if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
                extract && extract.getInput() == operation.getResult())
              extracts.push_back(extract);
          for (riscv::ExtractOp extract : extracts) {
            auto result = mlir::dyn_cast<riscv::ValueType>(
                extract.getResult().getType());
            auto source = load.getRegion().getType();
            if (!result) {
              extract.emitError(
                  "reload projection requires one shaped physical result");
              failed = true;
              continue;
            }
            llvm::SmallVector<int64_t> strides;
            llvm::SmallVector<int64_t> origins;
            for (int64_t axis : result.getAxisIds().asArrayRef()) {
              auto found = llvm::find(source.getAxisIds().asArrayRef(), axis);
              if (found == source.getAxisIds().asArrayRef().end()) {
                extract.emitError(
                    "reload projection introduced an axis absent from its memory descriptor");
                failed = true;
                break;
              }
              size_t index = static_cast<size_t>(
                  found - source.getAxisIds().asArrayRef().begin());
              strides.push_back(source.getStrides()[index]);
              origins.push_back(source.getOrigins()[index]);
            }
            if (failed)
              continue;
            auto descriptor = riscv::MemDescType::get(
                rewriter.getContext(), source.getEncoding(), result.getShape(),
                result.getAxisIds(), riscv_internal::integers(rewriter, strides),
                riscv_internal::integers(rewriter, origins),
                source.getAlignment(), "slice", source.getAccess(),
                source.getAliasSet(), source.getLayoutIdentity(),
                source.getStorageBits(), source.getElements(),
                source.getInterleaveRows());
            rewriter.setInsertionPoint(extract);
            auto slice = rewriter.create<riscv::SliceOp>(
                extract.getLoc(), descriptor, staged.getResult(),
                extract.getIndices(), extract.getSelectors());
            copyIdentity(extract, slice);
            llvm::StringRef form = extract.getAccess().getForm();
            auto leaf = riscv_internal::leaf(
                rewriter, "transfer", "load", ("rvv.load." + form).str(),
                ("rvv.load." + form).str(), 0, 0);
            auto projected = rewriter.create<riscv::LoadOp>(
                extract.getLoc(), result, slice.getResult(),
                mlir::Value(), extract.getAccess(), leaf);
            copyIdentity(extract, projected);
            extract.getResult().replaceAllUsesWith(projected.getResult());
            rewriter.eraseOp(extract);
          }
          if (failed)
            continue;
          llvm::SmallVector<mlir::OpOperand *> uses;
          for (mlir::OpOperand &use : operation.getResult().getUses())
            uses.push_back(&use);
          for (mlir::OpOperand *use : uses) {
            rewriter.setInsertionPoint(use->getOwner());
            auto cloned = rewriter.create<riscv::LoadOp>(
                operation.getLoc(), operation.getResult().getType(),
                staged.getResult(), mlir::Value(), load.getAccess(), load.getLeaf());
            copyIdentity(load, cloned);
            use->set(cloned.getResult());
          }
          rewriter.eraseOp(operation);
          if (load.getResult().use_empty())
            rewriter.eraseOp(load);
          continue;
        }
        // A non-memory pure producer cannot be represented as a lazy view.
        // Keep one explicit SSA value and let resource materialization decide
        // whether that computed value must later spill.
      }
      rewriter.setInsertionPoint(operation);
      mlir::Value input = operation.getInput();
      if (input.getType() != operation.getResult().getType()) {
        auto source = mlir::dyn_cast<riscv::ValueType>(input.getType());
        auto target =
            mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
        if (!source || !target) {
          operation.emitError(
              "register materialization cannot convert a non-shaped representation");
          failed = true;
          continue;
        }
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            operation.getLoc(), target, input,
            riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                             target.getLayout()),
            riscv::AccessAttr(),
            riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, conversion);
        input = conversion.getResult();
      }
      auto materialized = rewriter.create<riscv::RegisterMaterializeOp>(
          operation.getLoc(), operation.getResult().getType(),
          input, operation.getOwnerDomainId(),
          operation.getBirthId(), operation.getLifetimeEndDomainId(), "share");
      copyIdentity(operation, materialized);
      operation.getResult().replaceAllUsesWith(materialized.getResult());
      rewriter.eraseOp(operation);
    }
  }

  void lowerLocalGroupExtracts(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk([&](riscv::ExtractOp extract) {
      auto input =
          mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType());
      if (input && input.getLayout().getCarrier() == "local")
        extracts.push_back(extract);
    });
    for (riscv::ExtractOp extract : extracts) {
      auto input = mlir::cast<riscv::ValueType>(extract.getInput().getType());
      auto result =
          mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      auto binding = extract.getInput().getDefiningOp<riscv::LocalBindOp>();
      if (!result || result.getLayout().getCarrier() != "scalar" || !binding ||
          extract.getSelectors().size() != input.getShape().size())
        continue;

      size_t cursor = 0;
      int64_t groupDimension = -1;
      riscv::PhysicalPointOp groupPoint;
      bool supported = true;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= extract.getIndices().size() ||
            selector != "group_index" || groupDimension >= 0) {
          supported = false;
          break;
        }
        groupPoint = extract.getIndices()[cursor++]
                         .getDefiningOp<riscv::PhysicalPointOp>();
        groupDimension = static_cast<int64_t>(dimension);
      }
      auto resultParts = riscv_internal::staticProduct(
          result.getLayout().getReplicaFactors().asArrayRef());
      if (!supported || cursor != extract.getIndices().size() || !groupPoint ||
          groupDimension < 0 || !resultParts || *resultParts <= 0 ||
          llvm::any_of(result.getLayout().getTimeFactors().asArrayRef(),
                       [](int64_t factor) { return factor != 1; }) ||
          llvm::any_of(result.getLayout().getLaneFactors().asArrayRef(),
                       [](int64_t factor) { return factor != 1; }))
        continue;

      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(extract);
      mlir::Value groupCount = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(),
          input.getLayout().getLocalFactors()[groupDimension]);
      mlir::Value recordExtent = rewriter.create<mlir::arith::MulIOp>(
          extract.getLoc(), groupPoint.getPartition(), groupCount);
      mlir::Value relative = rewriter.create<mlir::arith::RemUIOp>(
          extract.getLoc(), groupPoint.getBase(), recordExtent);
      mlir::Value group = rewriter.create<mlir::arith::DivUIOp>(
          extract.getLoc(), relative, groupPoint.getPartition());

      llvm::SmallVector<mlir::Value> loaded;
      for (int64_t part = 0; part < *resultParts; ++part) {
        llvm::DenseMap<int64_t, int64_t> coordinates;
        int64_t remaining = part;
        for (int64_t dimension =
                 static_cast<int64_t>(result.getAxisIds().size()) - 1;
             dimension >= 0; --dimension) {
          const int64_t factor =
              result.getLayout().getReplicaFactors()[dimension];
          coordinates[result.getAxisIds()[dimension]] = remaining % factor;
          remaining /= factor;
        }
        mlir::Value linear = rewriter.create<mlir::arith::ConstantIndexOp>(
            extract.getLoc(), 0);
        for (size_t dimension = 0; dimension < input.getShape().size();
             ++dimension) {
          mlir::Value extent = rewriter.create<mlir::arith::ConstantIndexOp>(
              extract.getLoc(), input.getLayout().getLocalFactors()[dimension]);
          linear = rewriter.create<mlir::arith::MulIOp>(extract.getLoc(), linear,
                                                        extent);
          mlir::Value coordinate;
          if (static_cast<int64_t>(dimension) == groupDimension) {
            coordinate = group;
          } else {
            auto found = coordinates.find(input.getAxisIds()[dimension]);
            if (found == coordinates.end()) {
              supported = false;
              break;
            }
            coordinate = rewriter.create<mlir::arith::ConstantIndexOp>(
                extract.getLoc(), found->second);
          }
          linear = rewriter.create<mlir::arith::AddIOp>(extract.getLoc(), linear,
                                                        coordinate);
        }
        if (!supported)
          break;
        auto load = rewriter.create<riscv::LocalLoadOp>(
            extract.getLoc(), result.getElementType(), binding.getResult(), linear,
            riscv_internal::leaf(rewriter, "transfer", "local-load",
                                 "local.load.element", "local.load.element", 0,
                                 0, 0, 0));
        copyIdentity(extract, load);
        loaded.push_back(load.getResult());
      }
      if (!supported) {
        extract.emitError(
            "local grouped projection has no complete scalar replica mapping");
        failed = true;
        continue;
      }
      auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
          extract.getLoc(), result, loaded,
          riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                               "scalar.assemble-replicas",
                               "scalar.assemble-replicas", 0, 0));
      copyIdentity(extract, assembled);
      extract.getResult().replaceAllUsesWith(assembled.getResult());
      rewriter.eraseOp(extract);
    }
  }

  void lowerGroupedReductions(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp operation) { reductions.push_back(operation); });
    for (riscv::ReduceOp reduce : reductions) {
      auto widen =
          riscv_internal::stripRepresentationConversions(reduce.getInput())
              .getDefiningOp<riscv::WidenOp>();
      llvm::SmallVector<riscv::ConvertLayoutOp> bridges;
      mlir::Value macValue = widen ? widen.getInput() : mlir::Value();
      if (macValue)
        macValue = riscv_internal::stripRepresentationConversions(macValue,
                                                                  &bridges);
      auto mac = macValue ? macValue.getDefiningOp<riscv::MacGroupsOp>()
                          : riscv::MacGroupsOp();
      if (!widen || !mac)
        continue;
      auto implementation = mac->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if (!implementation || implementation.getEngine() != "rvv" ||
          implementation.getFamily() != "grouped-mac" ||
          implementation.getOperation() != "rvv.vwmaccsu.typed" ||
          implementation.getParameters().asArrayRef() !=
              llvm::ArrayRef<int64_t>({static_cast<int64_t>(mac.getGroup())})) {
        mac.emitError(
            "grouped MAC reduction has no selected mixed-sign widening implementation");
        failed = true;
        continue;
      }
      mlir::Value lhs = mac.getLhs();
      mlir::Value rhs = mac.getRhs();
      auto sourceLhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(lhs.getType()));
      auto sourceRhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(rhs.getType()));
      if (sourceLhsInteger && sourceRhsInteger &&
          sourceLhsInteger.isSigned() && sourceRhsInteger.isUnsigned())
        std::swap(lhs, rhs);
      riscv::AccessAttr lhsAccess = accessOf(lhs);
      riscv::AccessAttr rhsAccess = accessOf(rhs);
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      auto widenedType =
          mlir::dyn_cast<riscv::ValueType>(widen.getResult().getType());
      riscv::ValueType accumulatorType = resultType ? resultType : widenedType;
      auto lhsType = mlir::dyn_cast<riscv::ValueType>(lhs.getType());
      auto rhsType = mlir::dyn_cast<riscv::ValueType>(rhs.getType());
      auto lhsInteger = lhsType
                            ? mlir::dyn_cast<mlir::IntegerType>(
                                  lhsType.getElementType())
                            : mlir::IntegerType();
      auto rhsInteger = rhsType
                            ? mlir::dyn_cast<mlir::IntegerType>(
                                  rhsType.getElementType())
                            : mlir::IntegerType();
      if (!lhsAccess || !rhsAccess || !accumulatorType || !lhsType ||
          !rhsType || !lhsInteger || !rhsInteger) {
        reduce.emitError(
            "grouped MAC reduction has no typed operand access/result contract");
        failed = true;
        continue;
      }
      int64_t reductionAxis =
          mlir::cast<riscv::ValueType>(widen.getResult().getType())
              .getAxisIds()[reduce.getAxis()];
      riscv::ScheduleAttr schedule = mac.getSchedule();
      auto target = reduce->getParentOfType<riscv::KernelOp>().getTarget();
      auto lhsLayout = lhsType.getLayout();
      riscv::LayoutAttr resultLayout = accumulatorType.getLayout();
      riscv::LayoutAttr loadLayout = riscv_internal::projectLayout(
          rewriter, lhsType, resultLayout, target);
      auto partialElement = mlir::IntegerType::get(
          rewriter.getContext(), 16,
          mlir::IntegerType::SignednessSemantics::Signed);
      auto partialSeed = riscv::ValueType::get(
          rewriter.getContext(), partialElement, lhsType.getShape(),
          lhsType.getAxisIds(), lhsLayout);
      riscv::LayoutAttr partialLayout = riscv_internal::projectLayout(
          rewriter, partialSeed, resultLayout, target);
      if (!loadLayout || !partialLayout) {
        reduce.emitError(
            "grouped MAC operands have no result-anchored load/partial representation");
        failed = true;
        continue;
      }
      riscv::FieldOp lhsField = sourceField(lhs);
      riscv::FieldOp rhsField = sourceField(rhs);
      riscv::LoadOp lhsLoad = lhsField ? sourceLoad(lhsField.getOwner())
                                      : riscv::LoadOp();
      riscv::LoadOp rhsLoad = rhsField ? sourceLoad(rhsField.getOwner())
                                      : riscv::LoadOp();
      auto lhsMemory = lhsLoad ? lhsLoad.getRegion().getType()
                               : riscv::MemDescType();
      auto lhsEncoding = lhsField
                             ? mlir::dyn_cast<kernel::EncodingType>(
                                   riscv_internal::logicalElement(
                                       lhsField.getOwner().getType()))
                             : kernel::EncodingType();
      const int64_t interleaveRows =
          lhsField && lhsEncoding
              ? riscv_internal::interleaveRows(lhsField, lhsEncoding)
              : 0;
      mlir::Value lhsPoint = findOperandPoint(lhs, reductionAxis);
      mlir::Value rhsPoint = findOperandPoint(rhs, reductionAxis);
      mlir::Value point = lhsPoint ? lhsPoint : rhsPoint;
      int64_t laneAxis = 0;
      bool reductionLane = false;
      for (auto [axis, factor] :
           llvm::zip(lhsType.getAxisIds().asArrayRef(),
                     lhsLayout.getLaneFactors().asArrayRef())) {
        if (axis == reductionAxis && factor > 1)
          reductionLane = true;
        if (axis != reductionAxis && factor > 1) {
          laneAxis = axis;
          break;
        }
      }
      bool hasCohortStride = false;
      if (lhsMemory && laneAxis > 0)
        for (auto [axis, stride] :
             llvm::zip(lhsMemory.getAxisIds().asArrayRef(),
                       lhsMemory.getStrides().asArrayRef()))
          if (axis == laneAxis && stride != 0)
            hasCohortStride = true;
      const int64_t storageGroup = lhsAccess.getGroupSize();
      const int64_t storageLayer = lhsAccess.getLayerSize();
      auto storageWindow = lhsField
                               ? riscv_internal::storageWindowPlan(
                                     rewriter, lhsField, reductionAxis,
                                     /*projectionBase=*/0,
                                     /*projectionStride=*/1,
                                     /*projectionRepeat=*/1,
                                     /*projectionExtent=*/storageGroup,
                                     /*offsetAlignment=*/1)
                               : std::nullopt;
      const int64_t outputParts = resultParts(accumulatorType.getLayout());
      auto supplyPlan = riscv::groupedMacSupplyProjection(lhsType, accumulatorType);
      const bool exactGeometry =
          lhsInteger.isUnsigned() && lhsInteger.getWidth() < 8 &&
          rhsInteger.isSigned() && rhsInteger.getWidth() == 8 && lhsField &&
          rhsField && lhsLoad && rhsLoad && lhsPoint && rhsPoint &&
          lhsPoint == rhsPoint && storageWindow &&
          storageWindow->getKind() == "layered" &&
          rhsAccess.getMapping() == "natural" &&
          storageWindow->getLayerSize() % mac.getGroup() == 0 &&
          lhsAccess.getBitOffset() % 8 == 0 &&
          rhsAccess.getBitOffset() % 8 == 0 &&
          (interleaveRows > 0 || hasCohortStride) &&
          outputParts > 0 &&
          supplyPlan &&
          riscv::supportsRVVLayout(target, lhsLayout) &&
          riscv::supportsRVVLayout(target, loadLayout) &&
          riscv::supportsRVVLayout(target, partialLayout) &&
          riscv::supportsRVVLayout(target, accumulatorType.getLayout()) &&
          target.getHasWideningInteger();
      if (!exactGeometry) {
        reduce.emitError(
            "selected grouped MAC has no complete typed field, cohort-storage, "
            "RVV-layout, or shared reduction-point realization; lhs=")
            << lhsType << ", rhs=" << rhsType << ", lhs_access=" << lhsAccess
            << ", rhs_access=" << rhsAccess
            << ", lhs_field=" << static_cast<bool>(lhsField)
            << ", rhs_field=" << static_cast<bool>(rhsField)
            << ", lhs_load=" << static_cast<bool>(lhsLoad)
            << ", rhs_load=" << static_cast<bool>(rhsLoad)
            << ", shared_point=" << (lhsPoint && lhsPoint == rhsPoint)
            << ", reduction_lane=" << reductionLane
            << ", cohort_stride=" << hasCohortStride
            << ", interleave_rows="
            << (lhsMemory ? lhsMemory.getInterleaveRows() : -1)
            << ", result_parts="
            << (accumulatorType ? resultParts(accumulatorType.getLayout()) : -1)
            << ", supply_plan=" << static_cast<bool>(supplyPlan)
            << ", partial_layout=" << partialLayout
            << ", accumulator_layout="
            << (accumulatorType ? accumulatorType.getLayout()
                                : riscv::LayoutAttr())
            << ", lhs_layout_legal="
            << riscv::supportsRVVLayout(target, lhsLayout)
            << ", load_layout=" << loadLayout
            << ", load_layout_legal="
            << riscv::supportsRVVLayout(target, loadLayout)
            << ", partial_layout_legal="
            << riscv::supportsRVVLayout(target, partialLayout)
            << ", accumulator_layout_legal="
            << (accumulatorType && riscv::supportsRVVLayout(
                                       target, accumulatorType.getLayout()));
        failed = true;
        continue;
      }
      auto physicalPoint =
          point ? point.getDefiningOp<riscv::PhysicalPointOp>()
                : riscv::PhysicalPointOp();
      const bool exactWindow =
          physicalPoint &&
          physicalPoint.getResult().getType().getDomain().getTail() == "exact";
      mlir::Value active = physicalPoint ? physicalPoint.getActive()
                                         : fullAxisExtent(reduce, reductionAxis);
      if (!active) {
        reduce.emitError(
            "grouped MAC reduction has no explicit physical reduction extent");
        failed = true;
        continue;
      }
      auto groupedPlan = [&](llvm::StringRef kind) {
        llvm::SmallVector<int64_t> termOrder;
        const int64_t terms =
            static_cast<int64_t>(mac.getGroup()) * schedule.getUnroll();
        termOrder.reserve(static_cast<size_t>(terms));
        for (int64_t term = 0; term < terms; ++term)
          termOrder.push_back(term);
        const bool unitLoad = interleaveRows > 0;
        return riscv::GroupedMacPlanAttr::get(
            rewriter.getContext(), kind, mac.getGroup(), schedule.getUnroll(),
            reductionAxis, storageWindow->getGroupSize(),
            storageWindow->getLayerSize(), storageWindow->getElementBits(),
            storageWindow->getByteOffset(),
            rhsAccess.getBitOffset() / 8, lhsMemory.getElements(),
            interleaveRows, unitLoad ? "unit" : "strided",
            unitLoad ? 0 : laneAxis,
            unitLoad ? interleaveRows : 1,
            storageWindow->getPhysicalLayerBase(),
            storageWindow->getPhysicalLayerStep(),
            storageWindow->getShiftBase(), storageWindow->getShiftStep(),
            storageWindow->getMaskValue(),
            rewriter.getDenseI64ArrayAttr(termOrder));
      };
      rewriter.setInsertionPoint(reduce);
      mlir::Type element = accumulatorType.getElementType();
      mlir::TypedAttr zero = mlir::isa<mlir::FloatType>(element)
                                 ? mlir::cast<mlir::TypedAttr>(
                                       rewriter.getFloatAttr(element, 0.0))
                                 : mlir::cast<mlir::TypedAttr>(
                                       rewriter.getIntegerAttr(element, 0));
      auto scalarZero =
          rewriter.create<riscv::ConstantOp>(reduce.getLoc(), element, zero);
      auto accumulator = rewriter.create<riscv::RVVSplatOp>(
          reduce.getLoc(), accumulatorType, scalarZero,
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                               "rvv.splat", 0,
                               accumulatorType.getLayout().getRegisterGroups()));
      mlir::Value groupSize = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), mac.getGroup());
      mlir::Value groupCount = rewriter.create<mlir::arith::CeilDivUIOp>(
          reduce.getLoc(), active, groupSize);
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 0);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), schedule.getUnroll());
      auto loop = rewriter.create<mlir::scf::ForOp>(
          reduce.getLoc(), lower, groupCount, step,
          mlir::ValueRange{accumulator.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.schedule", schedule);
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      auto windowType = riscv::WindowType::get(
          rewriter.getContext(), "grouped-mac", lhs.getType(), rhs.getType(),
          accumulatorType, reductionAxis, schedule.getUnroll(), mac.getGroup(),
          resultParts(accumulatorType.getLayout()), partialLayout,
          accumulatorType.getLayout(),
          product({std::max<int64_t>(1, lhsLayout.getRegisterGroups()),
                   schedule.getUnroll(), static_cast<int64_t>(mac.getGroup())}));
      const int64_t termsPerWindow =
          product({static_cast<int64_t>(mac.getGroup()), schedule.getUnroll()});
      const bool compactWindow =
          termsPerWindow > 0 && storageLayer % termsPerWindow == 0;
      rewriter.setInsertionPointToStart(loop.getBody());
      auto load = rewriter.create<riscv::RVVGroupedMacLoadOp>(
          reduce.getLoc(), windowType, lhs, rhs, loop.getInductionVar(), active,
          loadLayout, partialLayout,
          rewriter.getDenseI64ArrayAttr(*supplyPlan),
          groupedPlan(compactWindow ? "compact" : "fragmented"),
          riscv_internal::leaf(
              rewriter, "rvv", "grouped-mac-load",
              "rvv.grouped-mac-load.u8-s8",
              "rvv.grouped-mac-load.u8-s8", 0,
              windowType.getResourceGroups(),
              0, 0, "none", exactWindow ? "exact" : "agnostic",
              {static_cast<int64_t>(mac.getGroup()), schedule.getUnroll(),
               reductionAxis}));
      auto stepOp = rewriter.create<riscv::RVVGroupedMacStepOp>(
          reduce.getLoc(), accumulatorType, load.getResult(),
          loop.getRegionIterArg(0),
          riscv_internal::leaf(
              rewriter, "rvv", "grouped-mac-step",
              "rvv.vwmaccsu.vx.grouped", "rvv.vwmaccsu.vx.grouped",
              windowType.getResourceGroups(),
              accumulatorType.getLayout().getRegisterGroups(),
              partialLayout.getRegisterGroups(), 0, "none",
              exactWindow ? "exact" : "agnostic",
              {static_cast<int64_t>(mac.getGroup()), schedule.getUnroll(),
               reductionAxis}));
      rewriter.setInsertionPointToEnd(loop.getBody());
      rewriter.create<mlir::scf::YieldOp>(reduce.getLoc(), stepOp.getResult());
      copyIdentity(reduce, load);
      copyIdentity(reduce, stepOp);
      if (resultType) {
        reduce.getResult().replaceAllUsesWith(loop.getResult(0));
        rewriter.eraseOp(reduce);
      } else {
        reduce.getInputMutable().assign(loop.getResult(0));
      }
      if (widen.getResult().use_empty())
        rewriter.eraseOp(widen);
      for (riscv::ConvertLayoutOp conversion : llvm::reverse(bridges))
        if (conversion.getResult().use_empty())
          rewriter.eraseOp(conversion);
      if (mac.getResult().use_empty())
        rewriter.eraseOp(mac);
    }
  }

  void lowerLoopCarriedWideningPartials(mlir::IRRewriter &rewriter,
                                        bool &failed) {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::scf::ForOp loop) { loops.push_back(loop); });

    for (mlir::scf::ForOp loop : loops) {
      if (!loop || loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
        continue;
      auto carryType =
          mlir::dyn_cast<mlir::IntegerType>(loop.getRegionIterArg(0).getType());
      auto initRange = riscv_internal::integerRange(loop.getInitArgs().front());
      if (!carryType || !carryType.isSigned() || carryType.getWidth() != 16 ||
          !initRange || initRange->minimum != 0 || initRange->maximum != 0 ||
          !loop.getRegionIterArg(0).hasOneUse() ||
          !loop.getResult(0).hasOneUse())
        continue;

      auto widen = mlir::dyn_cast<riscv::WidenOp>(
          *loop.getResult(0).getUsers().begin());
      auto widenedType = widen ? mlir::dyn_cast<mlir::IntegerType>(
                                    widen.getResult().getType())
                               : mlir::IntegerType();
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      auto add = yield && yield.getNumOperands() == 1
                     ? yield.getOperand(0).getDefiningOp<riscv::BinaryOp>()
                     : riscv::BinaryOp();
      if (!widen || !widenedType || !widenedType.isSigned() ||
          widenedType.getWidth() != 32 || !add || add.getKind() != "add" ||
          !add.getResult().hasOneUse() ||
          *add.getResult().getUsers().begin() != yield.getOperation())
        continue;

      mlir::Value contribution;
      if (add.getLhs() == loop.getRegionIterArg(0))
        contribution = add.getRhs();
      else if (add.getRhs() == loop.getRegionIterArg(0))
        contribution = add.getLhs();
      else
        continue;
      auto contract = contribution.getDefiningOp<riscv::ContractOp>();
      if (!contract || !contract.getResult().hasOneUse() ||
          contract.getResult().getType() != carryType ||
          contract.getOver().size() != 1)
        continue;

      const int64_t reductionAxis = contract.getOver()[0];
      auto lhs = mlir::dyn_cast<riscv::ValueType>(contract.getLhs().getType());
      auto rhs = mlir::dyn_cast<riscv::ValueType>(contract.getRhs().getType());
      auto lhsElement = lhs ? mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType())
                            : mlir::IntegerType();
      auto rhsElement = rhs ? mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType())
                            : mlir::IntegerType();
      auto lhsPosition = lhs ? axisPosition(lhs, reductionAxis)
                             : std::optional<size_t>();
      auto rhsPosition = rhs ? axisPosition(rhs, reductionAxis)
                             : std::optional<size_t>();
      if (!lhs || !rhs || !lhsElement || !rhsElement || lhsElement.isSignless() ||
          rhsElement.isSignless() ||
          (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
          std::max<unsigned>(8, lhsElement.getWidth()) !=
              std::max<unsigned>(8, rhsElement.getWidth()) ||
          2 * std::max<unsigned>(8, lhsElement.getWidth()) !=
              carryType.getWidth() ||
          !lhsPosition || !rhsPosition ||
          lhs.getLayout().getCarrier() != "rvv" ||
          rhs.getLayout().getCarrier() != "rvv" ||
          product(lhs.getLayout().getTimeFactors()) != 1 ||
          product(rhs.getLayout().getTimeFactors()) != 1 ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          lhs.getLayout().getLmulEighths() !=
              rhs.getLayout().getLmulEighths() ||
          lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
          lhs.getLayout().getLaneFactors()[*lhsPosition] !=
              rhs.getLayout().getLaneFactors()[*rhsPosition] ||
          lhs.getShape()[*lhsPosition] !=
              lhs.getLayout().getLaneFactors()[*lhsPosition] ||
          rhs.getShape()[*rhsPosition] !=
              rhs.getLayout().getLaneFactors()[*rhsPosition])
        continue;

      auto lower = constantInteger(loop.getLowerBound());
      auto upper = constantInteger(loop.getUpperBound());
      auto step = constantInteger(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
          (*upper - *lower) % *step)
        continue;
      const int64_t iterations = (*upper - *lower) / *step;
      auto lhsMagnitude = riscv_internal::maximumMagnitude(contract.getLhs());
      auto rhsMagnitude = riscv_internal::maximumMagnitude(contract.getRhs());
      const int64_t lanes = lhs.getLayout().getLaneFactors()[*lhsPosition];
      const __int128 bound =
          lhsMagnitude && rhsMagnitude
              ? static_cast<__int128>(*lhsMagnitude) * *rhsMagnitude * lanes *
                    iterations
              : static_cast<__int128>(std::numeric_limits<int64_t>::max());
      if (!lhsMagnitude || !rhsMagnitude || bound > 32767)
        continue;

      llvm::SmallVector<int64_t> axisIds{reductionAxis};
      llvm::SmallVector<int64_t> one{1};
      llvm::SmallVector<int64_t> lane{lanes};
      const int64_t partialLMUL = lhs.getLayout().getLmulEighths() * 2;
      const int64_t partialGroups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
      auto partialLayout = riscv::LayoutAttr::get(
          rewriter.getContext(), "rvv", riscv_internal::integers(rewriter, axisIds),
          riscv_internal::integers(rewriter, one),
          riscv_internal::integers(rewriter, lane),
          riscv_internal::integers(rewriter, one),
          riscv_internal::integers(rewriter, one),
          riscv_internal::integers(rewriter, one), carryType.getWidth(),
          partialLMUL, lhs.getLayout().getVl(), partialGroups,
          lhs.getLayout().getValidity());
      auto partialType = riscv::ValueType::get(
          rewriter.getContext(), carryType,
          riscv_internal::integers(rewriter, lane),
          riscv_internal::integers(rewriter, axisIds), partialLayout);
      auto kernel = loop->getParentOfType<riscv::KernelOp>();
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      if (!kernel || !kernel.getTarget().getHasWideningInteger() ||
          !llvm::is_contained(
              kernel.getTarget().getLegalLMULEighths().asArrayRef(), partialLMUL) ||
          !riscv::supportsRVVLayout(kernel.getTarget(), partialLayout) ||
          partialGroups + operandGroups >
              kernel.getTarget().getVectorRegisters())
        continue;

      rewriter.setInsertionPoint(loop);
      auto zero = rewriter.create<riscv::ConstantOp>(
          loop.getLoc(), carryType, rewriter.getIntegerAttr(carryType, 0));
      auto initial = rewriter.create<riscv::RVVSplatOp>(
          loop.getLoc(), partialType, zero.getResult(),
          riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat",
                               "rvv.splat", 0, partialGroups));
      auto replacement = rewriter.create<mlir::scf::ForOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
          loop.getStep(), mlir::ValueRange{initial.getResult()});
      for (mlir::NamedAttribute attribute : loop->getAttrs())
        replacement->setAttr(attribute.getName(), attribute.getValue());
      replacement->setAttr("weft.riscv.system_unroll",
                           rewriter.getStringAttr("disable"));
      if (iterations > 1 && iterations <= 8)
        replacement->setAttr(
            "weft.riscv.unroll_factor",
            rewriter.getI64IntegerAttr(iterations));
      if (auto defaultYield = mlir::dyn_cast<mlir::scf::YieldOp>(
              replacement.getBody()->getTerminator()))
        rewriter.eraseOp(defaultYield);

      mlir::IRMapping mapping;
      mapping.map(loop.getInductionVar(), replacement.getInductionVar());
      rewriter.setInsertionPointToStart(replacement.getBody());
      mlir::Value carried = replacement.getRegionIterArg(0);
      for (mlir::Operation &operation : loop.getBody()->without_terminator()) {
        if (&operation == contract.getOperation()) {
          mlir::Value mappedLhs = mapping.lookupOrDefault(contract.getLhs());
          mlir::Value mappedRhs = mapping.lookupOrDefault(contract.getRhs());
          auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
              contract.getLoc(), partialType, mappedLhs, mappedRhs, carried,
              rewriter.getDenseI64ArrayAttr({reductionAxis}),
              riscv_internal::leaf(
                  rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                  "rvv.vwmacc.partial", operandGroups + partialGroups,
                  partialGroups, 0, 0, "none", "agnostic"));
          copyIdentity(contract, accumulate);
          carried = accumulate.getResult();
          continue;
        }
        if (&operation == add.getOperation())
          continue;
        rewriter.clone(operation, mapping);
      }
      rewriter.setInsertionPointToEnd(replacement.getBody());
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), carried);

      rewriter.setInsertionPointAfter(replacement);
      auto setType = riscv::PartialSetType::get(
          rewriter.getContext(), partialType, reductionAxis, 1, lanes,
          partialGroups);
      const int64_t partialOwner = ownerDomain(replacement);
      auto captured = rewriter.create<riscv::RVVPartialCaptureOp>(
          widen.getLoc(), setType, replacement.getResult(0),
          partialOwner, nextPartialBirthId++, partialOwner,
          riscv_internal::leaf(rewriter, "rvv", "partial-capture",
                               "rvv.partial-capture", "rvv.partial-capture",
                               partialGroups, partialGroups, 0, 0, "none",
                               partialType.getLayout().getValidity() == "tail"
                                   ? "agnostic"
                                   : "exact"));
      copyIdentity(widen, captured);
      auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
          widen.getLoc(), widenedType, captured.getResult(), reductionAxis,
          riscv_internal::leaf(rewriter, "rvv", "partial-finalize",
                               "rvv.partial-finalize.reduce",
                               "rvv.partial-finalize.reduce", partialGroups, 0,
                               0, 0, "none", "exact",
                               {reductionAxis, 1}));
      copyIdentity(widen, finalized);
      widen.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(widen);
      rewriter.eraseOp(loop);
    }
  }

  void lowerContracts(mlir::IRRewriter &rewriter, bool &failed) {
    llvm::SmallVector<mlir::Operation *> contracts;
    getOperation().walk([&](mlir::Operation *operation) {
      if (mlir::isa<riscv::DotOp, riscv::ContractOp,
                    riscv::OuterContractOp>(operation))
        contracts.push_back(operation);
    });
    for (mlir::Operation *operation : contracts) {
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
      auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
      auto accType = operation->getAttrOfType<mlir::TypeAttr>("acc_type");
      auto laneOperand =
          operation->getAttrOfType<mlir::StringAttr>("lane_operand");
      auto laneMemory =
          operation->getAttrOfType<mlir::StringAttr>("lane_memory_form");
      auto schedule = operation->getAttrOfType<riscv::ScheduleAttr>("schedule");
      if (!implementation || !over || !laneOperand || !laneMemory || !schedule) {
        operation->emitError("contract has no complete typed physical contract");
        failed = true;
        continue;
      }
      if (implementation.getFamily() == "widen-dot" &&
          implementation.getOperation() == "rvv.vwmul-vwredsum") {
        riscv::ReduceOp fusedFreeAxisReduce;
        int64_t fusedFreeAxis = 0;
        if (resultType && operation->getResult(0).hasOneUse()) {
          auto reduce = mlir::dyn_cast<riscv::ReduceOp>(
              *operation->getResult(0).getUsers().begin());
          if (reduce && reduce.getKind() == "add" &&
              resultType.getAxisIds().size() == 1 && reduce.getAxis() == 0 &&
              mlir::isa<mlir::IntegerType>(reduce.getResult().getType())) {
            fusedFreeAxisReduce = reduce;
            fusedFreeAxis = resultType.getAxisIds()[0];
          }
        }

        mlir::Value lhsValue = operation->getOperand(0);
        mlir::Value rhsValue = operation->getOperand(1);
        auto streamFreeAxis = [&](mlir::Value value) -> mlir::Value {
          auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
          if (!type || !fusedFreeAxis)
            return value;
          auto axes = type.getAxisIds().asArrayRef();
          auto found = llvm::find(axes, fusedFreeAxis);
          if (found == axes.end())
            return value;
          const size_t position =
              static_cast<size_t>(found - axes.begin());
          auto layout = type.getLayout();
          llvm::SmallVector<int64_t> time(
              layout.getTimeFactors().asArrayRef().begin(),
              layout.getTimeFactors().asArrayRef().end());
          llvm::SmallVector<int64_t> lane(
              layout.getLaneFactors().asArrayRef().begin(),
              layout.getLaneFactors().asArrayRef().end());
          llvm::SmallVector<int64_t> replica(
              layout.getReplicaFactors().asArrayRef().begin(),
              layout.getReplicaFactors().asArrayRef().end());
          if (position >= time.size() || position >= lane.size() ||
              position >= replica.size()) {
            operation->emitError(
                "free-axis reduction cannot map its typed operand to issue time");
            failed = true;
            return {};
          }
          // Layout propagation may already coalesce the immediately reduced
          // free axis with the contraction axis into one complete product lane
          // carrier.  In that case no replica-to-time conversion is required:
          // the widened-dot slice planner below consumes both axes as physical
          // reduction lanes and validates that they form one legal contiguous
          // RVV carrier.  Reject mixed lane/time or lane/replica ownership here;
          // those still require an explicit conversion that this lowering does
          // not define.
          if (lane[position] > 1) {
            if (time[position] == 1 && replica[position] == 1)
              return value;
            operation->emitError(
                "free-axis reduction has an incomplete coalesced lane carrier");
            failed = true;
            return {};
          }
          const int64_t moved = replica[position];
          if (moved <= 1)
            return value;
          time[position] = product({time[position], moved});
          replica[position] = 1;
          const int64_t replicaProduct =
              riscv_internal::staticProduct(replica).value_or(-1);
          const int64_t groups =
              product({std::max<int64_t>(
                           1, (layout.getLmulEighths() + 7) / 8),
                       replicaProduct});
          auto streamedLayout = riscv::LayoutAttr::get(
              rewriter.getContext(), layout.getCarrier(), layout.getAxisIds(),
              riscv_internal::integers(rewriter, time),
              riscv_internal::integers(rewriter, lane),
              riscv_internal::integers(rewriter, replica),
              layout.getFragmentFactors(), layout.getLocalFactors(),
              layout.getSew(), layout.getLmulEighths(), layout.getVl(), groups,
              layout.getValidity());
          auto streamedType = mlir::cast<riscv::ValueType>(
              riscv_internal::withLayout(type, streamedLayout));
          rewriter.setInsertionPoint(operation);
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              operation->getLoc(), streamedType, value,
              riscv_internal::layoutConversion(rewriter, layout,
                                               streamedLayout),
              riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
          copyIdentity(operation, conversion);
          return conversion.getResult();
        };
        if (fusedFreeAxisReduce) {
          lhsValue = streamFreeAxis(lhsValue);
          rhsValue = streamFreeAxis(rhsValue);
          if (!lhsValue || !rhsValue || failed)
            continue;
        }
        auto lhsType =
            mlir::dyn_cast<riscv::ValueType>(lhsValue.getType());
        auto rhsType =
            mlir::dyn_cast<riscv::ValueType>(rhsValue.getType());
        auto lhsElement = lhsType
                              ? mlir::dyn_cast<mlir::IntegerType>(
                                    lhsType.getElementType())
                              : mlir::IntegerType();
        auto rhsElement = rhsType
                              ? mlir::dyn_cast<mlir::IntegerType>(
                                    rhsType.getElementType())
                              : mlir::IntegerType();
        auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
            riscv_internal::logicalElement(operation->getResult(0).getType()));
        auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
        llvm::SmallVector<int64_t> physicalReductionAxes(
            over.asArrayRef().begin(), over.asArrayRef().end());
        mlir::Type physicalResultType = operation->getResult(0).getType();
        riscv::ValueType laneResultType = resultType;
        if (fusedFreeAxisReduce) {
          physicalReductionAxes.push_back(fusedFreeAxis);
          physicalResultType = fusedFreeAxisReduce.getResult().getType();
          laneResultType = {};
        }
        // A complete integer contraction has one issue domain. Reconcile its
        // numeric operands at the use edge, preserving storage/index producer
        // layouts. Moving reduced replicas to time must cover every coordinate;
        // treating them as independent output replicas would drop contributions.
        if (!laneResultType && physicalReductionAxes.size() > 1 && lhsType && rhsType &&
            lhsType.getShape() == rhsType.getShape() && lhsType.getAxisIds() == rhsType.getAxisIds() &&
            lhsType.getAxisIds().size() == physicalReductionAxes.size() &&
            lhsType.getLayout().getCarrier() == "rvv" && rhsType.getLayout().getCarrier() == "rvv" &&
            lhsType.getLayout().getValidity() == "full" && rhsType.getLayout().getValidity() == "full" &&
            lhsType.getLayout().getSew() == rhsType.getLayout().getSew() &&
            llvm::all_of(lhsType.getAxisIds().asArrayRef(), [&](int64_t axis) {
              return llvm::is_contained(physicalReductionAxes, axis);
            })) {
          llvm::SmallVector<int64_t> lanes, time, ones(lhsType.getShape().size(), 1);
          bool complete = lhsType.getLayout().getReplicaGroup() == 1 && rhsType.getLayout().getReplicaGroup() == 1;
          for (size_t i = 0; i < lhsType.getShape().size(); ++i) {
            const int64_t lane = std::gcd(lhsType.getLayout().getLaneFactors()[i],
                                          rhsType.getLayout().getLaneFactors()[i]);
            const int64_t extent = lhsType.getShape()[i];
            complete &= lane > 0 && extent > 0 && extent % lane == 0 &&
                lhsType.getLayout().getFragmentFactors()[i] == 1 && rhsType.getLayout().getFragmentFactors()[i] == 1 &&
                lhsType.getLayout().getLocalFactors()[i] == 1 && rhsType.getLayout().getLocalFactors()[i] == 1;
            lanes.push_back(lane);
            time.push_back(lane > 0 ? extent / lane : 0);
          }
          const int64_t vl = riscv_internal::staticProduct(lanes).value_or(-1);
          riscv::LayoutAttr common;
          for (int64_t lmul : target.getLegalLMULEighths().asArrayRef()) {
            if (!complete || vl <= 0 ||
                !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), 2 * lmul)) continue;
            auto candidate = riscv::LayoutAttr::get(rewriter.getContext(), "rvv", lhsType.getAxisIds(),
                riscv_internal::integers(rewriter, time), riscv_internal::integers(rewriter, lanes),
                riscv_internal::integers(rewriter, ones), riscv_internal::integers(rewriter, ones),
                riscv_internal::integers(rewriter, ones), lhsType.getLayout().getSew(), lmul, vl,
                std::max<int64_t>(1, (lmul + 7) / 8), "full");
            if (riscv::supportsRVVLayout(target, candidate)) { common = candidate; break; }
          }
          if (common) {
            auto left = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(lhsType, common));
            auto right = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(rhsType, common));
            if (riscv::rvvLayoutRepackSlices(lhsType, left) && riscv::rvvLayoutRepackSlices(rhsType, right)) {
              auto convert = [&](mlir::Value &value, riscv::ValueType &type, riscv::ValueType next) {
                if (type == next) return;
                rewriter.setInsertionPoint(operation);
                auto converted = rewriter.create<riscv::ConvertLayoutOp>(operation->getLoc(), next, value,
                    riscv_internal::conversion(rewriter, "lane_repack", "pure",
                        riscv::rvvLayoutRepackTemporaryGroups(target, type, next)),
                    riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
                copyIdentity(operation, converted);
                value = converted.getResult();
                type = next;
              };
              convert(lhsValue, lhsType, left);
              convert(rhsValue, rhsType, right);
            }
          }
        }
        auto lhsLaneSlices =
            lhsType ? planWidenDotLaneSlices(lhsType, laneResultType,
                                             physicalReductionAxes, target)
                    : std::nullopt;
        auto rhsLaneSlices =
            rhsType ? planWidenDotLaneSlices(rhsType, laneResultType,
                                             physicalReductionAxes, target)
                    : std::nullopt;
        // A packed source may supply an m1 slice while an equally sized
        // storage window uses a fractional register. Give that window an
        // explicit m1 carrier; its logical lanes and register residency stay
        // unchanged, and both operands then use the same widening instruction.
        if (lhsLaneSlices && rhsLaneSlices &&
            lhsLaneSlices->reductionLanes == rhsLaneSlices->reductionLanes &&
            lhsLaneSlices->reductionStreams == rhsLaneSlices->reductionStreams &&
            lhsLaneSlices->sliceLmulEighths != rhsLaneSlices->sliceLmulEighths &&
            std::max(lhsLaneSlices->sliceLmulEighths,
                     rhsLaneSlices->sliceLmulEighths) == 8) {
          auto extendFractional = [&](mlir::Value &value, riscv::ValueType &type) {
            auto layout = type.getLayout();
            if (layout.getLmulEighths() >= 8) return;
            auto extended = riscv::LayoutAttr::get(
                rewriter.getContext(), layout.getCarrier(), layout.getAxisIds(),
                layout.getTimeFactors(), layout.getLaneFactors(),
                layout.getReplicaFactors(), layout.getFragmentFactors(),
                layout.getLocalFactors(), layout.getSew(), 8, layout.getVl(),
                layout.getRegisterGroups(), layout.getValidity());
            if (!riscv::supportsRVVLayout(target, extended)) return;
            type = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(type, extended));
            rewriter.setInsertionPoint(operation);
            auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
                operation->getLoc(), type, value,
                riscv_internal::layoutConversion(rewriter, layout, extended),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            copyIdentity(operation, conversion);
            value = conversion.getResult();
          };
          extendFractional(lhsValue, lhsType);
          extendFractional(rhsValue, rhsType);
          lhsLaneSlices = planWidenDotLaneSlices(lhsType, laneResultType, physicalReductionAxes, target);
          rhsLaneSlices = planWidenDotLaneSlices(rhsType, laneResultType, physicalReductionAxes, target);
        }
        const int64_t partialLMUL =
            lhsLaneSlices
                ? product({lhsLaneSlices->sliceLmulEighths, int64_t{2}})
                : -1;
        if (!lhsType || !rhsType || !lhsElement || !rhsElement ||
            !resultElement || lhsElement.isSignless() ||
            rhsElement.isSignless() || resultElement.isSignless() ||
            lhsElement.getWidth() > 16 || rhsElement.getWidth() > 16 ||
            std::max<unsigned>(8, lhsElement.getWidth()) !=
                std::max<unsigned>(8, rhsElement.getWidth()) ||
            (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
            !resultElement.isSigned() || resultElement.getWidth() != 32 ||
            over.empty() || !target.getHasWideningInteger() ||
            !riscv::supportsRVVLayout(target, lhsType.getLayout()) ||
            !riscv::supportsRVVLayout(target, rhsType.getLayout()) ||
            lhsType.getLayout().getSew() != rhsType.getLayout().getSew() ||
            !lhsLaneSlices || !rhsLaneSlices ||
            lhsLaneSlices->reductionLanes != rhsLaneSlices->reductionLanes ||
            lhsLaneSlices->reductionStreams != rhsLaneSlices->reductionStreams ||
            lhsLaneSlices->sliceLmulEighths !=
                rhsLaneSlices->sliceLmulEighths ||
            lhsLaneSlices->offsets.size() != rhsLaneSlices->offsets.size() ||
            lhsLaneSlices->parts.size() != rhsLaneSlices->parts.size() ||
            !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                                partialLMUL) ||
            (resultType && resultType.getLayout().getCarrier() != "scalar")) {
          operation->emitError(
              "selected RVV widening dot has no legal typed operands, scalar "
              "free-axis result, or target widening shape; lhs=")
              << operation->getOperand(0).getType() << ", rhs="
              << operation->getOperand(1).getType() << ", result="
              << operation->getResult(0).getType() << ", over=" << over
              << ", partial_lmul=" << partialLMUL << ", target=" << target;
          failed = true;
          continue;
        }
        const int64_t streams = lhsLaneSlices->reductionStreams;
        const unsigned partialWidth =
            std::max<unsigned>(8, lhsElement.getWidth()) * 2;
        const int64_t partialMaximum =
            (int64_t{1} << (partialWidth - 1)) - 1;
        auto lhsMagnitude = riscv_internal::maximumMagnitude(lhsValue);
        auto rhsMagnitude = riscv_internal::maximumMagnitude(rhsValue);
        auto hasFullReductionMapping = [&](riscv::ValueType value) {
          for (int64_t axis : over.asArrayRef()) {
            auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
            if (found == value.getAxisIds().asArrayRef().end())
              return false;
            const size_t position = static_cast<size_t>(
                found - value.getAxisIds().asArrayRef().begin());
            const int64_t extent = value.getShape()[position];
            if (extent <= 0 ||
                value.getLayout().getLaneFactors()[position] *
                        value.getLayout().getTimeFactors()[position] !=
                    extent)
              return false;
          }
          return true;
        };
        const bool fusedStreamsLegal =
            hasFullReductionMapping(lhsType) &&
            hasFullReductionMapping(rhsType) && streams > 1 &&
            lhsMagnitude && rhsMagnitude && *rhsMagnitude > 0 &&
            *lhsMagnitude <= partialMaximum / *rhsMagnitude / streams;
        const int64_t partialGroups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
        const int64_t outputParts =
            lhsLaneSlices ? static_cast<int64_t>(lhsLaneSlices->offsets.size())
                           : -1;
        // Topology is selected after all typed storage windows and downstream
        // reductions exist.  Until then reserve the conservative per-stream
        // footprint rather than making a second topology decision here.
        const int64_t livePartials = streams;
        const int64_t temporaryGroups =
            sum(product({livePartials, partialGroups}), 2);
        if (outputParts <= 0 || livePartials <= 0 || temporaryGroups <= 0) {
          operation->emitError(
              "selected RVV widening dot has invalid result/resource geometry");
          failed = true;
          continue;
        }
        rewriter.setInsertionPoint(operation);
        auto schedule = operation->getAttrOfType<riscv::ScheduleAttr>("schedule");
        if (!schedule) {
          operation->emitError(
              "selected RVV widening dot lost its typed physical schedule");
          failed = true;
          continue;
        }
        auto partialTopology = riscv::PartialTopologyAttr::get(
            rewriter.getContext(), "unassigned", -1,
            rewriter.getDenseI64ArrayAttr({}),
            rewriter.getDenseI64ArrayAttr({}), 0, 0, 0, 0, 0,
            rewriter.getDenseI64ArrayAttr({}), 0);
        auto widenedDot = rewriter.create<riscv::RVVWidenDotOp>(
            operation->getLoc(), physicalResultType, lhsValue, rhsValue,
            rewriter.getDenseI64ArrayAttr(physicalReductionAxes),
            partialTopology, fusedStreamsLegal, schedule.getUnroll(),
            lhsLaneSlices->reductionLanes,
            lhsLaneSlices->reductionStreams,
            lhsLaneSlices->sliceLmulEighths,
            rewriter.getDenseI64ArrayAttr(lhsLaneSlices->offsets),
            rewriter.getDenseI64ArrayAttr(rhsLaneSlices->offsets),
            rewriter.getDenseI64ArrayAttr(lhsLaneSlices->parts),
            rewriter.getDenseI64ArrayAttr(rhsLaneSlices->parts),
            riscv::PartialLayoutPlanAttr(),
            riscv::NestedPartialPlanAttr(),
            riscv::SequentialPartialPlanAttr(),
            riscv::ScaledPartialPlanAttr(),
            riscv::LayeredPartialPlanAttr(),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-dot", "rvv.vwmul-vwredsum",
                "rvv.vwmul-vwredsum",
                lhsType.getLayout().getRegisterGroups() +
                    rhsType.getLayout().getRegisterGroups(),
                0, temporaryGroups));
        copyIdentity(operation, widenedDot);
        // A multi-stream contraction owns an implicit issue loop.  A
        // single-stream contraction can still be nested directly in an
        // author-visible sub-Level over the same reduction axis; that sub-Level
        // is likewise an issue loop, not the outer storage-block traversal.
        // Preserve the selected unroll on either typed relation so the partial
        // topology planner sees one unexpanded reduction epoch.
        auto issueLoop = operation->getParentOfType<mlir::scf::ForOp>();
        auto issueLevel = issueLoop
                              ? issueLoop->getAttrOfType<riscv::LevelAttr>(
                                    "weft.riscv.level")
                              : riscv::LevelAttr();
        const bool explicitReductionIssue =
            issueLevel && issueLevel.getRelation() == "subs" &&
            llvm::is_contained(physicalReductionAxes, issueLevel.getAxisId());
        if (schedule.getUnroll() > 1 &&
            ((streams > 1 && !fusedStreamsLegal) || explicitReductionIssue)) {
          // The schedule belongs to the physical issue loop that directly
          // contains this contraction.  A more distant Level may carry the
          // same logical reduction axis while iterating whole storage blocks;
          // binding the local unroll there duplicates the complete block
          // program instead of its partial contributions.
          if (issueLoop) {
            auto existing = issueLoop->getAttrOfType<mlir::IntegerAttr>(
                "weft.riscv.unroll_factor");
            if (existing && existing.getInt() != schedule.getUnroll()) {
              operation->emitError(
                  "one physical Level received incompatible contraction unroll bindings");
              failed = true;
              rewriter.eraseOp(widenedDot);
              continue;
            }
            issueLoop->setAttr(
                "weft.riscv.unroll_factor",
                rewriter.getI64IntegerAttr(schedule.getUnroll()));
          }
        }
        if (fusedFreeAxisReduce) {
          fusedFreeAxisReduce.getResult().replaceAllUsesWith(
              widenedDot.getResult());
          rewriter.eraseOp(fusedFreeAxisReduce);
        } else {
          operation->getResult(0).replaceAllUsesWith(widenedDot.getResult());
        }
        rewriter.eraseOp(operation);
        continue;
      }
      if (!resultType || (accType && resultType.getElementType() != accType.getValue())) {
        operation->emitError("contraction requires a shaped result with its declared accumulator element");
        failed = true;
        continue;
      }
      if (schedule.getUnroll() < 1 || schedule.getUnroll() > 16 ||
          schedule.getPipelineDepth() != 1 || schedule.getBufferCount() != 1) {
        operation->emitError("register contraction requires a bounded unroll and one-stage schedule");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(operation);
      if (implementation.getEngine() == "ime") {
        auto result = riscv_internal::materializeFragmentProduct(
            rewriter, operation, implementation, resultType);
        if (mlir::failed(result)) { failed = true; continue; }
        rewriter.replaceOp(operation, *result);
        continue;
      }
      const int64_t axis = over.asArrayRef().front();
      const bool widening = implementation.getFamily() == "widen-float-contract";
      llvm::StringRef instruction;
      if (widening && implementation.getOperation() == "rvv.vfwmacc")
        instruction = "rvv.vfwmacc.vf";
      else if (implementation.getOperation() == "rvv.vfmacc")
        instruction = "rvv.vfmacc.vf";
      else if (implementation.getOperation() == "rvv.vmacc")
        instruction = "rvv.vmacc.vx";
      else {
        operation->emitError("contraction has no exact scalar/vector FMA instruction");
        failed = true;
        continue;
      }
      auto kernel = operation->getParentOfType<riscv::KernelOp>();
      auto element = resultType.getElementType();
      auto layout = resultType.getLayout();
      const unsigned laneIndex = laneOperand.getValue() == "rhs" ? 1 : 0;
      auto laneType = mlir::dyn_cast<riscv::ValueType>(operation->getOperand(laneIndex).getType());
      if (!laneType) {
        operation->emitError("contraction lane operand is not a numeric value");
        failed = true;
        continue;
      }

      // The source Level and the closed value shape independently bound K.
      // A source tail can be smaller than the declared register window.
      auto point = findOperandPoint(operation->getOperand(0), axis);
      if (!point) point = findOperandPoint(operation->getOperand(1), axis);
      auto physicalPoint = point ? point.getDefiningOp<riscv::PhysicalPointOp>()
                                 : riscv::PhysicalPointOp();
      std::optional<int64_t> staticExtent;
      for (mlir::Value input : operation->getOperands().take_front(2)) {
        auto value = mlir::dyn_cast<riscv::ValueType>(input.getType());
        if (!value) continue;
        auto position = axisPosition(value, axis);
        if (!position || value.getShape()[*position] <= 0) continue;
        int64_t bound = value.getShape()[*position];
        if (staticExtent && *staticExtent != bound) {
          operation->emitError("contraction operands disagree on their reduction extent");
          failed = true;
          break;
        }
        staticExtent = bound;
      }
      if (failed) continue;
      mlir::Value upper = physicalPoint ? physicalPoint.getActive() : fullAxisExtent(operation, axis);
      if (staticExtent) {
        auto bound = rewriter.create<mlir::arith::ConstantIndexOp>(operation->getLoc(), *staticExtent);
        upper = upper ? mlir::Value(rewriter.create<mlir::arith::MinUIOp>(operation->getLoc(), upper, bound))
                      : mlir::Value(bound);
      }
      if (!upper) {
        operation->emitError("contraction reduction axis has no physical extent");
        failed = true;
        continue;
      }
      auto projectedLane = riscv_internal::projectLayout(rewriter, laneType, layout, kernel.getTarget());
      if (!projectedLane || projectedLane.getCarrier() != "rvv") {
        operation->emitError("contraction has no legal free-axis lane representation");
        failed = true;
        continue;
      }

      // Integer addition uses the same declared accumulator overflow rule.
      // Floating adds remain outside the closed operation and are never fused.
      riscv::BinaryOp fusedAdd;
      mlir::Value initial;
      auto result = operation->getResult(0);
      if (mlir::isa<mlir::IntegerType>(element) && result.hasOneUse()) {
        auto &use = *result.getUses().begin();
        auto add = mlir::dyn_cast<riscv::BinaryOp>(use.getOwner());
        if (add && add.getKind() == "add" && add.getResult().getType() == resultType) {
          auto seed = use.getOperandNumber() == 0 ? add.getRhs() : add.getLhs();
          mlir::DominanceInfo dominance(kernel);
          llvm::SmallVector<mlir::Operation *> definitions;
          llvm::SmallPtrSet<mlir::Operation *, 8> seen;
          std::function<bool(mlir::Value)> available = [&](mlir::Value value) {
            if (dominance.dominates(value, operation)) return true;
            auto *definition = value.getDefiningOp();
            if (!definition || definition == operation || definition->getBlock() != operation->getBlock() ||
                definition->getNumRegions() || !mlir::isMemoryEffectFree(definition)) return false;
            if (!seen.insert(definition).second) return true;
            for (auto operand : definition->getOperands()) if (!available(operand)) return false;
            definitions.push_back(definition);
            return true;
          };
          if (available(seed)) {
            for (auto *definition : definitions) definition->moveBefore(operation);
            fusedAdd = add;
            initial = seed;
          }
        }
      }
      if (!initial) {
        auto zero = mlir::cast<mlir::TypedAttr>(rewriter.getZeroAttr(element));
        auto scalar = rewriter.create<riscv::ConstantOp>(operation->getLoc(), element, zero);
        initial = rewriter.create<riscv::RVVSplatOp>(operation->getLoc(), resultType, scalar,
            riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                                 layout.getRegisterGroups()));
        copyIdentity(operation, scalar);
        copyIdentity(operation, initial.getDefiningOp());
      }
      auto lower = rewriter.create<mlir::arith::ConstantIndexOp>(operation->getLoc(), 0);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(operation->getLoc(), schedule.getUnroll());
      // Within a closed integer contraction, two independent partials preserve
      // modular addition while shortening the unrolled MAC dependency chain.
      // Bound the extra residency to two register groups; the complete live
      // program still goes through ordinary resource materialization.
      const bool independent = mlir::isa<mlir::IntegerType>(element) &&
          layout.getValidity() == "full" && layout.getRegisterGroups() <= 2 &&
          schedule.getUnroll() >= 2 && schedule.getUnroll() % 2 == 0 &&
          staticExtent && *staticExtent >= schedule.getUnroll();
      llvm::SmallVector<mlir::Value> seeds{initial};
      if (independent) {
        auto zero = rewriter.create<riscv::ConstantOp>(operation->getLoc(), element,
            mlir::cast<mlir::TypedAttr>(rewriter.getZeroAttr(element)));
        auto seed = rewriter.create<riscv::RVVSplatOp>(operation->getLoc(), resultType, zero,
            riscv_internal::leaf(rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                                 layout.getRegisterGroups()));
        copyIdentity(operation, zero);
        copyIdentity(operation, seed);
        seeds.push_back(seed);
      }
      auto loop = rewriter.create<mlir::scf::ForOp>(operation->getLoc(), lower, upper, step,
                                                   seeds);
      loop->setAttr("weft.riscv.direction", rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll", rewriter.getStringAttr("disable"));
      copyIdentity(operation, loop);
      rewriter.setInsertionPointToStart(loop.getBody());

      // Each supply consumes K before the FMA. A dense view may be re-read only
      // after the common effect/axis proof; otherwise project the numeric SSA.
      // The latter includes snapshots and arbitrary computed operands.
      mlir::Value projectedInputs[2];
      auto supplyOperand = [&](unsigned input, mlir::Value index) -> mlir::FailureOr<mlir::Value> {
        auto original = operation->getOperand(input);
        auto type = mlir::dyn_cast<riscv::ValueType>(original.getType());
        if (!type || !llvm::is_contained(type.getAxisIds().asArrayRef(), axis)) return original;
        const bool isLane = input == laneIndex;
        auto projected = projectedReductionType(rewriter, type, axis);
        if (isLane) {
          auto candidate = riscv_internal::projectLayout(rewriter, projected, layout, kernel.getTarget());
          if (!candidate) return mlir::failure();
          projected = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(projected, candidate));
        }
        auto read = projectContractionRead(rewriter, original, projected, {{axis, index}}, operation);
        if (mlir::failed(read)) return mlir::failure();
        if (*read) return *read;
        auto source = sourceField(original);
        auto access = queryContractionSupply(original).access;
        if (source && access && access.getMapping() != "dense") {
          auto owner = sourceLoad(source.getOwner());
          if (!owner) return mlir::failure();
          auto encoding = mlir::cast<kernel::EncodingType>(owner.getRegion().getType().getEncoding());
          auto form = !isLane || riscv_internal::interleaveRows(source, encoding) > 0 ? "unit" : "strided";
          access = riscv::AccessAttr::get(rewriter.getContext(), form, access.getMapping(),
              access.getAlignment(), 0, 0, access.getGroupSize(), access.getLayerSize(),
              access.getJoinFields(), access.getJoinLowBits(), access.getJoinRole(),
              access.getBitOffset(), access.getStorageBits(), access.getOrder(), access.getFieldChunkBytes());
          llvm::SmallVector<mlir::Attribute> selectors;
          for (int64_t id : type.getAxisIds().asArrayRef())
            selectors.push_back(rewriter.getStringAttr(id == axis ? "index" : "all"));
          auto extract = rewriter.create<riscv::ExtractOp>(operation->getLoc(), projected, original,
              mlir::ValueRange{index}, rewriter.getArrayAttr(selectors), access,
              riscv_internal::unselectedLeaf(rewriter));
          copyIdentity(operation, extract);
          llvm::SmallVector<mlir::Value> indices;
          if (!riscv::fieldReadProjection(extract, &indices)) return mlir::failure();
          auto temporaries = riscv::fieldReadTemporaryGroups(extract, projected, access, indices);
          if (!temporaries) return mlir::failure();
          auto instruction = riscv::fieldReadInstruction(extract, projected, access);
          auto read = rewriter.create<riscv::FieldReadOp>(operation->getLoc(), projected, extract,
              indices, access, riscv_internal::leaf(rewriter, "transfer", "field-read", instruction,
                  instruction, 0, 0, *temporaries, 0, "none", "exact"));
          copyIdentity(operation, read);
          return read.getResult();
        }
        auto numeric = original;
        if (isLane && type.getLayout() != projectedLane) {
          // The complete conversion dominates every K projection. Emission may
          // only select a register part, never reload the original pointer.
          if (!projectedInputs[input]) {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(loop);
            auto converted = mlir::cast<riscv::ValueType>(riscv_internal::withLayout(type, projectedLane));
            auto conversion = rewriter.create<riscv::ConvertLayoutOp>(operation->getLoc(), converted, original,
                riscv_internal::layoutConversion(rewriter, type.getLayout(), projectedLane),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            copyIdentity(operation, conversion);
            projectedInputs[input] = conversion;
          }
          numeric = projectedInputs[input];
          projected = projectedReductionType(rewriter, mlir::cast<riscv::ValueType>(numeric.getType()), axis);
        }
        auto projection = rewriter.create<riscv::ProjectReductionOperandOp>(operation->getLoc(), projected,
            numeric, index, axis, riscv_internal::leaf(rewriter, "transfer", "reduction-projection",
                "rvv.project-reduction-operand", "rvv.project-reduction-operand", 0, 0, 1, 0));
        copyIdentity(operation, projection);
        return projection.getResult();
      };
      auto promote = [&](mlir::Value input) -> mlir::FailureOr<mlir::Value> {
        auto source = riscv_internal::logicalElement(input.getType());
        if (widening || source == element) return input;
        auto integer = mlir::dyn_cast<mlir::IntegerType>(source);
        auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(element);
        bool integerWiden = integer && targetInteger && integer.getWidth() < targetInteger.getWidth();
        bool floatWiden = (source.isF16() || source.isBF16()) && element.isF32();
        if (!integerWiden && !floatWiden) return mlir::failure();
        mlir::Type target = element;
        if (auto value = mlir::dyn_cast<riscv::ValueType>(input.getType())) {
          auto seed = riscv::ValueType::get(rewriter.getContext(), element, value.getShape(),
                                          value.getAxisIds(), value.getLayout());
          auto before = value.getLayout();
          auto candidate = before.getCarrier() == "scalar"
              ? riscv::LayoutAttr::get(rewriter.getContext(), "scalar", before.getAxisIds(),
                  before.getTimeFactors(), before.getLaneFactors(), before.getReplicaFactors(),
                  before.getFragmentFactors(), before.getLocalFactors(),
                  element.getIntOrFloatBitWidth(), 0, 1, 0, before.getValidity())
              : riscv_internal::projectLayout(rewriter, seed, before, kernel.getTarget());
          if (!candidate) return mlir::failure();
          target = riscv_internal::withLayout(seed, candidate);
        }
        mlir::Value converted;
        if (integerWiden)
          converted = rewriter.create<riscv::WidenOp>(operation->getLoc(), target, input,
                                                     riscv_internal::unselectedLeaf(rewriter));
        else
          converted = rewriter.create<riscv::CastOp>(operation->getLoc(), target, input,
                                                    riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, converted.getDefiningOp());
        return converted;
      };

      auto stepLeaf = riscv_internal::leaf(rewriter, "rvv", "contract-step",
          instruction, instruction, 0, 0, 0, 0, "none", "agnostic", over.asArrayRef());
      llvm::SmallVector<mlir::Value> partials(loop.getRegionIterArgs());
      bool invalid = false;
      for (int64_t offset = 0; offset < schedule.getUnroll(); ++offset) {
        mlir::Value &carried = partials[offset % partials.size()];
        mlir::Value index = loop.getInductionVar();
        if (offset) {
          auto delta = rewriter.create<mlir::arith::ConstantIndexOp>(operation->getLoc(), offset);
          index = rewriter.create<mlir::arith::AddIOp>(operation->getLoc(), index, delta);
        }
        mlir::scf::IfOp branch;
        if (offset) {
          auto valid = rewriter.create<mlir::arith::CmpIOp>(operation->getLoc(),
              mlir::arith::CmpIPredicate::ult, index, upper);
          branch = rewriter.create<mlir::scf::IfOp>(operation->getLoc(), mlir::TypeRange{resultType}, valid, true);
          copyIdentity(operation, branch);
          rewriter.setInsertionPointToStart(&branch.getThenRegion().front());
        }
        mlir::Value next;
        {
          mlir::Value operands[2];
          for (unsigned input = 0; input < 2; ++input) {
            auto supplied = supplyOperand(input, index);
            auto converted = mlir::succeeded(supplied) ? promote(*supplied) : mlir::FailureOr<mlir::Value>(mlir::failure());
            if (mlir::failed(converted)) {
              operation->emitError("contraction operand cannot be projected and converted as an explicit numeric SSA value")
                  << "; operand=" << input;
              invalid = true;
              break;
            }
            operands[input] = *converted;
          }
          if (invalid) break;
          auto update = rewriter.create<riscv::RVVContractStepOp>(operation->getLoc(), resultType,
              operands[0], operands[1], carried, mlir::Value(), axis, laneOperand.getValue(), stepLeaf);
          copyIdentity(operation, update);
          next = update;
        }
        if (branch) {
          rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), next);
          rewriter.setInsertionPointToStart(&branch.getElseRegion().front());
          rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), carried);
          rewriter.setInsertionPointAfter(branch);
          carried = branch.getResult(0);
        } else carried = next;
      }
      if (invalid) { failed = true; continue; }
      rewriter.create<mlir::scf::YieldOp>(operation->getLoc(), partials);
      rewriter.setInsertionPointAfter(loop);
      mlir::Value combined = loop.getResult(0);
      if (independent) {
        auto add = rewriter.create<riscv::BinaryOp>(operation->getLoc(), resultType,
            combined, loop.getResult(1), "add", riscv_internal::unselectedLeaf(rewriter));
        copyIdentity(operation, add);
        combined = add;
      }
      if (fusedAdd) {
        rewriter.replaceOp(fusedAdd, combined);
        rewriter.eraseOp(operation);
      } else rewriter.replaceOp(operation, combined);
    }
  }

};

class PrepareRISCVCompositesPass
    : public mlir::PassWrapper<PrepareRISCVCompositesPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override { return "weft-riscv-prepare-composites"; }
  llvm::StringRef getDescription() const override {
    return "Expose ordered loops, local storage and numerical frontiers before physical contraction selection";
  }
  void runOnOperation() override {
    if (mlir::failed(CompositeLowering(getOperation()).prepare())) signalPassFailure();
  }
};

class LowerRISCVCompositesPass
    : public mlir::PassWrapper<LowerRISCVCompositesPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit LowerRISCVCompositesPass(int64_t value = 8) { maximumLMUL = value; }
  LowerRISCVCompositesPass(const LowerRISCVCompositesPass &other) : PassWrapper(other) {
    copyOptionValuesFrom(&other);
  }
  llvm::StringRef getArgument() const override { return "weft-riscv-lower-composites"; }
  llvm::StringRef getDescription() const override {
    return "Materialize selected contractions and remaining typed local composites";
  }
  void runOnOperation() override {
    bool invalid = maximumLMUL < 1 || maximumLMUL > 64 || (maximumLMUL & (maximumLMUL - 1));
    getOperation().walk([&](riscv::KernelOp kernel) {
      invalid |= !llvm::is_contained(kernel.getTarget().getLegalLMULEighths().asArrayRef(), maximumLMUL.getValue());
    });
    if (invalid) {
      getOperation().emitError("composite lowering requires a legal target LMUL binding");
      signalPassFailure();
      return;
    }
    if (mlir::failed(CompositeLowering(getOperation()).finish(maximumLMUL))) signalPassFailure();
  }
private:
  Option<int64_t> maximumLMUL{*this, "lmul-eighths", llvm::cl::init(8),
      llvm::cl::desc("Instantiated maximum contraction LMUL in eighths")};
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createLowerRISCVCompositesPass(int64_t lmulEighths) {
  return std::make_unique<LowerRISCVCompositesPass>(lmulEighths);
}

std::unique_ptr<mlir::Pass> weft::createPrepareRISCVCompositesPass() {
  return std::make_unique<PrepareRISCVCompositesPass>();
}
