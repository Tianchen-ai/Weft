#ifndef WEFT_TARGET_PARTIAL_SUPPORT_H
#define WEFT_TARGET_PARTIAL_SUPPORT_H

#include "Weft/Target/RISCVPasses.h"

#include "../RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

namespace weft::riscv_partial {

void planPartialAddTrees(mlir::ModuleOp module, bool extractionsOnly);
bool materializePartialAddTrees(mlir::ModuleOp module,
                               mlir::IRRewriter &rewriter,
                               int64_t &nextPartialBirthId);

int64_t product(llvm::ArrayRef<int64_t> values);

std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values);

bool reachesReductionThroughPureOps(mlir::Value root);

bool checkedSubtract(int64_t lhs, int64_t rhs, int64_t &result);

bool checkedScale(int64_t value, int64_t factor, int64_t &result);

bool hasAxis(riscv::ValueType value, int64_t axis);

bool hasTopology(riscv::RVVWidenDotOp dot, llvm::StringRef kind);

int64_t ownerDomain(mlir::Operation *operation);

bool mayMoveReadAcross(mlir::Operation *operation);

void sinkReplicaSupplies(riscv::RVVPartialSetOp partial);

void placePureSliceAtFirstPostLoopUse(mlir::scf::ForOp loop,
                                      mlir::Value root);

std::optional<size_t> axisPosition(riscv::ValueType value, int64_t axis);

std::optional<int64_t>
laneProductForAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> axes);

std::optional<int64_t>
primaryReductionLaneAxis(riscv::ValueType partial,
                         llvm::ArrayRef<int64_t> reductionAxes);

riscv::ValueType projectOneWindow(mlir::Builder &builder, riscv::ValueType value,
                                  int64_t axis);

struct ProjectedRoot {
  mlir::Value value;
  riscv::FieldOp field;
  riscv::PhysicalPointOp origin;
  riscv::ValueType type;
  riscv::AccessAttr access;
  int64_t base = 0;
  int64_t stride = 1;
  int64_t repeat = 1;
  int64_t extent = 0;
  bool layered = false;
  riscv::LayeredStreamGeometryAttr geometry;
};

std::optional<ProjectedRoot> projectedRoot(mlir::Value value, int64_t axis);

void collectProjectedRoots(mlir::Value value, int64_t axis,
                           llvm::DenseSet<mlir::Operation *> &visited,
                           llvm::SmallVectorImpl<ProjectedRoot> &roots);

bool dependsOnAny(mlir::Value value,
                  const llvm::DenseMap<mlir::Value, mlir::Value> &replacements,
                  llvm::DenseMap<mlir::Value, bool> &cache);

bool canProjectWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::Builder &builder, llvm::DenseMap<mlir::Value, bool> &dependence,
    llvm::DenseMap<mlir::Value, bool> &cache);

mlir::FailureOr<mlir::Value> cloneWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::IRRewriter &rewriter, llvm::DenseMap<mlir::Value, mlir::Value> &clones,
    llvm::DenseMap<mlir::Value, bool> &dependence);

// Materialize one issue-time window of a shaped axis.  This differs from
// projectOneWindow: an upstream producer may carry a wider lane group than its
// downstream consumer and therefore needs to be sliced to the consumer's
// selected window extent, not merely to its own current lane factor.
riscv::ValueType issueWindowType(mlir::Builder &builder,
                                 riscv::TargetAttr target,
                                 riscv::ValueType source, int64_t axis,
                                 int64_t windowExtent);

struct IssueStorageWindowCandidate {
  riscv::FieldOp field;
  riscv::PhysicalPointOp point;
  riscv::ValueType resultType;
  riscv::StorageWindowPlanAttr plan;
  int64_t axis;
};

std::optional<IssueStorageWindowCandidate>
analyzeIssueStorageWindowCandidate(mlir::Builder &builder,
                                   riscv::ExtractOp extract);

struct IssueStorageSupplyFacts {
  bool closed = false;
  bool hasStorageWindow = false;
};

IssueStorageSupplyFacts analyzeIssueStorageSupply(
    mlir::Builder &builder, mlir::Value value, int64_t axis,
    int64_t windowExtent, riscv::TargetAttr target,
    llvm::DenseMap<mlir::Value, IssueStorageSupplyFacts> &memo);

bool supportsIssueStorageRematerialization(mlir::Builder &builder,
                                           mlir::Value value, int64_t axis,
                                           int64_t windowExtent,
                                           riscv::TargetAttr target);

mlir::FailureOr<mlir::Value> cloneIssueWindow(
    mlir::Value value, int64_t axis, int64_t windowExtent,
    mlir::Value windowIndex, riscv::TargetAttr target,
    mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &clones);

riscv::ValueType partialType(mlir::Builder &builder, riscv::ValueType lhs,
                             mlir::Type resultType, int64_t reductionAxis);

riscv::ValueType projectReplicaReductionOperandType(
    mlir::Builder &builder, riscv::ValueType input, int64_t removedAxis);

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    llvm::ArrayRef<int64_t> reductionAxes);

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    int64_t reductionAxis);

riscv::ValueType partialSlotType(mlir::Builder &builder, riscv::ValueType operand,
                                 int64_t reductionAxis);

riscv::ValueType partialSlotType(mlir::Builder &builder,
                                 riscv::RVVWidenDotOp dot);

riscv::ValueType issueSliceType(mlir::Builder &builder,
                                riscv::ValueType source,
                                llvm::ArrayRef<int64_t> reductionAxes);

riscv::ValueType sequentialIssueType(mlir::Builder &builder,
                                     riscv::RVVWidenDotOp dot,
                                     riscv::ValueType source);

riscv::ValueType groupedLanePartialSlotType(mlir::Builder &builder,
                                            riscv::TargetAttr target,
                                            riscv::ValueType lhs,
                                            riscv::ValueType rhs,
                                            int64_t reductionAxis,
                                            int64_t lanes);

riscv::ValueType splitPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source,
                                      int64_t reductionAxis, int64_t split);

std::optional<riscv::LeafAttr> partialRepackLeaf(
    mlir::Builder &builder, riscv::TargetAttr target,
    riscv::PartialSetType source, riscv::PartialSetType result, int64_t split);

riscv::ValueType widenPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source);

riscv::ValueType vectorPartialSlotType(
    mlir::Builder &builder, riscv::ValueType operand,
    riscv::ValueType result, int64_t reductionAxis);

std::optional<std::string>
partialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs);

std::optional<std::string>
sequentialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs);

riscv::ValueType reducedPartialSlotType(mlir::Builder &builder,
                                        riscv::ValueType partial,
                                        int64_t reductionAxis);

std::optional<llvm::StringRef>
partialFinalizeInstruction(riscv::PartialSetType input,
                           int64_t reductionAxis);

bool isIntegerZero(mlir::Value value);

struct PartialAddLeaf {
  riscv::RVVPartialFinalizeOp finalize;
  riscv::RVVWidenDotOp dot;
};

bool collectPartialAddTree(
    mlir::Value value, riscv::BinaryOp root,
    llvm::SmallVectorImpl<riscv::BinaryOp> &adds,
    llvm::SmallVectorImpl<PartialAddLeaf> &leaves);

std::optional<riscv::PartialSetType>
partialTypeForAddLeaf(mlir::Builder &builder, PartialAddLeaf leaf);

std::optional<int64_t> partialSplitFactor(riscv::PartialSetType source,
                                          riscv::PartialSetType target);

std::optional<int64_t> projectReplica(riscv::ValueType source,
                                      riscv::ValueType result,
                                      int64_t resultPart);

std::optional<int64_t> composeReplicaPart(
    riscv::ValueType value, llvm::ArrayRef<int64_t> outerAxes,
    int64_t outerPart, llvm::ArrayRef<int64_t> innerAxes, int64_t innerPart);

void eraseDeadChain(mlir::Value value,
                    const llvm::DenseSet<mlir::Value> &stops,
                    llvm::DenseSet<mlir::Operation *> &visited,
                    mlir::IRRewriter &rewriter);

bool isDeadChainCandidate(mlir::Operation *operation);

void collectDeadChainCandidates(
    mlir::Value value, const llvm::DenseSet<mlir::Value> &stops,
    llvm::DenseSet<mlir::Operation *> &candidates);

void sweepDeadChainCandidates(
    llvm::DenseSet<mlir::Operation *> &candidates,
    mlir::IRRewriter &rewriter);

riscv::ValueType scalarReplicaType(mlir::Builder &builder,
                                   riscv::ValueType source);

mlir::FailureOr<mlir::Value> rematerializeScalarReplicas(
    mlir::Value value, mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &memo);

bool supportsScalarReplicaRematerialization(
    mlir::Value value, llvm::DenseSet<mlir::Operation *> &visited);

std::optional<mlir::Value> narrowScaleSource(mlir::Value value);

riscv::ValueType plannedNarrowScaleType(mlir::Builder &builder,
                                        mlir::Value value,
                                        riscv::ValueType scalarScaleType);

std::optional<mlir::Value>
materializeExactNarrowScale(mlir::Value value, mlir::IRRewriter &rewriter);

struct ScaledPartialContribution {
  riscv::RVVWidenDotOp dot;
  riscv::BinaryOp scaleMultiply;
  riscv::BinaryOp carryAdd;
  mlir::Value scale;
  mlir::Value previous;
};

std::optional<ScaledPartialContribution>
matchScaledPartialContribution(mlir::Value value);

std::optional<llvm::SmallVector<ScaledPartialContribution>>
matchLevelScaledLoop(mlir::scf::ForOp loop);

std::optional<ScaledPartialContribution>
matchLevelScaledLoopSeed(mlir::scf::ForOp loop);

mlir::FailureOr<mlir::scf::ForOp>
expandPlannedLevelScaledIssueLoop(mlir::scf::ForOp loop, int64_t factor,
                                  mlir::IRRewriter &rewriter);

struct ReplicaScaledDotReduction {
  llvm::SmallVector<riscv::ReduceOp> reductions;
  llvm::SmallVector<riscv::ConvertLayoutOp> reductionConversions;
  llvm::SmallVector<int64_t> reducedAxes;
  riscv::BinaryOp scaleMultiply;
  riscv::RVVWidenDotOp dot;
  mlir::Value dotSide;
  mlir::Value scale;
};

std::optional<ReplicaScaledDotReduction>
matchReplicaScaledDotReduction(riscv::ReduceOp reduce);

int64_t replicaProductForAxes(riscv::ValueType value,
                              llvm::ArrayRef<int64_t> selectedAxes);

int64_t replicaProduct(riscv::ValueType value);

llvm::SmallVector<int64_t>
remainingAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> removed);

riscv::PartialTopologyAttr makePartialTopology(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot, llvm::StringRef kind,
    int64_t rootOperand, llvm::ArrayRef<int64_t> partialAxes,
    llvm::ArrayRef<int64_t> outputAxes, int64_t sourceSlots,
    int64_t partialSlots, int64_t outputReplicas, int64_t laneSplit,
    int64_t combineArity, int64_t resourceGroups);

struct LayeredTopologyFacts {
  int64_t reductionAxis = 0;
  int64_t rootOperand = -1;
  int64_t streams = 0;
  int64_t group = 0;
  int64_t layerExtent = 0;
  int64_t layers = 0;
  int64_t lanes = 0;
  int64_t windowsPerLayer = 0;
  int64_t windowCount = 0;
  riscv::ValueType lhsType;
  riscv::ValueType rhsType;
  ProjectedRoot layered;
  llvm::SmallVector<ProjectedRoot> roots;
  riscv::ValueType layeredWindowType;
  llvm::DenseMap<mlir::Value, riscv::ValueType> windowTypes;
  riscv::ValueType lhsIssueType;
  riscv::ValueType rhsIssueType;
  riscv::ValueType accumulatorType;
  mlir::IntegerType partialInteger;
};

std::optional<LayeredTopologyFacts>
analyzeLayeredTopology(mlir::Builder &builder, riscv::RVVWidenDotOp dot);

struct ScaledTopologyFacts {
  llvm::SmallVector<int64_t> partialAxes;
  llvm::SmallVector<int64_t> outputAxes;
  int64_t partialSlots = 0;
  int64_t outputReplicas = 0;
};

struct ScaledSourceLaneGeometry {
  riscv::ValueType sourceSlot;
  riscv::ValueType splitSourceSlot;
  riscv::ValueType partialSlot;
  riscv::ValueType reducedSlot;
  int64_t sourceSetGroups = 0;
  int64_t reducedGroups = 0;
};

struct SourceLanePlan {
  int64_t sourceSlots = 0;
  int64_t laneSplit = 1;
  riscv::ValueType lhsType;
  riscv::ValueType rhsType;
  ScaledSourceLaneGeometry geometry;
  llvm::SmallVector<int64_t> lhsParts;
  llvm::SmallVector<int64_t> rhsParts;
  llvm::SmallVector<int64_t> lhsOffsets;
  llvm::SmallVector<int64_t> rhsOffsets;
};

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis,
                                llvm::DenseSet<mlir::Operation *> &visited);

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis);

bool hasIndexedLookupSupply(mlir::Value value,
                            llvm::DenseSet<mlir::Operation *> &visited);

bool hasIndexedLookupSupply(mlir::Value value);

void collectConcreteFieldSupplies(
    mlir::Value value, llvm::SmallPtrSetImpl<mlir::Operation *> &supplies,
    llvm::SmallPtrSetImpl<mlir::Operation *> &visited);

bool sharesConcreteFieldSupply(mlir::Value lhs, mlir::Value rhs);

std::optional<riscv::ValueType>
coalescePartialOperand(mlir::Builder &builder, riscv::ValueType source,
                       llvm::ArrayRef<int64_t> partialAxes,
                       int64_t requiredLanes, riscv::TargetAttr target);

std::optional<ScaledSourceLaneGeometry>
deriveScaledSourceLaneGeometry(mlir::Builder &builder,
                               riscv::RVVWidenDotOp dot,
                               riscv::ValueType lhsSource,
                               riscv::ValueType rhsSource,
                               int64_t partialSlots, int64_t sourceSlots,
                               int64_t laneSplit);

std::optional<riscv::NestedPartialPlanAttr> planNestedPartialCarrier(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot,
    const ReplicaScaledDotReduction &match, int64_t windowAxis,
    int64_t issueStreams, int64_t windowExtent);

std::optional<SourceLanePlan>
planScaledSourceLanes(mlir::Builder &builder, riscv::RVVWidenDotOp dot,
                      const ScaledTopologyFacts &facts);

} // namespace weft::riscv_partial

#endif
