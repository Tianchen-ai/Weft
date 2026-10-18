#include "DenseProducerGraph.h"
#include "LocalMemory.h"
#include "RISCVPhysicalSupport.h"
#include "ValueLiveness.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"
#include "Weft/Dialect/RISCV/IR/DenseRelations.h"
#include "mlir/IR/Dominance.h"
#include <functional>
#include <limits>

using namespace mlir;
using namespace weft;
using namespace weft::riscv_internal;

bool riscv_internal::isDensePointwise(Operation *op) {
  return isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
             riscv::SelectOp, riscv::CastOp, riscv::BitcastOp,
             riscv::WidenOp, riscv::NarrowOp, riscv::RVVSplatOp>(op);
}

DenseProducerGraph::DenseProducerGraph(ArrayRef<Operation *> sinks)
    : root(sinks.back()), sinks(sinks) {
  for (Operation *sink : sinks) {
    if (auto store = dyn_cast<riscv::StoreOp>(sink)) {
      stores.push_back(store);
      values.push_back(store.getValue());
    } else if (auto reduce = dyn_cast<riscv::ReduceOp>(sink)) {
      reductions.push_back(reduce);
      values.push_back(reduce.getInput());
    } else {
      assert(isDensePointwise(sink) || isa<riscv::LookupOp>(sink));
      values.push_back(sink->getResult(0));
    }
  }
  result = values.front();
  domain = dyn_cast<riscv::ValueType>(result.getType());
  if (stores.size() == sinks.size())
    for (Value value : values) {
      auto type = dyn_cast<riscv::ValueType>(value.getType());
      if (type && domain && logicalBitWidth(type) > logicalBitWidth(domain)) domain = type;
    }
  if (domain) points.resize(domain.getShape().size());
}

bool DenseProducerGraph::ownsConsumer(Operation *op) const {
  return llvm::is_contained(sinks, op);
}

bool DenseProducerGraph::projects(riscv::ValueType type) const {
  for (auto [extent, axis] : llvm::zip(type.getShape().asArrayRef(), type.getAxisIds().asArrayRef())) {
    auto found = llvm::find(domain.getAxisIds().asArrayRef(), axis);
    if (found == domain.getAxisIds().asArrayRef().end()) return false;
    if (extent != 1 && extent != domain.getShape()[found - domain.getAxisIds().asArrayRef().begin()]) return false;
  }
  return true;
}

bool DenseProducerGraph::collect(Value value) {
  auto type = dyn_cast<riscv::ValueType>(value.getType());
  if (!type) return isa<IntegerType, FloatType, IndexType>(value.getType());
  if (!projects(type) || !isa<IntegerType, FloatType>(type.getElementType())) return false;
  // Once an observed value is an input, another path through a shared DAG
  // cannot turn that same SSA into a producer owned and erased by this graph.
  if (llvm::is_contained(inputs, value)) return true;
  if (type.getLayout().getCarrier() == "scalar" &&
      llvm::all_of(type.getShape().asArrayRef(), [](int64_t n) { return n == 1; }) &&
      (type.getShape() != domain.getShape() || type.getAxisIds() != domain.getAxisIds()))
    return boundary(value);
  if (type.getLayout().getCarrier() == "local") return boundary(value);
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != root->getBlock() || op->getNumResults() != 1 ||
      op->getNumRegions()) return boundary(value);
  if (members.size() >= 128) return false;
  if (members.contains(op)) return true;
  // A producer already observed before this sink must remain at its original
  // position. Supply its completed value instead of attempting to move its
  // entire graph past that observer. Reads retain their separate snapshot
  // handling below; only pure numerical producers use this boundary.
  auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op);
  if ((isDensePointwise(op) || isa<riscv::IotaOp>(op) ||
       (conversion && conversion.getConversion().getEffect() == "pure")) &&
      llvm::any_of(value.getUsers(), [&](Operation *user) {
        if (ownsConsumer(user) || members.contains(user)) return false;
        auto *observer = root->getBlock()->findAncestorOpInBlock(*user);
        return !observer || (observer != root && !root->isBeforeInBlock(observer));
      }))
    return boundary(value);
  members.insert(op);
  if (auto load = dyn_cast<riscv::LoadOp>(op)) {
    auto memory = load.getRegion().getType();
    auto encoding = dyn_cast<kernel::EncodingType>(memory.getEncoding());
    bool completePrivate = bool(load.getRegion().getDefiningOp<riscv::LocalViewOp>());
    if (!encoding || encoding.getKind() != "dense" || load.getSnapshotStorage() ||
        !riscv::isAddressableDenseDescriptor(load.getRegion()) ||
        memory.getShape() != type.getShape() || memory.getAxisIds() != type.getAxisIds() ||
        (logicalBitWidth(type) < 8 && !(type.getElementType().isInteger(1) && memory.getStorageBits() == 8))) return false;
    for (auto [position, axis] : llvm::enumerate(type.getAxisIds().asArrayRef())) {
      if (type.getShape()[position] == 1) continue;
      auto found = llvm::find(domain.getAxisIds().asArrayRef(), axis);
      size_t output = found - domain.getAxisIds().asArrayRef().begin();
      auto origin = originPoint(load.getRegion(), axis);
      // LocalView is the complete private array, not an external tail window.
      // A bounded consumer may project its active Level onto this snapshot;
      // the shared domain/point check below still closes that active range.
      if (type.getLayout().getValidity() == "tail" && !origin &&
          type.getShape()[position] > 0 && !completePrivate) return false;
      if (origin && points[output] && origin != points[output]) return false;
      if (origin) points[output] = origin;
    }
    reads.push_back(load);
  } else if (auto lookup = dyn_cast<riscv::LookupOp>(op); lookup && op == root) {
    // A large memory lookup reads its table at the original lookup boundary.
    // Only the index/value window is streamed; the table remains one address
    // operand and is never cloned or admitted as a full register value.
    if (!isa<riscv::MemDescType>(lookup.getTable().getType()) ||
        !riscv::isAddressableDenseDescriptor(lookup.getTable()) ||
        !collect(lookup.getIndices())) return false;
  } else if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op)) {
    if (isCompleteLocalRead(conversion)) localReads.push_back(conversion);
    else if (conversion.getConversion().getEffect() != "pure" || !collect(conversion.getInput())) return false;
  } else if (auto broadcast = dyn_cast<riscv::RVVAxisBroadcastOp>(op)) {
    // An internal broadcast is a coordinate embedding. Reproject its smaller
    // source with this issue instead of requiring the full expanded value to
    // reside in registers before the bounded pointwise consumer.
    if (!value.hasOneUse() || !members.contains(*value.getUsers().begin()) ||
        !collect(broadcast.getInput())) return false;
  } else if (isa<riscv::IotaOp>(op)) {
    if (type.getShape().size() != 1) return false;
  } else if (isDensePointwise(op)) {
    for (Value operand : op->getOperands())
      if (!collect(operand)) return false;
  } else {
    members.erase(op);
    return boundary(value);
  }
  order.push_back(op);
  return true;
}

bool DenseProducerGraph::boundary(Value value) {
  auto type = cast<riscv::ValueType>(value.getType());
  auto read = value.getDefiningOp<riscv::ConvertLayoutOp>();
  bool stored = read && isCompleteLocalRead(read) && !riscv::readCrossesWrite(read, root);
  if (type.getLayout().getCarrier() != "local" && !stored) {
    auto resident = residentRegisterGroups(value);
    auto target = root->getParentOfType<riscv::KernelOp>().getTarget();
    if (!resident || *resident > target.getVectorRegisters() ||
        llvm::any_of(type.getShape().asArrayRef(), [](int64_t n) { return n < 0; })) return false;
  }
  if (!llvm::is_contained(inputs, value)) inputs.push_back(value);
  return true;
}

bool DenseProducerGraph::collectConsumers() {
  // A scalar result observed before the joint boundary is a real dependence,
  // such as mean -> centered -> variance. It cannot join one streaming issue.
  for (auto reduce : reductions)
    for (Operation *user : reduce.getResult().getUsers()) {
      auto owner = root->getBlock()->findAncestorOpInBlock(*user);
      if (!owner || owner == root || !root->isBeforeInBlock(owner)) return false;
    }
  for (Operation *op : order) {
    bool external = false;
    for (Operation *user : op->getResult(0).getUsers()) {
      if (ownsConsumer(user) || members.contains(user)) continue;
      if (isa<riscv::LoadOp>(op) ||
          llvm::is_contained(localReads, dyn_cast<riscv::ConvertLayoutOp>(op))) continue;
      auto *observer = root->getBlock()->findAncestorOpInBlock(*user);
      if (!observer || observer == root || !root->isBeforeInBlock(observer)) return false;
      external = true;
    }
    if (external) {
      auto type = cast<riscv::ValueType>(op->getResult(0).getType());
      if (!isa<FloatType, IntegerType>(type.getElementType())) return false;
      exports.push_back(op->getResult(0));
    }
  }
  return true;
}

bool DenseProducerGraph::canDelayStores() {
  for (auto [position, store] : llvm::enumerate(stores)) {
    for (auto other : ArrayRef<riscv::StoreOp>(stores).drop_front(position + 1))
      if (!requireDisjoint(store.getRegion(), other.getRegion())) return false;
    if (store == root) continue;
    unsigned distance = 0;
    for (Operation *next = store->getNextNode(); next && next != root; next = next->getNextNode()) {
      if (++distance > 512 || next->getNumRegions()) return false;
      if (ownsConsumer(next)) continue;
      if (!riscv::hasOnlyReadEffects(next)) return false;
      auto effects = dyn_cast<MemoryEffectOpInterface>(next);
      if (!effects) {
        if (!isMemoryEffectFree(next)) return false;
        continue;
      }
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (auto effect : instances) {
        if (!isa<MemoryEffects::Read>(effect.getEffect()) || !effect.getValue() ||
            !requireDisjoint(store.getRegion(), effect.getValue())) return false;
      }
    }
  }
  return true;
}

bool DenseProducerGraph::requireDisjoint(Value lhs, Value rhs) {
  if (!riscv::mayAliasStorageAt(lhs, rhs, root)) return true;
  if (!allowAliasGuards) return false;
  Value a = riscv::fullDenseArgumentView(lhs), b = riscv::fullDenseArgumentView(rhs);
  if (!a || !b) return false;
  auto relation = riscv::denseProjectionRelation(lhs, rhs);
  bool distinct = relation && relation->disjoint;
  if (a == b && !distinct) return false;
  // This is an obligation, not a fact in the input program. The caller must
  // materialize the proof and re-analyze the selected true arm before rewriting.
  aliasRequirements.push_back({a, b, distinct});
  return true;
}

bool DenseProducerGraph::canStreamRead(riscv::LoadOp read) const {
  if (read->getBlock() != root->getBlock()) return false;
  unsigned distance = 0;
  for (Operation *next = read->getNextNode(); next != root; next = next->getNextNode()) {
    if (!next || ++distance > 512) return false;
    if (auto store = dyn_cast<riscv::StoreOp>(next); store && ownsConsumer(next)) {
      // All loads for a point are completed before its output stores. A later
      // point cannot observe a prior point's writes under this mapping proof.
      if (!riscv::densePointwiseIndependentAt(read.getRegion(), store.getRegion(), root)) return false;
    } else if (!riscv::hasOnlyReadEffects(next, true)) return false;
  }
  return true;
}

bool DenseProducerGraph::preservesOrder() const {
  for (Operation *op : order) {
    if (op == root) continue;
    // Numeric SSA operands are immutable snapshots. Moving a pure consumer to
    // its bounded issue does not move the reads that produced those operands.
    // In particular, intervening materialization of an unrelated predicate is
    // not a dependence of the numerical producer. Read nodes below still keep
    // their own storage-order proof.
    if (isDensePointwise(op) || isa<riscv::IotaOp, riscv::RVVAxisBroadcastOp>(op)) continue;
    if (auto conversion = dyn_cast<riscv::ConvertLayoutOp>(op);
        conversion && conversion.getConversion().getEffect() == "pure") continue;
    for (Operation *next = op->getNextNode(); next != root; next = next->getNextNode()) {
      // All nodes are in one finite block. Earlier materializers may insert
      // many address/extent operations; their count is not a storage effect.
      if (!next) return false;
      if (ownsConsumer(next)) continue;
      // A new private object does not modify an existing numerical source.
      // Storage for this very sink may be declared between its producers and
      // its store; that declaration must not force a whole register tuple.
      if (isa<riscv::LocalAllocOp>(next)) continue;
      if (llvm::is_contained(localReads, dyn_cast<riscv::ConvertLayoutOp>(op))) {
        if (riscv::mayWriteStorage(next, cast<riscv::ConvertLayoutOp>(op).getInput())) return false;
      } else if (!riscv::hasOnlyReadEffects(next, true)) return false;
    }
  }
  return true;
}

bool DenseProducerGraph::chooseIssue() {
  auto kernel = root->getParentOfType<riscv::KernelOp>();
  auto target = kernel.getTarget();
  Builder b(root->getContext());
  int64_t widest = 8;
  SmallVector<Value> excluded;
  for (Operation *op : order) {
    widest = std::max<int64_t>(widest, logicalBitWidth(op->getResult(0).getType()));
    excluded.push_back(op->getResult(0));
  }
  for (Value input : inputs)
    widest = std::max<int64_t>(widest, logicalBitWidth(input.getType()));
  int64_t maximum = domain.getLayout().getLaneFactors()[lanePosition];
  if (maximum <= 1) maximum = target.getVlenBits() / widest;
  maximum = std::min(maximum, target.getVlenBits() * 8 / widest);
  auto surrounding = surroundingRegisterGroups(root, excluded);
  if (!surrounding) return false;
  bool predicates = domain.getElementType().isInteger(1) ||
      llvm::any_of(inputs, [](Value value) { return logicalElement(value.getType()).isInteger(1); });
  if (predicates) maximum = 1;

  // Only an already declared active domain can provide validity for replicated
  // rows. A runtime ABI dimension without a physical point remains scalar in
  // this traversal; no source Level or logical cohort is invented here.
  int64_t freePosition = -1;
  for (int64_t position = int64_t(domain.getShape().size()) - 1; position >= 0; --position)
    if (position != lanePosition && points[position] && domain.getShape()[position] > 1) {
      freePosition = position;
      break;
    }
  std::optional<DenseIssueCost> selected;
  int64_t bestOverlap = std::numeric_limits<int64_t>::max();
  int64_t bestWork = 0, bestElements = 1;
  const int64_t maximumCopies = reductions.empty() ? kernel.getPointwiseIssueCopies() : 1;
  for (int64_t rows : {1, 2, 4}) {
    if (rows > 1 && (freePosition < 0 || domain.getShape()[freePosition] < rows)) continue;
    for (int64_t copies : {1, 2, 4}) {
      if (copies > maximumCopies || order.size() * copies > 384) continue;
      for (int64_t candidate = maximum; candidate >= 1; candidate /= 2) {
        if (candidate == 1 && !reductions.empty()) continue;
        int64_t extent = domain.getShape()[lanePosition];
        int64_t span = candidate * copies;
        if (extent > 0 && span > extent) continue;
        SmallVector<int64_t> tile(domain.getShape().size(), 1);
        tile[lanePosition] = span;
        if (freePosition >= 0) tile[freePosition] = rows;
        auto pieceOf = [&](riscv::ValueType type) {
          return valueTileType(b, type, domain.getAxisIds().asArrayRef(), tile,
              domain.getAxisIds()[lanePosition], target, true, candidate);
        };
        auto numeric = pieceOf(domain);
        if (!numeric) continue;
        int64_t accumulator = numeric.getLayout().getRegisterGroups() * reductions.size();
        bool legal = true;
        for (Value input : inputs) {
          auto supply = pieceOf(cast<riscv::ValueType>(input.getType()));
          if (!supply) { legal = false; break; }
          accumulator += supply.getLayout().getRegisterGroups() * *staticProduct(supply.getLayout().getTimeFactors().asArrayRef());
        }
        if (!legal) continue;
        auto cost = denseIssueCost(b, order, values, exports, domain, tile,
            domain.getAxisIds()[lanePosition], target, accumulator, candidate);
        if (!cost || cost->vectorGroups > target.getVectorRegisters()) continue;
        int64_t overlap = std::max<int64_t>(0, cost->vectorGroups + *surrounding - target.getVectorRegisters());
        int64_t work = cost->operations + 4 + rows * reductions.size();
        int64_t elements = span * rows;
        // Count padded free-axis iterations rather than rewarding unused slots.
        if (freePosition >= 0) {
          int64_t count = domain.getShape()[freePosition];
          work *= (count + rows - 1) / rows;
          elements = span * count;
        }
        if (extent > 0) {
          work *= (extent + span - 1) / span;
          elements = (elements / span) * extent;
        }
        bool better = !selected || overlap < bestOverlap ||
            (overlap == bestOverlap && (__int128(work) * bestElements < __int128(bestWork) * elements ||
             (__int128(work) * bestElements == __int128(bestWork) * elements && span > width)));
        if (better) {
          selected = cost;
          bestOverlap = overlap;
          bestWork = work;
          bestElements = elements;
          width = span;
          laneWidth = candidate;
          issueCopies = copies;
          tileShape = std::move(tile);
        }
      }
    }
  }
  if (!selected) return false;
  issueCost = *selected;
  return true;
}

bool DenseProducerGraph::analyze(bool required, bool allowGuards) {
  allowAliasGuards = allowGuards;
  failureReason = "unsupported numeric domain";
  if (!domain || domain.getShape().empty() || domain.getShape().size() > 8 ||
      llvm::any_of(domain.getShape().asArrayRef(), [](int64_t n) { return n == 0; }) ||
      (domain.getLayout().getCarrier() != "rvv" && domain.getLayout().getCarrier() != "scalar" &&
       domain.getLayout().getCarrier() != "local") ||
      !isa<FloatType, IntegerType>(domain.getElementType())) return false;
  for (auto [position, value] : llvm::enumerate(values)) {
    failureReason = "consumer domains disagree";
    auto type = dyn_cast<riscv::ValueType>(value.getType());
    if (!type || type.getShape() != domain.getShape() || type.getAxisIds() != domain.getAxisIds() ||
        (stores.size() != sinks.size() && type.getElementType() != domain.getElementType()) ||
        sinks[position]->getBlock() != root->getBlock()) return false;
    // A dense element type does not make an encoded field an affine address.
    // Only an explicit memory view may be split into ordinary dense subviews.
    if (auto store = dyn_cast<riscv::StoreOp>(sinks[position]); store &&
        (!riscv::isAddressableDenseDescriptor(store.getRegion()) ||
         store.getRegion().getType().getShape() != type.getShape() ||
         store.getRegion().getType().getAxisIds() != type.getAxisIds())) return false;
    auto previous = members;
    failureReason = "producer has no bounded representation";
    if (!collect(value)) return false;
    // A common producer is required. Merely adjacent independent kernels do
    // not justify joining their lifetimes or increasing one issue's live set.
    if (position && !llvm::any_of(order, [&](Operation *op) {
      return previous.contains(op) && llvm::any_of(op->getResult(0).getUsers(), [&](Operation *user) {
        return user == sinks[position] || (!previous.contains(user) && members.contains(user));
      });
    })) return false;
  }
  if (reads.empty() && localReads.empty() && inputs.empty() && !required) return false;
  // Preserve the original topological order rather than the order in which a
  // DFS first reached shared producers from its different output roots.
  llvm::stable_sort(order, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  failureReason = "producer has an earlier external consumer";
  if (!collectConsumers()) return false;
  // A full private value can be observed through a shorter destination View.
  // In a store-only graph with no numerical exports, that View bounds the
  // physical work. Its nominal type shape is the author partition capacity,
  // not permission to overwrite the inactive tail of the destination.
  if (reductions.empty() && exports.empty())
    for (auto store : stores)
      for (auto [position, axis] : llvm::enumerate(domain.getAxisIds().asArrayRef())) {
        auto origin = originPoint(store.getRegion(), axis);
        if (!origin) continue;
        if (points[position] && points[position] != origin) return false;
        points[position] = origin;
      }
  failureReason = "active domain has no dominating extent";
  for (auto [position, axis] : llvm::enumerate(domain.getAxisIds().asArrayRef())) {
    if (points[position] || domain.getShape()[position] <= 1 || domain.getLayout().getValidity() != "tail") continue;
    for (Operation *scope = root; scope && scope->getBlock() && !points[position];
         scope = scope->getParentOp())
      for (Operation *previous = scope->getPrevNode(); previous; previous = previous->getPrevNode())
        if (auto origin = dyn_cast<riscv::PhysicalPointOp>(previous);
            origin && origin.getType().getDomain().getAxisId() == axis) {
          points[position] = origin;
          break;
        }
    if (!points[position]) {
      // Tail validity applies to the whole layout. A separately introduced
      // fixed index axis can still be complete, even when a row axis is short.
      // Prove that fact through the typed suppliers instead of fabricating a
      // Level point for the independent index domain.
      llvm::DenseSet<Value> visiting;
      std::function<bool(Value)> complete = [&](Value value) {
        auto type = dyn_cast<riscv::ValueType>(value.getType());
        if (!type) return false;
        auto found = llvm::find(type.getAxisIds().asArrayRef(), axis);
        if (found == type.getAxisIds().asArrayRef().end() ||
            type.getShape()[found - type.getAxisIds().asArrayRef().begin()] != domain.getShape()[position])
          return false;
        if (type.getLayout().getValidity() == "full") return true;
        Operation *producer = value.getDefiningOp();
        if (auto broadcast = dyn_cast_or_null<riscv::RVVAxisBroadcastOp>(producer))
          if (!llvm::is_contained(broadcast.getInput().getType().getAxisIds().asArrayRef(), axis))
            return true;
        if (auto iota = dyn_cast_or_null<riscv::IotaOp>(producer))
          return iota.getEnd() - iota.getStart() == domain.getShape()[position];
        if (auto window = dyn_cast_or_null<riscv::RVVUnitEntryWindowLoadOp>(producer))
          return axis == static_cast<int64_t>(window.getPayloadAxis()) ||
                 llvm::is_contained(window.getEntryAxes(), axis);
        if (auto window = dyn_cast_or_null<riscv::RVVBitmaskWindowLoadOp>(producer))
          if (llvm::is_contained(window.getWindowAxes(), axis)) return true;
        if (auto entry = dyn_cast_or_null<riscv::RVVIndexedEntryLoadOp>(producer);
            entry && axis == static_cast<int64_t>(entry.getPayloadAxis())) return true;
        if (!producer || visiting.size() >= 128 || !visiting.insert(value).second ||
            !(isDensePointwise(producer) || isa<riscv::ConvertLayoutOp, riscv::RVVAxisBroadcastOp,
                 riscv::ExtractOp, riscv::LookupOp, riscv::RVVIndexedEntryLoadOp,
                 riscv::RegisterMaterializeOp, riscv::RVVWidenAddOp,
                 riscv::RVVWidenDotOp, riscv::ReduceOp>(producer))) return false;
        bool supplied = false, full = true;
        for (Value operand : producer->getOperands()) {
          auto input = dyn_cast<riscv::ValueType>(operand.getType());
          if (!input || !llvm::is_contained(input.getAxisIds().asArrayRef(), axis)) continue;
          supplied = true;
          full &= complete(operand);
        }
        visiting.erase(value);
        return supplied && full;
      };
      if (!complete(result)) return false;
    }
  }
  lanePosition = domain.getShape().size() - 1;
  for (auto [position, lanes] : llvm::enumerate(domain.getLayout().getLaneFactors().asArrayRef()))
    if (lanes > 1) lanePosition = position;
  if (!reductions.empty()) {
    failureReason = "reduction has no compatible issue axis";
    if (!domain.getElementType().isF32() && !domain.getElementType().isF64() &&
        !(domain.getElementType().isInteger(32) &&
          llvm::all_of(reductions, [](riscv::ReduceOp reduce) { return reduce.getKind() == "add"; }))) return false;
    lanePosition = reductions.front().getAxis();
    if (domain.getShape()[lanePosition] == 1 || llvm::any_of(reductions, [&](riscv::ReduceOp reduce) {
          return reduce.getAxis() != lanePosition || reduce.getResult().getType() != reductions.front().getResult().getType();
        })) return false;
  }
  point = points[lanePosition];
  failureReason = "store motion crosses an observable effect";
  if (!canDelayStores()) return false;
  failureReason = "producer motion crosses a storage write";
  if (!preservesOrder()) return false;
  failureReason = "no issue satisfies operand and resource contracts";
  if (!chooseIssue()) return false;
  failureReason = {};
  return required || needsRealization();
}

bool DenseProducerGraph::needsRealization() const {
  auto count = staticProduct(domain.getShape().asArrayRef());
  bool dynamic = llvm::any_of(domain.getShape().asArrayRef(), [](int64_t extent) { return extent < 0; });
  if ((!count && !dynamic) || (count && *count <= width)) return false;
  // A condition feeding several output stores has a long live range in the
  // whole-Value schedule. Keep its comparisons, mask logic and consumers in
  // the same bounded issue instead of retaining every predicate/data part
  // until all logical coordinates have been computed. Early byte layouts can
  // understate the pressure of the final per-data-part mask representation.
  if (stores.size() > 1 && llvm::any_of(order, [](Operation *op) {
        auto type = dyn_cast<riscv::ValueType>(op->getResult(0).getType());
        return type && type.getElementType().isInteger(1) &&
            (isa<riscv::CompareOp, riscv::BinaryOp>(op));
      })) return true;
  // A bounded integer reduction can keep its producer/scale graph in registers
  // for the partial-topology pass. Streaming a small, already legal graph here
  // would materialize the partials and sever that use-def relation.
  bool registerIntegerReduction = !reductions.empty() && domain.getElementType().isInteger(32) &&
      domain.getLayout().getCarrier() != "local";
  if (!dynamic && ((reductions.empty() && domain.getLayout().getCarrier() == "rvv") ||
                   registerIntegerReduction)) {
    bool scalarMath = llvm::any_of(order, [](Operation *op) {
      auto unary = dyn_cast<riscv::UnaryOp>(op);
      return unary && (unary.getKind() == "exp" || unary.getKind() == "exp2" || unary.getKind() == "log" ||
                      unary.getKind() == "sin" || unary.getKind() == "cos" || unary.getKind() == "tanh");
    });
    auto liveness = BlockRegisterLiveness::get(*root->getBlock());
    if (!liveness) return false;
    unsigned first = *liveness->position(root);
    for (Operation *op : order) first = std::min(first, *liveness->position(op));
    auto peak = liveness->peak(first, *liveness->position(root));
    if (!peak) return false;
    if (!scalarMath && localReads.empty() && exports.empty() &&
        peak->total() <= root->getParentOfType<riscv::KernelOp>().getTarget().getVectorRegisters())
      return false;
  }
  return true;
}
