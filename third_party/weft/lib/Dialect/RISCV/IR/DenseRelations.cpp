#include "Weft/Dialect/RISCV/IR/DenseRelations.h"
#include "Weft/Dialect/RISCV/IR/DenseBounds.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
using namespace weft::riscv;

namespace {

bool sameIndex(Value lhs, Value rhs) {
  return lhs == rhs || sameDenseIndex(lhs, rhs);
}

bool samePoint(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto a = lhs.getDefiningOp<PhysicalPointOp>();
  auto b = rhs.getDefiningOp<PhysicalPointOp>();
  return a && b && a.getType() == b.getType() && a.getParent() == b.getParent() &&
      sameIndex(a.getBase(), b.getBase()) && sameIndex(a.getActive(), b.getActive()) &&
      sameIndex(a.getPartition(), b.getPartition());
}

bool sameBound(OpFoldResult lhs, OpFoldResult rhs) {
  if (lhs == rhs) return true;
  auto a = getConstantIntValue(lhs), b = getConstantIntValue(rhs);
  if (a && b) return *a == *b;
  auto x = dyn_cast<Value>(lhs), y = dyn_cast<Value>(rhs);
  return x && y && sameIndex(x, y);
}

SmallVector<OpFoldResult> mixedBounds(ArrayRef<int64_t> values,
                                    ValueRange dynamic, unsigned &cursor,
                                    MLIRContext *context) {
  SmallVector<OpFoldResult> result;
  Builder builder(context);
  for (int64_t value : values) {
    if (value == -1) {
      if (cursor >= dynamic.size()) return {};
      result.push_back(dynamic[cursor++]);
    } else result.push_back(builder.getIndexAttr(value));
  }
  return result;
}

bool sameSubviewWindow(SubviewOp a, SubviewOp b) {
  if (a.getOffsets().size() != b.getOffsets().size() ||
      a.getResult().getType().getAxisIds() != b.getResult().getType().getAxisIds()) return false;
  unsigned x = 0, y = 0;
  auto aOffsets = mixedBounds(a.getOffsets(), a.getDynamicBounds(), x, a.getContext());
  auto aExtents = mixedBounds(a.getExtents(), a.getDynamicBounds(), x, a.getContext());
  auto bOffsets = mixedBounds(b.getOffsets(), b.getDynamicBounds(), y, b.getContext());
  auto bExtents = mixedBounds(b.getExtents(), b.getDynamicBounds(), y, b.getContext());
  if (aOffsets.size() != bOffsets.size() || aExtents.size() != bExtents.size() ||
      aOffsets.size() != aExtents.size()) return false;
  for (unsigned i = 0; i < aOffsets.size(); ++i)
    if (!sameBound(aOffsets[i], bOffsets[i]) || !sameBound(aExtents[i], bExtents[i])) return false;
  return true;
}

template <typename Predicate>
bool impliedAt(Operation *at, Predicate predicate) {
  if (!at) return false;
  auto implies = [&](auto &&self, Value condition, unsigned &budget) -> bool {
    if (!budget) return false;
    --budget;
    if (predicate(condition.getDefiningOp())) return true;
    if (auto both = condition.getDefiningOp<arith::AndIOp>())
      return self(self, both.getLhs(), budget) || self(self, both.getRhs(), budget);
    if (auto either = condition.getDefiningOp<arith::OrIOp>())
      return self(self, either.getLhs(), budget) && self(self, either.getRhs(), budget);
    return false;
  };
  for (Operation *child = at, *parent = child->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(parent);
    if (!branch || child->getParentRegion() != &branch.getThenRegion()) continue;
    unsigned budget = 64;
    if (implies(implies, branch.getCondition(), budget)) return true;
  }
  return false;
}

bool staticallyInjective(MemoryViewOp view) {
  struct Dimension { uint64_t stride, extent; };
  SmallVector<Dimension> dimensions;
  for (auto [size, step] : llvm::zip(view.getExtents(), view.getStrides())) {
    auto extent = denseConstantIndex(size), stride = denseConstantIndex(step);
    if (!extent || *extent < 0) return false;
    if (*extent == 0) return true;
    if (*extent == 1) continue;
    if (!stride || *stride <= 0) return false;
    dimensions.push_back({uint64_t(*stride), uint64_t(*extent)});
  }
  llvm::sort(dimensions, [](auto a, auto b) { return a.stride < b.stride; });
  uint64_t span = 1;
  for (auto dimension : dimensions) {
    if (dimension.stride < span || dimension.extent - 1 >
        (std::numeric_limits<uint64_t>::max() - span) / dimension.stride) return false;
    span += (dimension.extent - 1) * dimension.stride;
  }
  return true;
}
} // namespace

std::optional<DenseProjectionRelation> weft::riscv::denseProjectionRelation(
    Value lhs, Value rhs) {
  auto aType = dyn_cast<MemDescType>(lhs.getType());
  auto bType = dyn_cast<MemDescType>(rhs.getType());
  if (!aType || !bType || aType.getEncoding() != bType.getEncoding() ||
      aType.getShape() != bType.getShape() || aType.getAxisIds() != bType.getAxisIds() ||
      !isAddressableDenseDescriptor(lhs) || !isAddressableDenseDescriptor(rhs)) return std::nullopt;
  DenseProjectionRelation relation;
  relation.identical = relation.pointwise = true;
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (auto a = lhs.getDefiningOp<ReaxisOp>()) { lhs = a.getBase(); continue; }
    if (auto b = rhs.getDefiningOp<ReaxisOp>()) { rhs = b.getBase(); continue; }
    if (auto a = lhs.getDefiningOp<SliceOp>()) {
      auto b = rhs.getDefiningOp<SliceOp>();
      if (!b || a.getSelectors() != b.getSelectors() ||
          a.getIndices().size() != b.getIndices().size()) return std::nullopt;
      unsigned cursor = 0;
      for (Attribute attribute : a.getSelectors()) {
        StringRef kind = cast<StringAttr>(attribute).getValue();
        if (kind == "all") continue;
        Value x = a.getIndices()[cursor], y = b.getIndices()[cursor];
        ++cursor;
        if (kind == "domain") {
          if (!samePoint(x, y)) return std::nullopt;
        } else if (kind == "index") {
          bool same = sameIndex(x, y);
          relation.identical &= same;
          auto first = denseConstantIndex(x), second = denseConstantIndex(y);
          // Distinct fixed fields cannot overlap in an injective base. Dynamic
          // field selectors still preserve each retained point's coordinates.
          relation.disjoint |= first && second && *first >= 0 && *second >= 0 && *first != *second;
        } else return std::nullopt;
      }
      lhs = a.getBase(); rhs = b.getBase();
      continue;
    }
    if (auto a = lhs.getDefiningOp<SubviewOp>()) {
      auto b = rhs.getDefiningOp<SubviewOp>();
      // Different parent translations could cancel a child field difference.
      if (!b || !sameSubviewWindow(a, b)) return std::nullopt;
      lhs = a.getBase(); rhs = b.getBase();
      continue;
    }
    auto a = lhs.getDefiningOp<MemoryViewOp>(), b = rhs.getDefiningOp<MemoryViewOp>();
    if (!a || !b || lhs != fullDenseArgumentView(lhs) || rhs != fullDenseArgumentView(rhs) ||
        a.getResult().getType().getShape() != b.getResult().getType().getShape() ||
        a.getResult().getType().getEncoding() != b.getResult().getType().getEncoding()) return std::nullopt;
    relation.lhsBase = lhs;
    relation.rhsBase = rhs;
    return relation;
  }
  return std::nullopt;
}

SmallVector<Value> weft::riscv::denseSiblingFieldViews(ValueRange views) {
  SmallVector<Value> result(views);
  if (result.empty()) return {};
  for (unsigned depth = 0; depth < 32; ++depth) {
    auto first = result.front().getDefiningOp<SubviewOp>();
    if (!first) return result;
    for (Value view : result) {
      auto projected = view.getDefiningOp<SubviewOp>();
      if (!projected || !sameSubviewWindow(first, projected)) return {};
    }
    for (Value &view : result) view = cast<SubviewOp>(view.getDefiningOp()).getBase();
  }
  return {};
}

bool weft::riscv::hasInjectiveDenseMappingAt(Value fullView, Operation *at) {
  auto view = fullView.getDefiningOp<MemoryViewOp>();
  if (!view || fullView != fullDenseArgumentView(fullView)) return false;
  if (staticallyInjective(view)) return true;
  return impliedAt(at, [&](Operation *op) {
    if (auto injective = dyn_cast_or_null<DenseInjectiveOp>(op))
      return injective.getView() == fullView;
    if (auto alias = dyn_cast_or_null<DenseElementwiseAliasOp>(op))
      return alias.getLhs() == fullView || alias.getRhs() == fullView;
    return false;
  });
}

bool weft::riscv::haveEqualDenseMappingsAt(Value lhs, Value rhs, Operation *at) {
  if (lhs == rhs) return hasInjectiveDenseMappingAt(lhs, at);
  return impliedAt(at, [&](Operation *op) {
    auto alias = dyn_cast_or_null<DenseElementwiseAliasOp>(op);
    return alias && ((alias.getLhs() == lhs && alias.getRhs() == rhs) ||
                     (alias.getLhs() == rhs && alias.getRhs() == lhs));
  });
}

bool weft::riscv::haveDisjointDenseProjectionsAt(Value lhs, Value rhs, Operation *at) {
  auto relation = denseProjectionRelation(lhs, rhs);
  Value a = fullDenseArgumentView(lhs), b = fullDenseArgumentView(rhs);
  if (!a || !b) return false;
  bool distinct = relation && relation->disjoint;
  if (distinct && haveEqualDenseMappingsAt(a, b, at)) return true;
  auto matches = [&](Value x, Value y) { return (x == a && y == b) || (x == b && y == a); };
  return impliedAt(at, [&](Operation *op) {
    if (auto disjoint = dyn_cast_or_null<DenseDisjointOp>(op))
      return matches(disjoint.getLhs(), disjoint.getRhs());
    if (auto alias = dyn_cast_or_null<DenseElementwiseAliasOp>(op))
      return distinct && matches(alias.getLhs(), alias.getRhs());
    return false;
  });
}

bool weft::riscv::densePointwiseIndependentAt(Value lhs, Value rhs, Operation *at) {
  if (!mayAliasStorageAt(lhs, rhs, at)) return true;
  auto relation = denseProjectionRelation(lhs, rhs);
  if (!relation || !relation->pointwise) return false;
  Value a = relation->lhsBase, b = relation->rhsBase;
  if (haveEqualDenseMappingsAt(a, b, at)) return true;
  auto matches = [&](Value x, Value y) { return (x == a && y == b) || (x == b && y == a); };
  return impliedAt(at, [&](Operation *op) {
    if (auto disjoint = dyn_cast_or_null<DenseDisjointOp>(op))
      return matches(disjoint.getLhs(), disjoint.getRhs());
    if (auto alias = dyn_cast_or_null<DenseElementwiseAliasOp>(op))
      return matches(alias.getLhs(), alias.getRhs());
    return false;
  });
}

LogicalResult DenseInjectiveOp::verify() {
  auto view = getView().getDefiningOp<MemoryViewOp>();
  if (!view || getView() != fullDenseArgumentView(getView()) ||
      view->getParentOfType<KernelOp>() != (*this)->getParentOfType<KernelOp>())
    return emitOpError("requires a full dense ABI view of the containing kernel");
  return success();
}
