#include "PhysicalDiagnostics.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace weft;

namespace {
template <typename T> std::string spelling(T value) {
  std::string result;
  llvm::raw_string_ostream output(result);
  value.print(output);
  return result;
}

std::vector<int64_t> integers(DenseI64ArrayAttr value) {
  return {value.asArrayRef().begin(), value.asArrayRef().end()};
}

riscv::LayoutAttr layoutOf(Type type) {
  if (auto value = dyn_cast<riscv::ValueType>(type)) return value.getLayout();
  if (auto value = dyn_cast<riscv::FragmentType>(type)) return value.getLayout();
  return {};
}

RISCVLayoutDiagnostic describeLayout(Type type, StringRef role, int64_t index) {
  RISCVLayoutDiagnostic result;
  result.role = role.str();
  result.index = index;
  auto layout = layoutOf(type);
  if (auto value = dyn_cast<riscv::ValueType>(type)) {
    result.elementType = spelling(value.getElementType());
    result.shape = integers(value.getShape());
    result.axes = integers(value.getAxisIds());
  } else {
    auto fragment = cast<riscv::FragmentType>(type);
    result.elementType = spelling(fragment.getElementType());
    result.shape = integers(fragment.getShape());
    result.axes = integers(fragment.getAxisIds());
  }
  result.carrier = layout.getCarrier().str();
  result.validity = layout.getValidity().str();
  result.layoutAxes = integers(layout.getAxisIds());
  result.timeFactors = integers(layout.getTimeFactors());
  result.laneFactors = integers(layout.getLaneFactors());
  result.replicaFactors = integers(layout.getReplicaFactors());
  result.fragmentFactors = integers(layout.getFragmentFactors());
  result.localFactors = integers(layout.getLocalFactors());
  result.sew = layout.getSew();
  result.lmulEighths = layout.getLmulEighths();
  result.vl = layout.getVl();
  result.registerGroups = layout.getRegisterGroups();
  result.replicaGroup = layout.getReplicaGroup();
  return result;
}

SmallVector<std::pair<int64_t, Type>> layouts(TypeRange types) {
  SmallVector<std::pair<int64_t, Type>> result;
  for (auto [index, type] : llvm::enumerate(types))
    if (layoutOf(type)) result.emplace_back(index, type);
  return result;
}

RISCVAccessDiagnostic describeAccess(riscv::AccessAttr access) {
  RISCVAccessDiagnostic result;
  result.form = access.getForm().str();
  result.mapping = access.getMapping().str();
  result.alignment = access.getAlignment();
  result.indexSEW = access.getIndexSEW();
  result.segmentFields = access.getSegmentFields();
  result.groupSize = access.getGroupSize();
  result.layerSize = access.getLayerSize();
  result.joinFields = access.getJoinFields();
  result.joinLowBits = access.getJoinLowBits();
  result.joinRole = access.getJoinRole();
  result.bitOffset = access.getBitOffset();
  result.storageBits = access.getStorageBits();
  result.order = access.getOrder().str();
  result.fieldChunkBytes = access.getFieldChunkBytes();
  return result;
}
} // namespace

std::vector<RISCVOperationDiagnostic>
riscv_internal::collectPhysicalOperations(riscv::KernelOp kernel) {
  Builder builder(kernel.getContext());
  llvm::DenseMap<Attribute, size_t> groups;
  std::vector<RISCVOperationDiagnostic> result;
  kernel.walk([&](Operation *operation) {
    auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
    if (!leaf) return;
    auto access = operation->getAttrOfType<riscv::AccessAttr>("access");
    auto packet = operation->getAttrOfType<riscv::ScalarPacketAttr>("packet");
    auto inputs = layouts(operation->getOperandTypes());
    auto outputs = layouts(operation->getResultTypes());
    auto layoutKey = [&](ArrayRef<std::pair<int64_t, Type>> values) {
      SmallVector<Attribute> entries;
      for (const auto &[index, type] : values)
        entries.push_back(builder.getArrayAttr({builder.getI64IntegerAttr(index), TypeAttr::get(type)}));
      return builder.getArrayAttr(entries);
    };
    auto signature = builder.getArrayAttr({builder.getStringAttr(operation->getName().getStringRef()),
        leaf, access ? Attribute(access) : Attribute(builder.getUnitAttr()),
        packet ? Attribute(packet) : Attribute(builder.getUnitAttr()),
        builder.getI64IntegerAttr(operation->getNumOperands()), builder.getI64IntegerAttr(operation->getNumResults()),
        layoutKey(inputs), layoutKey(outputs)});
    auto [found, inserted] = groups.try_emplace(signature, result.size());
    size_t position = found->second;
    if (inserted) {
      RISCVOperationDiagnostic entry;
      entry.operation = operation->getName().getStringRef().str();
      entry.engine = leaf.getEngine().str();
      entry.family = leaf.getFamily().str();
      entry.instruction = leaf.getInstruction().str();
      entry.mask = leaf.getMask().str();
      entry.tail = leaf.getTail().str();
      if (access) entry.access = describeAccess(access);
      if (packet) entry.scalarPacket = RISCVScalarPacketDiagnostic{packet.getAxis(), packet.getSteps(),
          packet.getParts(), packet.getRegisters(), packet.getLiveRegisters()};
      entry.operandCount = operation->getNumOperands();
      entry.resultCount = operation->getNumResults();
      entry.operandGroups = leaf.getOperandGroups();
      entry.resultGroups = leaf.getResultGroups();
      entry.temporaryGroups = leaf.getTemporaryGroups();
      entry.fragmentGroups = leaf.getFragmentGroups();
      entry.localBytes = leaf.getLocalBytes();
      entry.parameters = integers(leaf.getParameters());
      for (const auto &[index, type] : inputs) entry.layouts.push_back(describeLayout(type, "operand", index));
      for (const auto &[index, type] : outputs) entry.layouts.push_back(describeLayout(type, "result", index));
      result.push_back(std::move(entry));
    }
    auto &entry = result[position];
    ++entry.occurrences;
    auto location = spelling(operation->getLoc());
    if (!llvm::is_contained(entry.locations, location)) entry.locations.push_back(std::move(location));
  });
  return result;
}
