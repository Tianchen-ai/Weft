#include "Weft/Target/RISCVPasses.h"
#include "PhysicalLoopControl.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Dialect/RISCV/IR/Layout.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <memory>
#include <limits>
#include <numeric>
#include <optional>
#include <string>

using namespace weft;

namespace {

bool isTerminalRISCVOperation(mlir::Operation *operation) {
  return operation->hasTrait<mlir::OpTrait::WeftRISCVTerminal>();
}

bool requiresLeaf(mlir::Operation *operation) {
  return operation->hasTrait<mlir::OpTrait::WeftRISCVLeaf>();
}

bool checkedAdd(int64_t &total, int64_t value) {
  if (value < 0 || total < 0 ||
      total > std::numeric_limits<int64_t>::max() - value)
    return false;
  total += value;
  return true;
}

riscv::EncodingDeclOp findEncoding(mlir::ModuleOp module,
                                   llvm::StringRef family) {
  riscv::EncodingDeclOp result;
  module.walk([&](riscv::EncodingDeclOp declaration) {
    if (!result && declaration.getSymName() == family)
      result = declaration;
  });
  return result;
}

riscv::DerivedEncodingOp findDerivedEncoding(
    mlir::ModuleOp module, kernel::EncodingType encoding) {
  riscv::DerivedEncodingOp result;
  module.walk([&](riscv::DerivedEncodingOp derived) {
    if (!result && derived.getResultFamily() == encoding.getFamily() &&
        derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
        derived.getParameterValues() == encoding.getParameters().asArrayRef())
      result = derived;
  });
  return result;
}

std::optional<int64_t> denseElementBits(kernel::EncodingType encoding) {
  if (encoding.getKind() != "dense")
    return std::nullopt;
  llvm::StringRef family = encoding.getFamily();
  if (family == "f8E4M3FN" || family == "f8E5M2") return 8;
  if (family == "f16" || family == "bf16")
    return 16;
  if (family == "f32")
    return 32;
  if (family == "f64")
    return 64;
  if (family.consume_front("i") || family.consume_front("u")) {
    int64_t width = 0;
    if (!family.getAsInteger(10, width) && width > 0)
      return width;
  }
  return std::nullopt;
}

mlir::LogicalResult verifyDescriptorFacts(mlir::Operation *owner,
                                          mlir::Value value,
                                          mlir::ModuleOp module) {
  auto descriptor = mlir::cast<riscv::MemDescType>(value.getType());
  auto encoding =
      mlir::dyn_cast<kernel::EncodingType>(descriptor.getEncoding());
  if (!encoding)
    return owner->emitError(
        "memory descriptor lost its canonical Encoding identity");
  if (descriptor.getLayoutIdentity() != encoding.getLayoutIdentity())
    return owner->emitError(
        "memory descriptor layout identity disagrees with its Encoding");
  if (encoding.getKind() == "dense") {
    auto expectedBits = denseElementBits(encoding);
    // FieldOp retains a logical projection into an encoded record. Its exact
    // storage bytes and bit packing are carried by the selected AccessAttr;
    // only an explicit FieldView exposes ordinary dense addressable storage.
    if (expectedBits && !value.getDefiningOp<riscv::FieldOp>())
      expectedBits = std::max<int64_t>(8, *expectedBits);
    if (!expectedBits || descriptor.getStorageBits() != *expectedBits ||
        descriptor.getElements() != 1 || descriptor.getInterleaveRows() != 0)
      return owner->emitError(
          "dense descriptor has non-dense record or interleave geometry");
    return mlir::success();
  }
  llvm::StringRef baseFamily = encoding.getFamily();
  int64_t interleaveRows = 0;
  if (encoding.getKind() == "derived_instance") {
    auto derived = findDerivedEncoding(module, encoding);
    if (!derived)
      return owner->emitError(
          "derived descriptor has no unique physical Encoding declaration");
    baseFamily = derived.getSourceFamily();
    interleaveRows = derived.getInterleaveRows();
  } else if (encoding.getKind() != "base") {
    return owner->emitError(
        "runtime memory descriptor cannot use an abstract Encoding family");
  }
  auto declaration = findEncoding(module, baseFamily);
  if (!declaration || descriptor.getStorageBits() != declaration.getStorageBits() ||
      descriptor.getElements() != declaration.getElements() ||
      descriptor.getAlignment() != declaration.getAlignment() ||
      descriptor.getInterleaveRows() != interleaveRows)
    return owner->emitError(
        "memory descriptor storage facts disagree with its physical Encoding declarations");
  return mlir::success();
}

bool requiresIntegerWidening(mlir::Operation *operation) {
  if (mlir::isa<riscv::RVVGroupedMacStepOp, riscv::RVVWidenDotOp,
                riscv::RVVWidenAddOp, riscv::RVVNarrowShiftRightOp,
                riscv::RVVPartialSetOp,
                riscv::RVVPartialCaptureOp, riscv::RVVPartialCollectOp,
                riscv::RVVPartialRepackOp,
                riscv::RVVPartialMergeOp,
                riscv::RVVPartialReduceOp,
                riscv::RVVPartialScaleCombineOp,
                riscv::RVVPartialPackedScaleOp,
                riscv::RVVPartialWidenScaleOp,
                riscv::RVVPartialCombineOp,
                riscv::RVVPartialFinalizeOp,
                riscv::RVVWidenAccumulateOp,
                riscv::RVVFinalizeWidenDotOp,
                riscv::RVVWidenReduceOp,
                riscv::RVVPartitionedWidenReduceStoreOp>(operation))
    return true;
  auto widen = mlir::dyn_cast<riscv::WidenOp>(operation);
  if (!widen)
    return false;
  return mlir::isa<mlir::IntegerType>(
      riscv_internal::logicalElement(widen.getInput().getType()));
}

bool requiresFloatWidening(mlir::Operation *operation) {
  auto widen = mlir::dyn_cast<riscv::WidenOp>(operation);
  return widen && mlir::isa<mlir::FloatType>(
                      riscv_internal::logicalElement(widen.getInput().getType()));
}

class VerifyFinalRISCVPass
    : public mlir::PassWrapper<VerifyFinalRISCVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyFinalRISCVPass)
  llvm::StringRef getArgument() const override {
    return "weft-riscv-verify-final";
  }
  llvm::StringRef getDescription() const override {
    return "Verify the complete target-aware RISC-V Physical IR contract";
  }

  void runOnOperation() override {
    if (getOperation()->hasAttr("weft.riscv.layout_input") || getOperation()->hasAttr("weft.riscv.contraction_input")) {
      getOperation().emitError("physical checkpoint must be consumed through its registered main suffix");
      signalPassFailure();
      return;
    }
    bool failed = false;
    mlir::ModuleOp module = getOperation();
    module.walk([&](riscv::KernelOp kernel) {
      auto used = riscv::usedInstructionExtensions(kernel);
      if (mlir::failed(used)) {
        failed = true;
        return;
      }
      for (mlir::Attribute attribute : kernel.getRequiredExtensions()) {
        auto extension = mlir::cast<mlir::StringAttr>(attribute).getValue();
        if (!llvm::is_contained(*used, extension)) {
          kernel.emitError("selected kernel does not use required extension: ") << extension;
          failed = true;
        }
      }
      int64_t grouped = 0, partitioned = 0;
      kernel.walk([&](riscv::IMEUnpackOp unpack) {
        if (unpack.getResult().getType().getLayout().getReplicaGroup() > 1)
          ++grouped;
        else
          ++partitioned;
      });
      auto strategy = kernel.getLocalTileStrategy();
      if ((strategy == "register-grouped" && (!grouped || partitioned)) ||
          (strategy == "atom-partitioned" && (!partitioned || grouped))) {
        kernel.emitError("explicit local tile strategy disagrees with the selected unpack entities: ")
            << strategy << ", grouped=" << grouped << ", partitioned=" << partitioned;
        failed = true;
      }
    });
    llvm::DenseSet<std::pair<int64_t, int64_t>> partialBirths;
    getOperation().walk([&](mlir::Operation *operation) {
      auto grouped = [](mlir::Type type) {
        auto value = mlir::dyn_cast<riscv::ValueType>(type);
        return value && value.getLayout().getReplicaGroup() > 1;
      };
      if ((llvm::any_of(operation->getOperandTypes(), grouped) ||
           llvm::any_of(operation->getResultTypes(), grouped)) &&
          !mlir::isa<riscv::IMEUnpackOp, riscv::NewOp, riscv::UnaryOp,
                     riscv::BinaryOp, riscv::FMAOp, riscv::CastOp, riscv::WidenOp, riscv::NarrowOp,
                     riscv::RVVAxisBroadcastOp, riscv::RVVSplatOp, riscv::ConvertLayoutOp,
                     riscv::RegisterMaterializeOp, mlir::scf::ForOp,
                     mlir::scf::IfOp, mlir::scf::YieldOp>(operation)) {
        operation->emitError("replica-grouped values require a selected pointwise/control relation or an explicit lane repack");
        failed = true;
      }
      mlir::Value region;
      mlir::Value memoryValue;
      if (auto load = mlir::dyn_cast<riscv::LoadOp>(operation)) {
        region = load.getRegion();
        memoryValue = load.getResult();
      }
      if (auto store = mlir::dyn_cast<riscv::StoreOp>(operation)) {
        region = store.getRegion();
        memoryValue = store.getValue();
      }
      if (auto type = memoryValue ? mlir::dyn_cast<riscv::ValueType>(memoryValue.getType())
                                  : riscv::ValueType();
          type && type.getLayout().getCarrier() == "rvv") {
        auto memory = mlir::dyn_cast<riscv::MemDescType>(region.getType());
        auto encoding = memory ? mlir::dyn_cast<kernel::EncodingType>(memory.getEncoding())
                               : kernel::EncodingType();
        if (encoding && encoding.getKind() == "dense" &&
            llvm::count_if(type.getLayout().getLaneFactors().asArrayRef(),
                           [](int64_t n) { return n > 1; }) > 1) {
          operation->emitError("dense transfer requires one lane axis and explicit carrier conversions");
          failed = true;
        }
      }
      if (region && kernel::hasDynamicSubviewExtent<riscv::SubviewOp, riscv::SliceOp, riscv::FieldOp>(region) &&
          llvm::any_of(riscv_internal::logicalShape(region.getType()), [](int64_t n) { return n < 0; })) {
        operation->emitError("dynamic shaped subview access has no explicit bounded value realization");
        failed = true;
      }
      if (auto cast = mlir::dyn_cast<riscv::CastOp>(operation);
          cast && mlir::failed(cast.verify()))
        failed = true;
      if (auto load = mlir::dyn_cast<riscv::LoadOp>(operation);
          load && !load.getSnapshotStorage() &&
          riscv_internal::needsReadSnapshot(load)) {
        load.emitError(
            "encoded read crosses a possibly aliasing write without a physical snapshot");
        failed = true;
      }
      auto verifyPartialBirth = [&](int64_t owner, int64_t birth) {
        if (!partialBirths.insert({owner, birth}).second) {
          operation->emitError()
              << "partial owner " << owner << " has duplicate birth " << birth;
          failed = true;
        }
      };
      if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
        verifyPartialBirth(partial.getOwnerDomainId(), partial.getBirthId());
      if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
        verifyPartialBirth(capture.getOwnerDomainId(), capture.getBirthId());
      if (auto collect = mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
        verifyPartialBirth(collect.getOwnerDomainId(), collect.getBirthId());
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "weft_kernel") {
        operation->emitError(
            "Canonical Kernel IR operation survived target conversion");
        failed = true;
      }
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "weft_riscv" &&
          !isTerminalRISCVOperation(operation)) {
        auto diagnostic = operation->emitError()
            << operation->getName()
            << " is not in the final RISC-V terminal allowlist";
        for (mlir::Value result : operation->getResults())
          for (mlir::OpOperand &use : result.getUses())
            diagnostic << " [used by " << use.getOwner()->getName() << "]";
        failed = true;
      }
      if (operation->getDialect() &&
          operation->getDialect()->getNamespace() == "weft_riscv" &&
          mlir::failed(riscv::verifyMemoryEffects(operation)))
        failed = true;
      if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
          leaf && leaf.getInstruction().ends_with(".dynamic-stride")) {
        operation->emitError("dynamic memory choice must become explicit control before final verification");
        failed = true;
      }
      if (operation->hasAttr("weft.riscv.dense_contraction")) {
        operation->emitError("dense contraction selection must be consumed before terminal emission");
        failed = true;
      }
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
          conversion && conversion.getConversion().getKind() == "local_load" &&
          riscv::storageOrigins(conversion.getInput()).isOwnedLocal()) {
        conversion.emitError("numeric local conversion must become an explicit descriptor read before final verification");
        failed = true;
      }
      if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation);
          reduce && reduce.getLeaf().getInstruction().starts_with("rvv.lane-reduce.")) {
        auto input = mlir::cast<riscv::ValueType>(reduce.getInput().getType());
        auto layout = input.getLayout();
        auto streams = riscv_internal::staticProduct(layout.getTimeFactors().asArrayRef());
        int64_t temporaries = 1;
        if (streams && *streams > 1 && layout.getValidity() == "full")
          temporaries += (layout.getLmulEighths() + 7) / 8;
        if (reduce.getLeaf().getTail() != "exact" ||
            reduce.getLeaf().getTemporaryGroups() != temporaries) {
          reduce.emitError("lane reduction must preserve its seed for empty parts and account for its local carriers");
          failed = true;
        }
      }
      for (mlir::Value result : operation->getResults())
        if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
          if (value.getLayout().getCarrier() == "unassigned") {
            operation->emitError("final shaped result has no physical layout");
            failed = true;
          }
          if (value.getLayout().getCarrier() == "ime") {
            operation->emitError(
                "IME representation must be carried by a typed fragment, not a generic value");
            failed = true;
          }
        }
      auto checkDescriptor = [&](mlir::Value value) {
        auto descriptor = mlir::dyn_cast<riscv::MemDescType>(value.getType());
        if (descriptor && mlir::failed(verifyDescriptorFacts(
                              operation, value, module)))
          failed = true;
      };
      for (mlir::Value result : operation->getResults())
        checkDescriptor(result);
      for (mlir::Region &region : operation->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            if (auto value =
                    mlir::dyn_cast<riscv::ValueType>(argument.getType())) {
              if (value.getLayout().getCarrier() == "unassigned") {
                operation->emitError(
                    "final shaped block argument has no physical layout");
                failed = true;
              }
            } else {
              checkDescriptor(argument);
            }
      if (auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf")) {
        if (auto broadcast = mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(operation)) {
          auto groups = riscv::axisBroadcastTemporaryGroups(broadcast);
          if (!groups || leaf.getTemporaryGroups() != *groups) {
            broadcast.emitError("axis broadcast local resource contract is stale");
            failed = true;
          }
        }
        if (leaf.getEngine() == "unselected" ||
            leaf.getInstruction().starts_with("pending-")) {
          operation->emitError()
              << "final target operation " << operation->getName()
              << " has no closed local leaf; leaf=" << leaf
              << ", operands=" << operation->getOperandTypes()
              << ", results=" << operation->getResultTypes();
          failed = true;
        }
        if (mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp,
                      riscv::CompareOp, riscv::SelectOp, riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp,
                      riscv::WidenOp, riscv::ReduceOp,
                      riscv::Fold2Op>(operation)) {
          std::string expected =
              riscv_internal::terminalInstruction(operation);
          if (expected.empty() || leaf.getInstruction() != expected ||
              leaf.getSpelling() != expected) {
            operation->emitError()
                << "final leaf does not match the typed operation; expected "
                << (expected.empty() ? llvm::StringRef("<unsupported>")
                                     : llvm::StringRef(expected))
                << ", got " << leaf.getInstruction();
            failed = true;
          }
          if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation);
              binary && leaf.getInstruction().ends_with(".broadcast")) {
            auto groups = riscv::binaryBroadcastTemporaryGroups(binary);
            if (!groups || leaf.getTemporaryGroups() != *groups) {
              binary.emitError("binary broadcast local resource contract is stale");
              failed = true;
            }
          }
          if (auto select = mlir::dyn_cast<riscv::SelectOp>(operation)) {
            auto groups = riscv::selectTemporaryGroups(select);
            if (!groups || leaf.getTemporaryGroups() != *groups) {
              select.emitError("select local resource contract is stale");
              failed = true;
            }
          }
          if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation);
              binary && leaf.getInstruction() == "rvv.mul.row-scale") {
            auto groups = riscv::rvvRowScaleTemporaryGroups(binary);
            if (!groups || leaf.getTemporaryGroups() != *groups) {
              binary.emitError("row scale local resource contract is stale");
              failed = true;
            }
          }
        }
      } else if (requiresLeaf(operation)) {
        operation->emitError(
            "final target-local operation has no exact leaf contract");
        failed = true;
      }
      if (operation->hasAttr("implementation")) {
        operation->emitError(
            "structural implementation anchor survived terminal leaf lowering");
        failed = true;
      }
      for (llvm::StringRef name :
           {"partial_layout_plan", "partial_combine_plan",
            "nested_partial_plan", "sequential_partial_plan",
            "scaled_partial_plan", "layered_partial_plan",
            "partial_add_tree_plan", "weft.riscv.unroll_factor",
            "weft.riscv.unroll_order", "weft.riscv.issue_operation_order",
            "weft.riscv.issue_iteration"})
        if (operation->hasAttr(name)) {
          operation->emitError()
              << "transient physical plan " << name
              << " survived materialization";
          failed = true;
        }
      if (mlir::isa<riscv::RVVGroupedMacLoadOp>(operation)) {
        auto loop = operation->getParentOfType<mlir::scf::ForOp>();
        auto systemUnroll =
            loop ? loop->getAttrOfType<mlir::StringAttr>(
                       "weft.riscv.system_unroll")
                 : mlir::StringAttr();
        if (!loop || !systemUnroll || systemUnroll.getValue() != "disable") {
          operation->emitError(
              "final fixed-window load requires a nearest parent loop with downstream unrolling disabled");
          failed = true;
        }
      }
      if (mlir::isa<riscv::RVVStreamReduceStepOp>(operation)) {
        auto loop = operation->getParentOfType<mlir::scf::ForOp>();
        auto systemUnroll =
            loop ? loop->getAttrOfType<mlir::StringAttr>(
                       "weft.riscv.system_unroll")
                 : mlir::StringAttr();
        if (!loop || !systemUnroll || systemUnroll.getValue() != "disable") {
          operation->emitError(
              "final stream step requires a nearest parent loop with "
              "downstream unrolling disabled");
          failed = true;
        }
      }
      auto scalarPacket = mlir::dyn_cast<riscv::ScalarTileLoadOp>(operation);
      if (mlir::isa<riscv::RVVFMAIssueOp>(operation) || (scalarPacket && scalarPacket.getPacket())) {
        if (auto loop = operation->getParentOfType<mlir::scf::ForOp>()) {
          auto unroll = loop->getAttrOfType<mlir::StringAttr>("weft.riscv.system_unroll");
          if (!unroll || unroll.getValue() != "disable") {
            operation->emitError("closed numeric issue requires its containing physical loop to disable downstream unrolling");
            failed = true;
          }
        }
      }
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
          conversion && riscv::fieldReadProjection(conversion.getInput())) {
        conversion.emitError("final encoded conversion input must be supplied by field_read");
        failed = true;
      }
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
          conversion &&
          (conversion.getConversion().getKind() == "register_to_lane" ||
           conversion.getConversion().getKind() == "time_to_lane")) {
        auto inputType =
            mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
        auto resultType =
            mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
        const bool rvvToRVV =
            inputType && resultType &&
            inputType.getLayout().getCarrier() == "rvv" &&
            resultType.getLayout().getCarrier() == "rvv";
        const bool closedRVVPack =
            rvvToRVV && static_cast<bool>(
                            riscv::rvvPartToLanePieces(inputType, resultType));
        mlir::Operation *definition = conversion.getInput().getDefiningOp();
        const bool invalidRVVPack = rvvToRVV && !closedRVVPack;
        const bool unmaterializedFieldRead =
            !rvvToRVV &&
            conversion.getConversion().getKind() == "time_to_lane";
        if (invalidRVVPack || unmaterializedFieldRead) {
          conversion.emitError(
              "final part-to-lane conversion requires a closed numeric pack; encoded supply must be an explicit field_read; input_def=")
              << (definition ? definition->getName().getStringRef()
                             : llvm::StringRef("<block-argument>"))
              << ", input_type=" << conversion.getInput().getType()
              << ", result_type=" << conversion.getResult().getType();
          failed = true;
        }
      }
      if (auto access = operation->getAttrOfType<riscv::AccessAttr>("access"))
        if (access.getForm() == "unassigned") {
          operation->emitError("final memory edge has no selected form");
          failed = true;
        }
      if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation)) {
        auto access = field.getAccess();
        auto leaf = field.getLeaf();
        const auto facts = riscv_internal::fieldFacts(field);
        if (!access || access.getMapping() == "opaque" ||
            leaf.getEngine() != "transfer" ||
            leaf.getFamily() != "encoded-field" ||
            leaf.getInstruction() !=
                ("rvv.encoded." + access.getMapping()).str()) {
          field.emitError(
              "final encoded field has no closed storage mapping and transfer leaf");
          failed = true;
        }
        if (access && access.getAlignment() > facts.alignment) {
          field.emitError(
              "final encoded field overstates its declared base-plus-offset alignment");
          failed = true;
        }
        auto value = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
        const unsigned elementBits =
            value ? riscv_internal::logicalBitWidth(value.getElementType()) : 0;
        if (access && value && value.getLayout().getCarrier() == "rvv" &&
            access.getMapping() == "natural" && elementBits > 8 &&
            elementBits % 8 == 0 &&
            access.getAlignment() < static_cast<int64_t>(elementBits / 8) &&
            !llvm::all_of(field.getResult().getUsers(), [](mlir::Operation *user) {
              return mlir::isa<riscv::FieldReadOp>(user);
            })) {
          field.emitError(
              "final natural RVV field requires a typed byte-load operation for its proven alignment");
          failed = true;
        }
      }
      if (auto entry =
              mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(operation)) {
        const int64_t availableAlignment =
            riscv_internal::indexedEntryAlignment(entry.getSource(),
                entry.getEntryOffsets(), entry.getEntryByteStride());
        if (entry.getAccess().getAlignment() > availableAlignment) {
          entry.emitError(
              "final indexed-entry load overstates source, byte-offset and entry-stride alignment")
              << "; access_alignment=" << entry.getAccess().getAlignment()
              << ", available_alignment=" << availableAlignment;
          failed = true;
        }
      }
      if (auto load = mlir::dyn_cast<riscv::LoadOp>(operation)) {
        auto access = load.getAccess();
        auto leaf = load.getLeaf();
        if (access && access.getForm() == "unit" &&
            riscv::hasDynamicDenseStride(load.getRegion().getType(), load.getResult().getType())) {
          auto type = mlir::cast<riscv::ValueType>(load.getResult().getType());
          for (auto [axis, lanes] : llvm::zip(type.getAxisIds().asArrayRef(), type.getLayout().getLaneFactors().asArrayRef()))
            if (lanes > 1 && !riscv::hasUnitStrideProof(load, load.getRegion(), axis)) {
              load.emitError("contiguous load from dynamic strides requires a dominating unit-stride proof");
              failed = true;
            }
        }
        if (!access) {
          load.emitError("final load has no selected memory access");
          failed = true;
          return;
        }
        const std::string expected = load.getSnapshotStorage()
                                         ? (load->getParentOfType<riscv::KernelOp>()
                                                    .getTarget().getHasRVV()
                                                ? "rvv.read-snapshot"
                                                : "scalar.read-snapshot")
                                         : ("rvv.load." + access.getForm()).str();
        if (leaf.getEngine() != "transfer" ||
            leaf.getFamily() != "load" ||
            leaf.getInstruction() != expected ||
            leaf.getSpelling() != leaf.getInstruction()) {
          load.emitError(
              "final load has no exact access-form transfer leaf");
          failed = true;
        }
      }
      if (auto store = mlir::dyn_cast<riscv::StoreOp>(operation)) {
        mlir::Value root = store.getRegion();
        while (mlir::isa_and_nonnull<riscv::SliceOp, riscv::SubviewOp>(root.getDefiningOp()))
          root = root.getDefiningOp()->getOperand(0);
        if (root != store.getRegion() && root.getDefiningOp<riscv::FieldOp>()) {
          store.emitError("projected encoded field store requires a materialized field view");
          failed = true;
          return;
        }
        auto access = store.getAccess();
        auto leaf = store.getLeaf();
        if (store.getRegion().getDefiningOp<riscv::FieldOp>()) {
          auto plan = store.getFieldStorePlanAttr();
          const bool scalar = plan && (plan.getKind() == "scalar" || plan.getKind() == "scalar_bytes");
          if (!plan || leaf.getFamily() != "encoded-field-store" ||
              leaf.getEngine() != (scalar ? "scalar" : "rvv") ||
              leaf.getInstruction() != (scalar ? "scalar.field-store" : "rvv.field-store") ||
              leaf.getSpelling() != leaf.getInstruction() ||
              mlir::failed(store.verify())) {
            store.emitError("final encoded field store has no closed physical packing/address plan");
            failed = true;
          }
          return;
        }
        auto value = mlir::dyn_cast<riscv::ValueType>(store.getValue().getType());
        if (access && access.getForm() == "unit" &&
            riscv::hasDynamicDenseStride(store.getRegion().getType(), store.getValue().getType()))
          for (auto [axis, lanes] : llvm::zip(value.getAxisIds().asArrayRef(), value.getLayout().getLaneFactors().asArrayRef()))
            if (lanes > 1 && !riscv::hasUnitStrideProof(store, store.getRegion(), axis)) {
              store.emitError("contiguous store to dynamic strides requires a dominating unit-stride proof");
              failed = true;
            }
        auto encoding = mlir::dyn_cast<kernel::EncodingType>(
            store.getRegion().getType().getEncoding());
        const bool scalarValue =
            !value || value.getLayout().getCarrier() == "scalar";
        const bool scalarStore =
            access && scalarValue &&
            (access.getForm() == "unit" || access.getForm() == "strided") &&
            access.getMapping() == "dense" && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "store" &&
            leaf.getInstruction() == "scalar.store" &&
            leaf.getSpelling() == leaf.getInstruction() && encoding &&
            encoding.getKind() == "dense" &&
            store.getRegion().getType().getElements() == 1 &&
            store.getRegion().getType().getInterleaveRows() == 0;
        const bool vectorStore =
            access && !scalarValue && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "store" &&
            leaf.getInstruction() == ("rvv.store." + access.getForm()).str() &&
            leaf.getSpelling() == leaf.getInstruction();
        if (!scalarStore && !vectorStore) {
          store.emitError(
              "final store has no exact access-form transfer leaf");
          failed = true;
        }
      }
      if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation)) {
        auto access = extract.getAccess();
        auto leaf = extract.getLeaf();
        const bool local = access && access.getForm() == "local";
        const bool localGather =
            access && access.getForm() == "indexed" &&
            leaf.getEngine() == "rvv" &&
            leaf.getFamily() == "local-gather" &&
            leaf.getInstruction() == "rvv.local-gather";
        const bool localExtract =
            local && leaf.getEngine() == "transfer" &&
            leaf.getFamily() == "local-extract" &&
            leaf.getInstruction() == "local.extract";
        const bool localDenseRead =
            access && access.getForm() == "unit" && access.getMapping() == "dense" &&
            leaf.getEngine() == "rvv" && leaf.getFamily() == "local-dense-read" &&
            leaf.getInstruction() == "rvv.local-dense-read" &&
            leaf.getSpelling() == leaf.getInstruction() && leaf.getTemporaryGroups() == 0 &&
            riscv_internal::isLocalDenseRowExtract(extract);
        const bool registerExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "rvv" && leaf.getFamily() == "extract" &&
            (leaf.getInstruction() == "rvv.extract.vrgather" ||
             leaf.getInstruction() == "rvv.extract.regular-scalar");
        const bool laneReplicaExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "rvv" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() == "rvv.extract.lane-to-replica";
        const bool scalarReplicaExtract =
            access && access.getForm() == "register" &&
            leaf.getEngine() == "scalar" && leaf.getFamily() == "extract" &&
            (leaf.getInstruction() == "scalar.replica-gather" ||
             leaf.getInstruction() == "scalar.replica-project");
        const bool ordinaryExtract =
            access && !local && !localGather && !registerExtract &&
            !laneReplicaExtract && !scalarReplicaExtract &&
            leaf.getEngine() == "transfer" && leaf.getFamily() == "extract" &&
            leaf.getInstruction() ==
                ("rvv.extract." + access.getForm()).str();
        if (!access ||
            (!localExtract && !localGather && !localDenseRead && !registerExtract &&
             !laneReplicaExtract &&
             !scalarReplicaExtract && !ordinaryExtract)) {
          extract.emitError(
              "final extract has no closed physical access and transfer leaf");
          failed = true;
        }
      }
      if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation)) {
        auto layout = update.getResult().getType().getLayout();
        auto leaf = update.getLeaf();
        bool local = layout.getCarrier() == "local" && leaf.getEngine() == "scalar" &&
            leaf.getInstruction() == "local.update";
        bool whole = update.coversWholeState() && update.getValue().getType() == update.getResult().getType() &&
            ((layout.getCarrier() == "rvv" && leaf.getEngine() == "rvv" && leaf.getInstruction() == "rvv.merge") ||
             (layout.getCarrier() == "scalar" && leaf.getEngine() == "scalar" && leaf.getInstruction() == "scalar.update"));
        if ((!local && !whole) || leaf.getFamily() != "state-update" ||
            leaf.getSpelling() != leaf.getInstruction()) {
          update.emitError(
              "final update requires a local store or complete register-state binding");
          failed = true;
        }
      }
      if (auto schedule =
              operation->getAttrOfType<riscv::ScheduleAttr>("schedule"))
        if (schedule.getPipelineDepth() > 1) {
          operation->emitError(
              "unexpanded pipeline schedule survived final physical lowering");
          failed = true;
        }
      if (operation->hasAttr("weft.riscv.schedule") ||
          operation->hasAttr("weft.riscv.pipeline_stage") ||
          operation->hasAttr("weft.riscv.pipeline_order") ||
          operation->hasAttr("weft.riscv.pipeline_invariant")) {
        operation->emitError(
            "typed physical schedule survived without structural expansion");
        failed = true;
      }
      if (auto state = mlir::dyn_cast<riscv::NewOp>(operation))
        if (auto value = mlir::dyn_cast<riscv::ValueType>(
                state.getResult().getType());
            value && value.getLayout().getCarrier() == "local") {
          state.emitError(
              "local state initialization survived without explicit local storage and control");
          failed = true;
        }
      if (auto store = mlir::dyn_cast<riscv::StoreOp>(operation))
        if (auto value =
                mlir::dyn_cast<riscv::ValueType>(store.getValue().getType());
            value && value.getLayout().getCarrier() == "local") {
          store.emitError(
              "local value commit survived without explicit element transfer control");
          failed = true;
        }
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation)) {
        auto level = loop->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
        auto control = riscv_internal::PhysicalLoopControl::get(loop);
        if (mlir::failed(control) || mlir::failed(control->verifyFiniteEndpoint(loop))) failed = true;
        auto systemUnroll = loop->getAttrOfType<mlir::StringAttr>(
            "weft.riscv.system_unroll");
        if (systemUnroll && systemUnroll.getValue() != "disable") {
          loop.emitError(
              "final ordered loop has an unsupported downstream-unroll contract");
          failed = true;
        }
        if (level) {
          riscv::PhysicalPointOp point;
          for (mlir::Operation &nested : loop.getBody()->without_terminator())
            if (!point)
              point = mlir::dyn_cast<riscv::PhysicalPointOp>(nested);
          if (!point) {
            loop.emitError(
                "final physical Level has no explicit point for its logical domain");
            failed = true;
          } else {
            auto domain = point.getResult().getType().getDomain();
            if (domain.getDomainId() != level.getDomainId() ||
                domain.getAxisId() != level.getAxisId() ||
                domain.getRelation() != level.getRelation()) {
              loop.emitError(
                  "final Level identity disagrees with its physical point domain");
              failed = true;
            }
          }
          llvm::DenseSet<int64_t> stateBirths;
          llvm::DenseSet<int64_t> stagedBirths;
          for (mlir::Operation &nested : loop.getBody()->without_terminator()) {
            if (auto staged = mlir::dyn_cast<riscv::LocalBindOp>(nested);
                staged && staged.getStorage().getType().getSchema() == "staged-value" &&
                staged.getStorage().getType().getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(staged.getStorage().getType().getBirthId());
            if (auto state = mlir::dyn_cast<riscv::NewOp>(nested);
                state && state.getOwnerDomainId() == level.getDomainId())
              stateBirths.insert(state.getBirthId());
            if (auto state = mlir::dyn_cast<riscv::LocalBindOp>(nested);
                state && state.getStorage().getType().getPurpose() == "handoff" &&
                state.getStorage().getType().getOwnerDomainId() ==
                    level.getDomainId())
              stateBirths.insert(state.getStorage().getType().getBirthId());
            if (auto staged =
                    mlir::dyn_cast<riscv::RegisterMaterializeOp>(nested);
                staged && staged.getRealization() != "physical-share" &&
                staged.getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(staged.getBirthId());
            if (auto staged = mlir::dyn_cast<riscv::StagedViewOp>(nested);
                staged && staged.getOwnerDomainId() == level.getDomainId())
              stagedBirths.insert(staged.getBirthId());
            if (auto staged =
                    mlir::dyn_cast<riscv::RVVLocalMaterializeOp>(nested);
                staged) {
              auto storage = staged.getDestination()
                                 .getDefiningOp<riscv::LocalBindOp>()
                                 .getStorage()
                                 .getType();
              // A compiler-created physical share has no author staged birth;
              // its lifetime is carried by LocalType and the resource pass.
              if (storage.getSchema() != "physical-share" &&
                  storage.getOwnerDomainId() == level.getDomainId())
                stagedBirths.insert(storage.getBirthId());
            }
            if (auto staged =
                    mlir::dyn_cast<riscv::EncodedLocalBindOp>(nested);
                staged) {
              auto storage = staged.getStorage().getType();
              if (storage.getOwnerDomainId() == level.getDomainId())
                stagedBirths.insert(storage.getBirthId());
            }
          }
          const int64_t actualState = stateBirths.size();
          const int64_t actualStaged = stagedBirths.size();
          if (level.getStateBirthCount() != actualState ||
              level.getStagedBirthCount() != actualStaged ||
              level.getHandoffCount() !=
                  static_cast<int64_t>(loop.getNumResults())) {
            loop.emitError()
                << "final physical Level lost a birth or handoff identity; expected state="
                << level.getStateBirthCount()
                << ", staged=" << level.getStagedBirthCount()
                << ", handoff=" << level.getHandoffCount()
                << ", observed state=" << actualState
                << ", staged=" << actualStaged
                << ", handoff=" << loop.getNumResults();
            failed = true;
          }
        }
      }
      if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation)) {
        auto level =
            branch->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
        if (!level)
          return;
        if (level.getHandoffCount() !=
            static_cast<int64_t>(branch.getNumResults())) {
          branch.emitError()
              << "pipelined physical Level lost its handoff identity; expected "
              << level.getHandoffCount() << ", observed "
              << branch.getNumResults();
          failed = true;
        }
        bool foundPoint = false;
        branch.getThenRegion().walk([&](riscv::PhysicalPointOp point) {
          auto domain = point.getResult().getType().getDomain();
          if (domain.getDomainId() == level.getDomainId() &&
              domain.getAxisId() == level.getAxisId() &&
              domain.getRelation() == level.getRelation())
            foundPoint = true;
        });
        if (!foundPoint) {
          branch.emitError(
              "pipelined physical Level has no point for its logical domain");
          failed = true;
        }
      }
    });
    for (riscv::KernelOp kernel : getOperation().getOps<riscv::KernelOp>()) {
      if (!kernel.getResourcesMaterialized()) {
        kernel.emitError(
            "final physical program has no closed resource materialization contract");
        failed = true;
      }
      for (mlir::BlockArgument argument : kernel.getBody().front().getArguments()) {
        if (!mlir::isa<riscv::MemDescType>(argument.getType()))
          continue;
        int64_t views = 0;
        for (mlir::OpOperand &use : argument.getUses()) {
          if (mlir::isa<riscv::StrideOp>(use.getOwner()))
            continue; // ABI metadata queries do not access the pinned storage.
          auto view = mlir::dyn_cast<riscv::MemoryViewOp>(use.getOwner());
          if (!view || use.getOperandNumber() != 0) {
            kernel.emitError(
                "kernel ABI descriptor bypasses its explicit runtime memory view");
            failed = true;
          } else {
            ++views;
          }
        }
        const bool noAccess = mlir::cast<mlir::StringAttr>(
            kernel.getArgAccess()[argument.getArgNumber()]).getValue() == "none";
        if (views != 1 && !(views == 0 && noAccess)) {
          kernel.emitError(
              "kernel ABI descriptor requires exactly one explicit runtime memory view");
          failed = true;
        }
      }
      int64_t totalRegisterPeak = kernel.getVectorRegisterPeak();
      auto actual = riscv_internal::physicalRegisterPressure(kernel);
      if (!actual || actual->vector != kernel.getVectorRegisterPeak() ||
          actual->fragment != kernel.getFragmentRegisterPeak()) {
        kernel.emitError("final register residency statistics disagree with physical SSA liveness");
        failed = true;
      }
      if (!checkedAdd(totalRegisterPeak, kernel.getFragmentRegisterPeak()) ||
          totalRegisterPeak > kernel.getTarget().getVectorRegisters()) {
        kernel.emitError("final physical resources exceed the target profile");
        failed = true;
      }
      if (kernel.getLocalStorageBytes() >
          kernel.getTarget().getMaxPrivateStackBytes()) {
        kernel.emitError(
            "final physical local storage exceeds the target profile");
        failed = true;
      }
      auto guardedBytes = riscv_internal::guardedLocalStorageBytes(kernel);
      if (mlir::failed(guardedBytes) || *guardedBytes > kernel.getLocalStorageBytes()) {
        kernel.emitError("final local storage bound disagrees with its lexical capacity guards");
        failed = true;
      }
      kernel.walk([&](mlir::Operation *operation) {
        auto checkType = [&](mlir::Type type) {
          auto value = mlir::dyn_cast<riscv::ValueType>(type);
          if (!value)
            return;
          auto layout = value.getLayout();
          if (layout.getReplicaGroup() > 1 && layout.getValidity() == "mask") {
            operation->emitError("replica-grouped values support only full or rectangular-tail validity");
            failed = true;
            return;
          }
          auto shape = value.getShape().asArrayRef();
          auto time = layout.getTimeFactors().asArrayRef();
          auto lane = layout.getLaneFactors().asArrayRef();
          auto replica = layout.getReplicaFactors().asArrayRef();
          auto fragment = layout.getFragmentFactors().asArrayRef();
          auto local = layout.getLocalFactors().asArrayRef();
          for (size_t index = 0; index < shape.size(); ++index) {
            if (shape[index] < 0)
              continue;
            int64_t factors[] = {time[index], lane[index], replica[index],
                                 fragment[index], local[index]};
            auto represented = riscv_internal::staticProduct(factors);
            if (!represented || *represented != shape[index]) {
              operation->emitError()
                  << "physical layout factors are invalid or do not represent axis "
                  << value.getAxisIds()[index] << " with extent " << shape[index];
              failed = true;
            }
          }
          if (layout.getCarrier() != "rvv")
            return;
          if (!kernel.getTarget().getHasRVV() ||
              kernel.getTarget().getVlenBits() <= 0 ||
              kernel.getTarget().getVectorRegisters() <= 0 ||
              !llvm::is_contained(kernel.getTarget().getSupportedSEW().asArrayRef(),
                                  layout.getSew()) ||
              !llvm::is_contained(
                  kernel.getTarget().getLegalLMULEighths().asArrayRef(),
                                  layout.getLmulEighths()) ||
              (value.getElementType().isF16() &&
               !kernel.getTarget().getHasVectorF16Conversion())) {
            operation->emitError(
                "RVV layout uses a SEW/LMUL outside the target profile");
            failed = true;
            return;
          }
          int64_t vlmax = kernel.getTarget().getVlenBits() *
                          layout.getLmulEighths() /
                          (8 * layout.getSew());
          auto lanes = riscv_internal::staticProduct(lane);
          auto replicas = riscv_internal::staticProduct(replica);
          const int64_t groupsPerValue =
              (layout.getLmulEighths() + 7) / 8;
          const bool groupsOverflow =
              !replicas ||
              *replicas > std::numeric_limits<int64_t>::max() / groupsPerValue;
          const int64_t expectedGroups =
              groupsOverflow ? -1 : groupsPerValue * *replicas / layout.getReplicaGroup();
          if (!lanes || *lanes > std::numeric_limits<int64_t>::max() / layout.getReplicaGroup() ||
              layout.getVl() > vlmax || *lanes * layout.getReplicaGroup() > layout.getVl() ||
              layout.getRegisterGroups() != expectedGroups) {
            operation->emitError(
                "RVV layout lane/vl/register assignment disagrees with the target vector shape: layout=")
                << layout << ", vlmax=" << vlmax << ", lanes="
                << (lanes ? *lanes : -1) << ", expected-groups="
                << expectedGroups;
            failed = true;
          }
        };
        for (mlir::Type type : operation->getResultTypes())
          checkType(type);
        for (mlir::Region &region : operation->getRegions())
          for (mlir::Block &block : region)
            for (mlir::BlockArgument argument : block.getArguments())
              checkType(argument.getType());
        for (mlir::NamedAttribute named : operation->getAttrs()) {
          auto access = mlir::dyn_cast<riscv::AccessAttr>(named.getValue());
          if (!access)
            continue;
          if ((access.getForm() == "indexed" &&
               !kernel.getTarget().getHasIndexedMemory()) ||
              (access.getForm() == "segment" &&
               !kernel.getTarget().getHasSegmentMemory())) {
            operation->emitError()
                << "memory form " << access.getForm()
                << " is unsupported by the final target profile";
            failed = true;
          }
        }
        if (auto laneForm =
                operation->getAttrOfType<mlir::StringAttr>("lane_memory_form")) {
          if ((laneForm.getValue() == "indexed" &&
               !kernel.getTarget().getHasIndexedMemory()) ||
              (laneForm.getValue() == "segment" &&
               !kernel.getTarget().getHasSegmentMemory())) {
            operation->emitError()
                << "lane memory form " << laneForm.getValue()
                << " is unsupported by the final target profile";
            failed = true;
          }
        }
        auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
        if ((leaf || mlir::isa<riscv::NewOp>(operation)) &&
            !kernel.getTarget().getHasVectorF16() &&
            riscv_internal::requiresFullVectorF16(operation)) {
          operation->emitError("selected half-vector operation requires full Zvfh rather than Zvfhmin");
          failed = true;
        }
        if (!leaf)
          return;
        if (leaf.getInstruction() == "rvv.local-gather") {
          auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation);
          auto groups = extract ? riscv_internal::localGatherTemporaryGroups(extract)
                                : std::optional<int64_t>();
          if (!groups || *groups != leaf.getTemporaryGroups()) {
            operation->emitError("local gather coordinate/resource contract is not closed");
            failed = true;
          }
        }
        if (leaf.getInstruction().starts_with("rvv.cmp.bool.")) {
          auto compare = mlir::dyn_cast<riscv::CompareOp>(operation);
          auto groups = compare ? riscv_internal::booleanComparisonTemporaryGroups(compare)
                                : std::optional<int64_t>();
          if (!groups || *groups != leaf.getTemporaryGroups() ||
              leaf.getInstruction() != ("rvv.cmp.bool." + compare.getPredicate()).str() ||
              leaf.getSpelling() != leaf.getInstruction()) {
            operation->emitError("Boolean comparison representation/resources are not closed");
            failed = true;
          }
        }
        if (leaf.getInstruction() == "rvv.f8e4m3fn-to-f32" ||
            leaf.getInstruction() == "rvv.f8e5m2-to-f32" ||
            leaf.getInstruction() == "rvv.bf16-to-f32" ||
            leaf.getInstruction() == "rvv.f32-to-bf16-rne") {
          auto groups = riscv_internal::storageFloatCastTemporaryGroups(operation);
          if (!groups || leaf.getTemporaryGroups() != *groups) {
            operation->emitError("BF16 conversion geometry/resources are not closed");
            failed = true;
          }
        }
        int64_t operandGroups = 0;
        int64_t resultGroups = 0;
        bool resourceOverflow = false;
        for (mlir::Value operand : operation->getOperands())
          resourceOverflow |=
              !checkedAdd(operandGroups, riscv_internal::issueRegisterGroups(operand));
        for (mlir::Value result : operation->getResults())
          resourceOverflow |=
              !checkedAdd(resultGroups, riscv_internal::issueRegisterGroups(result));
        if (resourceOverflow || leaf.getOperandGroups() != operandGroups ||
            leaf.getResultGroups() != resultGroups) {
          operation->emitError()
              << "leaf resource contract is stale: expected operands="
              << operandGroups << ", results=" << resultGroups
              << ", got operands=" << leaf.getOperandGroups()
              << ", results=" << leaf.getResultGroups();
          failed = true;
        }
        if (leaf.getEngine() == "rvv" && !kernel.getTarget().getHasRVV()) {
          operation->emitError("RVV leaf is illegal for a target without RVV");
          failed = true;
        }
        if (requiresIntegerWidening(operation) &&
            !kernel.getTarget().getHasWideningInteger()) {
          operation->emitError(
              "selected leaf requires integer widening unsupported by target");
          failed = true;
        }
        if (requiresFloatWidening(operation) &&
            !kernel.getTarget().getHasWideningFloat()) {
          operation->emitError(
              "selected leaf requires floating widening unsupported by target");
          failed = true;
        }
        auto update = mlir::dyn_cast<riscv::UpdateOp>(operation);
        const bool registerStateResult = update &&
            update.getResult().getType().getLayout().getCarrier() != "local";
        // A local update returns the same memory owner after a write. Its
        // scalar execution engine does not turn that owner into a register.
        // The exact local/state-update leaf is verified above.
        const bool arithmeticResult = registerStateResult || mlir::isa<
            riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp,
            riscv::CompareOp, riscv::SelectOp, riscv::CastOp, riscv::BitcastOp, riscv::NarrowOp, riscv::WidenOp,
            riscv::RVVSplatOp, riscv::RVVCompareSelectOp, riscv::RVVBitmaskDecodeOp,
            riscv::RVVSignedBitmaskReduceOp, riscv::RVVMaskedNegateOp,
            riscv::RVVWidenAddOp, riscv::RVVNarrowShiftRightOp,
            riscv::RVVBitmaskWindowLoadOp,
            riscv::RVVGroupedMacStepOp, riscv::RVVLayeredRecordLoadOp,
            riscv::RVVLayeredWindowOp,
            riscv::RVVLayeredStreamOp, riscv::RVVProjectedLayeredStreamOp,
            riscv::RVVContractStepOp, riscv::RVVTileContractOp, riscv::FMAOp, riscv::RVVFMAIssueOp>(operation);
        for (mlir::Value result : operation->getResults()) {
          if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
            llvm::StringRef carrier = value.getLayout().getCarrier();
            if (arithmeticResult &&
                ((leaf.getEngine() == "rvv" && carrier != "rvv") ||
                 (leaf.getEngine() == "scalar" && carrier != "scalar"))) {
              operation->emitError()
                  << "leaf engine " << leaf.getEngine()
                  << " and result physical carrier " << carrier
                  << " disagree for " << operation->getName();
              failed = true;
            }
          }
          if (mlir::isa<riscv::FragmentType>(result.getType()) &&
              leaf.getEngine() != "ime") {
            operation->emitError(
                "fragment result requires an IME target leaf");
            failed = true;
          }
        }
      });
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createVerifyFinalRISCVPass() {
  return std::make_unique<VerifyFinalRISCVPass>();
}
