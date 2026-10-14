#include "Weft/Target/RISCVPasses.h"
#include "Weft/Target/RISCVPipeline.h"
#include "Weft/Target/RISCVCompilation.h"
#include "Weft/Target/RISCVFragment.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Kernel/IR/SubviewBounds.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <memory>
#include <optional>

using namespace weft;

namespace {

struct EncodingFacts {
  int64_t alignment;
  int64_t storageBits;
  int64_t elements;
  int64_t interleaveRows;
};

class KernelConverter {
public:
  KernelConverter(mlir::ModuleOp module, RISCVCompilerOptions options,
                  riscv::TargetAttr target, mlir::OpBuilder &builder)
      : module(module), options(std::move(options)), target(target), builder(builder) {}

  mlir::LogicalResult run() {
    std::string bindingError;
    if (!options.validateBindings(bindingError))
      return invalidRISCVInput(module, bindingError);
    if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                            options.lmulEighths))
      return invalidRISCVInput(module, "LMUL binding is not supported by the explicit target");
    for (const auto &extension : options.requiredExtensions)
      if (!llvm::any_of(target.getFragments(), [&](mlir::Attribute attribute) {
            auto capability = mlir::cast<riscv::FragmentCapabilityAttr>(attribute);
            const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
            return contract && contract->extension == extension;
          }))
        return invalidRISCVInput(module,
            "required extension is absent from the explicit target capabilities: ") << extension;
    bool hasPhysical = false;
    module.walk([&](riscv::KernelOp) { hasPhysical = true; });
    if (hasPhysical)
      return invalidRISCVInput(module,
          "ConvertWeftToRISCV accepts Canonical Kernel IR, not an existing physical program");

    llvm::SmallVector<kernel::EncodingDeclOp> encodings(
        module.getOps<kernel::EncodingDeclOp>());
    llvm::SmallVector<kernel::DeriveOp> derives(module.getOps<kernel::DeriveOp>());
    llvm::SmallVector<kernel::KernelOp> kernels(module.getOps<kernel::KernelOp>());
    if (kernels.empty())
      return invalidRISCVInput(module, "canonical module contains no Weft kernels");
    if (failed(readSpecializedSourceParameters(kernels)))
      return mlir::failure();

    builder.setInsertionPointToStart(module.getBody());
    for (kernel::EncodingDeclOp encoding : encodings)
      cloneEncoding(encoding);
    for (kernel::DeriveOp derive : derives)
      if (failed(cloneDerivedEncoding(derive)))
        return mlir::failure();
    for (kernel::KernelOp kernel : kernels)
      if (failed(convertKernel(kernel)))
        return mlir::failure();

    for (kernel::KernelOp kernel : kernels)
      kernel.erase();
    for (kernel::DeriveOp derive : derives)
      derive.erase();
    for (kernel::EncodingDeclOp encoding : encodings)
      encoding.erase();
    module->setAttr("weft.riscv.physical",
                    builder.getUnitAttr());
    return mlir::success();
  }

private:
  mlir::LogicalResult readSpecializedSourceParameters(
      llvm::ArrayRef<kernel::KernelOp> kernels) {
    for (kernel::KernelOp kernel : kernels) {
      auto &bindings = sourceBindings[kernel.getOperation()];
      auto walked = kernel.getBody().walk([&](kernel::SymbolOp symbol) {
        if (symbol.getKind() != "source_auto") return mlir::WalkResult::advance();
        if (symbol.getChoices().size() != 1) {
          invalidRISCVInput(symbol, "source auto parameter '") << symbol.getName()
              << "' must be specialized by weft-bind-source-parameters before target conversion";
          return mlir::WalkResult::interrupt();
        }
        int64_t value = symbol.getChoices()[0];
        auto [position, inserted] = bindings.try_emplace(symbol.getName(), value);
        if (!inserted && position->second != value) {
          invalidRISCVInput(symbol, "source auto declarations require conflicting values within one kernel: ")
              << symbol.getName();
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (walked.wasInterrupted()) return mlir::failure();
    }
    return mlir::success();
  }

  void cloneEncoding(kernel::EncodingDeclOp source) {
    mlir::OperationState state(source.getLoc(),
                               riscv::EncodingDeclOp::getOperationName());
    state.addAttributes(source->getAttrs());
    mlir::Operation *operation = builder.create(state);
    riscv_internal::copyOrigin(source, operation);
  }

  mlir::LogicalResult cloneDerivedEncoding(kernel::DeriveOp source) {
    mlir::Block &body = source.getBody().front();
    if (body.getOperations().size() != 2)
      return source.emitError(
          "RISC-V physical ABI requires the declared single interleave derived-encoding program");
    auto interleave = mlir::dyn_cast<kernel::InterleaveOp>(body.front());
    auto yield = mlir::dyn_cast<kernel::DeriveYieldOp>(body.back());
    if (!interleave || !yield || interleave.getView() != body.getArgument(0) ||
        yield.getValue() != interleave.getResult() || interleave.getRows() <= 0)
      return source.emitError(
          "RISC-V physical ABI cannot replace an arbitrary derive body with an interleave artifact builder");
    const int64_t rows = interleave.getRows();
    mlir::OperationState state(source.getLoc(),
                               riscv::DerivedEncodingOp::getOperationName());
    state.addAttribute("sym_name", source.getSymNameAttr());
    state.addAttribute("source_family", source.getSourceFamilyAttr());
    state.addAttribute("result_family", source.getResultFamilyAttr());
    state.addAttribute("layout_identity", source.getLayoutIdentityAttr());
    state.addAttribute("parameter_names", source.getParameterNamesAttr());
    state.addAttribute("parameter_values", source.getParameterValuesAttr());
    state.addAttribute("interleave_rows", builder.getI64IntegerAttr(rows));
    state.addAttribute("field_chunks", interleave.getFieldChunksAttr());
    mlir::Operation *operation = builder.create(state);
    riscv_internal::copyOrigin(source, operation);
    return mlir::success();
  }

  kernel::DeriveOp findCanonicalDerive(kernel::EncodingType encoding) {
    kernel::DeriveOp result;
    module.walk([&](kernel::DeriveOp derive) {
      if (!result && derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derive;
    });
    return result;
  }

  mlir::LogicalResult createArtifactBuilder(kernel::KernelOp sourceKernel,
                                            unsigned argumentIndex,
                                            kernel::EncodingType encoding) {
    kernel::DeriveOp derive = findCanonicalDerive(encoding);
    if (!derive)
      return sourceKernel.emitError()
             << "derived kernel argument has no canonical artifact transform for "
             << encoding;
    kernel::EncodingDeclOp sourceEncoding =
        findCanonicalEncoding(derive.getSourceFamily());
    if (!sourceEncoding || sourceEncoding.getStorageBits() % 8)
      return derive.emitError(
          "artifact packing requires one byte-addressable source Encoding");
    auto interleave = *derive.getBody().getOps<kernel::InterleaveOp>().begin();
    const int64_t rows = interleave.getRows();
    const int64_t recordBytes = sourceEncoding.getStorageBits() / 8;
    const int64_t elements = sourceEncoding.getElements();
    if (rows <= 0 || recordBytes <= 0 || elements <= 0)
      return derive.emitError("artifact packing geometry is incomplete");
    auto spans = kernel::fieldInterleaveSpans(
        sourceEncoding.getStorageBits(), sourceEncoding.getFieldBitOffsets(),
        sourceEncoding.getFieldStorageBits(), sourceEncoding.getPadding(),
        interleave.getFieldChunks());
    if (mlir::failed(spans))
      return derive.emitError("artifact field interleave storage relation is not closed");

    llvm::StringRef argumentName = mlir::cast<mlir::StringAttr>(
        sourceKernel.getArgNames()[argumentIndex]).getValue();
    std::string symbol =
        (sourceKernel.getSymName() + "_" + argumentName).str();
    mlir::Location loc = derive.getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());

    mlir::OperationState state(loc, riscv::ArtifactPackOp::getOperationName());
    state.addAttribute("sym_name", builder.getStringAttr(symbol));
    state.addAttribute("target", this->target);
    state.addAttribute("source_family", derive.getSourceFamilyAttr());
    state.addAttribute("result_family", derive.getResultFamilyAttr());
    state.addAttribute("layout_identity", derive.getLayoutIdentityAttr());
    state.addAttribute("record_bytes", builder.getI64IntegerAttr(recordBytes));
    state.addAttribute("elements", builder.getI64IntegerAttr(elements));
    state.addAttribute("interleave_rows", builder.getI64IntegerAttr(rows));
    state.addAttribute("field_chunks", interleave.getFieldChunksAttr());
    state.addRegion();
    state.addRegion();
    auto artifact = mlir::cast<riscv::ArtifactPackOp>(builder.create(state));
    riscv_internal::copyOrigin(derive, artifact);

    auto *sizeBody = new mlir::Block();
    artifact.getSizeBody().push_back(sizeBody);
    mlir::BlockArgument sizeM =
        sizeBody->addArgument(builder.getIndexType(), loc);
    mlir::BlockArgument sizeK =
        sizeBody->addArgument(builder.getIndexType(), loc);
    builder.setInsertionPointToEnd(sizeBody);
    mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value rowsValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, rows);
    mlir::Value elementsValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, elements);
    mlir::Value bytesValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, recordBytes * rows);
    mlir::Value kIsZero = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, sizeK, zero);
    mlir::Value remainder =
        builder.create<mlir::arith::RemUIOp>(loc, sizeK, elementsValue);
    mlir::Value malformed = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, remainder, zero);
    mlir::Value invalid =
        builder.create<mlir::arith::OrIOp>(loc, kIsZero, malformed);
    auto sizeChoice = builder.create<mlir::scf::IfOp>(
        loc, mlir::TypeRange{builder.getIndexType()}, invalid, true);
    builder.setInsertionPointToStart(&sizeChoice.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(loc, zero);
    builder.setInsertionPointToStart(&sizeChoice.getElseRegion().front());
    mlir::Value rowGroups =
        builder.create<mlir::arith::CeilDivUIOp>(loc, sizeM, rowsValue);
    mlir::Value blocks =
        builder.create<mlir::arith::DivUIOp>(loc, sizeK, elementsValue);
    mlir::Value records =
        builder.create<mlir::arith::MulIOp>(loc, rowGroups, blocks);
    mlir::Value size =
        builder.create<mlir::arith::MulIOp>(loc, records, bytesValue);
    builder.create<mlir::scf::YieldOp>(loc, size);
    builder.setInsertionPointAfter(sizeChoice);
    builder.create<riscv::ArtifactSizeYieldOp>(loc, sizeChoice.getResult(0));

    auto *packBody = new mlir::Block();
    artifact.getPackBody().push_back(packBody);
    auto sourceType = requireConvertedType(
        derive, derive.getBody().front().getArgument(0).getType(), "pinned",
        "read", 0);
    auto canonicalYield =
        mlir::cast<kernel::DeriveYieldOp>(derive.getBody().front().getTerminator());
    auto targetType = requireConvertedType(
        derive, canonicalYield.getValue().getType(), "pinned", "write", 1);
    if (failed(sourceType) || failed(targetType))
      return mlir::failure();
    mlir::BlockArgument source = packBody->addArgument(*sourceType, loc);
    mlir::BlockArgument target = packBody->addArgument(*targetType, loc);
    mlir::BlockArgument packM =
        packBody->addArgument(builder.getIndexType(), loc);
    mlir::BlockArgument packK =
        packBody->addArgument(builder.getIndexType(), loc);
    builder.setInsertionPointToEnd(packBody);
    zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    rowsValue = builder.create<mlir::arith::ConstantIndexOp>(loc, rows);
    elementsValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, elements);
    mlir::Value recordBytesValue =
        builder.create<mlir::arith::ConstantIndexOp>(loc, recordBytes);
    blocks = builder.create<mlir::arith::DivUIOp>(loc, packK, elementsValue);
    rowGroups =
        builder.create<mlir::arith::CeilDivUIOp>(loc, packM, rowsValue);
    auto rgLoop = builder.create<mlir::scf::ForOp>(loc, zero, rowGroups, one);
    rgLoop->setAttr("weft.riscv.direction", builder.getStringAttr("ascending"));
    builder.setInsertionPointToStart(rgLoop.getBody());
    auto kbLoop = builder.create<mlir::scf::ForOp>(loc, zero, blocks, one);
    kbLoop->setAttr("weft.riscv.direction", builder.getStringAttr("ascending"));
    for (const auto &span : *spans) {
    builder.setInsertionPointToStart(kbLoop.getBody());
    mlir::Value fieldBegin =
        builder.create<mlir::arith::ConstantIndexOp>(loc, span.byteOffset);
    mlir::Value fieldEnd = builder.create<mlir::arith::ConstantIndexOp>(
        loc, span.byteOffset + span.byteCount);
    mlir::Value chunk =
        builder.create<mlir::arith::ConstantIndexOp>(loc, span.chunkBytes);
    auto byteLoop =
        builder.create<mlir::scf::ForOp>(loc, fieldBegin, fieldEnd, one);
    byteLoop->setAttr("weft.riscv.direction", builder.getStringAttr("ascending"));
    builder.setInsertionPointToStart(byteLoop.getBody());
    auto laneLoop = builder.create<mlir::scf::ForOp>(loc, zero, rowsValue, one);
    laneLoop->setAttr("weft.riscv.direction", builder.getStringAttr("ascending"));
    builder.setInsertionPointToStart(laneLoop.getBody());

    mlir::Value rowBase = builder.create<mlir::arith::MulIOp>(
        loc, rgLoop.getInductionVar(), rowsValue);
    mlir::Value row = builder.create<mlir::arith::AddIOp>(
        loc, rowBase, laneLoop.getInductionVar());
    mlir::Value sourceRecord = builder.create<mlir::arith::AddIOp>(
        loc, builder.create<mlir::arith::MulIOp>(loc, row, blocks),
        kbLoop.getInductionVar());
    mlir::Value sourceByte = builder.create<mlir::arith::AddIOp>(
        loc, builder.create<mlir::arith::MulIOp>(loc, sourceRecord,
                                                recordBytesValue),
        byteLoop.getInductionVar());
    mlir::Value groupRecord = builder.create<mlir::arith::AddIOp>(
        loc, builder.create<mlir::arith::MulIOp>(loc, rgLoop.getInductionVar(),
                                                blocks),
        kbLoop.getInductionVar());
    mlir::Value fieldByte = builder.create<mlir::arith::SubIOp>(
        loc, byteLoop.getInductionVar(), fieldBegin);
    mlir::Value chunkOffset = builder.create<mlir::arith::MulIOp>(
        loc, builder.create<mlir::arith::DivUIOp>(loc, fieldByte, chunk), chunk);
    mlir::Value byteBase = builder.create<mlir::arith::AddIOp>(
        loc, builder.create<mlir::arith::MulIOp>(loc, groupRecord, recordBytesValue),
        builder.create<mlir::arith::AddIOp>(loc, fieldBegin, chunkOffset));
    mlir::Value cohortByte = builder.create<mlir::arith::MulIOp>(loc, byteBase, rowsValue);
    mlir::Value rowChunk = builder.create<mlir::arith::MulIOp>(
        loc, laneLoop.getInductionVar(), chunk);
    mlir::Value withinChunk = builder.create<mlir::arith::RemUIOp>(loc, fieldByte, chunk);
    mlir::Value rowByte = builder.create<mlir::arith::AddIOp>(loc, rowChunk, withinChunk);
    mlir::Value targetByte = builder.create<mlir::arith::AddIOp>(loc, cohortByte, rowByte);
    mlir::Value inBounds = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ult, row, packM);
    auto byteType = builder.getI8Type();
    auto selectByte = builder.create<mlir::scf::IfOp>(
        loc, mlir::TypeRange{byteType}, inBounds, true);
    builder.setInsertionPointToStart(&selectByte.getThenRegion().front());
    auto load = builder.create<riscv::StorageLoadOp>(
        loc, byteType, source, sourceByte,
        riscv_internal::leaf(builder, "transfer", "artifact-storage",
                             "scalar.storage.load.u8",
                             "scalar.storage.load.u8", 0, 0));
    builder.create<mlir::scf::YieldOp>(loc, load.getResult());
    builder.setInsertionPointToStart(&selectByte.getElseRegion().front());
    mlir::Value zeroByte = builder.create<mlir::arith::ConstantOp>(
        loc, byteType, builder.getIntegerAttr(byteType, 0));
    builder.create<mlir::scf::YieldOp>(loc, zeroByte);
    builder.setInsertionPointAfter(selectByte);
    builder.create<riscv::StorageStoreOp>(
        loc, selectByte.getResult(0), target, targetByte,
        riscv_internal::leaf(builder, "transfer", "artifact-storage",
                             "scalar.storage.store.u8",
                             "scalar.storage.store.u8", 0, 0));
    }
    builder.setInsertionPointToEnd(packBody);
    builder.create<riscv::ArtifactReturnOp>(loc);
    return mlir::success();
  }

  mlir::LogicalResult createArtifactBuilders(kernel::KernelOp source) {
    for (auto [index, argument] :
         llvm::enumerate(source.getBody().front().getArguments())) {
      auto view = mlir::dyn_cast<kernel::ViewType>(argument.getType());
      if (!view)
        continue;
      auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
      if (encoding.getKind() == "derived_instance" &&
          failed(createArtifactBuilder(source, index, encoding)))
        return mlir::failure();
    }
    return mlir::success();
  }

  std::optional<int64_t> denseWidth(kernel::EncodingType encoding) const {
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

  mlir::Type denseElementType(kernel::EncodingType encoding) {
    if (encoding.getKind() != "dense")
      return {};
    llvm::StringRef family = encoding.getFamily();
    if (family == "f8E4M3FN") return mlir::Float8E4M3FNType::get(builder.getContext());
    if (family == "f8E5M2") return mlir::Float8E5M2Type::get(builder.getContext());
    if (family == "f16")
      return builder.getF16Type();
    if (family == "bf16")
      return builder.getBF16Type();
    if (family == "f32")
      return builder.getF32Type();
    if (family == "f64")
      return builder.getF64Type();
    mlir::IntegerType::SignednessSemantics signedness;
    if (family.consume_front("i"))
      signedness = mlir::IntegerType::Signed;
    else if (family.consume_front("u"))
      signedness = mlir::IntegerType::Unsigned;
    else
      return {};
    unsigned width = 0;
    if (family.getAsInteger(10, width) || width == 0)
      return {};
    if (width == 1 && signedness == mlir::IntegerType::Signed)
      return builder.getI1Type();
    return mlir::IntegerType::get(builder.getContext(), width, signedness);
  }

  kernel::EncodingDeclOp findCanonicalEncoding(llvm::StringRef family) {
    kernel::EncodingDeclOp result;
    module.walk([&](kernel::EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
    return result;
  }

  llvm::StringRef baseFamily(kernel::EncodingType encoding) {
    if (encoding.getKind() != "derived_instance")
      return encoding.getFamily();
    llvm::StringRef result;
    module.walk([&](kernel::DeriveOp derive) {
      if (result.empty() && derive.getResultFamily() == encoding.getFamily() &&
          derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derive.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derive.getSourceFamily();
    });
    return result;
  }

  std::optional<EncodingFacts> encodingFacts(kernel::EncodingType encoding) {
    if (auto width = denseWidth(encoding)) {
      // Dense logical predicates occupy one byte each; bit-packed fields are
      // described by an explicit record Encoding instead.
      int64_t storageBits = std::max<int64_t>(8, *width);
      return EncodingFacts{storageBits / 8, storageBits, 1, 0};
    }
    kernel::EncodingDeclOp declaration = findCanonicalEncoding(baseFamily(encoding));
    if (!declaration)
      return std::nullopt;
    EncodingFacts result{static_cast<int64_t>(declaration.getAlignment()),
                         static_cast<int64_t>(declaration.getStorageBits()),
                         static_cast<int64_t>(declaration.getElements()), 0};
    if (encoding.getKind() == "derived_instance")
      module.walk([&](kernel::DeriveOp derive) {
        if (derive.getResultFamily() == encoding.getFamily() &&
            derive.getLayoutIdentity() == encoding.getLayoutIdentity() &&
            derive.getParameterValues() == encoding.getParameters().asArrayRef())
          derive.getBody().walk([&](kernel::InterleaveOp interleave) {
            result.interleaveRows = interleave.getRows();
          });
      });
    if (encoding.getKind() == "derived_instance" && result.interleaveRows <= 0)
      return std::nullopt;
    return result;
  }

  llvm::SmallVector<int64_t>
  descriptorStrides(mlir::DenseI64ArrayAttr shape) const {
    llvm::SmallVector<int64_t> strides(shape.size(), -1);
    int64_t running = 1;
    for (int64_t dimension = static_cast<int64_t>(shape.size()) - 1;
         dimension >= 0; --dimension) {
      strides[dimension] = running;
      int64_t extent = shape[dimension];
      if (extent <= 0 || running < 0)
        running = -1;
      else
        running *= extent;
    }
    return strides;
  }

  mlir::DenseI64ArrayAttr instantiatedShape(mlir::DenseI64ArrayAttr shape) const {
    llvm::SmallVector<int64_t> extents;
    for (int64_t extent : shape.asArrayRef()) {
      auto binding = shapeBindings.find(extent);
      extents.push_back(binding == shapeBindings.end() ? extent : binding->second);
    }
    return riscv_internal::integers(builder, extents);
  }

  mlir::Type convertType(mlir::Type type, llvm::StringRef addressClass = "slice",
                         llvm::StringRef access = "none", int64_t aliasSet = -1) {
    if (auto view = mlir::dyn_cast<kernel::ViewType>(type)) {
      auto encoding = mlir::cast<kernel::EncodingType>(view.getEncoding());
      auto facts = encodingFacts(encoding);
      if (!facts)
        return {};
      auto shape = instantiatedShape(view.getShape());
      llvm::SmallVector<int64_t> strides = descriptorStrides(shape);
      llvm::SmallVector<int64_t> origins(view.getShape().size(), 0);
      return riscv::MemDescType::get(
          type.getContext(), encoding, shape, view.getAxisIds(),
          riscv_internal::integers(builder, strides),
          riscv_internal::integers(builder, origins), facts->alignment,
          addressClass, access, aliasSet, encoding.getLayoutIdentity(),
          facts->storageBits, facts->elements, facts->interleaveRows);
    }
    if (auto slice = mlir::dyn_cast<kernel::SliceType>(type)) {
      auto encoding = mlir::cast<kernel::EncodingType>(slice.getEncoding());
      auto facts = encodingFacts(encoding);
      if (!facts)
        return {};
      auto shape = instantiatedShape(slice.getShape());
      llvm::SmallVector<int64_t> strides = descriptorStrides(shape);
      llvm::SmallVector<int64_t> origins(slice.getShape().size(), 0);
      return riscv::MemDescType::get(
          type.getContext(), encoding, shape, slice.getAxisIds(),
          riscv_internal::integers(builder, strides),
          riscv_internal::integers(builder, origins), facts->alignment, "slice",
          access, aliasSet, encoding.getLayoutIdentity(), facts->storageBits,
          facts->elements, facts->interleaveRows);
    }
    if (auto value = mlir::dyn_cast<kernel::ValueType>(type)) {
      mlir::Type element = value.getElementType();
      if (auto encoding = mlir::dyn_cast<kernel::EncodingType>(element);
          encoding && encoding.getKind() == "dense") {
        element = denseElementType(encoding);
        if (!element)
          return {};
      }
      return riscv::ValueType::get(
          type.getContext(), element, instantiatedShape(value.getShape()),
          value.getAxisIds(), riscv_internal::unassignedLayout(
                                  builder, value.getAxisIds().asArrayRef()));
    }
    if (auto domain = mlir::dyn_cast<kernel::DomainType>(type))
      return riscv::DomainType::get(
          type.getContext(), domain.getAxisName(), domain.getDomainId(),
          domain.getParentDomainId(), domain.getAxisId(), domain.getRelation(),
          domain.getTail());
    if (auto point = mlir::dyn_cast<kernel::PointType>(type))
      return riscv::PointType::get(
          type.getContext(), mlir::cast<riscv::DomainType>(
                                 convertType(point.getDomain())));
    if (type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
      return type;
    return {};
  }

  mlir::FailureOr<mlir::Type>
  requireConvertedType(mlir::Operation *source, mlir::Type type,
                       llvm::StringRef addressClass = "slice",
                       llvm::StringRef access = "none", int64_t aliasSet = -1) {
    mlir::Type converted = convertType(type, addressClass, access, aliasSet);
    if (!converted) {
      source->emitError() << "canonical type has no RISC-V physical representation: "
                          << type;
      return mlir::failure();
    }
    return converted;
  }

  mlir::Value mapped(mlir::Value value) {
    mlir::Value result = mapping.lookupOrNull(value);
    if (!result)
      value.getParentBlock()->getParentOp()->emitError(
          "canonical value has no physical SSA mapping");
    return result;
  }

  mlir::Operation *createPhysical(mlir::Operation *source,
                                  llvm::StringRef name,
                                  mlir::ValueRange operands,
                                  mlir::TypeRange results,
                                  llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    mlir::OperationState state(source->getLoc(), name);
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(attrs);
    mlir::Operation *result = builder.create(state);
    result->setAttr("canonical_op",
                    builder.getStringAttr(source->getName().getStringRef()));
    riscv_internal::copyOrigin(source, result);
    for (auto [oldResult, newResult] :
         llvm::zip(source->getResults(), result->getResults()))
      mapping.map(oldResult, newResult);
    return result;
  }

  llvm::SmallVector<mlir::NamedAttribute>
  copiedAttrs(mlir::Operation *source) {
    llvm::SmallVector<mlir::NamedAttribute> attrs(source->getAttrs());
    return attrs;
  }

  void addLeaf(llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
    attrs.emplace_back(builder.getStringAttr("leaf"),
                       riscv_internal::unselectedLeaf(builder));
  }

  void addAccess(llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
    attrs.emplace_back(builder.getStringAttr("access"),
                       riscv_internal::unassignedAccess(builder));
  }

  void addSchedule(llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs,
                   bool localPipeline) {
    attrs.emplace_back(
        builder.getStringAttr("schedule"),
        riscv_internal::schedule(builder, options.unroll,
                                 localPipeline ? options.pipelineDepth : 1));
  }

  mlir::LogicalResult cloneStandard(mlir::Operation *source) {
    mlir::Operation *clone = source->cloneWithoutRegions(mapping);
    for (auto [oldResult, newResult] :
         llvm::zip(source->getResults(), clone->getResults())) {
      auto type = requireConvertedType(source, oldResult.getType());
      if (failed(type)) {
        clone->destroy();
        return mlir::failure();
      }
      newResult.setType(*type);
      mapping.map(oldResult, newResult);
    }
    builder.insert(clone);
    riscv_internal::copyOrigin(source, clone);
    for (auto [oldRegion, newRegion] :
         llvm::zip(source->getRegions(), clone->getRegions())) {
      for (mlir::Block &oldBlock : oldRegion) {
        auto *newBlock = new mlir::Block();
        newRegion.push_back(newBlock);
        for (mlir::BlockArgument argument : oldBlock.getArguments()) {
          auto type = requireConvertedType(source, argument.getType());
          if (failed(type))
            return mlir::failure();
          mlir::BlockArgument newArgument = newBlock->addArgument(
              *type, argument.getLoc());
          mapping.map(argument, newArgument);
        }
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(newBlock);
        for (mlir::Operation &operation : oldBlock)
          if (failed(convertOperation(&operation)))
            return mlir::failure();
      }
    }
    return mlir::success();
  }

  mlir::LogicalResult convertLevel(kernel::LevelOp source) {
    llvm::SmallVector<mlir::Value> carried;
    llvm::SmallVector<mlir::Type> resultTypes;
    for (mlir::Value value : source.getCarried())
      carried.push_back(mapped(value));
    for (mlir::Type sourceType : source.getResultTypes()) {
      auto type = requireConvertedType(source, sourceType);
      if (failed(type))
        return mlir::failure();
      resultTypes.push_back(*type);
    }

    mlir::OperationState state(source.getLoc(), riscv::LoopOp::getOperationName());
    state.addOperands(mapped(source.getDomain()));
    mlir::Value parentPoint = rootPoint;
    if (auto parentLevel = source->getParentOfType<kernel::LevelOp>())
      parentPoint = mapping.lookupOrNull(parentLevel.getBody().front().getArgument(0));
    if (!parentPoint)
      return source.emitError(
          "canonical Level has no explicit parent point in the physical value map");
    state.addOperands(parentPoint);
    state.addOperands(carried);
    state.addTypes(resultTypes);
    int64_t stateBirthCount = 0, stagedBirthCount = 0;
    for (mlir::Region *region : {&source.getStateBirths(), &source.getStagedBirths(),
                                 &source.getBody()})
      for (mlir::Operation &operation : region->front().without_terminator()) {
        stateBirthCount += mlir::isa<kernel::NewOp>(operation);
        stagedBirthCount += mlir::isa<kernel::MaterializeOp>(operation);
      }
    state.addAttribute("state_birth_count", builder.getI64IntegerAttr(stateBirthCount));
    state.addAttribute("staged_birth_count", builder.getI64IntegerAttr(stagedBirthCount));
    state.addAttribute("handoff_count",
                       builder.getI64IntegerAttr(resultTypes.size()));
    bool innermost = true;
    for (mlir::Operation &nested : source.getBody().front())
      nested.walk([&](kernel::LevelOp) { innermost = false; });
    const int64_t levelAxis =
        mlir::cast<kernel::DomainType>(source.getDomain().getType()).getAxisId();
    bool shapedReductionConsumesLevelAxis = false;
    source.getBody().walk([&](mlir::Operation *operation) {
      auto consumes = [&](mlir::DenseI64ArrayAttr over, mlir::Type resultType) {
        // A shaped contraction owns an explicit element-wise reduction loop in
        // LowerRISCVComposites and consumes the selected unroll there.  A
        // scalar dot instead closes one logical block with an RVV reduction;
        // its surrounding Level remains the owner of iteration unrolling.
        if (over && mlir::isa<kernel::ValueType>(resultType) &&
            llvm::is_contained(over.asArrayRef(), levelAxis))
          shapedReductionConsumesLevelAxis = true;
      };
      if (auto dot = mlir::dyn_cast<kernel::DotOp>(operation))
        consumes(dot.getOverAttr(), dot.getResult().getType());
      else if (auto contract = mlir::dyn_cast<kernel::ContractOp>(operation))
        consumes(contract.getOverAttr(), contract.getResult().getType());
      else if (auto outer = mlir::dyn_cast<kernel::OuterContractOp>(operation))
        consumes(outer.getOverAttr(), outer.getResult().getType());
    });
    const int64_t levelUnroll =
        innermost && !shapedReductionConsumesLevelAxis ? options.unroll : 1;
    const int64_t levelPipelineDepth =
        innermost && !carried.empty() ? options.pipelineDepth : 1;
    state.addAttribute(
        "schedule",
        riscv_internal::schedule(builder, levelUnroll, levelPipelineDepth));
    state.addRegion();
    mlir::Operation *rawLoop = builder.create(state);
    rawLoop->setAttr("canonical_op", builder.getStringAttr("weft_kernel.level"));
    riscv_internal::copyOrigin(source, rawLoop);
    auto loop = mlir::cast<riscv::LoopOp>(rawLoop);
    auto *body = new mlir::Block();
    loop.getBody().push_back(body);
    auto domain = mlir::cast<riscv::DomainType>(loop.getDomain().getType());
    mlir::BlockArgument point = body->addArgument(
        riscv::PointType::get(builder.getContext(), domain), source.getLoc());
    llvm::SmallVector<mlir::BlockArgument> carriedArguments;
    for (mlir::Value value : carried)
      carriedArguments.push_back(
          body->addArgument(value.getType(), source.getLoc()));

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(body);
    llvm::SmallVector<mlir::Value> stateBirths;
    llvm::SmallVector<mlir::Value> stagedBirths;
    const int64_t savedOwnerDomain = currentOwnerDomainId;
    const int64_t domainId = domain.getDomainId();
    currentOwnerDomainId = domainId;
    auto cloneBirths = [&](mlir::Region &region,
                           llvm::SmallVectorImpl<mlir::Value> &values) {
      mlir::Block &sourceBlock = region.front();
      mapping.map(sourceBlock.getArgument(0), point);
      for (mlir::Operation &operation : sourceBlock.without_terminator())
        if (failed(convertOperation(&operation)))
          return mlir::failure();
      auto yield = mlir::cast<kernel::BirthsYieldOp>(sourceBlock.getTerminator());
      for (mlir::Value value : yield.getValues())
        values.push_back(mapped(value));
      return mlir::success();
    };
    if (failed(cloneBirths(source.getStateBirths(), stateBirths)) ||
        failed(cloneBirths(source.getStagedBirths(), stagedBirths))) {
      currentOwnerDomainId = savedOwnerDomain;
      return mlir::failure();
    }

    mlir::Block &sourceBody = source.getBody().front();
    mapping.map(sourceBody.getArgument(0), point);
    size_t cursor = 1;
    for (mlir::BlockArgument argument : carriedArguments)
      mapping.map(sourceBody.getArgument(cursor++), argument);
    for (mlir::Value value : stateBirths)
      mapping.map(sourceBody.getArgument(cursor++), value);
    for (mlir::Value value : stagedBirths)
      mapping.map(sourceBody.getArgument(cursor++), value);
    for (mlir::Operation &operation : sourceBody.without_terminator())
      if (failed(convertOperation(&operation)))
        return mlir::failure();
    auto handoff = mlir::cast<kernel::HandoffOp>(sourceBody.getTerminator());
    llvm::SmallVector<mlir::Value> yielded;
    for (mlir::Value value : handoff.getValues())
      yielded.push_back(mapped(value));
    mlir::OperationState yieldState(source.getLoc(),
                                    riscv::YieldOp::getOperationName());
    yieldState.addOperands(yielded);
    builder.create(yieldState);

    for (auto [oldResult, newResult] : llvm::zip(source.getResults(),
                                                 loop.getResults()))
      mapping.map(oldResult, newResult);
    currentOwnerDomainId = savedOwnerDomain;
    return mlir::success();
  }

  mlir::LogicalResult convertOperation(mlir::Operation *source) {
    if (mlir::isa<mlir::arith::ConstantOp, mlir::arith::CeilDivUIOp,
                  mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::WhileOp,
                  mlir::scf::YieldOp, mlir::scf::ConditionOp>(source))
      return cloneStandard(source);

    if (auto level = mlir::dyn_cast<kernel::LevelOp>(source))
      return convertLevel(level);
    if (mlir::isa<kernel::BirthsYieldOp, kernel::HandoffOp>(source))
      return source->emitError("Level terminator escaped physical loop conversion");

    llvm::SmallVector<mlir::Value> operands;
    llvm::SmallVector<mlir::Type> results;
    for (mlir::Value operand : source->getOperands())
      operands.push_back(mapped(operand));
    for (mlir::Type sourceType : source->getResultTypes()) {
      auto type = requireConvertedType(source, sourceType);
      if (failed(type))
        return mlir::failure();
      results.push_back(*type);
    }
    auto attrs = copiedAttrs(source);
    llvm::StringRef targetName;

    if (auto symbol = mlir::dyn_cast<kernel::SymbolOp>(source)) {
      if (symbol.getKind() == "shape") {
        auto bound = currentBindings.find(symbol.getName());
        if (bound != currentBindings.end()) {
          auto constant = builder.create<mlir::arith::ConstantIndexOp>(symbol.getLoc(), bound->second);
          mapping.map(symbol.getResult(), constant.getResult());
          return mlir::success();
        }
      }
      if (symbol.getKind() == "source_auto") {
        auto found = currentBindings.find(symbol.getName());
        if (found == currentBindings.end())
          return symbol.emitError("source auto escaped complete binding validation");
        auto constant = builder.create<mlir::arith::ConstantIndexOp>(
            symbol.getLoc(), found->second);
        mapping.map(symbol.getResult(), constant.getResult());
        return mlir::success();
      }
      attrs.clear();
      attrs.emplace_back(builder.getStringAttr("name"), symbol.getNameAttr());
      targetName = riscv::SymbolOp::getOperationName();
    } else if (mlir::isa<kernel::LocalShapeOp>(source)) {
      targetName = riscv::LocalShapeOp::getOperationName();
    } else if (mlir::isa<kernel::EmptyOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("owner_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      attrs.emplace_back(builder.getStringAttr("birth_id"),
                         builder.getI64IntegerAttr(nextBirthId++));
      attrs.emplace_back(builder.getStringAttr("lifetime_end_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      targetName = riscv::EmptyOp::getOperationName();
    } else if (mlir::isa<kernel::RootDomainOp>(source)) {
      attrs.clear();
      targetName = riscv::RootDomainOp::getOperationName();
    } else if (mlir::isa<kernel::DomainOp>(source)) {
      attrs.clear();
      targetName = riscv::DomainOp::getOperationName();
    } else if (mlir::isa<kernel::ReturnOp>(source)) {
      attrs.clear();
      targetName = riscv::ReturnOp::getOperationName();
    } else if (mlir::isa<kernel::ConstantOp>(source)) {
      targetName = riscv::ConstantOp::getOperationName();
    } else if (mlir::isa<kernel::IotaOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::IotaOp::getOperationName();
    } else if (auto fresh = mlir::dyn_cast<kernel::NewOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("placement"),
                         builder.getStringAttr(
                             mlir::isa<riscv::ValueType>(results.front())
                                 ? "register"
                                 : "scalar"));
      attrs.emplace_back(builder.getStringAttr("owner_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      attrs.emplace_back(builder.getStringAttr("birth_id"),
                         builder.getI64IntegerAttr(nextBirthId++));
      attrs.emplace_back(builder.getStringAttr("lifetime_end_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      targetName = riscv::NewOp::getOperationName();
    } else if (mlir::isa<kernel::MaterializeOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("placement"),
                         builder.getStringAttr("shared"));
      attrs.emplace_back(builder.getStringAttr("schema"),
                         builder.getStringAttr("value-share"));
      attrs.emplace_back(builder.getStringAttr("owner_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      attrs.emplace_back(builder.getStringAttr("birth_id"),
                         builder.getI64IntegerAttr(nextBirthId++));
      attrs.emplace_back(builder.getStringAttr("lifetime_end_domain_id"),
                         builder.getI64IntegerAttr(currentOwnerDomainId));
      targetName = riscv::MaterializeOp::getOperationName();
    } else if (auto slice = mlir::dyn_cast<kernel::SliceOp>(source)) {
      auto base = mlir::cast<riscv::MemDescType>(operands.front().getType());
      auto result = mlir::cast<riscv::MemDescType>(results.front());
      llvm::SmallVector<int64_t> projectedStrides;
      llvm::SmallVector<int64_t> projectedOrigins;
      for (size_t dimension = 0; dimension < base.getShape().size(); ++dimension) {
        llvm::StringRef kind = "all";
        if (dimension < slice.getSelectors().size())
          kind = mlir::cast<mlir::StringAttr>(slice.getSelectors()[dimension])
                     .getValue();
        if (kind == "index" || kind == "group_index")
          continue;
        if (kind != "all" && kind != "domain")
          return slice.emitError(
              "RISC-V memory descriptor projection does not support this selector kind");
        projectedStrides.push_back(base.getStrides()[dimension]);
        projectedOrigins.push_back(base.getOrigins()[dimension]);
      }
      if (projectedStrides.size() != result.getShape().size())
        return slice.emitError(
            "slice selectors do not define one physical stride per result axis");
      results.front() = riscv::MemDescType::get(
          builder.getContext(), result.getEncoding(), result.getShape(),
          result.getAxisIds(), riscv_internal::integers(builder, projectedStrides),
          riscv_internal::integers(builder, projectedOrigins),
          base.getAlignment(), "slice", base.getAccess(), base.getAliasSet(),
          base.getLayoutIdentity(), base.getStorageBits(), base.getElements(),
          base.getInterleaveRows());
      targetName = riscv::SliceOp::getOperationName();
    } else if (auto subview = mlir::dyn_cast<kernel::SubviewOp>(source)) {
      auto base = mlir::cast<riscv::MemDescType>(operands.front().getType());
      auto result = mlir::cast<riscv::MemDescType>(results.front());
      results.front() = riscv::MemDescType::get(
          builder.getContext(), result.getEncoding(), result.getShape(),
          result.getAxisIds(), base.getStrides(), base.getOrigins(),
          base.getAlignment(), "slice", base.getAccess(), base.getAliasSet(),
          base.getLayoutIdentity(), base.getStorageBits(), base.getElements(),
          base.getInterleaveRows());
      targetName = riscv::SubviewOp::getOperationName();
    } else if (mlir::isa<kernel::ExtentOp>(source)) {
      targetName = riscv::ExtentOp::getOperationName();
    } else if (mlir::isa<kernel::ReshapeOp>(source)) {
      targetName = riscv::ReshapeOp::getOperationName();
    } else if (auto admit = mlir::dyn_cast<kernel::AdmitOp>(source)) {
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::LoadOp::getOperationName();
    } else if (auto commit = mlir::dyn_cast<kernel::CommitOp>(source)) {
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::StoreOp::getOperationName();
    } else if (mlir::isa<kernel::AtomicRMWOp>(source)) {
      attrs.push_back(builder.getNamedAttr("leaf", riscv_internal::leaf(
          builder, "scalar", "atomic-rmw", "scalar.atomic-add",
          "scalar.atomic-add", 0, 0)));
      targetName = riscv::AtomicRMWOp::getOperationName();
    } else if (mlir::isa<kernel::FieldOp>(source)) {
      auto owner =
          mlir::dyn_cast<riscv::MemDescType>(operands.front().getType());
      auto result = mlir::dyn_cast<riscv::MemDescType>(results.front());
      if (owner && result) {
        auto width = denseWidth(
            mlir::cast<kernel::EncodingType>(result.getEncoding()));
        if (!width)
          return source->emitError(
              "field projection requires its declared dense logical element width");
        // A field descriptor still names bits inside its owner's Encoding.
        // It is not a standalone byte-carried dense allocation: the selected
        // field access/store plan owns the actual record bytes and bit layout.
        results.front() = riscv::MemDescType::get(
            builder.getContext(), result.getEncoding(), result.getShape(),
            result.getAxisIds(), result.getStrides(), result.getOrigins(),
            result.getAlignment(), result.getAddressClass(), owner.getAccess(),
            owner.getAliasSet(), result.getLayoutIdentity(),
            *width, result.getElements(),
            result.getInterleaveRows());
      }
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::FieldOp::getOperationName();
    } else if (mlir::isa<kernel::ExtractOp>(source)) {
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::ExtractOp::getOperationName();
    } else if (mlir::isa<kernel::UpdateOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::UpdateOp::getOperationName();
    } else if (mlir::isa<kernel::UnaryOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::UnaryOp::getOperationName();
    } else if (mlir::isa<kernel::BinaryOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::BinaryOp::getOperationName();
    } else if (mlir::isa<kernel::CompareOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::CompareOp::getOperationName();
    } else if (mlir::isa<kernel::SelectOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::SelectOp::getOperationName();
    } else if (mlir::isa<kernel::BitcastOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::BitcastOp::getOperationName();
    } else if (mlir::isa<kernel::CastOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::CastOp::getOperationName();
    } else if (mlir::isa<kernel::NarrowOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::NarrowOp::getOperationName();
    } else if (mlir::isa<kernel::WidenOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::WidenOp::getOperationName();
    } else if (mlir::isa<kernel::MacGroupsOp>(source)) {
      addSchedule(attrs, true);
      addLeaf(attrs);
      targetName = riscv::MacGroupsOp::getOperationName();
    } else if (mlir::isa<kernel::ReduceOp>(source)) {
      addLeaf(attrs);
      targetName = riscv::ReduceOp::getOperationName();
    } else if (mlir::isa<kernel::Fold2Op>(source)) {
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::Fold2Op::getOperationName();
    } else if (mlir::isa<kernel::DotOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("lane_operand"),
                         builder.getStringAttr("unassigned"));
      attrs.emplace_back(builder.getStringAttr("lane_memory_form"),
                         builder.getStringAttr("unassigned"));
      addSchedule(attrs, riscv_internal::logicalElement(results.front()).isF32() ||
                         riscv_internal::logicalElement(results.front()).isF64());
      addLeaf(attrs);
      targetName = riscv::DotOp::getOperationName();
    } else if (mlir::isa<kernel::ContractOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("lane_operand"),
                         builder.getStringAttr("unassigned"));
      attrs.emplace_back(builder.getStringAttr("lane_memory_form"),
                         builder.getStringAttr("unassigned"));
      addSchedule(attrs, riscv_internal::logicalElement(results.front()).isF32() ||
                         riscv_internal::logicalElement(results.front()).isF64());
      addLeaf(attrs);
      targetName = riscv::ContractOp::getOperationName();
    } else if (mlir::isa<kernel::OuterContractOp>(source)) {
      attrs.emplace_back(builder.getStringAttr("lane_operand"),
                         builder.getStringAttr("unassigned"));
      attrs.emplace_back(builder.getStringAttr("lane_memory_form"),
                         builder.getStringAttr("unassigned"));
      addSchedule(attrs, riscv_internal::logicalElement(results.front()).isF32() ||
                         riscv_internal::logicalElement(results.front()).isF64());
      addLeaf(attrs);
      targetName = riscv::OuterContractOp::getOperationName();
    } else if (mlir::isa<kernel::LookupOp>(source)) {
      addAccess(attrs);
      addLeaf(attrs);
      targetName = riscv::LookupOp::getOperationName();
    } else {
      return source->emitError(
          "ConvertWeftToRISCV has no physical operation for this canonical op");
    }

    const bool bfloatPointwise =
        (mlir::isa<kernel::BinaryOp, kernel::CompareOp>(source) ||
         (mlir::isa<kernel::UnaryOp>(source) &&
          (mlir::cast<kernel::UnaryOp>(source).getKind() == "neg" ||
           mlir::cast<kernel::UnaryOp>(source).getKind() == "abs"))) &&
        !operands.empty() &&
        riscv_internal::logicalElement(operands.front().getType()).isBF16();
    mlir::Type originalResult;
    if (bfloatPointwise) {
      auto widenedType = [&](mlir::Type type) -> mlir::Type {
        if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
          return riscv::ValueType::get(builder.getContext(), builder.getF32Type(),
                                      value.getShape(), value.getAxisIds(),
                                      value.getLayout());
        return builder.getF32Type();
      };
      for (mlir::Value &operand : operands) {
        auto cast = builder.create<riscv::CastOp>(
            source->getLoc(), widenedType(operand.getType()), operand,
            riscv_internal::unselectedLeaf(builder));
        riscv_internal::copyOrigin(source, cast);
        operand = cast.getResult();
      }
      if (riscv_internal::logicalElement(results.front()).isBF16()) {
        originalResult = results.front();
        results.front() = widenedType(originalResult);
      }
    }
    mlir::Operation *physical =
        createPhysical(source, targetName, operands, results, attrs);
    if (originalResult) {
      auto rounded = builder.create<riscv::CastOp>(
          source->getLoc(), originalResult, physical->getResult(0),
          riscv_internal::unselectedLeaf(builder));
      riscv_internal::copyOrigin(source, rounded);
      mapping.map(source->getResult(0), rounded.getResult());
    }
    if (mlir::isa<kernel::RootDomainOp>(source)) {
      auto domain = mlir::cast<riscv::RootDomainOp>(physical);
      auto point = builder.create<riscv::RootPointOp>(
          source->getLoc(), riscv::PointType::get(builder.getContext(),
                                                  domain.getResult().getType()),
          domain.getResult());
      riscv_internal::copyOrigin(source, point);
      rootPoint = point.getResult();
    }
    return mlir::success();
  }

  mlir::LogicalResult convertKernel(kernel::KernelOp source) {
    mapping.clear();
    shapeBindings.clear();
    currentBindings.clear();
    rootPoint = {};
    currentOwnerDomainId = 0;
    nextBirthId = 0;
    llvm::SmallVector<std::string> parameterNames;
    llvm::SmallVector<int64_t> parameterValues;
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>> orderedBindings;
    auto resolved = sourceBindings.find(source.getOperation());
    if (resolved == sourceBindings.end())
      return source.emitError("kernel escaped complete source binding validation");
    currentBindings = resolved->second;
    for (auto [index, attribute] : llvm::enumerate(source.getShapeSymbols())) {
      auto name = mlir::cast<mlir::StringAttr>(attribute).getValue();
      auto bound = currentBindings.find(name);
      if (bound != currentBindings.end())
        shapeBindings[-static_cast<int64_t>(index) - 1] = bound->second;
    }
    for (auto &entry : currentBindings)
      orderedBindings.emplace_back(entry.first(), entry.second);
    llvm::sort(orderedBindings, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    for (const auto &entry : orderedBindings) {
      parameterNames.push_back(entry.first.str());
      parameterValues.push_back(entry.second);
    }

    mlir::OperationState state(source.getLoc(), riscv::KernelOp::getOperationName());
    state.addAttribute("sym_name", source.getSymNameAttr());
    state.addAttribute("arg_names", source.getArgNamesAttr());
    state.addAttribute("arg_access", source.getArgAccessAttr());
    state.addAttribute("arg_alias_sets", source.getArgAliasSetsAttr());
    if (auto declared = source.getStridedArgsAttr())
      state.addAttribute("strided_args", declared);
    state.addAttribute("shape_symbols", source.getShapeSymbolsAttr());
    state.addAttribute("source", source.getSourceAttr());
    state.addAttribute("target",
                       target);
    auto requirements = options.requiredExtensions;
    llvm::sort(requirements);
    requirements.erase(std::unique(requirements.begin(), requirements.end()), requirements.end());
    state.addAttribute("required_extensions", riscv_internal::strings(builder, requirements));
    state.addAttribute("local_tile_strategy", builder.getStringAttr(options.localTileStrategy));
    state.addAttribute("pointwise_issue_copies", builder.getI64IntegerAttr(options.pointwiseIssueCopies));
    state.addAttribute("contraction_column_copies", builder.getI64IntegerAttr(options.contractionColumnCopies));
    state.addAttribute("panel_transfer_fields", builder.getI64IntegerAttr(options.panelTransferFields));
    state.addAttribute("panel_transfer_strategy", builder.getStringAttr(options.panelTransferStrategy));
    state.addAttribute("contraction_panel_k", builder.getI64IntegerAttr(options.contractionPanelK));
    state.addAttribute("contraction_k_steps", builder.getI64IntegerAttr(options.contractionKSteps));
    state.addAttribute("parameter_names",
                       riscv_internal::strings(builder, parameterNames));
    state.addAttribute("parameter_values",
                       riscv_internal::integers(builder, parameterValues));
    state.addAttribute("vector_register_peak", builder.getI64IntegerAttr(0));
    state.addAttribute("fragment_register_peak", builder.getI64IntegerAttr(0));
    state.addAttribute("local_storage_bytes", builder.getI64IntegerAttr(0));
    state.addAttribute("resources_materialized", builder.getBoolAttr(false));
    state.addRegion();
    mlir::Operation *rawKernel = builder.create(state);
    riscv_internal::copyOrigin(source, rawKernel);
    auto physical = mlir::cast<riscv::KernelOp>(rawKernel);
    auto *body = new mlir::Block();
    physical.getBody().push_back(body);
    mlir::Block &sourceBody = source.getBody().front();
    for (auto [index, argument] : llvm::enumerate(sourceBody.getArguments())) {
      llvm::StringRef access = mlir::cast<mlir::StringAttr>(
                                   source.getArgAccess()[index])
                                   .getValue();
      auto type = requireConvertedType(
          source, argument.getType(), "pinned", access,
          source.getArgAliasSets()[index]);
      if (failed(type))
        return mlir::failure();
      if (source.getStridedArgsAttr() &&
          llvm::is_contained(source.getStridedArgsAttr(), source.getArgNames()[index])) {
        auto view = mlir::cast<riscv::MemDescType>(*type);
        llvm::SmallVector<int64_t> dynamic(view.getShape().size(), -1);
        type = riscv::MemDescType::get(
            builder.getContext(), view.getEncoding(), view.getShape(), view.getAxisIds(),
            riscv_internal::integers(builder, dynamic), view.getOrigins(), view.getAlignment(),
            view.getAddressClass(), view.getAccess(), view.getAliasSet(), view.getLayoutIdentity(),
            view.getStorageBits(), view.getElements(), view.getInterleaveRows());
      }
      mlir::BlockArgument mappedArgument =
          body->addArgument(*type, argument.getLoc());
      mapping.map(argument, mappedArgument);
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(body);
    for (mlir::Operation &operation : sourceBody)
      if (failed(convertOperation(&operation)))
        return mlir::failure();
    if (failed(materializeEntryMemoryViews(source, physical)))
      return mlir::failure();
    if (failed(createArtifactBuilders(source)))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult materializeEntryMemoryViews(kernel::KernelOp source,
                                                  riscv::KernelOp physical) {
    mlir::Block &sourceBody = source.getBody().front();
    mlir::Block &targetBody = physical.getBody().front();
    llvm::DenseMap<int64_t, mlir::Value> shapeExtents;
    for (mlir::Operation &operation : sourceBody) {
      auto symbol = mlir::dyn_cast<kernel::SymbolOp>(operation);
      if (!symbol || symbol.getKind() != "shape")
        continue;
      auto found = llvm::find(source.getShapeSymbols(), symbol.getNameAttr());
      if (found == source.getShapeSymbols().end())
        return symbol.emitError("shape symbol is absent from the kernel ABI");
      shapeExtents[static_cast<int64_t>(
          found - source.getShapeSymbols().begin()) + 1] = mapped(symbol.getResult());
    }

    mlir::Operation *insertion = nullptr;
    for (mlir::Operation &operation : targetBody)
      if (mlir::isa<riscv::RootDomainOp>(operation)) {
        insertion = &operation;
        break;
      }
    if (!insertion)
      return physical.emitError(
          "physical kernel has no root-domain boundary for ABI descriptor materialization");

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(insertion);
    for (auto [index, sourceArgument] :
         llvm::enumerate(sourceBody.getArguments())) {
      mlir::BlockArgument base = targetBody.getArgument(index);
      auto descriptor = mlir::dyn_cast<riscv::MemDescType>(base.getType());
      if (!descriptor)
        continue; // Scalar parameters already have their entry SSA binding.
      llvm::SmallVector<mlir::Value> extents;
      llvm::SmallVector<mlir::Value> strides(descriptor.getShape().size());
      llvm::SmallVector<mlir::Value> origins;
      for (auto [dimension, extent] :
           llvm::enumerate(descriptor.getShape().asArrayRef())) {
        if (extent > 0) {
          extents.push_back(builder.create<mlir::arith::ConstantIndexOp>(
              sourceArgument.getLoc(), extent));
        } else {
          int64_t axis = descriptor.getAxisIds()[dimension];
          mlir::Value dynamic = shapeExtents.lookup(-extent);
          if (!dynamic)
            return physical.emitError()
                   << "dynamic descriptor axis " << axis
                   << " has no explicit shape SSA value";
          extents.push_back(dynamic);
        }
        origins.push_back(builder.create<mlir::arith::ConstantIndexOp>(
            sourceArgument.getLoc(), 0));
      }
      mlir::Value running = builder.create<mlir::arith::ConstantIndexOp>(
          sourceArgument.getLoc(), 1);
      bool strided = source.getStridedArgsAttr() &&
                     llvm::is_contained(source.getStridedArgsAttr(), source.getArgNames()[index]);
      for (int64_t dimension = static_cast<int64_t>(extents.size()) - 1;
           dimension >= 0; --dimension) {
        strides[dimension] = strided
            ? builder.create<riscv::StrideOp>(sourceArgument.getLoc(), builder.getIndexType(),
                                              base, dimension).getResult()
            : running;
        if (!strided && dimension > 0)
          running = builder.create<mlir::arith::MulIOp>(
              sourceArgument.getLoc(), running, extents[dimension]);
      }
      auto view = builder.create<riscv::MemoryViewOp>(
          sourceArgument.getLoc(), descriptor, base, extents, strides, origins);
      riscv_internal::copyOrigin(source, view);
      for (mlir::OpOperand &use : llvm::make_early_inc_range(base.getUses()))
        if (use.getOwner() != view.getOperation() && !mlir::isa<riscv::StrideOp>(use.getOwner()))
          use.set(view.getResult());
      mapping.map(sourceArgument, view.getResult());
    }
    return mlir::success();
  }

  mlir::ModuleOp module;
  RISCVCompilerOptions options;
  riscv::TargetAttr target;
  mlir::OpBuilder &builder;
  mlir::IRMapping mapping;
  llvm::DenseMap<int64_t, int64_t> shapeBindings;
  llvm::StringMap<int64_t> currentBindings;
  llvm::DenseMap<mlir::Operation *, llvm::StringMap<int64_t>> sourceBindings;
  mlir::Value rootPoint;
  int64_t currentOwnerDomainId = 0;
  int64_t nextBirthId = 0;
};

class ConvertWeftToRISCVPass
    : public mlir::PassWrapper<ConvertWeftToRISCVPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  ConvertWeftToRISCVPass() = default;
  ConvertWeftToRISCVPass(const ConvertWeftToRISCVPass &other)
      : PassWrapper(other), hasSourceBindings(other.hasSourceBindings) {
    copyOptionValuesFrom(&other);
  }
  ConvertWeftToRISCVPass(llvm::StringRef targetText,
                         const RISCVCompilerOptions &options) {
    target = RISCVPipelineTarget{targetText.str()};
    hasSourceBindings = !options.metaBindings.empty();
    requirements = options.requiredExtensions;
    localTileStrategy = options.localTileStrategy;
    lmul = options.lmulEighths;
    unroll = options.unroll;
    pipeline = options.pipelineDepth;
    scalarPrime = options.scalarLoadPrime;
    pointwiseCopies = options.pointwiseIssueCopies;
    columnCopies = options.contractionColumnCopies;
    transferFields = options.panelTransferFields;
    transferStrategy = options.panelTransferStrategy;
    panelK = options.contractionPanelK;
    kSteps = options.contractionKSteps;
  }

  llvm::StringRef getArgument() const override {
    return "weft-convert-to-riscv";
  }
  llvm::StringRef getDescription() const override {
    return "Convert Canonical Kernel IR into one typed RISC-V Physical IR program";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<kernel::WEFTKernelDialect, riscv::WEFTRISCVDialect,
                    mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    if (hasSourceBindings) {
      invalidRISCVInput(getOperation(),
          "target conversion cannot accept source bindings; run weft-bind-source-parameters first");
      signalPassFailure();
      return;
    }
    auto parsed = mlir::dyn_cast_or_null<riscv::TargetAttr>(
        mlir::parseAttribute(target.getValue().text, &getContext()));
    if (!parsed) {
      getOperation().emitError("conversion requires a complete verified #weft_riscv.target attribute");
      signalPassFailure();
      return;
    }
    RISCVCompilerOptions options;
    options.requiredExtensions.assign(requirements.begin(), requirements.end());
    options.localTileStrategy = localTileStrategy;
    options.lmulEighths = lmul;
    options.unroll = unroll;
    options.pipelineDepth = pipeline;
    options.scalarLoadPrime = scalarPrime;
    options.pointwiseIssueCopies = pointwiseCopies;
    options.contractionColumnCopies = columnCopies;
    options.panelTransferFields = transferFields;
    options.panelTransferStrategy = transferStrategy;
    options.contractionPanelK = panelK;
    options.contractionKSteps = kSteps;
    mlir::OpBuilder builder(&getContext());
    KernelConverter converter(getOperation(), std::move(options), parsed, builder);
    if (failed(converter.run()))
      signalPassFailure();
  }

private:
  bool hasSourceBindings = false;
  Option<RISCVPipelineTarget> target{*this, "target",
      llvm::cl::desc("Complete #weft_riscv.target attribute")};
  ListOption<std::string> requirements{*this, "require-extension",
      llvm::cl::desc("Required selected extension instructions")};
  Option<std::string> localTileStrategy{*this, "local-tile-strategy",
      llvm::cl::init("target-default")};
  Option<int64_t> lmul{*this, "lmul-eighths", llvm::cl::init(8)};
  Option<int64_t> unroll{*this, "unroll", llvm::cl::init(1)};
  Option<int64_t> pipeline{*this, "pipeline-depth", llvm::cl::init(1)};
  Option<int64_t> scalarPrime{*this, "scalar-load-prime", llvm::cl::init(0)};
  Option<int64_t> pointwiseCopies{*this, "pointwise-issue-copies", llvm::cl::init(1)};
  Option<int64_t> columnCopies{*this, "contraction-column-copies", llvm::cl::init(4)};
  Option<int64_t> transferFields{*this, "panel-transfer-fields", llvm::cl::init(4)};
  Option<std::string> transferStrategy{*this, "panel-transfer-strategy", llvm::cl::init("issue-work")};
  Option<int64_t> panelK{*this, "contraction-panel-k", llvm::cl::init(128)};
  Option<int64_t> kSteps{*this, "contraction-k-steps", llvm::cl::init(4)};
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createConvertWeftToRISCVPass(RISCVCompilerOptions options) {
  return std::make_unique<ConvertWeftToRISCVPass>(
      serializeRISCVPipelineTarget(options.target), options);
}

std::unique_ptr<mlir::Pass> weft::createConvertWeftToRISCVPass(
    llvm::StringRef serializedTarget, const RISCVCompilerOptions &options) {
  return std::make_unique<ConvertWeftToRISCVPass>(serializedTarget, options);
}
