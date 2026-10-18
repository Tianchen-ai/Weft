#include "Emitter.h"
#include "llvm/Support/SaveAndRestore.h"

namespace weft::riscv_emission {

mlir::LogicalResult Emitter::validatePhysicalProgram() {
  bool invalid = false;
  kernel.walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults()) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(result.getType())) {
        auto layout = value.getLayout();
        if (!layout || layout.getCarrier() == "unassigned" ||
            (layout.getCarrier() == "rvv" &&
             (layout.getSew() <= 0 || layout.getLmulEighths() <= 0 ||
              layout.getVl() <= 0))) {
          operation->emitError("terminal emission received an incomplete physical value layout");
          invalid = true;
        }
      }
    }
    if (auto leaf = leafOf(operation); leaf &&
        (leaf.getEngine() == "unselected" || leaf.getInstruction().empty() ||
         leaf.getSpelling().empty())) {
      operation->emitError("terminal emission received an incomplete target leaf");
      invalid = true;
    }
    if (auto access = operation->getAttrOfType<riscv::AccessAttr>("access");
        access && access.getForm() == "unassigned") {
      operation->emitError("terminal emission received an unplanned memory edge");
      invalid = true;
    }
  });
  return mlir::failure(invalid);
}

mlir::LogicalResult Emitter::compileOperation(mlir::Operation &operation) {
  llvm::SaveAndRestore<mlir::Operation *> current(currentOperation, &operation);
  if (auto assumption = mlir::dyn_cast<riscv::AssumeOp>(operation)) {
    auto predicate = bindings.lookup(assumption.getCondition());
    if (predicate.kind != Binding::Kind::Scalar)
      return fail(assumption, "source assumption requires its scalar predicate");
    line("if (!(" + predicate.scalar + ")) __builtin_unreachable();");
    return mlir::success();
  }
  if (auto cast = mlir::dyn_cast<riscv::PointerCastOp>(operation)) {
    auto origin = bindings.lookup(cast.getInput());
    if (origin.kind != Binding::Kind::Memory) return fail(cast, "pointer cast requires its bound storage origin");
    origin.memory.encoding = cast.getResult().getType().getEncoding();
    origin.memory.elementType = *denseElementType(origin.memory.encoding);
    bindings[cast.getResult()] = std::move(origin);
    return mlir::success();
  }
  if (mlir::isa<riscv::PointerLoadOp, riscv::PointerStoreOp>(operation))
    return compilePointerMemory(&operation);
  if (auto atomic = mlir::dyn_cast<riscv::PointerAtomicOp>(operation))
    return compilePointerAtomic(atomic);
  if (auto helper = mlir::dyn_cast<riscv::HelperOp>(operation))
    return compileHelper(helper);
  if (auto ret = mlir::dyn_cast<riscv::HelperReturnOp>(operation))
    return compileHelperReturn(ret);
  if (auto shape = mlir::dyn_cast<riscv::LocalShapeOp>(operation)) {
    auto input = bindings.lookup(shape.getExtent());
    if (input.kind != Binding::Kind::Scalar)
      return fail(shape, "local shape requires its dominating scalar extent");
    std::string name = fresh("local_extent");
    line("const size_t " + name + " = " + input.scalar + ";");
    line("(void)" + name + ";");
    bindings[shape.getResult()] = scalar(name);
    return mlir::success();
  }
  if (auto size = mlir::dyn_cast<riscv::LocalSizeOp>(operation))
    return compileLocalSize(size);
  if (auto read = mlir::dyn_cast<riscv::FieldReadOp>(operation))
    return compileFieldRead(read);
  auto bindConstant = [&](mlir::Attribute value,
                          mlir::Value result) -> mlir::LogicalResult {
    std::string expression;
    mlir::Type element = riscv_internal::logicalElement(result.getType());
    if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(value)) {
      const llvm::APFloat &number = floating.getValue();
      if (usesRawFloatStorage(element))
        expression = std::to_string(number.bitcastToAPInt().getZExtValue());
      else if (number.isInfinity())
        expression = number.isNegative() ? "(-INFINITY)" : "INFINITY";
      else if (number.isNaN())
        expression = "NAN";
      else {
        llvm::SmallString<32> text;
        number.toString(text);
        expression = text.str().str();
        if (expression.find_first_of(".eEpP") == std::string::npos)
          expression += ".0";
        if (element.isF32())
          expression += "f";
        else if (element.isF16())
          expression = "((_Float16)(" + expression + "f))";
      }
    } else if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value);
               integer && integer.getValue().getBitWidth() == 64 && integer.getValue().isNegative()) {
      auto type = scalarCType(element);
      if (!type) return fail(&operation, "integer constant has no selected scalar carrier");
      llvm::SmallString<32> digits;
      integer.getValue().toString(digits, 16, false);
      // Decimal INT64_MIN is parsed by C as an out-of-range positive literal
      // followed by unary minus. Spell the exact bits in a typed carrier.
      expression = "((" + *type + ")UINT64_C(0x" + digits.str().str() + "))";
    } else {
      llvm::raw_string_ostream stream(expression);
      value.print(stream);
      stream.flush();
      size_t typeMarker = expression.find(" : ");
      if (typeMarker != std::string::npos)
        expression.resize(typeMarker);
    }
    bindings[result] = scalar(expression);
    return mlir::success();
  };
  if (auto stride = mlir::dyn_cast<riscv::StrideOp>(operation)) {
    auto argument = mlir::cast<mlir::BlockArgument>(stride.getView());
    bindings[stride.getResult()] = scalar(strideArgumentName(argument.getArgNumber(), stride.getAxis()));
    return mlir::success();
  }
  if (auto disjoint = mlir::dyn_cast<riscv::DenseDisjointOp>(operation))
    return compileDenseDisjoint(disjoint);
  if (auto alias = mlir::dyn_cast<riscv::DenseElementwiseAliasOp>(operation))
    return compileDenseElementwiseAlias(alias);
  if (auto injective = mlir::dyn_cast<riscv::DenseInjectiveOp>(operation))
    return compileDenseInjective(injective);
  if (auto guard = mlir::dyn_cast<riscv::DenseRowWindowGuardOp>(operation))
    return compileDenseRowWindowGuard(guard);
  if (auto compare = mlir::dyn_cast<riscv::RVVPredicateCompareOp>(operation))
    return compilePredicateCompare(compare);
  if (auto logic = mlir::dyn_cast<riscv::RVVPredicateLogicOp>(operation))
    return compilePredicateLogic(logic);
  if (auto splat = mlir::dyn_cast<riscv::RVVPredicateSplatOp>(operation))
    return compilePredicateSplat(splat);
  if (auto ingress = mlir::dyn_cast<riscv::RVVPredicateFromValueOp>(operation))
    return compilePredicateFromValue(ingress);
  if (auto egress = mlir::dyn_cast<riscv::RVVPredicateToValueOp>(operation))
    return compilePredicateToValue(egress);
  if (auto select = mlir::dyn_cast<riscv::RVVPredicateSelectOp>(operation))
    return compilePredicateSelect(select);
  if (auto symbol = mlir::dyn_cast<riscv::SymbolOp>(operation)) {
    if (auto binding = autoBindings.find(symbol.getName());
        binding != autoBindings.end()) {
      bindings[symbol.getResult()] = scalar(std::to_string(binding->second));
    } else {
      bindings[symbol.getResult()] = scalar(identifier(symbol.getName()));
    }
    return mlir::success();
  }
  if (auto root = mlir::dyn_cast<riscv::RootDomainOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.domainType = root.getResult().getType();
    bindings[root.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto root = mlir::dyn_cast<riscv::RootPointOp>(operation)) {
    Binding binding;
    binding.kind = Binding::Kind::Point;
    binding.point.axis = 0;
    binding.point.base = "0";
    binding.point.active = "1";
    binding.point.physicalExtent = 1;
    bindings[root.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto domain = mlir::dyn_cast<riscv::DomainOp>(operation)) {
    Binding extent = bindings.lookup(domain.getExtent());
    Binding partition = bindings.lookup(domain.getPartition());
    Binding multiplicity = bindings.lookup(domain.getMultiplicity());
    if (extent.kind != Binding::Kind::Scalar ||
        partition.kind != Binding::Kind::Scalar ||
        multiplicity.kind != Binding::Kind::Scalar)
      return fail(domain,
                  "domain extent, partition, and multiplicity require scalar index values");
    Binding binding;
    binding.kind = Binding::Kind::Domain;
    binding.parentDomain = domain.getParent();
    binding.domainType = domain.getResult().getType();
    binding.domainExtent = std::move(extent.scalar);
    binding.domainPartition = std::move(partition.scalar);
    binding.domainMultiplicity = std::move(multiplicity.scalar);
    bindings[domain.getResult()] = std::move(binding);
    return mlir::success();
  }
  if (auto point = mlir::dyn_cast<riscv::PhysicalPointOp>(operation))
    return compilePhysicalPoint(point);
  if (auto cohort = mlir::dyn_cast<riscv::RecordCohortOp>(operation))
    return compileRecordCohort(cohort);
  if (auto constant = mlir::dyn_cast<riscv::ConstantOp>(operation))
    return bindConstant(constant.getValue(), constant.getResult());
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(operation))
    return bindConstant(constant.getValue(), constant.getResult());
  if (operation.getDialect() && operation.getDialect()->getNamespace() == "arith")
    return compileScalarControl(operation);
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(operation))
    return compileFor(loop);
  if (auto branch = mlir::dyn_cast<mlir::scf::IfOp>(operation))
    return compileIf(branch);
  if (auto loop = mlir::dyn_cast<mlir::scf::WhileOp>(operation))
    return compileWhile(loop);
  if (auto view = mlir::dyn_cast<riscv::MemoryViewOp>(operation))
    return compileMemoryView(view);
  if (auto view = mlir::dyn_cast<riscv::LocalViewOp>(operation))
    return compileLocalView(view);
  if (auto snapshot = mlir::dyn_cast<riscv::DenseSnapshotOp>(operation))
    return compileDenseSnapshot(snapshot);
  if (auto copy = mlir::dyn_cast<riscv::LocalCopyOp>(operation))
    return compileLocalCopy(copy);
  if (auto load = mlir::dyn_cast<riscv::StorageLoadOp>(operation))
    return compileStorageLoad(load);
  if (auto store = mlir::dyn_cast<riscv::StorageStoreOp>(operation))
    return compileStorageStore(store);
  if (auto slice = mlir::dyn_cast<riscv::SliceOp>(operation))
    return compileSlice(slice);
  if (auto subview = mlir::dyn_cast<riscv::SubviewOp>(operation))
    return compileSubview(subview);
  if (auto extent = mlir::dyn_cast<riscv::ExtentOp>(operation))
    return compileExtent(extent);
  if (auto reshape = mlir::dyn_cast<riscv::ReshapeOp>(operation))
    return compileReshape(reshape);
  if (auto rename = mlir::dyn_cast<riscv::ReaxisOp>(operation))
    return compileReaxis(rename);
  if (auto view = mlir::dyn_cast<riscv::LinearViewOp>(operation))
    return compileLinearView(view);
  if (auto admit = mlir::dyn_cast<riscv::LoadOp>(operation))
    return compileAdmit(admit);
  if (auto staged = mlir::dyn_cast<riscv::StagedViewOp>(operation))
    return compileStagedView(staged);
  if (auto materialize =
          mlir::dyn_cast<riscv::RegisterMaterializeOp>(operation))
    return compileRegisterMaterialize(materialize);
  if (auto allocation = mlir::dyn_cast<riscv::LocalAllocOp>(operation))
    return compileLocalAlloc(allocation);
  if (auto reuse = mlir::dyn_cast<riscv::LocalReuseOp>(operation)) {
    bindings[reuse.getResult()] = bindings.lookup(reuse.getPrevious());
    return mlir::success();
  }
  if (auto guard = mlir::dyn_cast<riscv::LocalCapacityGuardOp>(operation))
    return compileLocalCapacityGuard(guard);
  if (auto binding = mlir::dyn_cast<riscv::LocalBindOp>(operation))
    return compileLocalBind(binding);
  if (auto load = mlir::dyn_cast<riscv::LocalLoadOp>(operation))
    return compileLocalLoad(load);
  if (auto store = mlir::dyn_cast<riscv::LocalStoreOp>(operation))
    return compileLocalStore(store);
  if (auto materialize =
          mlir::dyn_cast<riscv::RVVLocalMaterializeOp>(operation))
    return compileRVVLocalMaterialize(materialize);
  if (auto guard = mlir::dyn_cast<riscv::IndexMultipleGuardOp>(operation))
    return compileIndexMultipleGuard(guard);
  if (auto binding = mlir::dyn_cast<riscv::EncodedLocalBindOp>(operation))
    return compileEncodedLocalBind(binding);
  if (auto transfer =
          mlir::dyn_cast<riscv::RVVEncodedLocalPackTransferOp>(operation))
    return compileRVVEncodedLocalPackTransfer(transfer);
  if (auto spill = mlir::dyn_cast<riscv::SpillOp>(operation))
    return compileSpill(spill);
  if (auto reload = mlir::dyn_cast<riscv::ReloadOp>(operation))
    return compileReload(reload);
  if (auto reload = mlir::dyn_cast<riscv::ReloadSliceOp>(operation))
    return compileReloadSlice(reload);
  if (auto pack = mlir::dyn_cast<riscv::IMEPackOp>(operation))
    return compileIMEPack(pack);
  if (auto zero = mlir::dyn_cast<riscv::IMEZeroOp>(operation))
    return compileIMEZero(zero);
  if (auto load = mlir::dyn_cast<riscv::IMEFragmentStorageLoadOp>(operation))
    return compileIMEFragmentStorageLoad(load);
  if (auto load = mlir::dyn_cast<riscv::IMEFragmentWindowLoadOp>(operation))
    return compileIMEFragmentWindowLoad(load);
  if (auto load = mlir::dyn_cast<riscv::IMEFragmentDenseLoadOp>(operation))
    return compileIMEFragmentDenseLoad(load);
  if (auto decode = mlir::dyn_cast<riscv::IMEFragmentDecodeOp>(operation))
    return compileIMEFragmentDecode(decode);
  if (auto pointwise = mlir::dyn_cast<riscv::IMEFragmentPointwiseOp>(operation))
    return compileIMEFragmentPointwise(pointwise);
  if (auto broadcast = mlir::dyn_cast<riscv::IMEFragmentRowBroadcastOp>(operation))
    return compileIMEFragmentRowBroadcast(broadcast);
  if (auto mma = mlir::dyn_cast<riscv::IMEFragmentMMAOp>(operation))
    return compileIMEFragmentMMA(mma);
  if (auto unpack = mlir::dyn_cast<riscv::IMEUnpackOp>(operation))
    return compileIMEUnpack(unpack);
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation))
    return compileConvertLayout(conversion);
  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation))
    return compileIota(iota);
  if (auto broadcast = mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(operation))
    return compileRVVAxisBroadcast(broadcast);
  if (auto freshValue = mlir::dyn_cast<riscv::NewOp>(operation))
    return compileNew(freshValue);
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation))
    return compileField(field);
  if (auto view = mlir::dyn_cast<riscv::FieldViewOp>(operation))
    return compileFieldView(view);
  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation))
    return compileExtract(extract);
  if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation))
    return compileUpdate(update);
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation))
    return compileUnary(unary);
  if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation))
    return compileCompare(compare);
  if (auto select = mlir::dyn_cast<riscv::SelectOp>(operation))
    return compileSelect(select);
  if (auto select = mlir::dyn_cast<riscv::RVVCompareSelectOp>(operation))
    return compileRVVCompareSelect(select);
  if (auto cast = mlir::dyn_cast<riscv::CastOp>(operation))
    return compileCast(cast);
  if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation))
    return compileNarrow(narrow);
  if (auto widen = mlir::dyn_cast<riscv::WidenOp>(operation))
    return compileWiden(widen);
  if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation))
    return compileReduce(reduce);
  if (auto fold = mlir::dyn_cast<riscv::Fold2Op>(operation))
    return compileFold2(fold);
  if (auto merge = mlir::dyn_cast<riscv::RVVBitplaneMergeOp>(operation))
    return compileRVVBitplaneMerge(merge);
  if (auto merge = mlir::dyn_cast<riscv::PackedPlaneMergeOp>(operation))
    return compilePackedPlaneMerge(merge);
  if (auto decode = mlir::dyn_cast<riscv::RVVBitmaskDecodeOp>(operation))
    return compileRVVBitmaskDecode(decode);
  if (auto reduction =
          mlir::dyn_cast<riscv::RVVSignedBitmaskReduceOp>(operation))
    return compileRVVSignedBitmaskReduce(reduction);
  if (auto negate = mlir::dyn_cast<riscv::RVVMaskedNegateOp>(operation))
    return compileRVVMaskedNegate(negate);
  if (auto load =
          mlir::dyn_cast<riscv::RVVBitmaskWindowLoadOp>(operation))
    return compileRVVBitmaskWindowLoad(load);
  if (auto load = mlir::dyn_cast<riscv::RVVGroupedMacLoadOp>(operation))
    return compileGroupedMacLoad(load);
  if (auto step = mlir::dyn_cast<riscv::RVVGroupedMacStepOp>(operation))
    return compileGroupedMacStep(step);
  if (auto multiply = mlir::dyn_cast<riscv::RVVWidenMultiplyOp>(operation))
    return compileRVVWidenMultiply(multiply);
  if (auto add = mlir::dyn_cast<riscv::RVVWidenAddOp>(operation))
    return compileRVVWidenAdd(add);
  if (auto shift = mlir::dyn_cast<riscv::RVVNarrowShiftRightOp>(operation))
    return compileRVVNarrowShiftRight(shift);
  if (auto multiply =
          mlir::dyn_cast<riscv::RVVWidenScalarMultiplyOp>(operation))
    return compileRVVWidenScalarMultiply(multiply);
  if (auto multiply =
          mlir::dyn_cast<riscv::RVVMultiplyHighScalarOp>(operation))
    return compileRVVMultiplyHighScalar(multiply);
  if (auto index =
          mlir::dyn_cast<riscv::RVVRegularRepeatIndexOp>(operation))
    return compileRVVRegularRepeatIndex(index);
  if (auto gather =
          mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(operation))
    return compileRVVRegularRepeatGather(gather);
  if (auto load =
          mlir::dyn_cast<riscv::RVVRegularRepeatScalarLoadOp>(operation))
    return compileRVVRegularRepeatScalarLoad(load);
  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(operation))
    return compileRVVStorageWindow(window);
  if (auto load = mlir::dyn_cast<riscv::RVVLayeredRecordLoadOp>(operation))
    return compileRVVLayeredRecordLoad(load);
  if (auto load = mlir::dyn_cast<riscv::RVVLayeredStorageLoadOp>(operation))
    return compileRVVLayeredStorageLoad(load);
  if (auto decode =
          mlir::dyn_cast<riscv::RVVLayeredStorageDecodeOp>(operation))
    return compileRVVLayeredStorageDecode(decode);
  if (auto load = mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(operation))
    return compileRVVReplicaStorageLoad(load);
  if (auto load = mlir::dyn_cast<riscv::RVVSegmentPairLoadOp>(operation))
    return compileRVVSegmentPairLoad(load);
  if (auto load = mlir::dyn_cast<riscv::RVVRecordStorageLoadOp>(operation))
    return compileRVVRecordStorageLoad(load);
  if (auto decode = mlir::dyn_cast<riscv::RVVRecordStorageDecodeOp>(operation))
    return compileRVVRecordStorageDecode(decode);
  if (auto store = mlir::dyn_cast<riscv::RVVRecordStoreOp>(operation))
    return compileRVVRecordStore(store);
  if (auto slice = mlir::dyn_cast<riscv::RVVIssueSliceOp>(operation))
    return compileRVVIssueSlice(slice);
  if (auto assemble = mlir::dyn_cast<riscv::RVVAssembleValueOp>(operation))
    return compileRVVAssembleValue(assemble);
  if (auto accumulate =
          mlir::dyn_cast<riscv::RVVWidenAccumulateOp>(operation))
    return compileRVVWidenAccumulate(accumulate);
  if (auto finalize =
          mlir::dyn_cast<riscv::RVVFinalizeWidenDotOp>(operation))
    return compileRVVFinalizeWidenDot(finalize);
  if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
    return compileRVVPartialSet(partial);
  if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
    return compileRVVPartialCapture(capture);
  if (auto collect = mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
    return compileRVVPartialCollect(collect);
  if (auto repack = mlir::dyn_cast<riscv::RVVPartialRepackOp>(operation))
    return compileRVVPartialRepack(repack);
  if (auto merge = mlir::dyn_cast<riscv::RVVPartialMergeOp>(operation))
    return compileRVVPartialMerge(merge);
  if (auto reduce = mlir::dyn_cast<riscv::RVVPartialReduceOp>(operation))
    return compileRVVPartialReduce(reduce);
  if (auto combine =
          mlir::dyn_cast<riscv::RVVPartialScaleCombineOp>(operation))
    return compileRVVPartialScaleCombine(combine);
  if (auto packed =
          mlir::dyn_cast<riscv::RVVPartialPackedScaleOp>(operation))
    return compileRVVPartialPackedScale(packed);
  if (auto scale =
          mlir::dyn_cast<riscv::RVVPartialWidenScaleOp>(operation))
    return compileRVVPartialWidenScale(scale);
  if (auto combine = mlir::dyn_cast<riscv::RVVPartialCombineOp>(operation))
    return compileRVVPartialCombine(combine);
  if (auto finalize = mlir::dyn_cast<riscv::RVVPartialFinalizeOp>(operation))
    return compileRVVPartialFinalize(finalize);
  if (auto assemble =
          mlir::dyn_cast<riscv::RVVAssembleReplicasOp>(operation))
    return compileRVVAssembleReplicas(assemble);
  if (auto reduce = mlir::dyn_cast<riscv::RVVWidenReduceOp>(operation))
    return compileRVVWidenReduce(reduce);
  if (auto reduce =
          mlir::dyn_cast<riscv::RVVPartitionedWidenReduceStoreOp>(operation))
    return compileRVVPartitionedWidenReduceStore(reduce);
  if (auto window = mlir::dyn_cast<riscv::RVVLayeredWindowOp>(operation))
    return compileRVVLayeredWindow(window);
  if (auto stream = mlir::dyn_cast<riscv::RVVLayeredStreamOp>(operation))
    return compileRVVLayeredStream(stream);
  if (auto stream =
          mlir::dyn_cast<riscv::RVVProjectedLayeredStreamOp>(operation))
    return compileRVVProjectedLayeredStream(stream);
  if (auto load = mlir::dyn_cast<riscv::ScalarTileLoadOp>(operation))
    return compileScalarTileLoad(load);
  if (auto store = mlir::dyn_cast<riscv::ScalarTileStoreOp>(operation))
    return compileScalarTileStore(store);
  if (auto load = mlir::dyn_cast<riscv::RVVTileLoadOp>(operation))
    return compileRVVTileLoad(load);
  if (auto load = mlir::dyn_cast<riscv::RVVDenseSegmentLoadOp>(operation))
    return compileRVVDenseSegmentLoad(load);
  if (auto store = mlir::dyn_cast<riscv::RVVDenseSegmentStoreOp>(operation))
    return compileRVVDenseSegmentStore(store);
  if (auto slice = mlir::dyn_cast<riscv::RVVTileSliceOp>(operation))
    return compileRVVTileSlice(slice);
  if (auto store = mlir::dyn_cast<riscv::RVVTileStoreOp>(operation))
    return compileRVVTileStore(store);
  if (auto step = mlir::dyn_cast<riscv::RVVTileContractOp>(operation))
    return compileRVVTileContract(step);
  if (auto issue = mlir::dyn_cast<riscv::RVVFMAIssueOp>(operation))
    return compileRVVFMAIssue(issue);
  if (auto mac = mlir::dyn_cast<riscv::RVVIntegerMACOp>(operation))
    return compileRVVIntegerMAC(mac);
  if (auto fma = mlir::dyn_cast<riscv::FMAOp>(operation))
    return compileFMA(fma);
  if (auto step = mlir::dyn_cast<riscv::RVVStreamReduceStepOp>(operation))
    return compileRVVStreamReduceStep(step);
  if (auto step = mlir::dyn_cast<riscv::RVVStreamContractStepOp>(operation))
    return compileRVVStreamContractStep(step);
  if (auto finalize = mlir::dyn_cast<riscv::RVVStreamFinalizeOp>(operation))
    return compileRVVStreamFinalize(finalize);
  if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(operation))
    return compileLookup(lookup);
  if (auto byteGather = mlir::dyn_cast<riscv::RVVByteGatherOp>(operation))
    return compileRVVByteGather(byteGather);
  if (auto windows = mlir::dyn_cast<riscv::RVVByteWindowsLoadOp>(operation))
    return compileRVVByteWindowsLoad(windows);
  if (auto entryLoad =
          mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(operation))
    return compileRVVIndexedEntryLoad(entryLoad);
  if (auto window =
          mlir::dyn_cast<riscv::RVVUnitEntryWindowLoadOp>(operation))
    return compileRVVUnitEntryWindowLoad(window);
  if (auto splat = mlir::dyn_cast<riscv::RVVSplatOp>(operation))
    return compileRVVSplat(splat);
  if (auto projected =
          mlir::dyn_cast<riscv::ProjectReductionOperandOp>(operation))
    return compileProjectReductionOperand(projected);
  if (auto step = mlir::dyn_cast<riscv::RVVContractStepOp>(operation))
    return compileRVVContractStep(step);
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation))
    return compileBinary(binary);
  if (auto bitcast = mlir::dyn_cast<riscv::BitcastOp>(operation))
    return compileBitcast(bitcast);
  if (auto atomic = mlir::dyn_cast<riscv::AtomicRMWOp>(operation))
    return compileAtomicRMW(atomic);
  if (auto commit = mlir::dyn_cast<riscv::StoreOp>(operation))
    return compileCommit(commit);
  return fail(&operation, llvm::Twine("intrinsic-C emission has no rule for physical operation '") +
                              operation.getName().getStringRef() + "'");
}

mlir::LogicalResult
Emitter::compilePhysicalPoint(riscv::PhysicalPointOp point) {
  Binding base = bindings.lookup(point.getBase());
  Binding active = bindings.lookup(point.getActive());
  Binding partition = bindings.lookup(point.getPartition());
  if (base.kind != Binding::Kind::Scalar ||
      active.kind != Binding::Kind::Scalar ||
      partition.kind != Binding::Kind::Scalar)
    return fail(point, "physical point coordinates require scalar index values");
  int64_t extent = 0;
  if (llvm::StringRef(partition.scalar).getAsInteger(10, extent) || extent <= 0)
    return fail(point,
                "physical point partition must be a positive compile-time integer");
  Binding binding;
  binding.kind = Binding::Kind::Point;
  binding.point.axis = point.getResult().getType().getDomain().getAxisId();
  binding.point.base = std::move(base.scalar);
  binding.point.active = std::move(active.scalar);
  binding.point.physicalExtent = extent;
  bindings[point.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult
Emitter::compileRecordCohort(riscv::RecordCohortOp cohort) {
  Binding origin = bindings.lookup(cohort.getOrigin());
  auto type = cohort.getResult().getType();
  if (origin.kind != Binding::Kind::Point || type.getWidth() <= 1 ||
      type.getPartition() <= 0 ||
      origin.point.axis != type.getDomain().getAxisId() ||
      origin.point.physicalExtent != type.getPartition())
    return fail(cohort,
                "record cohort requires one matching physical Level point");
  Binding binding;
  binding.kind = Binding::Kind::RecordCohort;
  binding.point = std::move(origin.point);
  binding.cohortWidth = type.getWidth();
  binding.cohortPartition = type.getPartition();
  bindings[cohort.getResult()] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileFor(mlir::scf::ForOp operation) {
  Binding lower = bindings.lookup(operation.getLowerBound());
  Binding upper = bindings.lookup(operation.getUpperBound());
  Binding step = bindings.lookup(operation.getStep());
  if (lower.kind != Binding::Kind::Scalar ||
      upper.kind != Binding::Kind::Scalar || step.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered for bounds require selected scalar values");
  llvm::SmallVector<Binding, 0> carried;
  for (mlir::Value value : operation.getInitArgs()) {
    Binding source = bindings.lookup(value);
    if (source.kind == Binding::Kind::Memory || source.kind == Binding::Kind::Scalar || source.kind == Binding::Kind::Fragment ||
        source.kind == Binding::Kind::Vector || source.kind == Binding::Kind::ScalarTuple) {
      auto copy = declareMutableBinding(value, "for_carry");
      if (mlir::failed(copy) || mlir::failed(assignBinding(operation, value, *copy, value, source,
          "ordered for carry requires its complete selected register representation")))
        return mlir::failure();
      carried.push_back(std::move(*copy));
    } else if (source.kind == Binding::Kind::LocalArray) {
      carried.push_back(std::move(source));
    } else if (source.kind == Binding::Kind::Window) {
      Binding copy;
      copy.kind = Binding::Kind::Window;
      copy.windowFamily = source.windowFamily;
      auto declare = [&](llvm::StringRef expression, llvm::StringRef prefix) {
        std::string name = fresh(prefix);
        line("__auto_type " + name + " = " + expression.str() + ";");
        return name;
      };
      for (llvm::StringRef expression : source.windowLhs)
        copy.windowLhs.push_back(declare(expression, "for_window_lhs"));
      for (llvm::StringRef expression : source.windowRhs)
        copy.windowRhs.push_back(declare(expression, "for_window_rhs"));
      for (llvm::StringRef expression : source.windowValidity)
        copy.windowValidity.push_back(
            declare(expression, "for_window_valid"));
      for (llvm::StringRef validity : copy.windowValidity)
        line("(void)" + validity.str() + ";");
      carried.push_back(std::move(copy));
    } else {
      return fail(operation,
                  "ordered for carry has no selected scalar/vector/tuple/window/local handoff");
    }
  }
  std::string iterator = fresh("for_index");
  auto level =
      operation->getAttrOfType<riscv::LevelAttr>("weft.riscv.level");
  auto direction =
      operation->getAttrOfType<mlir::StringAttr>("weft.riscv.direction");
  llvm::StringRef ordered =
      level ? level.getDirection()
            : direction ? direction.getValue() : llvm::StringRef();
  if (ordered.empty())
    return fail(operation, "ordered for has no selected physical direction");
  const bool descending = ordered == "descending";
  const std::string indexType = descending ? "ptrdiff_t" : "size_t";
  const std::string comparison = descending ? " > " : " < ";
  auto systemUnroll = operation->getAttrOfType<mlir::StringAttr>(
      "weft.riscv.system_unroll");
  if (systemUnroll && systemUnroll.getValue() == "disable")
    line("#pragma GCC unroll 1");
  line("for (" + indexType + " " + iterator + " = (" + indexType + ")(" +
       lower.scalar + "); " + iterator + comparison + "(" + indexType + ")(" +
       upper.scalar + "); " + iterator + " += " + step.scalar + ") {");
  ++indent;
  llvm::SmallVector<Binding, 0> arguments{scalar(iterator)};
  mlir::Block &body = *operation.getBody();
  for (auto [index, binding] : llvm::enumerate(carried)) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInitArgs()[index], body.getArgument(index + 1), binding);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered for argument requires an explicit physical layout conversion");
    arguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(body, arguments)))
    return mlir::failure();
  auto yield = mlir::cast<mlir::scf::YieldOp>(body.getTerminator());
  if (yield.getResults().size() != carried.size())
    return fail(operation,
                "ordered for yield count does not match its carried state");
  llvm::SmallVector<Binding, 0> staged;
  for (mlir::Value nextValue : yield.getResults()) {
    Binding next = bindings.lookup(nextValue);
    if (next.kind == Binding::Kind::LocalArray) {
      staged.push_back(std::move(next));
      continue;
    }
    if (next.kind == Binding::Kind::Window) {
      Binding temporary;
      temporary.kind = Binding::Kind::Window;
      temporary.windowFamily = next.windowFamily;
      auto stageWindow = [&](llvm::ArrayRef<std::string> expressions,
                             llvm::SmallVectorImpl<std::string> &results) {
        for (llvm::StringRef expression : expressions) {
          std::string name = fresh("for_next_window");
          line("__auto_type " + name + " = " + expression.str() + ";");
          results.push_back(std::move(name));
        }
      };
      stageWindow(next.windowLhs, temporary.windowLhs);
      stageWindow(next.windowRhs, temporary.windowRhs);
      stageWindow(next.windowValidity, temporary.windowValidity);
      staged.push_back(std::move(temporary));
      continue;
    }
    mlir::FailureOr<Binding> temporary =
        declareMutableBinding(nextValue, "for_next");
    if (mlir::failed(temporary) ||
        mlir::failed(assignBinding(
            operation, nextValue, *temporary, nextValue, next,
            "ordered for body requires an explicit physical layout conversion")))
      return mlir::failure();
    staged.push_back(std::move(*temporary));
  }
  for (auto [index, nextValue] : llvm::enumerate(yield.getResults())) {
    if (mlir::failed(assignBinding(
            operation, operation.getInitArgs()[index], carried[index], nextValue,
            staged[index],
            "ordered for body requires an explicit physical layout conversion")))
      return mlir::failure();
  }
  --indent;
  line("}");
  for (auto [index, result] : llvm::enumerate(operation.getResults())) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInitArgs()[index], result, carried[index]);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered for result requires an explicit physical layout conversion");
    bindings[result] = std::move(*projected);
  }
  return mlir::success();
}

mlir::LogicalResult Emitter::compileIf(mlir::scf::IfOp operation) {
  Binding condition = bindings.lookup(operation.getCondition());
  if (condition.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered if requires a selected scalar condition");

  llvm::SmallVector<Binding, 0> results;
  for (mlir::Value resultValue : operation.getResults()) {
    if (auto value = mlir::dyn_cast<riscv::ValueType>(resultValue.getType());
        value && value.getLayout().getCarrier() == "local") {
      Binding result;
      result.kind = Binding::Kind::None;
      results.push_back(std::move(result));
      continue;
    }
    auto result = declareMutableBinding(resultValue, "if_result");
    if (mlir::failed(result))
      return fail(operation, "ordered if result has no complete selected representation");
    results.push_back(std::move(*result));
  }

  auto compileBranch = [&](mlir::Region &region) -> mlir::LogicalResult {
    mlir::Block &block = region.front();
    if (mlir::failed(compileBlock(block, {})))
      return mlir::failure();
    auto yield = mlir::cast<mlir::scf::YieldOp>(block.getTerminator());
    if (yield.getResults().size() != results.size())
      return fail(operation,
                  "ordered if yield count does not match its results");
    for (auto [index, value] : llvm::enumerate(yield.getResults())) {
      Binding &target = results[index];
      Binding source = bindings.lookup(value);
      if (mlir::failed(assignBinding(
              operation, operation.getResult(index), target, value, source,
              "ordered if branch requires an explicit physical layout conversion")))
        return mlir::failure();
    }
    return mlir::success();
  };

  line("if ((bool)(" + condition.scalar + ")) {");
  ++indent;
  if (mlir::failed(compileBranch(operation.getThenRegion())))
    return mlir::failure();
  --indent;
  if (!operation.getElseRegion().empty()) {
    line("} else {");
    ++indent;
    if (mlir::failed(compileBranch(operation.getElseRegion())))
      return mlir::failure();
    --indent;
  }
  line("}");
  for (auto [resultValue, binding] : llvm::zip(operation.getResults(), results))
    bindings[resultValue] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileWhile(mlir::scf::WhileOp operation) {
  llvm::SmallVector<Binding, 0> carried;
  for (mlir::Value value : operation.getInits()) {
    mlir::FailureOr<Binding> storage = declareMutableBinding(value, "while_carry");
    if (mlir::failed(storage))
      return fail(operation,
                  "ordered while carry has no mutable scalar/vector representation");
    Binding source = bindings.lookup(value);
    if (mlir::failed(assignBinding(
            operation, value, *storage, value, source,
            "ordered while initial carry requires an explicit physical conversion")))
      return mlir::failure();
    carried.push_back(std::move(*storage));
  }

  mlir::Block &before = operation.getBefore().front();
  mlir::Block &after = operation.getAfter().front();
  llvm::SmallVector<Binding, 0> results;
  for (mlir::Value value : operation.getResults()) {
    auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (value.use_empty() || (type && type.getLayout().getCarrier() == "local")) {
      results.emplace_back();
      continue;
    }
    auto storage = declareMutableBinding(value, "while_result");
    if (mlir::failed(storage))
      return fail(operation, "ordered while result has no complete selected representation");
    results.push_back(std::move(*storage));
  }
  line("while (1) {");
  ++indent;
  llvm::SmallVector<Binding, 0> beforeArguments;
  for (auto [index, binding] : llvm::enumerate(carried)) {
    mlir::FailureOr<Binding> projected = projectBinding(
        operation.getInits()[index], before.getArgument(index), binding);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered while condition carry requires an explicit physical conversion");
    beforeArguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(before, beforeArguments)))
    return mlir::failure();
  auto condition = mlir::cast<mlir::scf::ConditionOp>(before.getTerminator());
  Binding predicate = bindings.lookup(condition.getCondition());
  if (predicate.kind != Binding::Kind::Scalar)
    return fail(operation, "ordered while condition requires a scalar predicate");
  // scf.condition supplies the exit values, which may differ in arity, type,
  // and value from the before inputs. Capture them on the false edge while
  // their definitions are still in scope, instead of returning the backedge.
  line("if (!(" + predicate.scalar + ")) {");
  ++indent;
  for (auto [index, value] : llvm::enumerate(condition.getArgs())) {
    mlir::Value result = operation.getResult(index);
    if (result.use_empty()) continue;
    Binding source = bindings.lookup(value);
    if (mlir::failed(assignBinding(operation, result, results[index], value, source,
            "ordered while exit requires an explicit physical conversion")))
      return mlir::failure();
  }
  line("break;");
  --indent;
  line("}");

  llvm::SmallVector<Binding, 0> afterArguments;
  for (auto [index, value] : llvm::enumerate(condition.getArgs())) {
    Binding source = bindings.lookup(value);
    mlir::FailureOr<Binding> projected =
        projectBinding(value, after.getArgument(index), source);
    if (mlir::failed(projected))
      return fail(operation,
                  "ordered while body carry requires an explicit physical conversion");
    afterArguments.push_back(std::move(*projected));
  }
  if (mlir::failed(compileBlock(after, afterArguments)))
    return mlir::failure();
  auto yield = mlir::cast<mlir::scf::YieldOp>(after.getTerminator());
  if (yield.getResults().size() != carried.size())
    return fail(operation,
                "ordered while yield count does not match its carried state");
  llvm::SmallVector<Binding, 0> staged;
  for (mlir::Value nextValue : yield.getResults()) {
    Binding next = bindings.lookup(nextValue);
    if (next.kind == Binding::Kind::LocalArray) {
      staged.push_back(std::move(next));
      continue;
    }
    if (next.kind == Binding::Kind::Window) {
      Binding temporary;
      temporary.kind = Binding::Kind::Window;
      temporary.windowFamily = next.windowFamily;
      auto stageWindow = [&](llvm::ArrayRef<std::string> expressions,
                             llvm::SmallVectorImpl<std::string> &results) {
        for (llvm::StringRef expression : expressions) {
          std::string name = fresh("while_next_window");
          line("__auto_type " + name + " = " + expression.str() + ";");
          results.push_back(std::move(name));
        }
      };
      stageWindow(next.windowLhs, temporary.windowLhs);
      stageWindow(next.windowRhs, temporary.windowRhs);
      stageWindow(next.windowValidity, temporary.windowValidity);
      staged.push_back(std::move(temporary));
      continue;
    }
    mlir::FailureOr<Binding> temporary =
        declareMutableBinding(nextValue, "while_next");
    if (mlir::failed(temporary) ||
        mlir::failed(assignBinding(
            operation, nextValue, *temporary, nextValue, next,
            "ordered while body requires an explicit physical layout conversion")))
      return mlir::failure();
    staged.push_back(std::move(*temporary));
  }
  for (auto [index, value] : llvm::enumerate(yield.getResults())) {
    if (mlir::failed(assignBinding(
            operation, operation.getInits()[index], carried[index], value,
            staged[index],
            "ordered while body requires an explicit physical conversion")))
      return mlir::failure();
  }
  --indent;
  line("}");
  for (auto [value, binding] : llvm::zip(operation.getResults(), results))
    bindings[value] = std::move(binding);
  return mlir::success();
}

mlir::LogicalResult Emitter::compileNew(riscv::NewOp operation) {
  if (!operation.getInitialized() || !operation.getInitial())
    return fail(operation,
                "physical new has no explicit canonical initializer");
  Binding initial = bindings.lookup(operation.getInitial());
  if (mlir::isa<riscv::FragmentType>(operation.getResult().getType())) {
    auto result = declareMutableBinding(operation.getResult(), "fragment_state");
    if (mlir::failed(result) || mlir::failed(assignBinding(operation,
            operation.getResult(), *result, operation.getInitial(), initial,
            "fragment state birth requires an exact register initializer")))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
    return mlir::success();
  }
  auto selectedKind = selectedBindingKind(operation.getResult());
  if (selectedKind && *selectedKind == Binding::Kind::Vector) {
    mlir::FailureOr<Binding> result =
        initial.kind == Binding::Kind::Slice
            ? loadDenseBlock(operation.getResult(), initial)
            : makeVector(operation.getResult(), "state", initial,
                         operation.getInitial());
    if (mlir::failed(result))
      return mlir::failure();
    bindings[operation.getResult()] = std::move(*result);
    return mlir::success();
  }
  if (selectedKind && *selectedKind == Binding::Kind::ScalarTuple) {
    auto type =
        scalarCType(riscv_internal::logicalElement(operation.getResult().getType()));
    const int64_t parts = scalarPartCount(operation.getResult());
    if (!type || parts <= 0)
      return fail(operation,
                  "selected register tuple state has no scalar C representation");
    Binding result;
    result.kind = Binding::Kind::ScalarTuple;
    llvm::SmallVector<int64_t, 4> axes(layoutOf(operation.getResult()).getAxisIds().asArrayRef());
    for (int64_t part = 0; part < parts; ++part) {
      std::string expression;
      if (initial.kind == Binding::Kind::Scalar) {
        expression = initial.scalar;
      } else if (initial.kind == Binding::Kind::ScalarTuple) {
        auto projected = projectPart(operation.getInitial(), operation.getResult(), part);
        if (!projected || *projected >= initial.parts.size())
          return fail(operation,
                      "register tuple initializer mapping is not projectable");
        expression = initial.parts[*projected];
      } else if (initial.kind == Binding::Kind::Slice) {
        auto coordinates = scalarCoordinates(operation.getResult(), part);
        if (!coordinates || coordinates->size() != axes.size())
          return fail(operation,
                      "register tuple initializer has no coordinate projection");
        llvm::SmallVector<std::pair<int64_t, std::string>> offsets;
        std::string active = "1";
        for (auto [axis, coordinate] : llvm::zip(axes, *coordinates)) {
          offsets.push_back({axis, std::to_string(coordinate)});
          auto scope = axisScopes.find(axis);
          if (scope != axisScopes.end() && !scope->second.empty())
            active += " && " + std::to_string(coordinate) + " < " +
                      scope->second.back().active;
        }
        auto address = denseAddress(initial.slice, offsets);
        if (!address)
          return fail(operation,
                      "register tuple initializer has no dense address relation");
        expression = "((" + active + ") ? *(const " + *type + " *)(" +
                     *address + ") : (" + *type + ")0)";
      } else {
        return fail(operation,
                    "register tuple new requires scalar, tuple, or dense initializer");
      }
      std::string name = fresh("state");
      line(*type + " " + name + " = " + expression + ";");
      result.parts.push_back(std::move(name));
    }
    bindings[operation.getResult()] = std::move(result);
    return mlir::success();
  }
  if (!selectedKind || *selectedKind != Binding::Kind::Scalar)
    return fail(operation, "intrinsic-C emission has no realization for selected new value");
  if (initial.kind != Binding::Kind::Scalar)
    return fail(operation, "scalar new requires a scalar initializer");
  Binding result = initial;
  result.scalar = fresh("state");
  auto type = scalarCType(riscv_internal::logicalElement(operation.getResult().getType()));
  if (!type)
    return fail(operation, "intrinsic-C emission does not support selected scalar state type");
  line(*type + " " + result.scalar + " = " + initial.scalar + ";");
  bindings[operation.getResult()] = std::move(result);
  return mlir::success();
}

} // namespace weft::riscv_emission
