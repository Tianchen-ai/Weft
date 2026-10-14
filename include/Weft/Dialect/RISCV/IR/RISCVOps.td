#ifndef WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
#define WEFT_DIALECT_RISCV_IR_RISCVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_RISCV_Dialect : Dialect {
  let name = "weft_riscv";
  let cppNamespace = "::weft::riscv";
  let summary = "Target-aware non-SIMT RISC-V physical program";
  let description = [{
    The dialect is Weft's second and final program IR. It carries logical-axis
    preserving physical representations, explicit conversions, memory edges,
    local storage, physical loops, and selected RVV/IME local operations. It is
    a real SSA program rather than a planning record beside Canonical Kernel IR.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class WeftRISCV_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WEFT_RISCV_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class WeftRISCV_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WEFT_RISCV_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WeftRISCV_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_RISCV_Dialect, mnemonic, traits>;

def WeftRISCV_Terminal : NativeOpTrait<"WeftRISCVTerminal">;
def WeftRISCV_Leaf : NativeOpTrait<"WeftRISCVLeaf">;

class WeftRISCV_TerminalOp<string mnemonic, list<Trait> traits = []>
    : WeftRISCV_Op<mnemonic, !listconcat([WeftRISCV_Terminal], traits)>;

class WeftRISCV_LeafOp<string mnemonic, list<Trait> traits = []>
    : WeftRISCV_TerminalOp<mnemonic, !listconcat([WeftRISCV_Leaf], traits)>;

def WeftRISCV_LayoutAttr : WeftRISCV_Attr<"Layout", "layout"> {
  let parameters = (ins
    StringRefParameter<"unassigned, scalar, rvv, ime, or local">:$carrier,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::mlir::DenseI64ArrayAttr":$timeFactors,
    "::mlir::DenseI64ArrayAttr":$laneFactors,
    "::mlir::DenseI64ArrayAttr":$replicaFactors,
    "::mlir::DenseI64ArrayAttr":$fragmentFactors,
    "::mlir::DenseI64ArrayAttr":$localFactors,
    "int64_t":$sew,
    "int64_t":$lmulEighths,
    "int64_t":$vl,
    "int64_t":$registerGroups,
    StringRefParameter<"unassigned, full, tail, or mask">:$validity,
    DefaultValuedParameter<"int64_t", "1", "adjacent replica tiles per RVV carrier">:$replicaGroup
  );
  let assemblyFormat = [{
    `<` $carrier `,` $axisIds `,` $timeFactors `,` $laneFactors `,`
        $replicaFactors `,` $fragmentFactors `,` $localFactors `,`
        $sew `,` $lmulEighths `,` $vl `,` $registerGroups `,` $validity
        (`,` `group` `=` $replicaGroup^)? `>`
  }];
  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [AttrBuilder<(ins
    "::llvm::StringRef":$carrier, "::mlir::DenseI64ArrayAttr":$axisIds,
    "::mlir::DenseI64ArrayAttr":$timeFactors, "::mlir::DenseI64ArrayAttr":$laneFactors,
    "::mlir::DenseI64ArrayAttr":$replicaFactors, "::mlir::DenseI64ArrayAttr":$fragmentFactors,
    "::mlir::DenseI64ArrayAttr":$localFactors, "int64_t":$sew, "int64_t":$lmulEighths,
    "int64_t":$vl, "int64_t":$registerGroups, "::llvm::StringRef":$validity,
    CArg<"int64_t", "1">:$replicaGroup), [{
      return $_get($_ctxt, carrier, axisIds, timeFactors, laneFactors, replicaFactors,
                   fragmentFactors, localFactors, sew, lmulEighths, vl, registerGroups,
                   validity, replicaGroup);
  }]>];
}

def WeftRISCV_FragmentPackingAttr
    : WeftRISCV_Attr<"FragmentPacking", "fragment_packing"> {
  let parameters = (ins
    StringRefParameter<"target fragment packing identity">:$schema,
    "::mlir::DenseI64ArrayAttr":$axisOrder,
    "int64_t":$rowsPerTile,
    "int64_t":$columnsPerTile,
    StringRefParameter<"row_major or column_major tile order">:$tileOrder,
    StringRefParameter<"row_major or column_major element order">:$elementOrder,
    "int64_t":$storageBits,
    "int64_t":$alignment
  );
  let assemblyFormat = [{
    `<` $schema `,` $axisOrder `,` $rowsPerTile `,` $columnsPerTile `,` $tileOrder `,`
        $elementOrder `,` $storageBits `,` $alignment `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FragmentCapabilityAttr
    : WeftRISCV_Attr<"FragmentCapability", "fragment_capability"> {
  let parameters = (ins
    StringRefParameter<"instruction identity">:$instruction,
    StringRefParameter<"lhs signedness">:$lhsSignedness,
    StringRefParameter<"rhs signedness">:$rhsSignedness,
    "int64_t":$lhsBits,
    "int64_t":$rhsBits,
    "int64_t":$accumulatorBits,
    "int64_t":$mFactor,
    "int64_t":$nFactor,
    "int64_t":$kFactor,
    "int64_t":$activeMFactor,
    "int64_t":$lhsResourceGroups,
    "int64_t":$rhsResourceGroups,
    "int64_t":$accumulatorResourceGroups,
    "::weft::riscv::FragmentPackingAttr":$lhsPacking,
    "::weft::riscv::FragmentPackingAttr":$rhsPacking,
    "::weft::riscv::FragmentPackingAttr":$accumulatorPacking,
    "int64_t":$mmaGroups,
    "int64_t":$mmaChunks,
    "::mlir::ArrayAttr":$clobbers,
    "bool":$volatileAsm,
    "bool":$memoryClobber
  );
  let assemblyFormat = [{
    `<` $instruction `,` $lhsSignedness `,` $rhsSignedness `,`
        $lhsBits `,` $rhsBits `,` $accumulatorBits `,`
        $mFactor `,` $nFactor `,` $kFactor `,` $activeMFactor `,`
        $lhsResourceGroups `,` $rhsResourceGroups `,`
        $accumulatorResourceGroups `,`
        $lhsPacking `,` $rhsPacking `,` $accumulatorPacking `,`
        $mmaGroups `,` $mmaChunks `,`
        $clobbers `,` $volatileAsm `,` $memoryClobber `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_DataCacheAttr : WeftRISCV_Attr<"DataCache", "data_cache"> {
  let parameters = (ins "int64_t":$sizeBytes, "int64_t":$lineBytes,
                        "int64_t":$ways, "int64_t":$sets);
  let assemblyFormat = "`<` $sizeBytes `,` $lineBytes `,` $ways `,` $sets `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_PanelTransferTileAttr : WeftRISCV_Attr<"PanelTransferTile", "panel_transfer_tile"> {
  let parameters = (ins "int64_t":$axis, "int64_t":$fields,
    "int64_t":$lanes, "int64_t":$lmulEighths, "int64_t":$registers,
    "int64_t":$loads, "int64_t":$stores, "int64_t":$work);
  let assemblyFormat = [{
    `<` $axis `,` $fields `,` $lanes `,` $lmulEighths `,` $registers `,`
        $loads `,` $stores `,` $work `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PanelTransferCaseAttr : WeftRISCV_Attr<"PanelTransferCase", "panel_transfer_case"> {
  let parameters = (ins "::weft::riscv::PanelTransferTileAttr":$main,
                        "::weft::riscv::PanelTransferTileAttr":$tail);
  let assemblyFormat = "`<` $main `,` $tail `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_PanelTransferPlanAttr : WeftRISCV_Attr<"PanelTransferPlan", "panel_transfer_plan"> {
  let parameters = (ins "int64_t":$unitAxis, "int64_t":$nonunitAxis,
                        "::mlir::ArrayAttr":$cases);
  let assemblyFormat = "`<` $unitAxis `,` $nonunitAxis `,` $cases `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_ScalarPacketAttr : WeftRISCV_Attr<"ScalarPacket", "scalar_packet"> {
  let parameters = (ins "int64_t":$axis, "int64_t":$steps,
                        "int64_t":$parts, "int64_t":$registers,
                        "int64_t":$liveRegisters);
  let assemblyFormat = "`<` $axis `,` $steps `,` $parts `,` $registers `,` $liveRegisters `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_ContractionTileAttr : WeftRISCV_Attr<"ContractionTile", "contraction_tile"> {
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$shape,
    "int64_t":$laneAxis, "int64_t":$laneInput,
    "int64_t":$lanes, "int64_t":$lmulEighths,
    "int64_t":$scalarSteps, "int64_t":$registers,
    "int64_t":$loads, "int64_t":$conversions
  );
  let assemblyFormat = [{
    `<` $shape `,` $laneAxis `,` $laneInput `,` $lanes `,` $lmulEighths `,`
        $scalarSteps `,` $registers `,` $loads `,` $conversions `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ContractionEpilogueAttr : WeftRISCV_Attr<"ContractionEpilogue", "contraction_epilogue"> {
  let parameters = (ins
    StringRefParameter<"scalar, direct, or vector">:$mode,
    "::mlir::DenseI64ArrayAttr":$shape,
    "int64_t":$laneAxis, "int64_t":$lanes, "int64_t":$lmulEighths,
    "int64_t":$registers
  );
  let assemblyFormat = "`<` $mode `,` $shape `,` $laneAxis `,` $lanes `,` $lmulEighths `,` $registers `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_ContractionCaseAttr : WeftRISCV_Attr<"ContractionCase", "contraction_case"> {
  let parameters = (ins
    "int64_t":$singletonAxis,
    "::weft::riscv::ContractionTileAttr":$tile,
    "::mlir::DenseI64ArrayAttr":$panelShape,
    "int64_t":$panelK,
    "::weft::riscv::ContractionTileAttr":$updateTile,
    "::weft::riscv::ContractionTileAttr":$outputTile,
    "::weft::riscv::ContractionEpilogueAttr":$directEpilogue,
    "::weft::riscv::ContractionEpilogueAttr":$epilogue
  );
  let assemblyFormat = [{
    `<` $singletonAxis `,` $tile `,` $panelShape `,` $panelK `,` $updateTile `,` $outputTile `,` $directEpilogue `,` $epilogue `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ContractionPlanAttr : WeftRISCV_Attr<"ContractionPlan", "contraction_plan"> {
  let parameters = (ins
    "::mlir::DenseI64ArrayAttr":$outputAxes,
    "::mlir::DenseI64ArrayAttr":$reductionAxes,
    "::mlir::ArrayAttr":$cases,
    "int64_t":$maximumLMUL,
    "int64_t":$unroll, "int64_t":$pipelineDepth,
    "bool":$streamedStore
  );
  let assemblyFormat = [{
    `<` $outputAxes `,` $reductionAxes `,` $cases `,` $maximumLMUL `,`
        $unroll `,` $pipelineDepth `,` $streamedStore `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_TargetAttr : WeftRISCV_Attr<"Target", "target"> {
  let parameters = (ins
    StringRefParameter<"target triple">:$triple,
    StringRefParameter<"RISC-V ISA string">:$march,
    StringRefParameter<"ABI">:$abi,
    "bool":$hasRVV,
    "bool":$hasVectorF16,
    "bool":$hasVectorF16Conversion,
    "bool":$hasIndexedMemory,
    "bool":$hasSegmentMemory,
    "bool":$hasWideningInteger,
    "bool":$hasWideningFloat,
    "int64_t":$vlenBits,
    "int64_t":$vectorRegisters,
    "int64_t":$maxPrivateStackBytes,
    "::mlir::DenseI64ArrayAttr":$supportedSEW,
    "::mlir::DenseI64ArrayAttr":$legalLMULEighths,
    "int64_t":$maxWideningCombineGroups,
    StringRefParameter<"partial combine structural priority">:$partialCombinePolicy,
    StringRefParameter<"record-axis placement structural priority">:$recordAxisPolicy,
    "::mlir::ArrayAttr":$fragments,
    "::mlir::Attribute":$dataCache
  );
  let assemblyFormat = [{
    `<` $triple `,` $march `,` $abi `,` $hasRVV `,` $hasVectorF16 `,` $hasVectorF16Conversion `,`
        $hasIndexedMemory `,` $hasSegmentMemory `,` $hasWideningInteger `,`
        $hasWideningFloat `,` $vlenBits `,` $vectorRegisters `,`
        $maxPrivateStackBytes `,` $supportedSEW `,` $legalLMULEighths `,`
        $maxWideningCombineGroups `,` $partialCombinePolicy `,`
        $recordAxisPolicy `,` $fragments `,` $dataCache `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_AccessAttr : WeftRISCV_Attr<"Access", "access"> {
  let parameters = (ins
    StringRefParameter<"unassigned, unit, strided, indexed, segment, register, or local">:$form,
    StringRefParameter<"dense, natural, grouped_layered, joined, or opaque">:$mapping,
    "int64_t":$alignment,
    "int64_t":$indexSEW,
    "int64_t":$segmentFields,
    "int64_t":$groupSize,
    "int64_t":$layerSize,
    "int64_t":$joinFields,
    "int64_t":$joinLowBits,
    "int64_t":$joinRole,
    "int64_t":$bitOffset,
    "int64_t":$storageBits,
    StringRefParameter<"none, lo_first, or hi_first">:$order,
    "int64_t":$fieldChunkBytes
  );
  let assemblyFormat = [{
    `<` $form `,` $mapping `,` $alignment `,` $indexSEW `,`
        $segmentFields `,` $groupSize `,` $layerSize `,` $joinFields `,`
        $joinLowBits `,` $joinRole `,` $bitOffset `,`
        $storageBits `,` $order `,` $fieldChunkBytes `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ScheduleAttr : WeftRISCV_Attr<"Schedule", "schedule"> {
  let parameters = (ins
    StringRefParameter<"sequential or pipelined">:$kind,
    "int64_t":$unroll,
    "int64_t":$pipelineDepth,
    "int64_t":$bufferCount,
    "int64_t":$prefetchDistance
  );
  let assemblyFormat = [{
    `<` $kind `,` $unroll `,` $pipelineDepth `,` $bufferCount `,`
        $prefetchDistance `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FieldStorePlanAttr
    : WeftRISCV_Attr<"FieldStorePlan", "field_store_plan"> {
  let parameters = (ins
    StringRefParameter<"scalar, scalar_bytes, natural, or layered">:$kind,
    "int64_t":$logicalBits,
    "int64_t":$storeBits,
    "int64_t":$lanes,
    "int64_t":$sourcesPerStore,
    "::mlir::DenseI64ArrayAttr":$byteOffsets,
    "::mlir::DenseI64ArrayAttr":$sourceParts,
    "::mlir::DenseI64ArrayAttr":$bitShifts
  );
  let assemblyFormat = [{
    `<` $kind `,` $logicalBits `,` $storeBits `,` $lanes `,`
        $sourcesPerStore `,` $byteOffsets `,` $sourceParts `,` $bitShifts `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PartialTopologyAttr
    : WeftRISCV_Attr<"PartialTopology", "partial_topology"> {
  let parameters = (ins
    StringRefParameter<"unassigned, sequential_fused, sequential_per_stream, independent, replica_reduced, scaled, reduced_scaled, level_scaled, merged, or layered">:$kind,
    "int64_t":$rootOperand,
    "::mlir::DenseI64ArrayAttr":$partialAxes,
    "::mlir::DenseI64ArrayAttr":$outputAxes,
    "int64_t":$sourceSlots,
    "int64_t":$partialSlots,
    "int64_t":$outputReplicas,
    "int64_t":$laneSplit,
    "int64_t":$combineArity,
    "::mlir::DenseI64ArrayAttr":$slotOrder,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $kind `,` $rootOperand `,` $partialAxes `,` $outputAxes `,`
        $sourceSlots `,` $partialSlots `,` $outputReplicas `,` $laneSplit `,`
        $combineArity `,` $slotOrder `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PackedPlaneMergePlanAttr
    : WeftRISCV_Attr<"PackedPlaneMergePlan", "packed_plane_merge_plan"> {
  let parameters = (ins
    "int64_t":$logicalAxis,
    "int64_t":$logicalElements,
    "int64_t":$lowBits,
    "int64_t":$highBits,
    "int64_t":$groupSize,
    "int64_t":$lowLayerBytes,
    "int64_t":$highLayerBytes,
    "int64_t":$lowByteOffset,
    "int64_t":$highByteOffset,
    "int64_t":$insertBit
  );
  let assemblyFormat = [{
    `<` $logicalAxis `,` $logicalElements `,` $lowBits `,` $highBits `,`
        $groupSize `,` $lowLayerBytes `,` $highLayerBytes `,`
        $lowByteOffset `,` $highByteOffset `,` $insertBit `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PartialLayoutPlanAttr
    : WeftRISCV_Attr<"PartialLayoutPlan", "partial_layout_plan"> {
  let parameters = (ins
    "::mlir::Type":$sourceLhsType,
    "::mlir::Type":$sourceRhsType,
    "::mlir::Type":$sourceSlotType,
    "::mlir::Type":$splitSourceSlotType,
    "::mlir::Type":$partialSlotType,
    "::mlir::Type":$widenedSlotType,
    "::mlir::Type":$reducedSlotType,
    "::mlir::Type":$vectorSlotType,
    "::mlir::Type":$sourceSetType,
    "::mlir::Type":$repackedSetType,
    "::mlir::Type":$partialSetType,
    "::mlir::Type":$scaledSetType,
    "::mlir::Type":$directSetType,
    "::mlir::Type":$reducedSetType,
    "::mlir::Type":$scaleCombinedSetType,
    "::mlir::Type":$fullScaledSetType,
    "::mlir::DenseI64ArrayAttr":$sourceLhsParts,
    "::mlir::DenseI64ArrayAttr":$sourceRhsParts,
    "::mlir::DenseI64ArrayAttr":$sourceLhsOffsets,
    "::mlir::DenseI64ArrayAttr":$sourceRhsOffsets,
    "::mlir::Attribute":$repackLeaf
  );
  let assemblyFormat = [{
    `<` $sourceLhsType `,` $sourceRhsType `,`
        $sourceSlotType `,` $splitSourceSlotType `,` $partialSlotType `,`
        $widenedSlotType `,` $reducedSlotType `,` $vectorSlotType `,`
        $sourceSetType `,` $repackedSetType `,` $partialSetType `,`
        $scaledSetType `,` $directSetType `,` $reducedSetType `,`
        $scaleCombinedSetType `,` $fullScaledSetType `,`
        $sourceLhsParts `,` $sourceRhsParts `,`
        $sourceLhsOffsets `,` $sourceRhsOffsets `,` $repackLeaf `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PartialCombinePlanAttr
    : WeftRISCV_Attr<"PartialCombinePlan", "partial_combine_plan"> {
  let parameters = (ins
    StringRefParameter<"level_scaled, independent_vector, or independent_scalar">:$realization,
    "::mlir::Type":$sourceSetType,
    "::mlir::Type":$reducedSetType,
    "::mlir::Type":$finalSetType,
    "::mlir::ArrayAttr":$combineSetTypes,
    "::mlir::DenseI64ArrayAttr":$combineArities,
    "::mlir::DenseI64ArrayAttr":$lhsParts,
    "::mlir::DenseI64ArrayAttr":$rhsParts,
    "::mlir::DenseI64ArrayAttr":$scaleParts,
    StringRefParameter<"selected widening multiply instruction">:$multiplyInstruction,
    StringRefParameter<"selected partial finalization instruction">:$finalizeInstruction,
    "int64_t":$outputParts,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $realization `,` $sourceSetType `,` $reducedSetType `,`
        $finalSetType `,` $combineSetTypes `,` $combineArities `,`
        $lhsParts `,` $rhsParts `,` $scaleParts `,`
        $multiplyInstruction `,` $finalizeInstruction `,`
        $outputParts `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PartialAddTreePlanAttr
    : WeftRISCV_Attr<"PartialAddTreePlan", "partial_add_tree_plan"> {
  let parameters = (ins
    "::mlir::ArrayAttr":$leafSetTypes,
    "::mlir::ArrayAttr":$normalizedSetTypes,
    "::mlir::ArrayAttr":$leafMultiplyInstructions,
    "::mlir::DenseI64ArrayAttr":$splits,
    "::mlir::Type":$mergedSetType,
    StringRefParameter<"selected partial finalization instruction">:$finalizeInstruction,
    "int64_t":$totalSlots,
    "int64_t":$totalTerms,
    "int64_t":$totalResourceGroups,
    "::mlir::ArrayAttr":$repackLeaves
  );
  let assemblyFormat = [{
    `<` $leafSetTypes `,` $normalizedSetTypes `,`
        $leafMultiplyInstructions `,` $splits `,` $mergedSetType `,`
        $finalizeInstruction `,` $totalSlots `,` $totalTerms `,`
        $totalResourceGroups `,` $repackLeaves `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_NestedPartialPlanAttr
    : WeftRISCV_Attr<"NestedPartialPlan", "nested_partial_plan"> {
  let parameters = (ins
    "int64_t":$windowAxis,
    "int64_t":$issueStreams,
    "int64_t":$windowExtent,
    "::mlir::DenseI64ArrayAttr":$partialAxes,
    "::mlir::DenseI64ArrayAttr":$partialAxisExtents,
    "int64_t":$partialSlots,
    "int64_t":$issueUnroll,
    "::mlir::DenseI64ArrayAttr":$outputAxes,
    "int64_t":$outputReplicas,
    "::mlir::DenseI64ArrayAttr":$issueMaterializationOrder,
    "::mlir::DenseI64ArrayAttr":$lhsSourceParts,
    "::mlir::DenseI64ArrayAttr":$rhsSourceParts,
    "::mlir::DenseI64ArrayAttr":$lhsLaneOffsets,
    "::mlir::DenseI64ArrayAttr":$rhsLaneOffsets,
    "::mlir::DenseI64ArrayAttr":$scaleReplicas,
    "::mlir::Type":$issueLhsType,
    "::mlir::Type":$issueRhsType,
    "::mlir::Type":$sourceSlotType,
    "::mlir::Type":$splitSlotType,
    "::mlir::Type":$reducedSlotType,
    "::mlir::Type":$scaleReplicaType,
    "::mlir::Type":$sourceSetType,
    "::mlir::Type":$repackedSetType,
    "::mlir::Type":$reducedSetType,
    "::mlir::Type":$scaleCombinedSetType,
    "::mlir::ArrayAttr":$combineSetTypes,
    "::mlir::DenseI64ArrayAttr":$combineAxes,
    "::mlir::DenseI64ArrayAttr":$combineArities,
    StringRefParameter<"selected scale supply form">:$scaleSupply,
    StringRefParameter<"selected scale supply stage">:$scaleSupplyStage,
    StringRefParameter<"selected widening multiply instruction">:$multiplyInstruction,
    StringRefParameter<"selected partial reduction instruction">:$reduceInstruction,
    StringRefParameter<"selected partial finalization instruction">:$finalizeInstruction,
    "int64_t":$resourceGroups,
    "::mlir::Attribute":$repackLeaf
  );
  let assemblyFormat = [{
    `<` $windowAxis `,` $issueStreams `,` $windowExtent `,`
        $partialAxes `,`
        $partialAxisExtents `,` $partialSlots `,` $issueUnroll `,`
        $outputAxes `,` $outputReplicas `,`
        $issueMaterializationOrder `,`
        $lhsSourceParts `,`
        $rhsSourceParts `,` $lhsLaneOffsets `,` $rhsLaneOffsets `,`
        $scaleReplicas `,`
        $issueLhsType `,` $issueRhsType `,` $sourceSlotType `,`
        $splitSlotType `,` $reducedSlotType `,` $scaleReplicaType `,`
        $sourceSetType `,` $repackedSetType `,` $reducedSetType `,`
        $scaleCombinedSetType `,` $combineSetTypes `,`
        $combineAxes `,` $combineArities `,` $scaleSupply `,`
        $scaleSupplyStage `,`
        $multiplyInstruction `,`
        $reduceInstruction `,` $finalizeInstruction `,` $resourceGroups `,`
        $repackLeaf `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ScaledPartialPlanAttr
    : WeftRISCV_Attr<"ScaledPartialPlan", "scaled_partial_plan"> {
  let parameters = (ins
    StringRefParameter<"scaled or reduced_scaled">:$realization,
    "::mlir::Type":$scalarScaleType,
    "::mlir::Type":$narrowScaleType,
    StringRefParameter<"scalar-rematerialize or vector-convert">:$scaleSupply,
    StringRefParameter<"selected widening multiply instruction">:$multiplyInstruction,
    StringRefParameter<"selected partial reduction instruction or none">:$reduceInstruction,
    StringRefParameter<"selected partial finalization instruction">:$finalizeInstruction,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $realization `,` $scalarScaleType `,` $narrowScaleType `,`
        $scaleSupply `,` $multiplyInstruction `,` $reduceInstruction `,`
        $finalizeInstruction `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_SequentialPartialPlanAttr
    : WeftRISCV_Attr<"SequentialPartialPlan", "sequential_partial_plan"> {
  let parameters = (ins
    StringRefParameter<"fused, per_stream, or fused_partial">:$realization,
    "int64_t":$issueCount,
    "::mlir::DenseI64ArrayAttr":$reductionAxes,
    "::mlir::Type":$lhsIssueType,
    "::mlir::Type":$rhsIssueType,
    "::mlir::Type":$accumulatorType,
    StringRefParameter<"slice or storage-rematerialize">:$lhsIssueSupply,
    StringRefParameter<"slice or storage-rematerialize">:$rhsIssueSupply,
    "::mlir::DenseI64ArrayAttr":$lhsSourceParts,
    "::mlir::DenseI64ArrayAttr":$rhsSourceParts,
    "::mlir::DenseI64ArrayAttr":$lhsLaneOffsets,
    "::mlir::DenseI64ArrayAttr":$rhsLaneOffsets,
    StringRefParameter<"selected widened multiply instruction">:$multiplyInstruction,
    StringRefParameter<"selected widened accumulate instruction">:$accumulateInstruction,
    StringRefParameter<"selected final reduction instruction">:$finalizeInstruction,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $realization `,` $issueCount `,` $reductionAxes `,`
        $lhsIssueType `,` $rhsIssueType `,`
        $accumulatorType `,` $lhsIssueSupply `,` $rhsIssueSupply `,`
        $lhsSourceParts `,` $rhsSourceParts `,`
        $lhsLaneOffsets `,` $rhsLaneOffsets `,`
        $multiplyInstruction `,` $accumulateInstruction `,`
        $finalizeInstruction `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LayeredStreamGeometryAttr
    : WeftRISCV_Attr<"LayeredStreamGeometry", "layered_stream_geometry"> {
  let parameters = (ins
    "int64_t":$axis,
    "int64_t":$groupSize,
    "int64_t":$layerSize,
    "int64_t":$laneCount,
    "int64_t":$streamCount,
    "int64_t":$replicaCount,
    "::mlir::DenseI64ArrayAttr":$windowForStream,
    "::mlir::DenseI64ArrayAttr":$representativeStreamForWindow,
    "::mlir::DenseI64ArrayAttr":$physicalLayerForStream,
    "::mlir::DenseI64ArrayAttr":$physicalLayerForLogicalLayer,
    "::mlir::DenseI64ArrayAttr":$shiftAmountForLogicalLayer,
    "::mlir::DenseI64ArrayAttr":$maskValueForLogicalLayer,
    "::mlir::DenseI64ArrayAttr":$shiftAmountForStream,
    "::mlir::DenseI64ArrayAttr":$maskValueForStream,
    "::mlir::DenseI64ArrayAttr":$groupForWindow,
    "::mlir::DenseI64ArrayAttr":$withinForWindow
  );
  let assemblyFormat = [{
    `<` $axis `,` $groupSize `,` $layerSize `,` $laneCount `,`
        $streamCount `,` $replicaCount `,` $windowForStream `,`
        $representativeStreamForWindow `,` $physicalLayerForStream `,`
        $physicalLayerForLogicalLayer `,`
        $shiftAmountForLogicalLayer `,` $maskValueForLogicalLayer `,`
        $shiftAmountForStream `,` $maskValueForStream `,`
        $groupForWindow `,` $withinForWindow `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_StorageWindowPlanAttr
    : WeftRISCV_Attr<"StorageWindowPlan", "storage_window_plan"> {
  let parameters = (ins
    StringRefParameter<"unit, strided, repeat, layered, interleaved_natural, or interleaved_joined">:$kind,
    "int64_t":$reductionAxis,
    "int64_t":$projectionBase,
    "int64_t":$projectionStride,
    "int64_t":$projectionRepeat,
    "int64_t":$projectionExtent,
    "int64_t":$offsetAlignment,
    "int64_t":$recordElements,
    "int64_t":$byteOffset,
    "int64_t":$elementBits,
    "int64_t":$groupSize,
    "int64_t":$layerSize,
    "int64_t":$physicalLayerBase,
    "int64_t":$physicalLayerStep,
    "int64_t":$shiftBase,
    "int64_t":$shiftStep,
    "int64_t":$maskValue
  );
  let assemblyFormat = [{
    `<` $kind `,` $reductionAxis `,` $projectionBase `,`
        $projectionStride `,` $projectionRepeat `,` $projectionExtent `,`
        $offsetAlignment `,` $recordElements `,` $byteOffset `,`
        $elementBits `,` $groupSize `,` $layerSize `,`
        $physicalLayerBase `,` $physicalLayerStep `,` $shiftBase `,`
        $shiftStep `,` $maskValue `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LocalPackPlanAttr
    : WeftRISCV_Attr<"LocalPackPlan", "local_pack_plan"> {
  let parameters = (ins
    StringRefParameter<"currently interleave">:$kind,
    "int64_t":$rowAxis,
    "int64_t":$recordAxis,
    "int64_t":$interleaveRows,
    "int64_t":$recordBytes,
    "int64_t":$recordElements,
    "int64_t":$localBytes
  );
  let assemblyFormat = [{
    `<` $kind `,` $rowAxis `,` $recordAxis `,` $interleaveRows `,`
        $recordBytes `,` $recordElements `,` $localBytes `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LayeredPartialPlanAttr
    : WeftRISCV_Attr<"LayeredPartialPlan", "layered_partial_plan"> {
  let parameters = (ins
    "int64_t":$rootIndex,
    "::mlir::Type":$decodedWindowType,
    "::mlir::Type":$storageWindowType,
    "::mlir::Type":$lhsIssueType,
    "::mlir::Type":$rhsIssueType,
    "::mlir::Type":$accumulatorType,
    "::mlir::ArrayAttr":$rootWindowTypes,
    "::mlir::ArrayAttr":$rootStoragePlans,
    "::mlir::ArrayAttr":$rootWindowInstructions,
    "::mlir::ArrayAttr":$decodeInstructions,
    StringRefParameter<"selected final reduction instruction">:$finalizeInstruction,
    "::weft::riscv::LayeredStreamGeometryAttr":$geometry,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $rootIndex `,` $decodedWindowType `,` $storageWindowType `,`
        $lhsIssueType `,` $rhsIssueType `,` $accumulatorType `,`
        $rootWindowTypes `,` $rootStoragePlans `,`
        $rootWindowInstructions `,` $decodeInstructions `,`
        $finalizeInstruction `,` $geometry `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_GroupedMacPlanAttr
    : WeftRISCV_Attr<"GroupedMacPlan", "grouped_mac_plan"> {
  let parameters = (ins
    StringRefParameter<"reduce, compact, or fragmented">:$kind,
    "int64_t":$group,
    "int64_t":$unroll,
    "int64_t":$reductionAxis,
    "int64_t":$storageGroup,
    "int64_t":$storageLayer,
    "int64_t":$logicalWidth,
    "int64_t":$lhsBitOffsetBytes,
    "int64_t":$rhsBitOffsetBytes,
    "int64_t":$recordElements,
    "int64_t":$interleaveRows,
    StringRefParameter<"unit or strided">:$loadForm,
    "int64_t":$rowStrideAxis,
    "int64_t":$termByteStride,
    "int64_t":$physicalLayerBase,
    "int64_t":$physicalLayerStep,
    "int64_t":$shiftBase,
    "int64_t":$shiftStep,
    "int64_t":$maskValue,
    "::mlir::DenseI64ArrayAttr":$termOrder
  );
  let assemblyFormat = [{
    `<` $kind `,` $group `,` $unroll `,` $reductionAxis `,`
        $storageGroup `,` $storageLayer `,` $logicalWidth `,`
        $lhsBitOffsetBytes `,` $rhsBitOffsetBytes `,` $recordElements `,`
        $interleaveRows `,` $loadForm `,` $rowStrideAxis `,`
        $termByteStride `,` $physicalLayerBase `,` $physicalLayerStep `,`
        $shiftBase `,` $shiftStep `,`
        $maskValue `,` $termOrder `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LevelAttr : WeftRISCV_Attr<"Level", "level"> {
  let parameters = (ins
    "int64_t":$domainId,
    "int64_t":$axisId,
    StringRefParameter<"rows, cols, tiles, blocks, or subs">:$relation,
    "int64_t":$stateBirthCount,
    "int64_t":$stagedBirthCount,
    "int64_t":$handoffCount,
    StringRefParameter<"ascending or descending">:$direction
  );
  let assemblyFormat = [{
    `<` $domainId `,` $axisId `,` $relation `,` $stateBirthCount `,`
        $stagedBirthCount `,` $handoffCount `,` $direction `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ImplementationAttr
    : WeftRISCV_Attr<"Implementation", "implementation"> {
  let parameters = (ins
    StringRefParameter<"scalar, rvv, ime, or transfer">:$engine,
    StringRefParameter<"local semantic implementation family">:$family,
    StringRefParameter<"target operation class">:$operation,
    "::mlir::DenseI64ArrayAttr":$parameters
  );
  let assemblyFormat =
      "`<` $engine `,` $family `,` $operation `,` $parameters `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_LeafAttr : WeftRISCV_Attr<"Leaf", "leaf"> {
  let parameters = (ins
    StringRefParameter<"unselected, scalar, rvv, ime, or transfer">:$engine,
    StringRefParameter<"local operation family">:$family,
    StringRefParameter<"instruction identity">:$instruction,
    StringRefParameter<"terminal spelling key">:$spelling,
    "int64_t":$operandGroups,
    "int64_t":$resultGroups,
    "int64_t":$temporaryGroups,
    "int64_t":$fragmentGroups,
    StringRefParameter<"none, explicit, or tail">:$mask,
    StringRefParameter<"exact or agnostic">:$tail,
    "::mlir::DenseI64ArrayAttr":$parameters,
    "int64_t":$localBytes
  );
  let assemblyFormat = [{
    `<` $engine `,` $family `,` $instruction `,` $spelling `,`
        $operandGroups `,` $resultGroups `,` $temporaryGroups `,`
        $fragmentGroups `,` $mask `,` $tail `,` $parameters `,`
        $localBytes `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ConversionAttr : WeftRISCV_Attr<"Conversion", "conversion"> {
  let parameters = (ins
    StringRefParameter<"splat, extract, reshape, tuple, register_to_lane, time_to_lane, lane_to_register, rvv_to_fragment, fragment_to_rvv, local_load, or local_store">:$kind,
    StringRefParameter<"pure, read, write, or handoff">:$effect,
    "int64_t":$temporaryGroups
  );
  let assemblyFormat = "`<` $kind `,` $effect `,` $temporaryGroups `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_ValueType : WeftRISCV_Type<"Value", "value"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::weft::riscv::LayoutAttr":$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape `,` $axisIds `,` $layout `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_MemDescType : WeftRISCV_Type<"MemDesc", "memdesc"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::mlir::DenseI64ArrayAttr":$strides,
    "::mlir::DenseI64ArrayAttr":$origins,
    "int64_t":$alignment,
    StringRefParameter<"pinned, slice, or local">:$addressClass,
    StringRefParameter<"none, read, write, or readwrite">:$access,
    "int64_t":$aliasSet,
    StringRefParameter<"pinned storage layout identity">:$layoutIdentity,
    "int64_t":$storageBits,
    "int64_t":$elements,
    "int64_t":$interleaveRows
  );
  let assemblyFormat = [{
    `<` $encoding `,` $shape `,` $axisIds `,` $strides `,` $origins `,`
        $alignment `,` $addressClass `,` $access `,` $aliasSet `,`
        $layoutIdentity `,` $storageBits `,` $elements `,` $interleaveRows `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LocalType : WeftRISCV_Type<"Local", "local"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "int64_t":$sizeBytes,
    "int64_t":$alignment,
    "int64_t":$aliasSet,
    StringRefParameter<"spill, pack, pipeline, or handoff">:$purpose,
    "int64_t":$ownerDomainId,
    "int64_t":$birthId,
    "int64_t":$lifetimeEndDomainId,
    StringRefParameter<"physical storage schema">:$schema
  );
  let assemblyFormat = [{
    `<` $elementType `,` $shape `,` $axisIds `,` $sizeBytes `,` $alignment `,`
        $aliasSet `,` $purpose `,` $ownerDomainId `,` $birthId `,`
        $lifetimeEndDomainId `,` $schema `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FragmentType : WeftRISCV_Type<"Fragment", "fragment"> {
  let parameters = (ins
    StringRefParameter<"fragment family">:$family,
    StringRefParameter<"lhs, rhs, or accumulator fragment role">:$role,
    "::weft::riscv::FragmentPackingAttr":$packing,
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::weft::riscv::LayoutAttr":$layout,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $family `,` $role `,` $packing `,` $elementType `,` $shape `,`
        $axisIds `,` $layout `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_WindowType : WeftRISCV_Type<"Window", "window"> {
  let parameters = (ins
    StringRefParameter<"grouped-mac, encoded-dot, or contract">:$family,
    "::mlir::Type":$lhsType,
    "::mlir::Type":$rhsType,
    "::mlir::Type":$resultType,
    "int64_t":$reductionAxis,
    "int64_t":$slots,
    "int64_t":$termsPerSlot,
    "int64_t":$resultParts,
    "::weft::riscv::LayoutAttr":$partialLayout,
    "::weft::riscv::LayoutAttr":$resultLayout,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $family `,` $lhsType `,` $rhsType `,` $resultType `,` $reductionAxis `,` $slots `,`
        $termsPerSlot `,` $resultParts `,` $partialLayout `,` $resultLayout `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LayeredWindowType
    : WeftRISCV_Type<"LayeredWindow", "layered_window"> {
  let parameters = (ins
    "::weft::riscv::ValueType":$fieldType,
    "::weft::riscv::ValueType":$resultType,
    "int64_t":$reductionAxis,
    "int64_t":$layers,
    "int64_t":$windowsPerLayer,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $fieldType `,` $resultType `,` $reductionAxis `,` $layers `,`
        $windowsPerLayer `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PartialSetType
    : WeftRISCV_Type<"PartialSet", "partial_set"> {
  let parameters = (ins
    "::weft::riscv::ValueType":$partialType,
    "int64_t":$reductionAxis,
    "int64_t":$slots,
    "int64_t":$termsPerSlot,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $partialType `,` $reductionAxis `,` $slots `,` $termsPerSlot `,`
        $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_DomainType : WeftRISCV_Type<"Domain", "domain"> {
  let parameters = (ins
    StringRefParameter<"axis name">:$axisName,
    "int64_t":$domainId,
    "int64_t":$parentDomainId,
    "int64_t":$axisId,
    StringRefParameter<"root, rows, cols, tiles, blocks, or subs">:$relation,
    StringRefParameter<"exact or tail">:$tail
  );
  let assemblyFormat = [{
    `<` $axisName `,` $domainId `,` $parentDomainId `,` $axisId `,`
        $relation `,` $tail `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PointType : WeftRISCV_Type<"Point", "point"> {
  let parameters = (ins WeftRISCV_DomainType:$domain);
  let assemblyFormat = "`<` $domain `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_RecordCohortType
    : WeftRISCV_Type<"RecordCohort", "record_cohort"> {
  let parameters = (ins
    WeftRISCV_DomainType:$domain,
    "int64_t":$width,
    "int64_t":$partition
  );
  let assemblyFormat = "`<` $domain `,` $width `,` $partition `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_EncodingDeclOp
    : WeftRISCV_TerminalOp<"encoding_decl", [Symbol, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
                       StrAttr:$layout_identity, StrAttr:$bit_order,
                       StrAttr:$byte_order, I64Attr:$alignment,
                       I64Attr:$elements, I64Attr:$storage_bits,
                       ArrayAttr:$field_names, ArrayAttr:$field_types,
                       ArrayAttr:$field_shapes, ArrayAttr:$field_layouts,
                       DenseI64ArrayAttr:$field_bit_offsets,
                       DenseI64ArrayAttr:$field_storage_bits,
                       DenseI64ArrayAttr:$padding);
  let hasVerifier = 1;
}

def WeftRISCV_DerivedEncodingOp
    : WeftRISCV_TerminalOp<"derived_encoding", [Symbol, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$source_family,
                       StrAttr:$result_family, StrAttr:$layout_identity,
                       ArrayAttr:$parameter_names,
                       DenseI64ArrayAttr:$parameter_values,
                       I64Attr:$interleave_rows,
                       DenseI64ArrayAttr:$field_chunks);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactPackOp
    : WeftRISCV_TerminalOp<"artifact_pack", [Symbol, IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       WeftRISCV_TargetAttr:$target,
                       StrAttr:$source_family, StrAttr:$result_family,
                       StrAttr:$layout_identity, I64Attr:$record_bytes,
                       I64Attr:$elements, I64Attr:$interleave_rows,
                       DenseI64ArrayAttr:$field_chunks);
  let regions = (region SizedRegion<1>:$size_body,
                        SizedRegion<1>:$pack_body);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactSizeYieldOp
    : WeftRISCV_TerminalOp<"artifact_size_yield", [Terminator]> {
  let arguments = (ins Index:$value);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactReturnOp
    : WeftRISCV_TerminalOp<"artifact_return", [Terminator]> {
  let hasVerifier = 1;
}

def WeftRISCV_StorageLoadOp
    : WeftRISCV_LeafOp<"storage_load"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source, Index:$byte_index,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyInteger:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StorageStoreOp
    : WeftRISCV_LeafOp<"storage_store"> {
  let arguments = (ins AnyInteger:$value, Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       Index:$byte_index, WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_KernelOp
    : WeftRISCV_TerminalOp<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins SymbolNameAttr:$sym_name, ArrayAttr:$arg_names,
                       ArrayAttr:$arg_access,
                       DenseI64ArrayAttr:$arg_alias_sets,
                       OptionalAttr<StrArrayAttr>:$strided_args,
                       ArrayAttr:$shape_symbols, StrAttr:$source,
                       WeftRISCV_TargetAttr:$target,
                       StrArrayAttr:$required_extensions,
                       StrAttr:$local_tile_strategy,
                       I64Attr:$pointwise_issue_copies,
                       I64Attr:$contraction_column_copies,
                       I64Attr:$panel_transfer_fields,
                       StrAttr:$panel_transfer_strategy,
                       I64Attr:$contraction_panel_k,
                       I64Attr:$contraction_k_steps,
                       ArrayAttr:$parameter_names,
                       DenseI64ArrayAttr:$parameter_values,
                       I64Attr:$vector_register_peak,
                       I64Attr:$fragment_register_peak,
                       I64Attr:$local_storage_bytes,
                       BoolAttr:$resources_materialized);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftRISCV_ReturnOp : WeftRISCV_TerminalOp<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftRISCV_RootDomainOp : WeftRISCV_TerminalOp<"root_domain", [Pure]> {
  let results = (outs WeftRISCV_DomainType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SymbolOp : WeftRISCV_TerminalOp<"symbol", [Pure]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StrideOp : WeftRISCV_TerminalOp<"stride", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$view, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DenseDisjointOp : WeftRISCV_TerminalOp<"dense_disjoint", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$lhs, WeftRISCV_MemDescType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DenseElementwiseAliasOp : WeftRISCV_TerminalOp<"dense_elementwise_alias", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$lhs, WeftRISCV_MemDescType:$rhs);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DenseInjectiveOp : WeftRISCV_TerminalOp<"dense_injective", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$view);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DomainOp : WeftRISCV_TerminalOp<"domain", [Pure]> {
  let arguments = (ins WeftRISCV_DomainType:$parent, Index:$extent,
                       Index:$partition, Index:$multiplicity);
  let results = (outs WeftRISCV_DomainType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LoopOp : WeftRISCV_Op<"loop", [RecursiveMemoryEffects]> {
  let arguments = (ins WeftRISCV_DomainType:$domain,
                       WeftRISCV_PointType:$parent_point,
                       Variadic<AnyType>:$carried,
                       I64Attr:$state_birth_count,
                       I64Attr:$staged_birth_count,
                       I64Attr:$handoff_count,
                       WeftRISCV_ScheduleAttr:$schedule);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftRISCV_YieldOp : WeftRISCV_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftRISCV_RootPointOp : WeftRISCV_TerminalOp<"root_point"> {
  let arguments = (ins WeftRISCV_DomainType:$domain);
  let results = (outs WeftRISCV_PointType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_PhysicalPointOp : WeftRISCV_TerminalOp<"physical_point"> {
  let arguments = (ins WeftRISCV_PointType:$parent, Index:$base,
                       Index:$active, Index:$partition);
  let results = (outs WeftRISCV_PointType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RecordCohortOp : WeftRISCV_TerminalOp<"record_cohort"> {
  let arguments = (ins WeftRISCV_PointType:$origin);
  let results = (outs WeftRISCV_RecordCohortType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ConstantOp : WeftRISCV_TerminalOp<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IotaOp : WeftRISCV_LeafOp<"iota", [Pure]> {
  let arguments = (ins I64Attr:$start, I64Attr:$end, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_NewOp : WeftRISCV_TerminalOp<"new", [Pure]> {
  let arguments = (ins Optional<AnyType>:$initial, BoolAttr:$initialized,
                       StrAttr:$placement, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalShapeOp : WeftRISCV_TerminalOp<"local_shape", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins Index:$extent, I64Attr:$id);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftRISCV_EmptyOp : WeftRISCV_Op<"empty"> {
  let arguments = (ins Variadic<Index>:$extents, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id);
  let results = (outs Res<WeftRISCV_ValueType, "", [MemAlloc]>:$result);
  let hasVerifier = 1;
}

def WeftRISCV_MaterializeOp : WeftRISCV_Op<"materialize", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$placement,
                       StrAttr:$schema,
                       OptionalAttr<WeftRISCV_LocalPackPlanAttr>:$local_pack_plan,
                       I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StagedViewOp : WeftRISCV_TerminalOp<"staged_view", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$source, StrAttr:$schema,
                       I64Attr:$owner_domain_id, I64Attr:$birth_id,
                       I64Attr:$lifetime_end_domain_id);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_MemoryViewOp
    : WeftRISCV_TerminalOp<"memory_view", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins WeftRISCV_MemDescType:$base,
                       Variadic<Index>:$extents,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$origins);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_FieldViewOp : WeftRISCV_TerminalOp<"field_view", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$field, Index:$origin);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SliceOp : WeftRISCV_TerminalOp<"slice", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$base,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SubviewOp : WeftRISCV_TerminalOp<"subview", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$base,
                       Variadic<Index>:$dynamic_bounds,
                       DenseI64ArrayAttr:$offsets,
                       DenseI64ArrayAttr:$extents);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ExtentOp : WeftRISCV_TerminalOp<"extent", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$base, I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}

def WeftRISCV_ReaxisOp : WeftRISCV_TerminalOp<"reaxis", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$base);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ReshapeOp : WeftRISCV_TerminalOp<"reshape", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       DenseI64ArrayAttr:$order);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LoadOp : WeftRISCV_LeafOp<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins WeftRISCV_MemDescType:$region,
                       Optional<WeftRISCV_LocalType>:$snapshot_storage,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StoreOp : WeftRISCV_LeafOp<"store"> {
  let arguments = (ins AnyType:$value, Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$region,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf,
                       OptionalAttr<WeftRISCV_FieldStorePlanAttr>:$field_store_plan);
  let hasVerifier = 1;
}

def WeftRISCV_AtomicRMWOp : WeftRISCV_LeafOp<"atomic_rmw"> {
  let arguments = (ins AnyType:$value, Arg<WeftRISCV_MemDescType, "", [MemRead, MemWrite]>:$region,
                       StrAttr:$kind, StrAttr:$ordering,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_FieldOp : WeftRISCV_LeafOp<"field", [Pure]> {
  let arguments = (ins AnyType:$owner, StrAttr:$name,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ExtractOp : WeftRISCV_LeafOp<"extract", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_UpdateOp : WeftRISCV_LeafOp<"update", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, AnyType:$value,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let extraClassDeclaration = [{ bool coversWholeState(); }];
  let hasVerifier = 1;
}

def WeftRISCV_UnaryOp : WeftRISCV_LeafOp<"unary", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_BitcastOp : WeftRISCV_LeafOp<"bitcast", [Pure]> {
  let arguments = (ins AnyType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_BinaryOp : WeftRISCV_LeafOp<"binary", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_CompareOp : WeftRISCV_LeafOp<"compare", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SelectOp : WeftRISCV_LeafOp<"select", [Pure]> {
  let arguments = (ins AnyType:$condition, AnyType:$true_value,
                       AnyType:$false_value, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVCompareSelectOp : WeftRISCV_LeafOp<"rvv_compare_select", [Pure]> {
  let summary = "One typed vector comparison consumed directly as a merge mask";
  let arguments = (ins WeftRISCV_ValueType:$lhs, AnyType:$rhs,
                       AnyType:$true_value, WeftRISCV_ValueType:$false_value,
                       StrAttr:$predicate, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateCompareOp : WeftRISCV_LeafOp<"rvv_predicate_compare", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs, AnyType:$rhs,
                       StrAttr:$predicate, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateSplatOp : WeftRISCV_LeafOp<"rvv_predicate_splat", [Pure]> {
  let arguments = (ins AnyType:$scalar, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateFromValueOp : WeftRISCV_LeafOp<"rvv_predicate_from_value", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateToValueOp : WeftRISCV_LeafOp<"rvv_predicate_to_value", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateLogicOp : WeftRISCV_LeafOp<"rvv_predicate_logic", [Pure]> {
  let arguments = (ins Variadic<WeftRISCV_ValueType>:$inputs, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPredicateSelectOp : WeftRISCV_LeafOp<"rvv_predicate_select", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$condition, AnyType:$true_value,
                       WeftRISCV_ValueType:$false_value, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

class WeftRISCV_ConversionOp<string mnemonic>
    : WeftRISCV_LeafOp<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_CastOp : WeftRISCV_ConversionOp<"cast">;

def WeftRISCV_NarrowOp : WeftRISCV_LeafOp<"narrow", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$rounding, BoolAttr:$saturate,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_WidenOp : WeftRISCV_ConversionOp<"widen">;

def WeftRISCV_MacGroupsOp : WeftRISCV_Op<"mac_groups", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$group,
                       StrAttr:$overflow, WeftRISCV_ScheduleAttr:$schedule,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ReduceOp : WeftRISCV_LeafOp<"reduce", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind, I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_Fold2Op : WeftRISCV_LeafOp<"fold2", [Pure]> {
  let arguments = (ins AnyType:$input, WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftRISCV_ContractLikeOp<string mnemonic>
    : WeftRISCV_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       DenseI64ArrayAttr:$over,
                       OptionalAttr<TypeAttr>:$acc_type,
                       StrAttr:$lane_operand,
                       StrAttr:$lane_memory_form,
                       WeftRISCV_ScheduleAttr:$schedule,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DotOp : WeftRISCV_ContractLikeOp<"dot">;
def WeftRISCV_ContractOp : WeftRISCV_ContractLikeOp<"contract">;
def WeftRISCV_OuterContractOp
    : WeftRISCV_ContractLikeOp<"outer_contract">;

def WeftRISCV_LookupOp : WeftRISCV_LeafOp<"lookup", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins AnyType:$table, AnyType:$indices, StrAttr:$bounds,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVByteGatherOp
    : WeftRISCV_LeafOp<"rvv_byte_gather"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$source,
                       WeftRISCV_ValueType:$byte_offsets,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVByteWindowsLoadOp
    : WeftRISCV_LeafOp<"rvv_byte_windows_load"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$source, Variadic<AnyType>:$byte_bases,
                       I64Attr:$window_bytes,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVIndexedEntryLoadOp
    : WeftRISCV_LeafOp<"rvv_indexed_entry_load"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$source,
                       AnyType:$entry_offsets,
                       I64Attr:$source_axis,
                       I64Attr:$payload_axis,
                       I64Attr:$payload_extent,
                       I64Attr:$entry_byte_stride,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVUnitEntryWindowLoadOp
    : WeftRISCV_LeafOp<"rvv_unit_entry_window_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$source,
                       AnyType:$entry_base,
                       I64Attr:$source_axis,
                       DenseI64ArrayAttr:$entry_axes,
                       DenseI64ArrayAttr:$entry_extents,
                       I64Attr:$payload_axis,
                       I64Attr:$payload_extent,
                       I64Attr:$entry_stride,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ConvertLayoutOp : WeftRISCV_LeafOp<"convert_layout",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_ConversionAttr:$conversion,
                       OptionalAttr<WeftRISCV_AccessAttr>:$source_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_FieldReadOp : WeftRISCV_LeafOp<"field_read"> {
  let summary = "Read an encoded projection into an explicit numeric representation";
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$input,
                       Variadic<AnyType>:$indices,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalAllocOp
    : WeftRISCV_TerminalOp<"local_alloc"> {
  let arguments = (ins Index:$size_bytes);
  let results = (outs Res<WeftRISCV_LocalType, "", [MemAlloc]>:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalSizeOp : WeftRISCV_LeafOp<"local_size", [MemoryEffects<[]>]> {
  let arguments = (ins Variadic<Index>:$extents, I64Attr:$element_bytes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalReuseOp
    : WeftRISCV_TerminalOp<"local_reuse"> {
  let summary = "Transfer a dead private storage slot to a new local object";
  let arguments = (ins Arg<WeftRISCV_LocalType, "", [MemWrite]>:$previous, Index:$size_bytes);
  let results = (outs WeftRISCV_LocalType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalCapacityGuardOp : WeftRISCV_LeafOp<"local_capacity_guard"> {
  let arguments = (ins Index:$size_bytes, Index:$live_bytes, I64Attr:$limit_bytes,
                       I64Attr:$birth_id, WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_LocalBindOp : WeftRISCV_TerminalOp<"local_bind", [Pure]> {
  let arguments = (ins WeftRISCV_LocalType:$storage, Index:$elements,
                       Variadic<Index>:$extents);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalViewOp
    : WeftRISCV_TerminalOp<"local_view", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins WeftRISCV_ValueType:$source,
                       Variadic<Index>:$extents, Variadic<Index>:$strides);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DenseSnapshotOp
    : WeftRISCV_LeafOp<"dense_snapshot"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       Arg<WeftRISCV_LocalType, "", [MemWrite]>:$storage,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalCopyOp : WeftRISCV_LeafOp<"local_copy"> {
  let summary = "Copy one bounded byte issue between distinct complete local values";
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       Index:$byte_offset, Index:$active_bytes,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_LocalLoadOp
    : WeftRISCV_LeafOp<"local_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$source, Index:$index,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalStoreOp
    : WeftRISCV_LeafOp<"local_store"> {
  let arguments = (ins AnyType:$value, Arg<WeftRISCV_ValueType, "", [MemWrite]>:$destination,
                       Index:$index,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLocalMaterializeOp
    : WeftRISCV_LeafOp<"rvv_local_materialize"> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Arg<WeftRISCV_ValueType, "", [MemWrite]>:$destination,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_EncodedLocalPackOp
    : WeftRISCV_Op<"encoded_local_pack"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$input,
                       Arg<WeftRISCV_LocalType, "", [MemWrite]>:$storage,
                       WeftRISCV_PointType:$row_point,
                       WeftRISCV_PointType:$record_point,
                       WeftRISCV_LocalPackPlanAttr:$plan,
                       WeftRISCV_LayoutAttr:$transfer_layout,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IndexMultipleGuardOp
    : WeftRISCV_LeafOp<"index_multiple_guard"> {
  let arguments = (ins Index:$value, I64Attr:$multiple,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_EncodedLocalBindOp
    : WeftRISCV_TerminalOp<"encoded_local_bind", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_LocalType:$storage,
                       Index:$block_count,
                       WeftRISCV_LocalPackPlanAttr:$plan);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVEncodedLocalPackTransferOp
    : WeftRISCV_LeafOp<"rvv_encoded_local_pack_transfer"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$input,
                       Arg<WeftRISCV_LocalType, "", [MemWrite]>:$storage,
                       Index:$row_group, Index:$row_base,
                       Index:$block, Index:$block_count,
                       Index:$byte, Index:$transfer_vl,
                       WeftRISCV_LocalPackPlanAttr:$plan,
                       WeftRISCV_LayoutAttr:$transfer_layout,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_SpillOp
    : WeftRISCV_LeafOp<"spill"> {
  let arguments = (ins AnyType:$input, Arg<WeftRISCV_LocalType, "", [MemWrite]>:$slot,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_ReloadOp
    : WeftRISCV_LeafOp<"reload"> {
  let arguments = (ins Arg<WeftRISCV_LocalType, "", [MemRead]>:$slot, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEZeroOp : WeftRISCV_LeafOp<"ime_zero", [Pure]> {
  let arguments = (ins WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentRowBroadcastOp
    : WeftRISCV_LeafOp<"ime_fragment_row_broadcast", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentStorageLoadOp
    : WeftRISCV_LeafOp<"ime_fragment_storage_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$input, WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentWindowLoadOp
    : WeftRISCV_LeafOp<"ime_fragment_window_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field, Index:$logical_base,
                       WeftRISCV_AccessAttr:$access, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DenseRowWindowGuardOp
    : WeftRISCV_TerminalOp<"dense_row_window_guard", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$source,
                       I64Attr:$row_axis, I64Attr:$inner_axis,
                       I64Attr:$rows, I64Attr:$columns, I64Attr:$alignment);
  let results = (outs I1:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentDenseLoadOp
    : WeftRISCV_LeafOp<"ime_fragment_dense_load"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       WeftRISCV_AccessAttr:$access, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentDecodeOp
    : WeftRISCV_LeafOp<"ime_fragment_decode", [Pure]> {
  let arguments = (ins WeftRISCV_FragmentType:$input, Index:$shift,
                       I64Attr:$element_bits, BoolAttr:$signed_elements,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentPointwiseOp
    : WeftRISCV_LeafOp<"ime_fragment_pointwise", [Pure]> {
  let arguments = (ins WeftRISCV_FragmentType:$lhs, AnyType:$rhs,
                       StrAttr:$kind, StrAttr:$lhs_policy, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentMMAOp
    : WeftRISCV_LeafOp<"ime_fragment_mma",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let arguments = (ins WeftRISCV_FragmentType:$lhs,
                       WeftRISCV_FragmentType:$rhs,
                       WeftRISCV_FragmentType:$accumulator,
                       StrAttr:$accumulator_policy,
                       I64Attr:$groups, I64Attr:$chunks,
                       ArrayAttr:$asm_clobbers, BoolAttr:$volatile_asm,
                       BoolAttr:$memory_clobber,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEPackOp
    : WeftRISCV_LeafOp<"ime_pack"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$input, StrAttr:$role,
                       WeftRISCV_FragmentPackingAttr:$packing,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEUnpackOp : WeftRISCV_LeafOp<"ime_unpack", [Pure]> {
  let arguments = (ins WeftRISCV_FragmentType:$input,
                       WeftRISCV_ConversionAttr:$conversion,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RegisterMaterializeOp
    : WeftRISCV_TerminalOp<"register_materialize", [Pure]> {
  let arguments = (ins AnyType:$input, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id,
                       StrAttr:$realization);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVBitplaneMergeOp
    : WeftRISCV_LeafOp<"rvv_bitplane_merge"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$low,
                       Arg<WeftRISCV_ValueType, "", [MemRead]>:$plane,
                       I64Attr:$insert_bit,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_PackedPlaneMergeOp
    : WeftRISCV_LeafOp<"packed_plane_merge"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$low_field,
                       Arg<WeftRISCV_ValueType, "", [MemRead]>:$high_field,
                       WeftRISCV_PackedPlaneMergePlanAttr:$plan,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVBitmaskDecodeOp
    : WeftRISCV_LeafOp<"rvv_bitmask_decode"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$origin,
                       WeftRISCV_PointType:$point,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVSignedBitmaskReduceOp
    : WeftRISCV_LeafOp<"rvv_signed_bitmask_reduce", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$data,
                       WeftRISCV_ValueType:$mask,
                       I64Attr:$reduction_axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVMaskedNegateOp
    : WeftRISCV_LeafOp<"rvv_masked_negate", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$data,
                       WeftRISCV_ValueType:$mask,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVBitmaskWindowLoadOp
    : WeftRISCV_LeafOp<"rvv_bitmask_window_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       AnyType:$byte_base,
                       I64Attr:$source_axis,
                       DenseI64ArrayAttr:$window_axes,
                       DenseI64ArrayAttr:$window_extents,
                       DenseI64ArrayAttr:$part_bit_offsets,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVGroupedMacLoadOp
    : WeftRISCV_LeafOp<"rvv_grouped_mac_load"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$lhs,
                       Arg<AnyType, "", [MemRead]>:$rhs, Index:$group_index,
                       Index:$active_terms,
                       WeftRISCV_LayoutAttr:$load_layout,
                       WeftRISCV_LayoutAttr:$partial_layout,
                       DenseI64ArrayAttr:$packed_supply_for_result,
                       WeftRISCV_GroupedMacPlanAttr:$plan,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_WindowType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVGroupedMacStepOp
    : WeftRISCV_LeafOp<"rvv_grouped_mac_step", [Pure]> {
  let arguments = (ins WeftRISCV_WindowType:$window,
                       WeftRISCV_ValueType:$accumulator,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenDotOp : WeftRISCV_Op<"rvv_widen_dot", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       DenseI64ArrayAttr:$over,
                       WeftRISCV_PartialTopologyAttr:$partial_topology,
                       BoolAttr:$fused_streams_legal,
                       I64Attr:$partial_unroll,
                       I64Attr:$reduction_lanes,
                       I64Attr:$reduction_streams,
                       I64Attr:$slice_lmul_eighths,
                       DenseI64ArrayAttr:$lhs_lane_offsets,
                       DenseI64ArrayAttr:$rhs_lane_offsets,
                       DenseI64ArrayAttr:$lhs_parts,
                       DenseI64ArrayAttr:$rhs_parts,
                       OptionalAttr<WeftRISCV_PartialLayoutPlanAttr>:$partial_layout_plan,
                       OptionalAttr<WeftRISCV_NestedPartialPlanAttr>:$nested_partial_plan,
                       OptionalAttr<WeftRISCV_SequentialPartialPlanAttr>:$sequential_partial_plan,
                       OptionalAttr<WeftRISCV_ScaledPartialPlanAttr>:$scaled_partial_plan,
                       OptionalAttr<WeftRISCV_LayeredPartialPlanAttr>:$layered_partial_plan,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenAddOp
    : WeftRISCV_LeafOp<"rvv_widen_add", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVNarrowShiftRightOp
    : WeftRISCV_LeafOp<"rvv_narrow_shift_right", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       AnyType:$amount,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenMultiplyOp
    : WeftRISCV_LeafOp<"rvv_widen_multiply", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenScalarMultiplyOp
    : WeftRISCV_LeafOp<"rvv_widen_scalar_multiply", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       AnyType:$rhs,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVMultiplyHighScalarOp
    : WeftRISCV_LeafOp<"rvv_multiply_high_scalar", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       AnyType:$rhs,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRegularRepeatIndexOp
    : WeftRISCV_LeafOp<"rvv_regular_repeat_index", [Pure]> {
  let arguments = (ins I64Attr:$reduction_axis,
                       I64Attr:$repeat,
                       ArrayAttr:$part_bases,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs Variadic<WeftRISCV_ValueType>:$results);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRegularRepeatGatherOp
    : WeftRISCV_LeafOp<"rvv_regular_repeat_gather"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       Variadic<WeftRISCV_ValueType>:$indices,
                       I64Attr:$source_axis,
                       I64Attr:$reduction_axis,
                       AnyType:$source_base,
                       I64Attr:$source_count,
                       I64Attr:$repeat,
                       ArrayAttr:$part_bases,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs Variadic<WeftRISCV_ValueType>:$results);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRegularRepeatScalarLoadOp
    : WeftRISCV_LeafOp<"rvv_regular_repeat_scalar_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       I64Attr:$source_axis,
                       I64Attr:$reduction_axis,
                       AnyType:$source_base,
                       I64Attr:$source_count,
                       I64Attr:$repeat,
                       DenseI64ArrayAttr:$part_bases,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStorageWindowOp
    : WeftRISCV_LeafOp<"rvv_storage_window"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$origin,
                       Index:$logical_offset,
                       WeftRISCV_StorageWindowPlanAttr:$plan,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredRecordLoadOp
    : WeftRISCV_LeafOp<"rvv_layered_record_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$origin,
                       Index:$logical_offset,
                       WeftRISCV_StorageWindowPlanAttr:$plan,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredStorageLoadOp
    : WeftRISCV_LeafOp<"rvv_layered_storage_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$origin,
                       Index:$window_index,
                       WeftRISCV_StorageWindowPlanAttr:$plan,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_LayeredWindowType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredStorageDecodeOp
    : WeftRISCV_LeafOp<"rvv_layered_storage_decode", [Pure]> {
  let arguments = (ins WeftRISCV_LayeredWindowType:$window,
                       I64Attr:$layer,
                       I64Attr:$physical_layer,
                       I64Attr:$shift_amount,
                       I64Attr:$mask_value,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVReplicaStorageLoadOp
    : WeftRISCV_LeafOp<"rvv_replica_storage_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       AnyType:$logical_base,
                       WeftRISCV_StorageWindowPlanAttr:$plan,
                       I64Attr:$record_rank,
                       DenseI64ArrayAttr:$window_offsets,
                       DenseI64ArrayAttr:$record_coordinates_for_window,
                       DenseI64ArrayAttr:$window_for_part,
                       DenseI64ArrayAttr:$layer_for_part,
                       DenseI64ArrayAttr:$physical_layer_for_part,
                       DenseI64ArrayAttr:$shift_offset_for_part,
                       DenseI64ArrayAttr:$shift_base_factor_for_part,
                       DenseI64ArrayAttr:$mask_value_for_part,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVSegmentPairLoadOp
    : WeftRISCV_LeafOp<"rvv_segment_pair_load"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       I64Attr:$logical_base,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$first,
                      WeftRISCV_ValueType:$second);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRecordStorageLoadOp
    : WeftRISCV_LeafOp<"rvv_record_storage_load"> {
  let arguments = (ins Arg<AnyType, "", [MemRead]>:$field,
                       WeftRISCV_RecordCohortType:$cohort,
                       I64Attr:$storage_index,
                       I64Attr:$storage_width,
                       I64Attr:$record_byte_stride,
                       DenseI64ArrayAttr:$logical_indices,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRecordStorageDecodeOp
    : WeftRISCV_LeafOp<"rvv_record_storage_decode", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       I64Attr:$logical_index,
                       I64Attr:$shift_amount,
                       I64Attr:$mask_value,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVRecordStoreOp
    : WeftRISCV_LeafOp<"rvv_record_store"> {
  let arguments = (ins WeftRISCV_ValueType:$value,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       WeftRISCV_RecordCohortType:$cohort,
                       I64Attr:$element_offset,
                       I64Attr:$record_byte_stride,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVIssueSliceOp
    : WeftRISCV_LeafOp<"rvv_issue_slice", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       DenseI64ArrayAttr:$source_parts,
                       DenseI64ArrayAttr:$lane_offsets,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVAssembleValueOp : WeftRISCV_LeafOp<"rvv_assemble_value", [Pure]> {
  let summary = "Assemble contiguous physical issue values into one numeric carrier";
  let arguments = (ins Variadic<WeftRISCV_ValueType>:$pieces,
                       I64Attr:$axis, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenAccumulateOp
    : WeftRISCV_LeafOp<"rvv_widen_accumulate", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       WeftRISCV_ValueType:$accumulator,
                       DenseI64ArrayAttr:$reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVFinalizeWidenDotOp
    : WeftRISCV_LeafOp<"rvv_finalize_widen_dot", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$partial,
                       DenseI64ArrayAttr:$reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialSetOp
    : WeftRISCV_LeafOp<"rvv_partial_set",
                   [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<WeftRISCV_ValueType>:$lhs,
                       Variadic<WeftRISCV_ValueType>:$rhs,
                       DenseI64ArrayAttr:$lhs_operands,
                       DenseI64ArrayAttr:$rhs_operands,
                       DenseI64ArrayAttr:$lhs_parts,
                       DenseI64ArrayAttr:$rhs_parts,
                       DenseI64ArrayAttr:$lhs_lane_offsets,
                       DenseI64ArrayAttr:$rhs_lane_offsets,
                       I64Attr:$reduction_axis,
                       StrAttr:$multiply_instruction,
                       I64Attr:$owner_domain_id,
                       I64Attr:$birth_id,
                       I64Attr:$lifetime_end_domain_id,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialCaptureOp
    : WeftRISCV_LeafOp<"rvv_partial_capture", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       I64Attr:$owner_domain_id,
                       I64Attr:$birth_id,
                       I64Attr:$lifetime_end_domain_id,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialCollectOp
    : WeftRISCV_LeafOp<"rvv_partial_collect", [Pure]> {
  let arguments = (ins Variadic<WeftRISCV_ValueType>:$inputs,
                       I64Attr:$owner_domain_id,
                       I64Attr:$birth_id,
                       I64Attr:$lifetime_end_domain_id,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialRepackOp
    : WeftRISCV_LeafOp<"rvv_partial_repack", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       I64Attr:$split_factor,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialMergeOp
    : WeftRISCV_LeafOp<"rvv_partial_merge", [Pure]> {
  let arguments = (ins Variadic<WeftRISCV_PartialSetType>:$inputs,
                       StrAttr:$topology,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialReduceOp
    : WeftRISCV_LeafOp<"rvv_partial_reduce", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialScaleCombineOp
    : WeftRISCV_LeafOp<"rvv_partial_scale_combine", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       Variadic<AnyType>:$scales,
                       DenseI64ArrayAttr:$scale_replicas,
                       DenseI64ArrayAttr:$slot_order,
                       DenseI64ArrayAttr:$logical_reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialPackedScaleOp
    : WeftRISCV_LeafOp<"rvv_partial_packed_scale", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<WeftRISCV_PartialSetType>:$inputs,
                       Variadic<WeftRISCV_ValueType>:$scales,
                       DenseI64ArrayAttr:$slot_order,
                       DenseI64ArrayAttr:$logical_reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialWidenScaleOp
    : WeftRISCV_LeafOp<"rvv_partial_widen_scale", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       Variadic<WeftRISCV_ValueType>:$scales,
                       DenseI64ArrayAttr:$scale_replicas,
                       I64Attr:$combine_arity,
                       DenseI64ArrayAttr:$logical_reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialCombineOp
    : WeftRISCV_LeafOp<"rvv_partial_combine", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       I64Attr:$arity,
                       StrAttr:$topology,
                       DenseI64ArrayAttr:$logical_reduction_axes,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_PartialSetType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartialFinalizeOp
    : WeftRISCV_LeafOp<"rvv_partial_finalize", [Pure]> {
  let arguments = (ins WeftRISCV_PartialSetType:$input,
                       I64Attr:$reduction_axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVAssembleReplicasOp
    : WeftRISCV_LeafOp<"rvv_assemble_replicas", [Pure]> {
  let arguments = (ins Variadic<AnyInteger>:$values,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenReduceOp
    : WeftRISCV_LeafOp<"rvv_widen_reduce", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       StrAttr:$kind,
                       I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartitionedWidenReduceStoreOp
    : WeftRISCV_LeafOp<"rvv_partitioned_widen_reduce_store"> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       Index:$base_index,
                       I64Attr:$partition,
                       I64Attr:$count,
                       I64Attr:$destination_axis,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredWindowOp
    : WeftRISCV_LeafOp<"rvv_layered_window"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$point,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_StorageWindowPlanAttr:$plan,
                       DenseI64ArrayAttr:$physical_layer_for_result,
                       DenseI64ArrayAttr:$shift_for_result,
                       DenseI64ArrayAttr:$mask_for_result,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$first,
                      WeftRISCV_ValueType:$second);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredStreamOp
    : WeftRISCV_LeafOp<"rvv_layered_stream"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LayeredStreamGeometryAttr:$geometry,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVProjectedLayeredStreamOp
    : WeftRISCV_LeafOp<"rvv_projected_layered_stream"> {
  let arguments = (ins Arg<WeftRISCV_ValueType, "", [MemRead]>:$field,
                       WeftRISCV_PointType:$origin,
                       I64Attr:$reduction_axis,
                       I64Attr:$projection_base,
                       I64Attr:$projection_stride,
                       I64Attr:$projection_repeat,
                       I64Attr:$projection_extent,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LayeredStreamGeometryAttr:$geometry,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}


def WeftRISCV_DensePanelCopyOp : WeftRISCV_Op<"dense_panel_copy"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       I64Attr:$copy_axis, I64Attr:$maximum_lmul,
                       I64Attr:$maximum_fields,
                       OptionalAttr<WeftRISCV_PanelTransferPlanAttr>:$plan);
  let hasVerifier = 1;
}

def WeftRISCV_ScalarTileLoadOp
    : WeftRISCV_LeafOp<"scalar_tile_load"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       Variadic<Index>:$extents, BoolAttr:$guarded,
                       WeftRISCV_LeafAttr:$leaf,
                       OptionalAttr<WeftRISCV_ScalarPacketAttr>:$packet);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVTileSliceOp : WeftRISCV_LeafOp<"rvv_tile_slice", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       DenseI64ArrayAttr:$offsets, I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ScalarTileStoreOp : WeftRISCV_LeafOp<"scalar_tile_store"> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       Variadic<Index>:$extents, BoolAttr:$guarded,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVTileLoadOp : WeftRISCV_LeafOp<"rvv_tile_load"> {
  let arguments = (ins Arg<WeftRISCV_MemDescType, "", [MemRead]>:$source,
                       Variadic<Index>:$extents, I64Attr:$axis,
                       WeftRISCV_AccessAttr:$access, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVTileStoreOp : WeftRISCV_LeafOp<"rvv_tile_store"> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Arg<WeftRISCV_MemDescType, "", [MemWrite]>:$destination,
                       Variadic<Index>:$extents, I64Attr:$axis,
                       WeftRISCV_AccessAttr:$access, WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVDenseSegmentLoadOp
    : WeftRISCV_LeafOp<"rvv_dense_segment_load", [AttrSizedOperandSegments]> {
  let arguments = (ins Arg<Variadic<WeftRISCV_MemDescType>, "", [MemRead]>:$sources,
                       Variadic<Index>:$extents, I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs Variadic<WeftRISCV_ValueType>:$values);
  let hasVerifier = 1;
}

def WeftRISCV_RVVDenseSegmentStoreOp
    : WeftRISCV_LeafOp<"rvv_dense_segment_store", [AttrSizedOperandSegments]> {
  let arguments = (ins Variadic<WeftRISCV_ValueType>:$values,
                       Arg<Variadic<WeftRISCV_MemDescType>, "", [MemWrite]>:$destinations,
                       Variadic<Index>:$extents, I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVTileContractOp : WeftRISCV_LeafOp<"rvv_tile_contract", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, WeftRISCV_ValueType:$accumulator,
                       Variadic<Index>:$extents, I64Attr:$axis,
                       I64Attr:$reduction_axis, StrAttr:$lane_operand,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVFMAIssueOp : WeftRISCV_LeafOp<"rvv_fma_issue", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$vector, AnyType:$multiplier,
                       WeftRISCV_ValueType:$accumulator, Variadic<Index>:$lengths,
                       DenseI64ArrayAttr:$group_ends, StrAttr:$tail,
                       StrAttr:$accumulator_policy, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamReduceStepOp
    : WeftRISCV_LeafOp<"rvv_stream_reduce_step", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_ValueType:$accumulator,
                       Index:$active, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamContractStepOp
    : WeftRISCV_LeafOp<"rvv_stream_contract_step", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       WeftRISCV_ValueType:$accumulator,
                       Index:$active, I64Attr:$reduction_axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamFinalizeOp
    : WeftRISCV_LeafOp<"rvv_stream_finalize", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       I64Attr:$axis, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVSplatOp : WeftRISCV_LeafOp<"rvv_splat", [Pure]> {
  let arguments = (ins AnyType:$scalar, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVAxisBroadcastOp
    : WeftRISCV_LeafOp<"rvv_axis_broadcast", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ProjectReductionOperandOp
    : WeftRISCV_LeafOp<"project_reduction_operand", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, Index:$reduction_index,
                       I64Attr:$reduction_axis, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVContractStepOp : WeftRISCV_LeafOp<"rvv_contract_step", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       WeftRISCV_ValueType:$accumulator,
                       Optional<Index>:$active,
                       I64Attr:$reduction_axis, StrAttr:$lane_operand,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
