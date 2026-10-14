#include "Weft/Target/RISCVTargetCatalog.h"
#include "Weft/Target/RISCVExtensions.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"

namespace weft {
namespace {

template <typename Range> llvm::json::Array numbers(const Range &values) {
  llvm::json::Array result;
  for (auto value : values) result.push_back(static_cast<int64_t>(value));
  return result;
}

llvm::StringRef signedness(RISCVFragmentSignedness value) {
  switch (value) {
  case RISCVFragmentSignedness::Signed: return "signed";
  case RISCVFragmentSignedness::Unsigned: return "unsigned";
  }
  llvm_unreachable("unknown registered fragment signedness");
}

llvm::json::Object packing(const RISCVFragmentPacking &value) {
  return llvm::json::Object{{"schema", value.schema}, {"axis_order", numbers(value.axisOrder)},
      {"rows_per_tile", value.rowsPerTile}, {"columns_per_tile", value.columnsPerTile},
      {"tile_order", value.tileOrder}, {"element_order", value.elementOrder},
      {"storage_bits", value.storageBits}, {"alignment", value.alignment}};
}

llvm::json::Object fragment(const RISCVFragmentCapability &value) {
  llvm::json::Array clobbers;
  for (const auto &name : value.clobbers) clobbers.push_back(name);
  return llvm::json::Object{{"instruction", value.name}, {"extension", value.extension},
      {"xlen", value.xlen}, {"vlen_bits", value.vlenBits},
      {"lhs_signedness", signedness(value.lhsSignedness)},
      {"rhs_signedness", signedness(value.rhsSignedness)},
      {"accumulator_signedness", signedness(value.accumulatorSignedness)},
      {"lhs_element_bits", value.lhsElementBits}, {"rhs_element_bits", value.rhsElementBits},
      {"accumulator_element_bits", value.accumulatorElementBits},
      {"m", value.mFactor}, {"n", value.nFactor}, {"k", value.kFactor},
      {"active_m", value.activeMFactor}, {"lhs_packing", packing(value.lhsPacking)},
      {"rhs_packing", packing(value.rhsPacking)},
      {"accumulator_packing", packing(value.accumulatorPacking)},
      {"resources", llvm::json::Object{
          {"lhs_groups", value.lhsResourceGroups}, {"rhs_groups", value.rhsResourceGroups},
          {"accumulator_groups", value.accumulatorResourceGroups},
          {"pack_temporary_groups", value.packTemporaryGroups},
          {"unpack_temporary_groups", value.unpackTemporaryGroups},
          {"lhs_pack_local_bytes", value.lhsPackLocalBytes},
          {"rhs_pack_local_bytes", value.rhsPackLocalBytes}, {"mma_local_bytes", value.mmaLocalBytes}}},
      {"assembly", llvm::json::Object{{"groups", value.mmaGroups}, {"chunks", value.mmaChunks},
          {"clobbers", std::move(clobbers)}, {"volatile", value.volatileAsm},
          {"memory_clobber", value.memoryClobber}}}};
}

llvm::json::Array fragments(llvm::ArrayRef<RISCVFragmentCapability> values) {
  llvm::json::Array result;
  for (const auto &value : values) result.push_back(fragment(value));
  return result;
}

llvm::json::Object extension(const RISCVExtension &value) {
  llvm::json::Array matches;
  for (const auto &match : value.nativeMatches)
    matches.push_back(llvm::json::Object{{"xlen", match.xlen}, {"vlen_bits", match.vlenBits},
        {"vendor_id", "0x" + llvm::utohexstr(match.vendorId)},
        {"architecture_id", "0x" + llvm::utohexstr(match.architectureId)},
        {"isa_feature", match.isaFeature}});
  return llvm::json::Object{{"name", value.name}, {"engine", value.engine}, {"description", value.description},
      {"native_matches", std::move(matches)}, {"fragments", fragments(value.fragments)}};
}

} // namespace

llvm::json::Object riscvExtensionCatalogJSON() {
  llvm::json::Array entries;
  for (const auto &value : riscvExtensions()) entries.push_back(extension(value));
  return llvm::json::Object{{"kind", "weft-extension-catalog"}, {"extensions", std::move(entries)}};
}

bool riscvTargetCapabilitiesJSON(const RISCVTargetProfile &profile,
                                 llvm::json::Object &result, std::string &error) {
  RISCVISAFacts facts;
  if (!parseRISCVISAFacts(profile.march, profile.vlenBits, facts, error)) return false;
  llvm::json::Object isa;
  for (const auto &[name, version] : facts.extensions)
    isa[name] = llvm::json::Array{version.first, version.second};
  llvm::json::Array enabled;
  llvm::StringSet<> seen;
  for (const auto &fragment : profile.fragmentCapabilities)
    if (seen.insert(fragment.extension).second) enabled.push_back(fragment.extension);
  result = llvm::json::Object{{"kind", "weft-target-capabilities"},
      {"target", llvm::json::Object{{"march", profile.march}, {"abi", profile.abi},
          {"xlen", profile.xlen}, {"vlen_bits", profile.vlenBits}}},
      {"isa_extensions", std::move(isa)},
      {"capabilities", llvm::json::Object{
          {"rvv", profile.hasRVV}, {"scalar_f32", profile.hasF}, {"scalar_f64", profile.hasD},
          {"scalar_f16_arithmetic", profile.hasScalarF16},
          {"vector_f16_arithmetic", profile.hasVectorF16},
          {"vector_f16_conversion", profile.hasVectorF16Conversion},
          {"indexed_memory", profile.hasIndexedMemory}, {"segment_memory", profile.hasSegmentMemory},
          {"widening_integer", profile.hasWideningInteger}, {"widening_float", profile.hasWideningFloat},
          {"vector_registers", profile.vectorRegisters}, {"private_stack_bytes", profile.maxPrivateStackBytes},
          {"max_widening_combine_groups", profile.maxWideningCombineGroups},
          {"supported_sew", numbers(profile.supportedSEW)}, {"legal_lmul_eighths", numbers(profile.legalLMULEighths)},
          {"partial_combine_policy", profile.partialCombinePolicy == RISCVPartialCombinePolicy::Sequential
              ? "sequential" : "independent-multilevel"},
          {"record_axis_policy", profile.recordAxisPolicy == RISCVRecordAxisPolicy::AcrossRecords
              ? "across-records" : "within-record"}}},
      {"matrix_extensions", std::move(enabled)}, {"fragments", fragments(profile.fragmentCapabilities)}};
  return true;
}

} // namespace weft
