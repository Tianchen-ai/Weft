#ifndef WEFT_TARGET_RISCVCOMPILER_H
#define WEFT_TARGET_RISCVCOMPILER_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <utility>

namespace weft {

class RISCVCompilationReport;

struct RISCVCompilerOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
  std::vector<std::string> requiredExtensions;
  std::string localTileStrategy = "target-default";
  int64_t lmulEighths = 8;
  int64_t unroll = 1;
  int64_t pipelineDepth = 1;
  int64_t scalarLoadPrime = 0;
  int64_t pointwiseIssueCopies = 1;
  int64_t contractionColumnCopies = 4;
  int64_t panelTransferFields = 4;
  std::string panelTransferStrategy = "issue-work";
  int64_t contractionPanelK = 128;
  int64_t contractionKSteps = 4;

  // Syntax/domain validity of one binding is independent of whether a target
  // can realize it. Resource/capability rejection happens later in real passes.
  bool validateBindings(std::string &error) const;
};

struct RISCVPhysicalizationResult {
  std::string riscvIR;
};

struct RISCVArgumentABI {
  std::string name;
  std::string kind;
  std::string dtype;
  std::string access;
  bool strided = false;
  std::string cType;
  std::string encoding;
  std::vector<std::string> shape;
  int64_t storageBytes = 0;
  int64_t recordElements = 0;
  int64_t interleaveRows = 0;
  int64_t alignment = 0;
  int64_t aliasSet = 0;
  bool writable = false;
};

struct RISCVLayoutDiagnostic {
  std::string role;
  int64_t index = 0;
  std::string elementType;
  std::string carrier;
  std::string validity;
  std::vector<int64_t> shape;
  std::vector<int64_t> axes;
  std::vector<int64_t> layoutAxes;
  std::vector<int64_t> timeFactors;
  std::vector<int64_t> laneFactors;
  std::vector<int64_t> replicaFactors;
  std::vector<int64_t> fragmentFactors;
  std::vector<int64_t> localFactors;
  int64_t sew = 0;
  int64_t lmulEighths = 0;
  int64_t vl = 0;
  int64_t registerGroups = 0;
  int64_t replicaGroup = 1;
};

struct RISCVAccessDiagnostic {
  std::string form;
  std::string mapping;
  int64_t alignment = 0;
  int64_t indexSEW = 0;
  int64_t segmentFields = 0;
  int64_t groupSize = 0;
  int64_t layerSize = 0;
  int64_t joinFields = 0;
  int64_t joinLowBits = 0;
  int64_t joinRole = 0;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
  std::string order;
  int64_t fieldChunkBytes = 0;
};

struct RISCVScalarPacketDiagnostic {
  int64_t axis;
  int64_t steps;
  int64_t parts;
  int64_t registers;
  int64_t liveRegisters;
};

struct RISCVOperationDiagnostic {
  std::string operation;
  std::string engine;
  std::string family;
  std::string instruction;
  std::string mask;
  std::string tail;
  std::optional<RISCVAccessDiagnostic> access;
  std::optional<RISCVScalarPacketDiagnostic> scalarPacket;
  int64_t occurrences = 0;
  int64_t operandCount = 0;
  int64_t resultCount = 0;
  int64_t operandGroups = 0;
  int64_t resultGroups = 0;
  int64_t temporaryGroups = 0;
  int64_t fragmentGroups = 0;
  int64_t localBytes = 0;
  std::vector<int64_t> parameters;
  std::vector<std::string> locations;
  std::vector<RISCVLayoutDiagnostic> layouts;
};

struct RISCVKernelABI {
  std::string symbol;
  std::string declaration;
  std::string march;
  std::string abi;
  int64_t vlenBits = 0;
  std::optional<RISCVDataCache> dataCache;
  std::map<std::string, std::pair<unsigned, unsigned>> isaExtensions;
  std::vector<std::string> matrixExtensions;
  std::vector<std::string> requiredExtensions;
  std::vector<std::string> usedExtensions;
  std::vector<RISCVArgumentABI> arguments;
  std::vector<std::string> shapeParameters;
  std::vector<std::pair<std::string, int64_t>> bindings;
  std::string localTileStrategy;
  std::vector<std::pair<std::string, int64_t>> localTileRealizations;
  // Verified Physical IR occupancy and capacity, not system-compiler stack
  // usage or measured hardware register allocation.
  std::map<std::string, int64_t> physicalResources;
  // Static final-IR operation groups, including mutually exclusive paths.
  // These are inspection facts, not dynamic instruction counts or a cost model.
  std::vector<RISCVOperationDiagnostic> physicalOperations;
};

struct RISCVCompilationResult {
  std::string riscvIR;
  std::string intrinsicC;
  std::vector<RISCVKernelABI> kernels;
};

/// In-process physical program entry used by embedding compilers.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                 RISCVCompilationReport *report = nullptr);

/// Convert one instantiated Canonical Kernel IR candidate into a complete,
/// target-aware RISC-V Physical IR program.
mlir::FailureOr<RISCVPhysicalizationResult>
physicalizeRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                      RISCVCompilationReport *report = nullptr);

enum class RISCVCheckpoint { Contractions, PanelTransfers, Layout };

/// Save a selected Physical IR boundary and its remaining compilation binding.
mlir::FailureOr<RISCVPhysicalizationResult>
prepareRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options, RISCVCheckpoint checkpoint,
                         RISCVCompilationReport *report = nullptr);

/// Consume a saved Physical IR boundary through the corresponding main suffix.
mlir::FailureOr<RISCVCompilationResult>
completeRISCVModule(mlir::ModuleOp module, RISCVCheckpoint checkpoint,
                    RISCVCompilationReport *report = nullptr);

/// Inspect typed selections after checking their numerical and target context.
/// The returned JSON describes static physical choices, never measured costs.
mlir::FailureOr<std::string> inspectRISCVContractions(mlir::ModuleOp module);
mlir::FailureOr<std::string> inspectRISCVPanelTransfers(mlir::ModuleOp module);

/// Translate a verified RISC-V Physical IR program mechanically to intrinsic C.
mlir::FailureOr<RISCVCompilationResult>
compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options,
                   RISCVCompilationReport *report = nullptr);

/// Verify and mechanically translate an already-physicalized RISC-V program.
/// This is the terminal boundary used after independent Physical IR rewrites.
mlir::FailureOr<RISCVCompilationResult>
translateRISCVModule(mlir::ModuleOp module, RISCVCompilationReport *report = nullptr);

} // namespace weft

#endif // WEFT_TARGET_RISCVCOMPILER_H
