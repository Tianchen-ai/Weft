add_subdirectory(Target/Extensions)
add_library(WeftRISCVContracts STATIC
  ${WEFT_RISCV_EXTENSION_CONTRACT_SOURCES}
  Target/RISCVFragment.cpp
  Target/RISCVExtensions.cpp
  Target/RISCVDataCache.cpp
)
set_target_properties(WeftRISCVContracts PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(WeftRISCVContracts PRIVATE ${WEFT_RISCV_EXTENSION_REGISTRY_DIR})

# Compiler and deployment share one ISA parser and native capability probe.
add_library(WeftRISCVPlatform STATIC Target/RISCVTargetProfile.cpp Target/RISCVISA.cpp
  Target/RISCVTargetCatalog.cpp Target/RISCVNativeTarget.cpp Target/RISCVNativeCache.cpp)
set_target_properties(WeftRISCVPlatform PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(WeftRISCVPlatform PUBLIC WeftRISCVContracts)
if(LLVM_LINK_LLVM_DYLIB)
  llvm_config(WeftRISCVPlatform USE_SHARED TargetParser Support)
else()
  llvm_config(WeftRISCVPlatform TargetParser Support)
endif()

if(NOT TRITON_WEFT_RUNTIME_ONLY)
  add_subdirectory(Dialect)
  add_subdirectory(Target)
endif()
add_subdirectory(Runtime)
