add_library(WeftRISCVContracts STATIC
  Target/IME/FragmentContracts.cpp
  Target/RISCVExtensions.cpp
  Target/RISCVDataCache.cpp
)
set_target_properties(WeftRISCVContracts PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

add_subdirectory(Dialect)
add_subdirectory(Target)
add_subdirectory(Runtime)
