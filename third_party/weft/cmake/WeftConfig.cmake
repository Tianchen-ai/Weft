get_filename_component(Weft_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(Weft_LLAMA_CPP_DIR "${Weft_ROOT}/share/llama.cpp")
if(NOT TARGET Weft::Runtime)
  add_library(Weft::Runtime SHARED IMPORTED)
  set_target_properties(Weft::Runtime PROPERTIES
    IMPORTED_LOCATION "${Weft_ROOT}/lib/libweft_runtime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${Weft_ROOT}/include")
endif()
set(Weft_FOUND TRUE)
