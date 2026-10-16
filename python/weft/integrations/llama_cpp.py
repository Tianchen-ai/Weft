from __future__ import annotations

import re
from pathlib import Path

from weft.runtime.driver import CompilationError


# Framework storage contracts, not lowering rules or alternative math kernels.
# The same facts are checked against the actual linked GGML library on load.
VEC_DOT = {
    "q4_0": ("Q4_0", "Q8_0", 32, 18, 34),
    "q4_1": ("Q4_1", "Q8_1", 32, 20, 36),
    "q5_0": ("Q5_0", "Q8_0", 32, 22, 34),
    "q5_1": ("Q5_1", "Q8_1", 32, 24, 36),
    "q8_0": ("Q8_0", "Q8_0", 32, 34, 34),
    "q2_K": ("Q2_K", "Q8_K", 256, 84, 292),
    "q3_K": ("Q3_K", "Q8_K", 256, 110, 292),
    "q4_K": ("Q4_K", "Q8_K", 256, 144, 292),
    "q5_K": ("Q5_K", "Q8_K", 256, 176, 292),
    "q6_K": ("Q6_K", "Q8_K", 256, 210, 292),
}


def validate_vec_dot(artifact, weight_type):
    if weight_type not in VEC_DOT:
        raise NotImplementedError(f"GGML adapter has no declared storage contract for {weight_type}")
    weight, activation, block, weight_bytes, activation_bytes = VEC_DOT[weight_type]
    parameters = artifact.interface.arguments
    if len(parameters) != 3 or artifact.interface.shape_parameters != ("K",):
        raise ValueError("GGML vec-dot requires exactly W, X, Y Views and one dynamic K")
    for parameter, encoding, shape, access, elements, size in zip(
            parameters, (weight, activation, "dense.f32"), (("K",), ("K",), ("1",)),
            ("read", "read", "write"), (block, block, 1), (weight_bytes, activation_bytes, 4)):
        if (parameter.kind, parameter.encoding, parameter.shape, parameter.access,
            parameter.record_elements, parameter.storage_bytes, parameter.strided, parameter.interleave_rows) != (
                "view", encoding, shape, access, elements, size, False, 0):
            raise ValueError(f"{parameter.name} does not match the GGML {weight_type} vec-dot ABI")
    if artifact.metadata["used_extensions"]:
        raise ValueError("GGML vec-dot registration only accepts ordinary RVV entries; matrix paths have different contracts")


def validate_quantizer(artifact, activation_type):
    contracts = {"q8_0": ("Q8_0", 32, 34, 2), "q8_K": ("Q8_K", 256, 292, 4)}
    if activation_type not in contracts:
        raise NotImplementedError(f"GGML activation quantizer has no storage contract for {activation_type}")
    encoding, block, size, alignment = contracts[activation_type]
    arguments = artifact.interface.arguments
    if len(arguments) != 2 or artifact.interface.shape_parameters != ("K",):
        raise ValueError("GGML activation quantization requires X[K], Y[K] and one runtime K")
    for arg, enc, access, elements, storage, align in zip(
            arguments, ("dense.f32", encoding), ("read", "write"), (1, block), (4, size), (4, alignment)):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", enc, ("K",), access, elements, storage, False, 0) or arg.alignment > align:
            raise ValueError(f"{arg.name} does not match the GGML activation row ABI")
    if artifact.metadata["used_extensions"]:
        raise ValueError("GGML activation quantization requires an ordinary CPU row entry")


def validate_ime1(artifact, role):
    contracts = {"q4_centered_prefill": (False, 4), "q4_affine_prefill": (True, 4),
                 "q4_centered_decode": (False, 1), "q4_affine_decode": (True, 1)}
    if role not in contracts:
        raise NotImplementedError(f"GGML IME row tile has no contract for {role}")
    affine, rows = contracts[role]
    arguments = artifact.interface.arguments
    if len(arguments) != 3 or artifact.interface.shape_parameters != ("N", "K", "M"):
        raise ValueError("GGML IME row tile requires W, Xq, Y and N, K, M dimensions")
    encodings = ("IMEQ4Affine_I[16]" if affine else "IMEQ4Centered_I[16]", f"IMEQ8_I[{rows}]", "dense.f32")
    for i, arg in enumerate(arguments):
        if (arg.kind, arg.encoding, arg.shape, arg.access, arg.record_elements,
                arg.storage_bytes, arg.strided, arg.interleave_rows) != (
                "view", encodings[i], (("N", "K"), ("M", "K"), ("M", "N"))[i],
                "write" if i == 2 else "read", 1 if i == 2 else 32,
                ((19 if affine else 18), 36, 4)[i], i == 2, (16, rows, 0)[i]) or arg.alignment > ((1 if affine else 2), 4, 4)[i]:
            raise ValueError(f"{arg.name} does not match the GGML IME row-tile ABI")
    if tuple(artifact.metadata["used_extensions"]) != ("spacemit-ime1",):
        raise ValueError("GGML IME binding requires an artifact using the IME1 capability")


HOOK_HEADER = '''/* Weft CPU call-site integration. Mathematics stays in generated artifacts. */
#pragma once
#if defined(GGML_USE_WEFT)
#include <weft/ggml.h>
#define WEFT_GGML_INIT() ggml_weft_initialize()
#define WEFT_GGML_DOT_TYPE ggml_weft_dot_binding
#define WEFT_GGML_DOT(type, original) ggml_weft_select_vec_dot((type), (original))
#define WEFT_GGML_CALL_DOT(binding, ...) ggml_weft_call_dot((binding), __VA_ARGS__)
#define WEFT_GGML_FROM_FLOAT(fn, x, y, n) ggml_weft_from_float((fn), (x), (y), (n))
#define WEFT_GGML_NODE(params, tensor) ggml_weft_begin_node((tensor), (params)->ith, (params)->use_ref)
#define WEFT_GGML_GRAPH(graph, plan) ggml_weft_begin_graph((graph)->n_nodes, (graph)->nodes, (plan)->abort_callback != NULL)
#define WEFT_GGML_END_NODE() ggml_weft_end_node()
#define WEFT_GGML_EXTRA(params, tensor) ggml_weft_extra_node((tensor), (params)->ith)
#else
#define WEFT_GGML_INIT() ((void)0)
#define WEFT_GGML_DOT_TYPE ggml_vec_dot_t
#define WEFT_GGML_DOT(type, original) (original)
#define WEFT_GGML_CALL_DOT(binding, ...) (binding)(__VA_ARGS__)
#define WEFT_GGML_FROM_FLOAT(fn, x, y, n) (fn)((x), (y), (n))
#define WEFT_GGML_NODE(params, tensor) ((void)0)
#define WEFT_GGML_GRAPH(graph, plan) ((void)0)
#define WEFT_GGML_END_NODE() ((void)0)
#define WEFT_GGML_EXTRA(params, tensor) ((void)0)
#endif
'''


# The reviewed upstream vendor source uses C++20 and has unconditional source
# entries whose files reject absent ISA macros. Apply its own probe results to
# those entries in both ordinary and Weft builds; retain every available engine.
NATIVE_BUILD = '''
    if (GGML_CPU_RISCV64_SPACEMIT)
        target_compile_features(${GGML_CPU_NAME} PRIVATE cxx_std_20)
        set_property(TARGET ${GGML_CPU_NAME} PROPERTY CXX_SCAN_FOR_MODULES OFF)
        get_target_property(native_cpu_sources ${GGML_CPU_NAME} SOURCES)
        if (NOT SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
            list(FILTER native_cpu_sources EXCLUDE REGEX "/ime1_kernels[.]cpp$")
        endif()
        if (NOT (SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4 AND SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK AND SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK))
            list(FILTER native_cpu_sources EXCLUDE REGEX "/ime2_kernels[.]cpp$")
        endif()
        set_property(TARGET ${GGML_CPU_NAME} PROPERTY SOURCES "${native_cpu_sources}")
    endif()'''


def _once(source, before, after):
    if source.count(before) != 1:
        raise CompilationError(f"llama.cpp CPU source does not expose the supported integration boundary: {before!r}")
    return source.replace(before, after, 1)


def install_source(source_directory):
    """Add thin build/call-site hooks to an explicitly selected upstream tree.

    This does not disable repack, OpenMP, IME, BLAS or any other upstream backend.
    All expected boundaries are checked before the first write. An upstream
    layout change is unsupported until reviewed, rather than guessed around.
    """
    root = Path(source_directory).expanduser().resolve(strict=True)
    directory = root / "ggml/src/ggml-cpu"
    path, cmake = directory / "ggml-cpu.c", directory / "CMakeLists.txt"
    source, build = path.read_text(encoding="utf-8"), cmake.read_text(encoding="utf-8")
    ime_path = directory / "spacemit/ime.cpp"
    ime = ime_path.read_text(encoding="utf-8")
    ime_header = '#if defined(GGML_USE_WEFT)\n#include <weft/ggml.h>\n#endif\n'
    ime_anchor = '                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;'
    ime_call = ime_anchor + '''
#if defined(GGML_USE_WEFT)
                if (!params->use_ref) {
                    auto *binding = ggml_weft_select_ime1(spacemit_kernels::ime1::gemm_kernel_i8i4, src0->type);
                    if (binding) gemm_kernel = [binding](size_t block, const uint8_t *x,
                            const uint8_t *w, const uint8_t *zero, float *y, size_t rows,
                            size_t columns, size_t blocks, size_t stride) {
                        return ggml_weft_call_ime1(binding, block, x, w, zero, y, rows, columns, blocks, stride);
                    };
                }
#endif'''
    marker = '#include "weft-ggml-hooks.h"'
    header = directory / "weft-ggml-hooks.h"
    if marker in source:
        if header.read_text(encoding="utf-8") != HOOK_HEADER or "weft_add_llama_backend" not in build or NATIVE_BUILD not in build or ime_call not in ime:
            raise CompilationError("llama.cpp contains an incomplete Weft integration")
        return {"source": str(root), "changed": False}
    source = _once(source, '#include "ggml-cpu-impl.h"', '#include "ggml-cpu-impl.h"\n' + marker)
    if ime.count(ime_anchor) != 2:
        raise CompilationError("llama.cpp IME MUL_MAT and MUL_MAT_ID row-tile boundaries have changed")
    ime = ime_header + ime.replace(ime_anchor, ime_call)
    source, count = re.subn(r'ggml_vec_dot_t(\s+const vec_dot\s*=\s*)(type_traits_cpu\[type\]\.vec_dot);',
                            r'WEFT_GGML_DOT_TYPE\1WEFT_GGML_DOT(type, \2);', source)
    if count != 2:
        raise CompilationError("llama.cpp needs the reviewed MUL_MAT and MUL_MAT_ID vec-dot call sites")
    source, count = re.subn(r'\bfrom_float\(', 'WEFT_GGML_FROM_FLOAT(from_float, ', source)
    if count != 4:
        raise CompilationError("llama.cpp activation quantization boundaries have changed")
    source = _once(source, '    // extra_buffer op?\n', '    WEFT_GGML_NODE(params, tensor);\n\n    // extra_buffer op?\n')
    source, count = re.subn(r'(?m)^(\s*)vec_dot\(', r'\1WEFT_GGML_CALL_DOT(vec_dot, ', source)
    if count != 2:
        raise CompilationError("llama.cpp ordinary and indirect vec-dot invocation boundaries have changed")
    source = _once(source, '    if (ggml_cpu_extra_compute_forward(params, tensor)) {\n',
                   '    if (ggml_cpu_extra_compute_forward(params, tensor)) {\n        WEFT_GGML_EXTRA(params, tensor);\n')
    source = _once(source, '        is_first_call = false;\n', '        WEFT_GGML_INIT();\n        is_first_call = false;\n')
    source = _once(source, '            ggml_compute_forward(&params, node);',
                   '            ggml_compute_forward(&params, node);\n            WEFT_GGML_END_NODE();')
    graph_loop = '    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {'
    source = _once(source, graph_loop, '    WEFT_GGML_GRAPH(cgraph, cplan);\n\n' + graph_loop)
    build = 'option(GGML_WEFT "Use an explicit Weft kernel package at CPU call sites" OFF)\n\n' + build
    anchor = '    target_sources(${GGML_CPU_NAME} PRIVATE ${GGML_CPU_SOURCES})'
    build = _once(build, anchor, anchor + NATIVE_BUILD + '''
    if (GGML_WEFT)
        find_package(Weft CONFIG REQUIRED)
        include("${Weft_LLAMA_CPP_DIR}/WeftLlama.cmake")
        weft_add_llama_backend(${GGML_CPU_NAME})
    endif()''')
    if header.exists():
        raise FileExistsError(header)
    header.write_text(HOOK_HEADER, encoding="utf-8")
    path.write_text(source, encoding="utf-8")
    cmake.write_text(build, encoding="utf-8")
    ime_path.write_text(ime, encoding="utf-8")
    return {"source": str(root), "changed": True, "files": [str(path), str(cmake), str(header), str(ime_path)]}
