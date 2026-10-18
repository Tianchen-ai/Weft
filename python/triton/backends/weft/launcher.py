"""Typed CPU launch code; grid coordinates stay outside canonical semantics."""
import copy


GRID_DECLARATION = """typedef int (*triton_check_thread)(const void *, char *, size_t);
int triton_weft_grid(void (*entry)(void), void *const *arguments,
                    size_t grid0, size_t grid1, size_t grid2, int threads,
                    const void *scope, triton_check_thread check, char *error, size_t error_size);"""


def single_instance_abi(abi):
    result = copy.deepcopy(abi)
    result["symbol"] += "_instance"
    result["arguments"] = [arg for arg in result["arguments"] if not arg["name"].startswith("__triton_")]
    parameters = [f"{arg['c_type']} {arg['name']}" for arg in result["arguments"]]
    extras = len(abi["shape_parameters"]) + sum(len(arg["shape"]) for arg in abi["arguments"] if arg["strided"])
    names = {arg["name"] for arg in result["arguments"]}
    for i in range(extras):
        name = f"extra_{i}"
        while name in names:
            name += "_"
        names.add(name)
        parameters.append(f"size_t {name}")
    result["declaration"] = f"void {result['symbol']}({', '.join(parameters) or 'void'});"
    return result


def launch_source(abi):
    from triton.runtime.weft.dispatch import dispatch_source
    symbol = abi["symbol"]
    if not symbol.isascii() or not symbol.isidentifier():
        raise ValueError("invalid native kernel symbol")
    values, declarations, fields = [], [], []
    for index, argument in enumerate(abi["arguments"]):
        name = argument["name"]
        if name.startswith("__triton_pid_"):
            values.append("pid" + name[-1])
        elif name.startswith("__triton_grid_"):
            values.append("grid" + name[-1])
        elif argument["kind"] in ("view", "pointer"):
            fields.append(f"{argument['c_type']} value_{index};")
            declarations.append(f"decoded.value_{index} = ({argument['c_type']})arguments[{index}];")
            values.append(f"decoded->value_{index}")
        else:
            fields.append(f"{argument['c_type']} value_{index};")
            declarations.append(f"memcpy(&decoded.value_{index}, arguments[{index}], sizeof(decoded.value_{index}));")
            values.append(f"decoded->value_{index}")
    count = len(abi["arguments"])
    extras = len(abi["shape_parameters"]) + sum(len(arg["shape"]) for arg in abi["arguments"] if arg["strided"])
    for index in range(count, count + extras):
        fields.append(f"size_t value_{index};")
        declarations.append(f"memcpy(&decoded.value_{index}, arguments[{index}], sizeof(decoded.value_{index}));")
        values.append(f"decoded->value_{index}")
    invocation = f"(({symbol}__weft_entry_type)entry)({', '.join(values)});"
    dispatch = dispatch_source(abi).replace('#include "kernel.h"', '')
    single = single_instance_abi(abi)
    # Private wrapper parameter names must not hide the kernel symbol. Resolve
    # the target in file scope, just as the typed indirect dispatch does.
    parameters, single_args = [], []
    for arg in abi["arguments"]:
        if arg["name"].startswith("__triton_pid_"):
            single_args.append("0")
        elif arg["name"].startswith("__triton_grid_"):
            single_args.append("1")
        else:
            name = f"weft_arg_{len(parameters)}"
            parameters.append(f"{arg['c_type']} {name}")
            single_args.append(name)
    for i in range(extras):
        name = f"weft_arg_{len(parameters)}"
        parameters.append(f"size_t {name}")
        single_args.append(name)
    direct = symbol + "__weft_direct"
    single_body = (f"static {symbol}__weft_entry_type const {direct} = &{symbol};\n"
                   f"void {single['symbol']}({', '.join(parameters) or 'void'}) {{ "
                   + direct + "(" + ", ".join(single_args) + "); }")
    instance = symbol + "__weft_launch_instance"
    single_dispatch = dispatch_source(single).replace('#include "kernel.h"', '').replace(
        'void weft_kernel_dispatch(', 'void triton_weft_instance_dispatch(')
    return f'''#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
{abi['declaration']}
{dispatch}
{single_body}
{single_dispatch}
{GRID_DECLARATION}
struct launch_arguments {{ {' '.join(fields) if fields else 'unsigned char unused;'} }};
static inline void {instance}(void (*entry)(void), const struct launch_arguments *decoded,
                            size_t grid0, size_t grid1, size_t grid2, size_t linear) {{
  const size_t pid0 = linear % grid0;
  const size_t pid1 = (linear / grid0) % grid1;
  const size_t pid2 = linear / (grid0 * grid1);
  {invocation}
}}
int triton_weft_grid(void (*entry)(void), void *const *arguments,
                    size_t grid0, size_t grid1, size_t grid2, int threads,
                    const void *scope, triton_check_thread check, char *error, size_t error_size) {{
  if (!grid0 || !grid1 || !grid2) return 0;
  if (grid0 > SIZE_MAX / grid1 || grid0 * grid1 > SIZE_MAX / grid2) return 1;
  size_t count = grid0 * grid1 * grid2;
  if (count > PTRDIFF_MAX || threads < 0 || !check) return 1;
  struct launch_arguments decoded;
  {' '.join(declarations)}
  if (threads == 0) threads = omp_get_max_threads();
  if (count == 1 || threads == 1) {{
    int status = check(scope, error, error_size);
    if (status) return status;
    for (size_t i = 0; i < count; ++i) {instance}(entry, &decoded, grid0, grid1, grid2, i);
    return 0;
  }}
  int failed = 0;
  #pragma omp parallel num_threads(threads) shared(failed)
  {{
    char local_error[1024] = {{0}};
    int status = check(scope, local_error, sizeof(local_error));
    if (status) {{
      #pragma omp critical
      {{
        if (!failed) {{
          failed = status;
          if (error && error_size) {{ strncpy(error, local_error, error_size - 1); error[error_size - 1] = 0; }}
        }}
      }}
    }}
    #pragma omp barrier
    if (!failed) {{
      #pragma omp for schedule(static)
      for (ptrdiff_t i = 0; i < (ptrdiff_t)count; ++i)
        {instance}(entry, &decoded, grid0, grid1, grid2, (size_t)i);
    }}
  }}
  return failed;
}}
'''
