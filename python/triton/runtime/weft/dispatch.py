from __future__ import annotations

from .errors import CompilationError


def dispatch_source(metadata: dict) -> str:
    """Mechanical C ABI marshalling for the already compiled kernel declaration.

    View entries are base addresses. Scalar, extent and stride entries point to
    objects of their exact ABI type. No argument expression is an algorithm or
    physical decision; the compiler's typed declaration remains authoritative.
    The original symbol address is supplied by its own dlopen handle, avoiding
    ELF interposition between two configurations of the same kernel symbol.
    """
    symbol = metadata["symbol"]
    if not symbol.isascii() or not symbol.isidentifier():
        raise CompilationError("native dispatch needs a compiler-emitted C identifier")
    entry_type = symbol + "__weft_entry_type"
    lines = ['#include "kernel.h"', '#include <string.h>', '',
             f'typedef __typeof__(&{symbol}) {entry_type};',
             'void weft_kernel_dispatch(void (*entry)(void), void *const *arguments) {']
    values = []
    for index, argument in enumerate(metadata["arguments"]):
        ctype = argument["c_type"]
        if argument["kind"] in ("view", "pointer"):
            values.append(f"({ctype})arguments[{index}]")
        else:
            lines += [f"    {ctype} value_{index};",
                      f"    memcpy(&value_{index}, arguments[{index}], sizeof(value_{index}));"]
            values.append(f"value_{index}")
    count = len(values)
    extras = len(metadata["shape_parameters"]) + sum(
        len(argument["shape"]) for argument in metadata["arguments"] if argument["strided"])
    for index in range(count, count + extras):
        lines += [f"    size_t value_{index};",
                  f"    memcpy(&value_{index}, arguments[{index}], sizeof(value_{index}));"]
        values.append(f"value_{index}")
    if not values:
        lines.append("    (void)arguments;")
    lines += [f"    (({entry_type})entry)({', '.join(values)});", "}", ""]
    return "\n".join(lines)
