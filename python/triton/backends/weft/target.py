"""Inspect the registered RISC-V compiler and native execution capabilities.

Catalog and explicit-target queries work on a cross-build host and require no
C compiler, sysroot or device. Native discovery reports the current CPU set;
it neither changes affinity nor grants permission to execute an extension.
"""
from dataclasses import asdict

from triton._C.libtriton import weft
from triton.backends.compiler import CPUTarget
from .compiler import WeftOptions

__all__ = ["extension_catalog", "target_capabilities", "native_capabilities"]


def extension_catalog() -> dict:
    """Return the local extensions built into this Triton compiler and SDK."""
    return weft.extension_catalog()


def target_capabilities(target: CPUTarget, *, matrix_extension: str = "none") -> dict:
    """Describe an explicit build target using the compiler's target validator.

    ``matrix_extension`` has the same meaning as the backend option. A positive
    result proves that this compiler implements the target contract, not that
    the host CPU or selected C toolchain can execute or assemble it.
    """
    if not isinstance(target, CPUTarget) or target.backend != "weft" or target.arch != "riscv64":
        raise TypeError("target_capabilities requires a Weft riscv64 CPUTarget")
    if not isinstance(matrix_extension, str):
        raise TypeError("matrix_extension must be a registered extension name or 'none'")
    options = asdict(WeftOptions(matrix_extension=matrix_extension))
    options.update(march=target.march, abi=target.abi, vlen_bits=target.vlen_bits)
    return weft.target_capabilities(options)


def native_capabilities() -> dict:
    """Discover ISA facts and eligible extension CPUs in the current affinity.

    Actual workers still undergo the SDK's execution checks at launch time.
    This query is supported on RISC-V Linux; it does not emulate a native target
    on a cross-build host.
    """
    return weft.native_target()
