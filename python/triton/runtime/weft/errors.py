from triton.errors import TritonError


class CompilationError(TritonError):
    """An invalid compiled CPU artifact or native ABI contract."""
