"""CPU math library selected through Triton's backend module map."""
from triton.language.math import exp, exp2, log, sqrt, sqrt_rn, rsqrt, abs

__all__ = ["exp", "exp2", "log", "sqrt", "sqrt_rn", "rsqrt", "abs"]
