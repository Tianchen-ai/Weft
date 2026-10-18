"""Framework hooks for Triton's extended programming models.

The PyTorch Triton HOP owns capture, functionalization, and graph scheduling.
These hooks supply the model's semantic facts and direct native launch.
"""
from .effects import kernel_accesses, kernel_interface

__all__ = ["kernel_accesses", "kernel_interface"]
