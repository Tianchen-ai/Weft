"""CPU Tensor forward and explicit backward using the same Triton JIT route."""
import torch
from .affine_backward import affine_backward
from .affine_strided import affine_strided


class AffineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale, bias):
        result = torch.empty_like(value)
        affine_strided[(1,)](value, result, scale, bias=bias, B=64)
        ctx.scale = scale
        return result

    @staticmethod
    def backward(ctx, gradient):
        result = torch.empty_like(gradient)
        affine_backward[(1,)](gradient, result, ctx.scale)
        return result, None, None


class Affine(torch.nn.Module):
    def __init__(self, scale=1.0, bias=0.0):
        super().__init__()
        self.scale, self.bias = scale, bias

    def forward(self, value):
        return AffineFunction.apply(value, self.scale, self.bias)
