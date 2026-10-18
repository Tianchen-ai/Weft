"""Use a hand-written Triton kernel and Inductor's CPU Triton backend.

Install the supported PyTorch source seam once with python -m triton.tools.pytorch.
"""
import torch
import triton

from source.weft.dense.triton_ops import affine


def main():
    x = torch.linspace(-1, 1, 131, dtype=torch.float32, device="cpu")
    y = torch.empty_like(x)
    affine[(triton.cdiv(x.numel(), 64),)](x, y, x.numel(), 1.25, -0.5, B=64, num_cpu_threads=2)
    print("Hand-written kernel:", y[:8])

    def normalize(value):
        centered = value - value.mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt((centered * centered).mean(dim=-1, keepdim=True) + 1e-5)

    compiled = torch.compile(normalize, backend="inductor", fullgraph=True,
                             options={"cpu_backend": "triton"})
    matrix = torch.linspace(-2, 2, 4 * 64, dtype=torch.float32).reshape(4, 64)
    print("Inductor-generated kernel:", compiled(matrix)[0, :8])


if __name__ == "__main__":
    main()
