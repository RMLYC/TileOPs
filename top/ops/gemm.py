import torch
from .op import Op
from top.kernels import gemm_kernel, gemm_bwd_kernel, Kernel
from typing import Optional, Dict

__all__ = ['Gemm', 'Gemm_bwd']


class Gemm(Op):

    def __init__(self,
                 M: int,
                 N: int,
                 K: int,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.M = M
        self.N = N
        self.K = K

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemm_kernel"](M, N, K, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"gemm_kernel": gemm_kernel}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.kernel(A, B)


class Gemm_bwd(Op):

    def __init__(self,
                 M: int,
                 N: int,
                 K: int,
                 dtype=torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune=False):
        self.M = M
        self.N = N
        self.K = K

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gemm_bwd_kernel"](M, N, K, self.dtype, tune=tune)

    @property
    def default_kernel_map(self):
        return {"gemm_bwd_kernel": gemm_bwd_kernel}

    def forward(self, A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor) -> torch.Tensor:
        dC = dC.contiguous()
        return self.kernel(A, B, dC)
    