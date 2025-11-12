from benchmarks.benchmark import Benchmark
from top.ops import Gemm, Gemm_bwd
import torch


class gemm_benchmark(Benchmark): 

    op_type = Gemm

    def __init__(self, M, N, K, dtype):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

    @property
    def total_flops(self):
        return 2.0 * self.M * self.N * self.K

    @property
    def total_memory(self):
        return (self.M * self.K + self.K * self.N + self.M * self.N) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.M, self.K, device='cuda', dtype=self.dtype)
        B = torch.randn(self.K, self.N, device='cuda', dtype=self.dtype)
        return A, B

    def ref_program(self, A: torch.Tensor, B: torch.Tensor):
        return torch.matmul(A, B)
    

class gemm_bwd_benchmark(Benchmark): 

    op_type = Gemm_bwd

    def __init__(self, M, N, K, dtype):
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

    @property
    def total_flops(self):
        return 4.0 * self.M * self.N * self.K

    @property
    def total_memory(self):
        return (2 * self.M * self.K + 2 * self.K * self.N + self.M * self.N) * self.dtype.itemsize

    def gen_inputs(self):
        A = torch.randn(self.M, self.K, device='cuda', dtype=self.dtype, requires_grad=True)
        B = torch.randn(self.K, self.N, device='cuda', dtype=self.dtype, requires_grad=True)
        dC = torch.randn(self.M, self.N, device='cuda', dtype=self.dtype)
        return A, B, dC

    def ref_program(self, A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
        output = torch.matmul(A, B)
        output.backward(dC)
        return A.grad, B.grad