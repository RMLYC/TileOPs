import tilelang
import tilelang.language as T
from typing import Optional
import itertools
import torch
from top.utils import get_sm_version
from top.kernels import Kernel

__all__ = ["gemm_kernel", "gemm_bwd_kernel"]

def _gemm_kernel(M, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(out_idx=[-1], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_func(block_M, block_N, block_K, threads, num_stages, enable_rasteration):

        @T.prim_func
        def _gemm_main(
                A: T.Tensor((M, K), dtype),  # type: ignore
                B: T.Tensor((K, N), dtype),  # type: ignore
                C: T.Tensor((M, N), dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)

                T.annotate_layout({
                    C_shared: tilelang.layout.make_swizzled_layout(C_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                T.clear(C_local)

                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])

        return _gemm_main

    return _gemm_func


class gemm_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self, M, N, K, dtype, config: Optional[dict] = None, tune=False):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _gemm_kernel(M, N, K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        sm_version = get_sm_version()
        if sm_version in {80}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True
            }
        elif sm_version in {90}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 64,
                "num_stages": 3,
                "threads": 256,
                "enable_rasteration": True
            }
        else:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 0,
                "threads": 128,
                "enable_rasteration": True
            }

    @property
    def autotune_configs(self) -> list[dict]:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        enable_rasteration = [True, False]
        _configs = list(
            itertools.product(block_M, block_N, block_K, num_stages, threads, enable_rasteration))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4],
            'enable_rasteration': c[5]
        } for c in _configs]
        return configs

    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return self.kernel(**self.config)(A, B)


def _gemm_bwd_kernel(M, N, K, dtype='float16'):
    accum_dtype = "float"

    @tilelang.jit(out_idx=[3,4], compile_flags=["-O3", "-DENABLE_BF16"])
    def _gemm_bwd_func(block_M, block_N, block_K, threads, num_stages, enable_rasteration):

        @T.macro
        def _gemm_bwd_A(
                B: T.Tensor((K, N), dtype),  # type: ignore
                dC: T.Tensor((M, N), dtype),  # type: ignore
                dA: T.Tensor((M, K), dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(M, block_M), T.ceildiv(K, block_K), threads=threads) as (bx, by):
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                dC_shared = T.alloc_shared((block_M, block_N), dtype)
                dA_shared = T.alloc_shared((block_M, block_K), dtype)
                dA_local = T.alloc_fragment((block_M, block_K), accum_dtype)

                T.annotate_layout({
                    dA_shared: tilelang.layout.make_swizzled_layout(dA_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                T.clear(dA_local)
                for n in T.Pipelined(T.ceildiv(N, block_N), num_stages=num_stages):
                    T.copy(B[by * block_K, n * block_N], B_shared)
                    T.copy(dC[bx * block_M, n * block_N], dC_shared)
                    T.gemm(dC_shared, B_shared, dA_local, transpose_B=True)
                
                T.copy(dA_local, dA_shared)
                T.copy(dA_shared, dA[bx * block_M, by * block_K])

        @T.macro
        def _gemm_bwd_B(
                A: T.Tensor((M, K), dtype),  # type: ignore
                dC: T.Tensor((M, N), dtype),  # type: ignore
                dB: T.Tensor((K, N), dtype),  # type: ignore
        ):
            with T.Kernel(
                    T.ceildiv(N, block_N), T.ceildiv(K, block_K), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                dC_shared = T.alloc_shared((block_M, block_N), dtype)
                dB_shared = T.alloc_shared((block_K, block_N), dtype)
                dB_local = T.alloc_fragment((block_K, block_N), accum_dtype)

                T.annotate_layout({
                    dB_shared: tilelang.layout.make_swizzled_layout(dB_shared),
                })
                T.use_swizzle(10, enable=enable_rasteration)

                T.clear(dB_local)
                for m in T.Pipelined(T.ceildiv(M, block_M), num_stages=num_stages):
                    T.copy(A[m * block_M, by * block_K], A_shared)
                    T.copy(dC[m * block_M, bx * block_N], dC_shared)
                    T.gemm(A_shared, dC_shared, dB_local, transpose_A=True)
                
                T.copy(dB_local, dB_shared)
                T.copy(dB_shared, dB[by * block_K, bx * block_N])

        @T.prim_func
        def _gemm_bwd_main(
                A: T.Tensor((M, K), dtype),  # type: ignore
                B: T.Tensor((K, N), dtype),  # type: ignore
                dC: T.Tensor((M, N), dtype),  # type: ignore
                dA: T.Tensor((M, K), dtype),  # type: ignore
                dB: T.Tensor((K, N), dtype),  # type: ignore
        ):
            _gemm_bwd_A(B, dC, dA)
            _gemm_bwd_B(A, dC, dB)

        return _gemm_bwd_main

    return _gemm_bwd_func


class gemm_bwd_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self, M, N, K, dtype, config: Optional[dict] = None, tune=False):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.dtype = dtype

        self.kernel = _gemm_bwd_kernel(M, N, K, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        sm_version = get_sm_version()
        if sm_version in {80}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 2,
                "threads": 128,
                "enable_rasteration": True
            }
        elif sm_version in {90}:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 64,
                "num_stages": 2,
                "threads": 256,
                "enable_rasteration": True
            }
        else:
            return {
                "block_M": 128,
                "block_N": 256,
                "block_K": 32,
                "num_stages": 0,
                "threads": 128,
                "enable_rasteration": True
            }

    @property
    def autotune_configs(self) -> list[dict]:
        # From tilelang/examples/gemm/example_gemm_autotune.py
        block_M = [64, 128, 256]
        block_N = [64, 128, 256]
        block_K = [32, 64]
        num_stages = [0, 1, 2, 3]
        threads = [128, 256]
        enable_rasteration = [True, False]
        _configs = list(
            itertools.product(block_M, block_N, block_K, num_stages, threads, enable_rasteration))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'block_K': c[2],
            'num_stages': c[3],
            'threads': c[4],
            'enable_rasteration': c[5]
        } for c in _configs]
        return configs

    def forward(self, A: torch.Tensor, B: torch.Tensor, dC: torch.Tensor):
        return self.kernel(**self.config)(A, B, dC)
# TODO: add persistent, split-k, steam-k...