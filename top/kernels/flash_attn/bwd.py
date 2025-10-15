import tilelang
import tilelang.language as T
from typing import Optional
from top.kernels.kernel import Kernel
import itertools

__all__ = ['mha_bwd_preprocess_kernel', 'mha_bwd_kernel', 'mha_bwd_wgmma_pipelined_kernel']


@tilelang.jit(out_idx=[2])
def _mha_bwd_preprocess_kernel(batch, heads, seq_len, dim, dtype):
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(
            O: T.Tensor(shape, dtype),  # type: ignore
            dO: T.Tensor(shape, dtype),  # type: ignore
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o)
                T.copy(dO[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk:(by + 1) * blk])

    return flash_bwd_prep


class mha_bwd_preprocess_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self, batch, heads, seq_len, dim, dtype):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype

        self.kernel = _mha_bwd_preprocess_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                                 self.dtype_str)


def _mha_bwd_kernel(batch, heads, seq_len, dim, is_causal, dtype="float16"):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_bwd_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_bwd_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                dO: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(shape, accum_dtype),  # type: ignore
                dK: T.Tensor(shape, dtype),  # type: ignore
                dV: T.Tensor(shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                # should not store K to local if dim is large
                # K_local = T.alloc_fragment([block_M, dim], dtype)
                # K_local_T = T.alloc_fragment([block_M, dim], dtype)
                # V_local = T.alloc_fragment([block_M, dim], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)

                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                    T.clear(qkT)
                    T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j,
                                                       qkT[i, j], 0)
                    T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    T.copy(dq, dq_shared)
                    T.atomic_add(dQ[bz, k * block_N:(k + 1) * block_N, bx, :], dq_shared)

                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

        return _mha_bwd_main

    return _mha_bwd_func


def _mha_bwd_wgmma_pipelined_kernel(batch, heads, seq_len, dim, is_causal, dtype="float16"):
    sm_scale = (1.0 / dim)**0.5
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    shape = [batch, seq_len, heads, dim]
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[7, 8],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mha_bwd_wgmma_pipelined_func(block_M, block_N, num_stages, threads):

        @T.prim_func
        def _mha_bwd_wgmma_pipelined_main(
                Q: T.Tensor(shape, dtype),  # type: ignore
                K: T.Tensor(shape, dtype),  # type: ignore
                V: T.Tensor(shape, dtype),  # type: ignore
                dO: T.Tensor(shape, dtype),  # type: ignore
                lse: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
                dQ: T.Tensor(shape, accum_dtype),  # type: ignore
                dK: T.Tensor(shape, dtype),  # type: ignore
                dV: T.Tensor(shape, dtype),  # type: ignore
        ):
            with T.Kernel(
                    heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                # should not store K to local if dim is large
                # K_local = T.alloc_fragment([block_M, dim], dtype)
                # K_local_T = T.alloc_fragment([block_M, dim], dtype)
                # V_local = T.alloc_fragment([block_M, dim], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({
                    dq_shared: tilelang.layout.make_swizzled_layout(dq_shared),
                    dv_shared: tilelang.layout.make_swizzled_layout(dv_shared),
                    dk_shared: tilelang.layout.make_swizzled_layout(dk_shared),
                })

                T.copy(K[bz, by * block_M:(by + 1) * block_M, bx, :], K_shared)
                T.copy(V[bz, by * block_M:(by + 1) * block_M, bx, :], V_shared)
                T.clear(dv)
                T.clear(dk)

                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N:(k + 1) * block_N, bx, :], q)
                    T.clear(qkT)
                    T.gemm(
                        K_shared,
                        q,
                        qkT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.copy(dO[bz, k * block_N:(k + 1) * block_N, bx, :], do)
                    T.clear(dsT)
                    T.gemm(
                        V_shared,
                        do,
                        dsT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                        wg_wait=-1)
                    T.wait_wgmma(1)

                    T.copy(lse[bz, bx, k * block_N:(k + 1) * block_N], lse_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(by * block_M + i <= k * block_N + j,
                                                       qkT[i, j], 0)
                    T.wait_wgmma(0)
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow, wg_wait=-1)

                    T.copy(Delta[bz, bx, k * block_N:(k + 1) * block_N], delta)

                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow, wg_wait=1)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True, wg_wait=1)
                    T.wait_wgmma(0)
                    T.copy(dq, dq_shared)
                    T.atomic_add(dQ[bz, k * block_N:(k + 1) * block_N, bx, :], dq_shared)

                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, by * block_M:(by + 1) * block_M, bx, :])
                T.copy(dk_shared, dK[bz, by * block_M:(by + 1) * block_M, bx, :])

        return _mha_bwd_wgmma_pipelined_main

    return _mha_bwd_wgmma_pipelined_func


class mha_bwd_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_bwd_kernel(self.batch, self.heads, self.seq_len, self.dim,
                                      self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 64,
            "block_N": 64 if self.dim <= 64 else 32,
            "num_stages": 1,
            "threads": 128
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs


class mha_bwd_wgmma_pipelined_kernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 dim,
                 is_causal,
                 dtype,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

        self.kernel = _mha_bwd_wgmma_pipelined_kernel(self.batch, self.heads, self.seq_len,
                                                      self.dim, self.is_causal, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_M": 128,
            "block_N": 128 if self.dim <= 64 else 32,
            "num_stages": 2,
            "threads": 256
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_M = [32, 64, 128]
        block_N = [32, 64, 128]
        num_stages = [1, 2, 3]
        threads = [128, 256]
        _configs = list(itertools.product(block_M, block_N, num_stages, threads))

        configs = [{
            'block_M': c[0],
            'block_N': c[1],
            'num_stages': c[2],
            'threads': c[3]
        } for c in _configs]
        return configs
