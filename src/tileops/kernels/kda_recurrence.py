"""
KDA (Kimi Delta Attention) full-sequence recurrent forward.

Per step t (state S in R^{DK x DV}, per-channel log decay g_t in R^{DK}):

    S      <- S * exp(g_t)[:, None]                # per-channel decay
    old     = S^T @ k_t                            # delta-rule read-out
    v_new   = beta_t * (v_t - old)
    S      <- S + outer(k_t, v_new)                # delta-rule write
    o_t     = scale * S^T @ q_t

Layout: BTHD (batch, seq_len, head, dim), matching the FLA
``fla.ops.kda.fused_recurrent_kda`` convention. The state is held in fp32
shared memory for the whole sequence loop, so multi-step numerics track the
fp32 PyTorch reference.

Optimization:
  - One thread block per (batch, head, V tile); the V dimension tiles
    trivially because every state column is independent.
  - fp32 shared-memory state: no per-step HBM round trip for S.
  - fp32 scalar accumulation for the recurrent matvecs (the same reason the
    GDN/GLA decode kernels avoid T.gemm: TF32 mantissa truncation compounds
    over the recurrence).
"""
import functools
from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["KDARecurrentFwdKernel"]

_LOG2E = 1.4426950408889634
_DEFAULT_V_TILE = 32


@functools.lru_cache(maxsize=32)
def _kda_recurrent_fwd_tl(
    batch: int,
    seq_len: int,
    head: int,
    dim_k: int,
    dim_v: int,
    v_tile: int = _DEFAULT_V_TILE,
    dtype: str = "float32",
    scale: float = -1.0,
):
    """Build the JIT-compiled KDA recurrent forward for one shape specialization.

    Every argument is baked into the TileLang program as a compile-time
    constant, so the builder is memoized with ``lru_cache``: repeat calls with
    the same (batch, seq_len, head, dim_k, dim_v, v_tile, dtype, scale) reuse
    the cached builder instead of re-tracing. Returns the ``@tilelang.jit``
    decorator factory; calling it with ``(threads)`` yields the runnable
    kernel function.
    """
    # State accumulation always happens in fp32 regardless of the I/O dtype:
    # the recurrence feeds each step's output state back in as input, so any
    # precision loss compounds over seq_len steps.
    accum_dtype = "float32"
    if dim_v % v_tile != 0:
        raise ValueError(f"dim_v={dim_v} must be divisible by v_tile={v_tile}")

    # scale <= 0 is the "unset" sentinel: resolve to the attention-style
    # default 1/sqrt(DK), matching FLA fused_recurrent_kda.
    if scale <= 0:
        scale = dim_k ** -0.5

    @tilelang.jit(
        # o and final_state are the last two prim_func params; the JIT wrapper
        # allocates and returns them, so callers pass only the six inputs.
        out_idx=[-2, -1],
        pass_configs={
            # Fast math would allow reassociation of the serial accumulations;
            # keep it off so multi-step numerics track the fp32 reference.
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        },
        compile_flags=["-O3", "-DENABLE_BF16"],
    )
    def _fwd_func(threads=32):
        @T.prim_func
        def kda_recurrent_fwd(
            q: T.Tensor([batch, seq_len, head, dim_k], dtype),
            k: T.Tensor([batch, seq_len, head, dim_k], dtype),
            v: T.Tensor([batch, seq_len, head, dim_v], dtype),
            g: T.Tensor([batch, seq_len, head, dim_k], dtype),
            beta: T.Tensor([batch, seq_len, head], dtype),
            initial_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
            o: T.Tensor([batch, seq_len, head, dim_v], dtype),
            final_state: T.Tensor([batch, head, dim_k, dim_v], dtype),
        ):
            # Grid: one block per (batch, head, V tile). The recurrence is
            # sequential in t and every state column is independent, so the V
            # dimension is the free axis for both tiling and the thread-parallel
            # inner loops below.
            with T.Kernel(batch, head, dim_v // v_tile, threads=threads) as (bid, hid, vid):
                # s_tile: this block's [dim_k, v_tile] slice of the recurrent
                # state S, held in fp32 shared memory for the whole sequence
                # (no per-step HBM round trip for S).
                s_tile = T.alloc_shared([dim_k, v_tile], accum_dtype)
                # v_new: per-column delta-rule write vector for the current
                # step (register fragment, one lane per state column).
                v_new = T.alloc_fragment([v_tile], accum_dtype)
                # o_frag: per-column output accumulator for the current step.
                o_frag = T.alloc_fragment([v_tile], accum_dtype)

                # Load this block's initial-state slice once; the sequence
                # loop then runs entirely out of shared memory.
                for i, j in T.Parallel(dim_k, v_tile):
                    s_tile[i, j] = T.cast(
                        initial_state[bid, hid, i, vid * v_tile + j], accum_dtype)
                T.sync_threads()

                # Sequential recurrence over the sequence dimension. Per step:
                #   S <- S * exp(g_t)[:, None]                (per-channel decay)
                #   S <- S + beta_t k_t (v_t - S^T k_t)^T     (delta-rule write)
                #   o_t = scale * S^T q_t                     (read-out)
                for t in T.serial(seq_len):
                    beta_val = T.cast(beta[bid, t, hid], accum_dtype)

                    # Pass 1: decay S row-wise, accumulate old = (S * alpha)^T k.
                    # Serial over dim_k (each lane reduces over the full K
                    # extent of its own column), parallel over the v_tile lanes.
                    T.fill(v_new, 0.0)
                    for i in T.Serial(dim_k):
                        # alpha_i = exp(g_i), computed as exp2(g_i * log2(e)).
                        alpha_i = T.exp2(
                            T.cast(g[bid, t, hid, i], accum_dtype) * _LOG2E)
                        k_val = T.cast(k[bid, t, hid, i], accum_dtype)
                        for j in T.Parallel(v_tile):
                            # Decay first, then read the decayed value into the
                            # reduction so `old` sees this step's gate.
                            s_tile[i, j] = s_tile[i, j] * alpha_i
                            v_new[j] = v_new[j] + s_tile[i, j] * k_val
                    T.sync_threads()

                    # v_new = beta * (v - old): the delta-rule correction,
                    # turning the read-out into the write vector.
                    for j in T.Parallel(v_tile):
                        v_new[j] = beta_val * (
                            T.cast(v[bid, t, hid, vid * v_tile + j], accum_dtype)
                            - v_new[j])
                    T.sync_threads()

                    # Pass 2: S += outer(k, v_new); o = scale * S^T q on the
                    # updated state (the current token's write is included).
                    T.fill(o_frag, 0.0)
                    for i in T.Serial(dim_k):
                        k_val = T.cast(k[bid, t, hid, i], accum_dtype)
                        q_val = T.cast(q[bid, t, hid, i], accum_dtype)
                        for j in T.Parallel(v_tile):
                            s_tile[i, j] = s_tile[i, j] + k_val * v_new[j]
                            o_frag[j] = o_frag[j] + s_tile[i, j] * q_val
                    # Write the step output in the I/O dtype; the state itself
                    # stays fp32 in shared memory.
                    for j in T.Parallel(v_tile):
                        o[bid, t, hid, vid * v_tile + j] = T.cast(
                            scale * o_frag[j], dtype)
                    # Barrier before the next step's pass 1 rewrites s_tile.
                    T.sync_threads()

                # Sequence finished: flush the fp32 state slice back to global
                # memory, cast to the I/O dtype.
                for i, j in T.Parallel(dim_k, v_tile):
                    final_state[bid, hid, i, vid * v_tile + j] = T.cast(
                        s_tile[i, j], dtype)

        return kda_recurrent_fwd

    return _fwd_func


@torch.library.custom_op("tileops::kda_recurrent_fwd_kernel", mutates_args=())
def _kda_recurrent_fwd_wrapped_kernel(
    batch: int,
    seq_len: int,
    head: int,
    dim_k: int,
    dim_v: int,
    v_tile: int,
    dtype: str,
    scale: float,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kernel_fn = _kda_recurrent_fwd_tl(
        batch, seq_len, head, dim_k, dim_v, v_tile, dtype, scale,
    )(threads)
    return kernel_fn(q, k, v, g, beta, initial_state)


@_kda_recurrent_fwd_wrapped_kernel.register_fake
def _kda_recurrent_fwd_wrapped_kernel_fake(
    batch: int,
    seq_len: int,
    head: int,
    dim_k: int,
    dim_v: int,
    v_tile: int,
    dtype: str,
    scale: float,
    threads: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty(batch, seq_len, head, dim_v, dtype=q.dtype, device=q.device)
    final_state = torch.empty(batch, head, dim_k, dim_v, dtype=q.dtype, device=q.device)
    return o, final_state


class KDARecurrentFwdKernel(Kernel):
    """KDA full-sequence recurrent forward kernel.

    One block per (batch, head, V tile); the state tile lives in fp32 shared
    memory across the whole sequence loop. Supports fp16/bf16/fp32 with fp32
    accumulation, so multi-step numerics follow the fp32 PyTorch reference.
    """

    supported_archs: list[int] = [80, 89, 90]
    general = True

    def __init__(
        self,
        batch: int,
        seq_len: int,
        head: int,
        dim_k: int,
        dim_v: int,
        scale: float = -1.0,
        dtype: str = "float32",
        config: Optional[dict] = None,
        tune: bool = False,
    ):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.head = head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = scale if scale > 0 else dim_k ** -0.5
        self.dtype = dtype

        if tune:
            self._autotune_with_v_tile()
        else:
            self.init_config(config, tune=False)

        # Cache the JIT-compiled kernel to avoid re-creation overhead
        # on every forward call (_kda_recurrent_fwd_wrapped_kernel is kept
        # for torch.compile compatibility).
        self._kernel_fn = _kda_recurrent_fwd_tl(
            batch, seq_len, head, dim_k, dim_v,
            self.config["v_tile"], self.dtype_str, self.scale,
        )(self.config["threads"])

    def _autotune_with_v_tile(self) -> None:
        """Autotune across v_tile and threads."""
        from tilelang.profiler import do_bench

        best_time = float("inf")
        best_config = self.default_config

        B, S, H, DK, DV = (
            self.batch, self.seq_len, self.head, self.dim_k, self.dim_v)
        torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                       "bfloat16": torch.bfloat16}[self.dtype_str]
        q = torch.randn(B, S, H, DK, device="cuda", dtype=torch_dtype)
        k = torch.randn(B, S, H, DK, device="cuda", dtype=torch_dtype)
        v = torch.randn(B, S, H, DV, device="cuda", dtype=torch_dtype)
        g = -torch.rand(B, S, H, DK, device="cuda", dtype=torch_dtype)
        beta = torch.rand(B, S, H, device="cuda", dtype=torch_dtype)
        initial_state = torch.randn(B, H, DK, DV, device="cuda", dtype=torch_dtype)

        print(f"Start autotuning {self.__class__.__name__}...")
        for v_tile in [16, 32, 64]:
            if DV % v_tile != 0:
                continue
            for threads in [32, 64, 128]:
                try:
                    fn = _kda_recurrent_fwd_tl(
                        B, S, H, DK, DV, v_tile, self.dtype_str, self.scale,
                    )(threads)
                    t = do_bench(
                        lambda _fn=fn: _fn(q, k, v, g, beta, initial_state),
                        warmup=10, rep=20)
                    if t < best_time:
                        best_time = t
                        best_config = {"threads": threads, "v_tile": v_tile}
                except Exception:
                    continue

        self.config = best_config
        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def default_config(self) -> dict:
        return {
            "threads": 32,
            "v_tile": _DEFAULT_V_TILE,
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_fn(q, k, v, g, beta, initial_state)
