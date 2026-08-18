"""Kimi Delta Attention (KDA) operators (host-side Op layer).

Provides:
  - KDARecurrentFwdOp: full-sequence recurrent forward
"""

from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kda_recurrence import KDARecurrentFwdKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["KDARecurrentFwdOp"]


class KDARecurrentFwdOp(Op):
    """KDA (Kimi Delta Attention) full-sequence recurrent forward.

    Computes the per-channel-gated delta rule step by step:
        S_t = S_{t-1} * exp(g_t)[:, None] + beta_t k_t (v_t - (S_{t-1} * exp(g_t)[:, None])^T k_t)^T
        o_t = scale * S_t^T q_t

    Layout: BTHD (batch, seq_len, head, dim), matching the FLA
    ``fla.ops.kda.fused_recurrent_kda`` convention (H == HV, plain log-decay
    ``g``). State ownership is functional: ``initial_state`` is read-only and
    the op returns a fresh ``final_state`` tensor.

    Supports float32, float16, and bfloat16 with fp32 state accumulation.

    Args:
        scale: Output scale applied to q; a value <= 0 resolves to
            ``DK ** -0.5`` at kernel build time (default -1.0).
        kernel_map: Optional override for kernel dispatch.
        tune: Whether to autotune (default False).
    """

    def __init__(
        self,
        scale: float = -1.0,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.scale = scale
        self.tune = tune
        self.batch = None
        self.seq_len = None
        self.heads = None
        self.dim_k = None
        self.dim_v = None
        self.dtype = None

        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"KDARecurrentFwdKernel": KDARecurrentFwdKernel}

    def _infer_output_shapes(
        self,
        q_shape: tuple[int, ...],
        k_shape: tuple[int, ...],
        v_shape: tuple[int, ...],
        g_shape: tuple[int, ...],
        beta_shape: tuple[int, ...],
        initial_state_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        del k_shape, g_shape, beta_shape
        return {
            "o": (q_shape[0], q_shape[1], q_shape[2], v_shape[-1]),
            "final_state": tuple(initial_state_shape),
        }

    def _validate_dtypes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> None:
        dtype = q.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported dtype: {dtype}")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
        ):
            if tensor.dtype != dtype:
                raise ValueError(f"{name}.dtype must be {dtype}, got {tensor.dtype}")

    def _validate_shapes(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> None:
        if q.ndim != 4:
            raise ValueError("q must have shape [batch, seq_len, heads, dim_k]")
        batch, seq_len, heads, dim_k = q.shape
        if v.ndim != 4 or v.shape[:3] != (batch, seq_len, heads):
            raise ValueError("v must have shape [batch, seq_len, heads, dim_v]")
        dim_v = v.shape[-1]
        q_shape = (batch, seq_len, heads, dim_k)
        v_shape = (batch, seq_len, heads, dim_v)
        beta_shape = (batch, seq_len, heads)
        state_shape = (batch, heads, dim_k, dim_v)
        expected_shapes = (
            ("q", q, q_shape),
            ("k", k, q_shape),
            ("v", v, v_shape),
            ("g", g, q_shape),
            ("beta", beta, beta_shape),
            ("initial_state", initial_state, state_shape),
        )
        for name, tensor, expected in expected_shapes:
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if not all(tensor.is_cuda for tensor in (q, k, v, g, beta, initial_state)):
            raise ValueError("q, k, v, g, beta, and initial_state must be CUDA tensors")
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = q.dtype

    def eval_roofline(self) -> tuple[int, int]:
        from tileops.perf.formulas import kda_recurrent_fwd_roofline

        return kda_recurrent_fwd_roofline(self)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run KDA recurrent forward.

        Args:
            q: Query tensor [B, S, H, DK].
            k: Key tensor [B, S, H, DK].
            v: Value tensor [B, S, H, DV].
            g: Per-channel log-decay gate [B, S, H, DK].
            beta: Delta-rule write strength [B, S, H].
            initial_state: Initial recurrent state [B, H, DK, DV].

        Returns:
            Tuple of (o [B, S, H, DV], final_state [B, H, DK, DV]).
        """
        self._validate_dtypes(q, k, v, g, beta, initial_state)
        self._validate_shapes(q, k, v, g, beta, initial_state)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        initial_state = initial_state.contiguous()
        batch, seq_len, heads, dim_k, dim_v = (
            self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v)
        kernel = self.get_or_build_kernel(
            "KDARecurrentFwdKernel",
            (q, k, v, g, beta, initial_state),
            key=((batch, seq_len, heads, dim_k, dim_v, self.scale), q.dtype),
            build=lambda: self.kernel_map["KDARecurrentFwdKernel"](
                batch,
                seq_len,
                heads,
                dim_k,
                dim_v,
                scale=self.scale,
                dtype=Kernel.dtype_to_str(q.dtype),
                tune=self.tune,
            ),
        )
        return kernel(q, k, v, g, beta, initial_state)
