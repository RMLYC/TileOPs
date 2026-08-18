import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import KDARecurrentFwdOp
from workloads.linear_attention import KDARecurrentFwdWorkload


class KDARecurrentFwdTest(KDARecurrentFwdWorkload, TestBase):
    pass


# Correctness tests


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        # fp32 state accumulation over a full sequence still compounds small
        # rounding differences against the step-by-step reference.
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 1e-2, "rtol": 1e-2}
    else:  # bfloat16
        return {"atol": 2e-2, "rtol": 2e-2}


class KDARecurrentFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, dtype, tune", [
            # dtype coverage: one typical shape per supported dtype (smoke)
            pytest.param(1, 64, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
            # shape coverage: multi-batch, DK=DV=128 (Kimi Linear head dims),
            # non-power-of-2 seq_len (serial-loop tail), degenerate S=1
            pytest.param(2, 128, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 100, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 128, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 128, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 256, 32, 128, 128, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


@KDARecurrentFwdFixture
def test_kda_recurrent_fwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    test = KDARecurrentFwdTest(batch, seq_len, heads, dim_k, dim_v, dtype)
    op = KDARecurrentFwdOp(tune=tune)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@pytest.mark.smoke
def test_kda_recurrent_fwd_explicit_scale() -> None:
    """Feature coverage: a caller-supplied positive scale overrides DK**-0.5."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    test = KDARecurrentFwdTest(1, 64, 4, 64, 64, dtype, scale=0.25)
    op = KDARecurrentFwdOp(scale=0.25)
    tols = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), **tols)


@pytest.mark.smoke
def test_kda_recurrent_fwd_zero_initial_state() -> None:
    """Boundary coverage: zero initial state == no history."""
    torch.manual_seed(42)
    dtype = torch.float32
    test = KDARecurrentFwdTest(1, 64, 4, 64, 64, dtype)
    q, k, v, g, beta, _ = test.gen_inputs()
    initial_state = torch.zeros(1, 4, 64, 64, device="cuda", dtype=dtype)
    op = KDARecurrentFwdOp()
    tols = _get_tolerances(dtype)
    test.check(op, q, k, v, g, beta, initial_state, **tols)


@pytest.mark.smoke
def test_kda_recurrent_fwd_rejects_manifest_shape_mismatch() -> None:
    op = KDARecurrentFwdOp()

    q = torch.empty(2, 5, 3, 4)
    k = torch.empty(2, 5, 3, 4)
    v = torch.empty(2, 5, 3, 6)
    g = torch.empty(2, 5, 3, 5)
    beta = torch.empty(2, 5, 3)
    initial_state = torch.empty(2, 3, 4, 6)

    with pytest.raises(ValueError, match="g must have shape"):
        op.forward(q, k, v, g, beta, initial_state)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
