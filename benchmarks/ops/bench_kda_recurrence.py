import pytest
import torch

from benchmarks.benchmark_base import ManifestBenchmark
from benchmarks.ops.attention.manifest_params import manifest_params
from tileops.manifest import load_workloads
from tileops.ops import KDARecurrentFwdOp
from workloads.linear_attention import KDARecurrentFwdWorkload
from workloads.workload_base import FixtureBase

_OP_NAME = "KDARecurrentFwdOp"


class KDARecurrentFwdBenchFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, dtype, tune",
            manifest_params(
                load_workloads(_OP_NAME),
                lambda w: (
                    w["q_shape"][0],
                    w["q_shape"][1],
                    w["q_shape"][2],
                    w["q_shape"][3],
                    w["v_shape"][3],
                ),
                tune=False,
            ),
        ),
    ]


@KDARecurrentFwdBenchFixture
def test_kda_recurrent_fwd_bench(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = KDARecurrentFwdWorkload(batch, seq_len, heads, dim_k, dim_v, dtype)
    inputs = test.gen_inputs()

    op = KDARecurrentFwdOp(tune=tune)
    bm = ManifestBenchmark(_OP_NAME, op, test)
    # Pure-PyTorch step-by-step recurrence as the independent baseline.
    functors = {"tileops": op, "torch-ref": test.ref_program}

    bm.compare(functors, *inputs, record_as=op, params=locals())
