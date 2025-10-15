import torch
from abc import ABC, abstractmethod
from tilelang.profiler import do_bench
from top.ops import Op


class Benchmark(ABC):

    op_type: type[Op]

    @property
    def total_flops(self):
        raise NotImplementedError

    @property
    def total_memory(self):
        raise NotImplementedError

    def gen_inputs(self):
        raise NotImplementedError
        #TODo: impl this?

    @abstractmethod
    def ref_program(self, *inputs):
        raise NotImplementedError

    def check(self, op, *inputs, atol=1e-2, rtol=1e-2):
        """Check the correctness of the op"""
        assert isinstance(op, self.op_type), f"op is not instance of {self.op_type.__name__}"

        try:
            outputs_ref = self.ref_program(*inputs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            else:
                raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        with torch.no_grad():
            outputs = op(*inputs)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), "outputs and outputs_ref have different size"
        for i, (output, output_ref) in enumerate(zip(outputs, outputs_ref)):
            # print(f"outputs[{i}] max err: {(output - output_ref).abs().max()}")
            if output_ref is not None:  # skip checking for None placeholders in ref
                assert torch.allclose(output, output_ref, atol=atol, rtol=rtol), \
                    f"outputs[{i}] is not close to outputs_ref[{i}], max err: {(output - output_ref).abs().max()}"

        print(f"All checks passed for {op.__class__.__name__}.✅")

    def profile(self, op, *inputs, warmup=25, rep=100):
        assert isinstance(op, self.op_type), f"op is not instance of {self.op_type.__name__}"

        print(f"===== Profiling {op.__class__.__name__} =====")
        with torch.no_grad():
            # Always use cupti backend for better accuracy
            latency = do_bench(lambda: op(*inputs), warmup=warmup, rep=rep, backend='cupti')

        print(f"{op.__class__.__name__} latency: {latency:.2f} ms")
        if self.total_flops is not None:
            print(f"{op.__class__.__name__} TFlops: {self.total_flops / latency * 1e-9:.2f} TFlops")
        if self.total_memory is not None:
            print(
                f"{op.__class__.__name__} Bandwidth: {self.total_memory / latency * 1e-9:.2f} GB/s")
