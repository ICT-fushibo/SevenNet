"""SevenNet-owned e3nn convolution/gate region selection."""
import math
import torch
from torch import nn
from md_benchmark.opt4_fx import install_tp_regions
from md_benchmark.opt4_registry import FusionSetupError


class _NativeTPVJP(torch.autograd.Function):
    @staticmethod
    def forward(reference, candidate, *args):
        return candidate(*args)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.reference = inputs[0]
        ctx.save_for_backward(*inputs[2:])

    @staticmethod
    def backward(ctx, grad):
        # One original, UNPARTITIONED graph preserves the full TP derivative
        # grouping. Per-region native backwards do not preserve this grouping.
        # This deliberately recomputes the native TP; report its cost honestly.
        if torch.is_grad_enabled():
            raise RuntimeError("SevenNet Opt4 native TP VJP supports MD first derivatives only")
        args = tuple(x.detach().requires_grad_(needed)
                     for x, needed in zip(ctx.saved_tensors, ctx.needs_input_grad[2:]))
        active = tuple(x for x in args if x.requires_grad)
        with torch.enable_grad():
            out = ctx.reference(*args)
            gradients = iter(torch.autograd.grad(out, active, grad, allow_unused=True))
        return (None, None, *(next(gradients) if x.requires_grad else None for x in args))


class NativeTPBackward(nn.Module):
    """Candidate forward, explicit original TP VJP; MD first derivatives only."""
    def __init__(self, reference, candidate):
        super().__init__()
        self.reference = reference
        self.candidate = candidate

    def forward(self, *args):
        return _NativeTPVJP.apply(self.reference, self.candidate, *args)


class CompactSupportCutoff(nn.Module):
    """Keep far sink edges out of the polynomial's unbounded extension.

    Mask AFTER evaluating at a bounded radius: where(mask, poly(r), 0) alone
    still permits inf intermediates and 0*inf in the force backward.
    """
    def __init__(self, original, cutoff):
        super().__init__()
        self.original = original
        self.cutoff = float(cutoff)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise FusionSetupError("SevenNet Opt4 requires a finite positive cutoff")

    def forward(self, radius):
        value = self.original(radius.clamp(max=self.cutoff))
        return torch.where(radius < self.cutoff, value, 0.0)


def install(model, passes, report):
    # This model instance belongs to Opt4. Do not change EdgeEmbedding globally
    # or alter Opt3/off. Neighbor connectivity and skin/CAP are unchanged.
    bounded = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ == "EdgeEmbedding":
            cutoff = module.cutoff_function
            if isinstance(cutoff, CompactSupportCutoff):
                continue
            if type(cutoff).__name__ not in ("PolynomialCutoff", "XPLORCutoff"):
                raise FusionSetupError("Unsupported SevenNet cutoff for Opt4 sink isolation")
            radius = getattr(cutoff, "cutoff_length", getattr(cutoff, "r_cut", None))
            module.cutoff_function = CompactSupportCutoff(cutoff, radius)
            bounded.append(path)
    originals = {path: module for path, module in model.named_modules()
                 if isinstance(module, torch.fx.GraphModule) and "_compiled_main" in path}
    install_tp_regions(model, passes, report,
        lambda path: "convolution" in path or "self_connection" in path or "gate" in path,
        backward_policy="aten")
    for path, reference in originals.items():
        candidate = model.get_submodule(path)
        if candidate is not reference:
            parent, _, leaf = path.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, leaf, NativeTPBackward(reference, candidate))
    for entry in report["passes"].values():
        entry.update(backward_policy="unpartitioned-native-TP-vjp", backward_recomputes_reference=True,
                     fusion_scope="forward-only", performance_gate="must remeasure recomputation cost")
        entry["sink_cutoff"] = {"modules": bounded, "policy": "bounded-evaluation-then-zero-outside-cutoff",
                                "counted_as_fusion": False}
