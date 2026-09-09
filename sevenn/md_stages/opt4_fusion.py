"""SevenNet-owned e3nn convolution/gate region selection."""
import math
import torch
from torch import nn
from md_benchmark.opt4_fx import install_tp_regions
from md_benchmark.opt4_registry import FusionSetupError


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
    install_tp_regions(model, passes, report,
        lambda path: "convolution" in path or "self_connection" in path or "gate" in path,
        backward_policy="aten")
    for entry in report["passes"].values():
        entry["sink_cutoff"] = {"modules": bounded, "policy": "bounded-evaluation-then-zero-outside-cutoff",
                                "counted_as_fusion": False}
