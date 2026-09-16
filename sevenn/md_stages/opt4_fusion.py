"""FastEq-inspired full SevenNet convolution boundary for Opt4.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


class CompactSupportCutoff(nn.Module):
    """Keep sink edges exactly outside the released cutoff polynomial."""

    def __init__(self, original, cutoff):
        super().__init__()
        self.original = original
        self.cutoff = float(cutoff)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise FusionSetupError("SevenNet FastEq adapter requires a positive cutoff")

    def forward(self, radius):
        value = self.original(radius.clamp(max=self.cutoff))
        return torch.where(radius < self.cutoff, value, 0.0)


class _Uniform1DConvolution(nn.Module):
    """Gather + channelwise TP + fixed destination reduction + epilogue."""

    def __init__(self, convolution, edge_rows, rows):
        super().__init__()
        object.__setattr__(self, "_convolution", convolution)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def set_layout(self, edge_rows, rows):
        self.edge_rows = edge_rows
        self.rows = int(rows)

    def forward(self, x, edge_filter, weight, edge_src, denominator):
        message = self._convolution(x.index_select(0, edge_src), edge_filter, weight)
        out = message.new_zeros((self.rows, message.shape[-1]))
        out.index_add_(0, self.edge_rows, message)
        return out.div(denominator)


def _layout(options, parameter):
    row_ptr, edge_rows, _max_row = fixed_csr_layout(
        options,
        parameter,
        extra_rows=int(options.get("cuda_graph_dummy_atoms", 32)),
    )
    return edge_rows, int(row_ptr.shape[0] - 1)


def refresh(model, options) -> None:
    edge_rows, rows = _layout(options, next(model.parameters()))
    for module in model.modules():
        region = getattr(module, "_opt4_fasteq_uniform1d", None)
        if isinstance(region, CheckedRegion):
            module._opt4_edge_capacity = int(edge_rows.numel())
            region.reference.set_layout(edge_rows, rows)
            region.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_uniform1d_conv" not in passes:
        return
    bounded = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "EdgeEmbedding":
            continue
        cutoff = module.cutoff_function
        if isinstance(cutoff, CompactSupportCutoff):
            continue
        if type(cutoff).__name__ not in ("PolynomialCutoff", "XPLORCutoff"):
            raise FusionSetupError("unsupported SevenNet cutoff for sink isolation")
        radius = getattr(cutoff, "cutoff_length", getattr(cutoff, "r_cut", None))
        module.cutoff_function = CompactSupportCutoff(cutoff, radius)
        bounded.append(path)

    edge_rows, rows = _layout(options, next(model.parameters()))
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "IrrepsConvolution":
            continue
        if module.convolution is None:
            raise FusionSetupError("SevenNet convolution must be instantiated before Opt4")
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        boundary = _Uniform1DConvolution(module.convolution, edge_rows, rows)
        module._opt4_fasteq_uniform1d = CheckedRegion(boundary, detail)
        module._opt4_edge_capacity = int(edge_rows.numel())
        modules.append(detail)
    record(
        report,
        "fasteq_uniform1d_conv",
        len(modules),
        "inductor-triton-full-boundary-aot-vjp",
        modules=modules,
        fused_boundaries=[
            "source-gather",
            "tp-instruction-chain",
            "destination-reduce",
            "denominator",
        ],
        gemm="original-e3nn",
        backward="aot-compiled-complete-input-vjp",
        replay_runtime_compile=False,
        sink_cutoff={"modules": bounded, "counted_as_fusion": False},
    )
