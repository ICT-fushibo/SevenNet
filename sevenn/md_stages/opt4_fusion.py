"""SevenNet Opt4: fixed-slot convolution destination reduction."""
from __future__ import annotations

import math
import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_ops import csr_segment_sum
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


class CompactSupportCutoff(nn.Module):
    """Keep far sink edges outside the released cutoff polynomial."""

    def __init__(self, original, cutoff):
        super().__init__()
        self.original = original
        self.cutoff = float(cutoff)
        if not math.isfinite(self.cutoff) or self.cutoff <= 0:
            raise FusionSetupError("SevenNet Opt4 requires a finite positive cutoff")

    def forward(self, radius):
        value = self.original(radius.clamp(max=self.cutoff))
        return torch.where(radius < self.cutoff, value, 0.0)


class _NativeScatter(nn.Module):
    def __init__(self, edge_rows, rows):
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def forward(self, message):
        out = message.new_zeros((self.rows, *message.shape[1:]))
        out.index_add_(0, self.edge_rows, message)
        return out


class _FixedCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row):
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row):
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.max_row = int(max_row)

    def forward(self, message):
        return csr_segment_sum(
            message.contiguous(), self.row_ptr, self.edge_rows, self.max_row
        )


def _layout(options, parameter):
    return fixed_csr_layout(
        options,
        parameter,
        extra_rows=int(options.get("cuda_graph_dummy_atoms", 32)),
    )


def refresh(model, options):
    row_ptr, edge_rows, max_row = _layout(options, next(model.parameters()))
    for module in model.modules():
        region = getattr(module, "_opt4_conv_csr", None)
        if isinstance(region, CheckedRegion):
            region.reference.edge_rows = edge_rows
            region.reference.rows = row_ptr.shape[0] - 1
            region.compiled.set_layout(row_ptr, edge_rows, max_row)
            region.signatures.clear()


def install(model, passes, report, options):
    if "conv_tp_reduce_vjp" not in passes:
        return
    bounded = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ == "EdgeEmbedding":
            cutoff = module.cutoff_function
            if isinstance(cutoff, CompactSupportCutoff):
                continue
            if type(cutoff).__name__ not in ("PolynomialCutoff", "XPLORCutoff"):
                raise FusionSetupError("Unsupported SevenNet cutoff for sink isolation")
            radius = getattr(cutoff, "cutoff_length", getattr(cutoff, "r_cut", None))
            module.cutoff_function = CompactSupportCutoff(cutoff, radius)
            bounded.append(path)

    row_ptr, edge_rows, max_row = _layout(options, next(model.parameters()))
    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "IrrepsConvolution":
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        module._opt4_conv_csr = CheckedRegion(
            _NativeScatter(edge_rows, row_ptr.shape[0] - 1),
            detail,
            _FixedCSR(row_ptr, edge_rows, max_row),
        )
        modules.append(detail)
    record(
        report,
        "conv_tp_reduce_vjp",
        len(modules),
        "triton-fixed-csr-explicit-vjp",
        modules=modules,
        tensor_product="native-e3nn-gemm-and-instructions-unchanged",
        fused_boundaries=["convolution-destination-reduce"],
        backward_recomputes_reference=False,
        fusion_scope="forward-and-backward",
        sink_cutoff={"modules": bounded, "counted_as_fusion": False},
    )
