"""SevenNet Opt4 route with the existing cuEquivariance fused scatter path."""

from __future__ import annotations

from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.opt4_policy import run_opt4_with_opt3

from . import opt3


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "sevennet" or request.stage != "opt4":
        raise ValueError(
            f"SevenNet Opt4 route received {request.model}/{request.stage}"
        )
    result, policy = run_opt4_with_opt3(request, opt3.run_md, model="sevennet")
    result.stage = "opt4"
    result.metadata.update(policy.metadata)
    result.metadata.update(
        {
            "opt4_model_strategy": "cuequivariance-tensor-product-scatter",
            "opt4_fused_components": [
                "tensor-product",
                "edge-message",
                "destination-scatter",
            ],
            "opt4_fixed_address_buffers": True,
            "opt4_custom_kernel": False,
            "opt4_reverse_edge_verified": False,
        }
    )
    validate_result(request, result)
    return result


__all__ = ["run_md"]
