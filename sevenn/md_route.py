"""Stable SevenNet MD route for the shared acceleration benchmark.

``stage=baseline, backend=eager`` is the permanent scientific-correctness
reference: the regular SevenNet/e3nn ASE calculator, its normal CPU matscipy
neighbor list, float32 model arithmetic, and no tensor-product accelerator.
The ``cueq``, ``flash``, and ``oeq`` backends are retained only as explicit
comparators for accelerators that already exist upstream.

All non-baseline stages use the permanent dynamic dispatch convention.  Thus
``stage=opt3`` resolves to ``sevenn.md_stages.opt3.run_md`` and its
``whole-step-cuda-graph`` backend without changing baseline/Opt1/Opt2 behavior.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    run_ase_baseline,
    run_optimized_stage,
)

_ACCELERATOR_ENV = (
    'SEVENNET_ENABLE_CUEQ',
    'SEVENNET_ENABLE_FLASH',
    'SEVENNET_ENABLE_OEQ',
)


@contextlib.contextmanager
def _isolated_accelerator_environment() -> Iterator[None]:
    """Prevent a shell-level accelerator flag from changing route semantics."""
    saved = {name: os.environ.get(name) for name in _ACCELERATOR_ENV}
    try:
        for name in _ACCELERATOR_ENV:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != 'sevennet':
        raise ValueError(f'sevenn.md_route does not own model {request.model!r}')
    if request.stage != 'baseline':
        return run_optimized_stage(request, module_prefix='sevenn.md_stages')

    from sevenn.calculator import SevenNetCalculator

    supported = {'eager', 'cueq', 'flash', 'oeq'}
    if request.backend not in supported:
        raise ValueError(f'SevenNet backend must be one of {sorted(supported)}')
    # This controls matmul precision only.  SevenNet itself converts floating
    # model inputs to float32.  ASE's MD state remains float64, matching the
    # upstream Matbench setup even when its CLI says ``--dtype float64`` (that
    # option is not forwarded to SevenNetCalculator upstream either).
    configure_torch_baseline()
    with _isolated_accelerator_environment():
        calculator = SevenNetCalculator(
            model=request.model_path,
            device=request.config.device,
            modal=request.options.get('modal'),
            enable_cueq=request.backend == 'cueq',
            enable_flash=request.backend == 'flash',
            enable_oeq=request.backend == 'oeq',
        )
    return run_ase_baseline(
        request,
        calculator,
        metadata={
            'calculator': 'SevenNetCalculator',
            'model_arithmetic': 'float32',
            'tf32': False,
            'tp_backend': request.backend,
            'neighbor_list': 'matscipy_cpu_or_ase_fallback',
            'gpu_resident': False,
            'modal': request.options.get('modal'),
        },
    )
