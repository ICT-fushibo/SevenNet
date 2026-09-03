"""SevenNet Opt2: model-only CUDA Graph in a GPU-resident MD loop.

The CUDA neighbor list, fixed-capacity input preparation, thermostat, and MD
integration remain eager and outside the graph.  Only the regular SevenNet/e3nn
energy and conservative-force path is captured.  Ragged neighbor graphs are
copied into fixed-address buffers; padding edges are isolated on one dummy atom
and the total-energy reduction explicitly excludes that atom.

This stage intentionally does not enable tensor-product accelerators,
``torch.compile``, TF32, AMP, whole-step capture, or model-specific fusion.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from ase import units
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)
from torch import nn

import sevenn._keys as key
from sevenn.atom_graph_data import AtomGraphData
from sevenn.md_stages.opt1 import (
    _berendsen_step,
    _configure_output,
    _frame,
    _initial_momenta,
    _ModelOutput,
    _nhc_step,
    _NoseHooverChain,
    _SingleSystemPotential,
)


class CUDAGraphCapacityError(RuntimeError):
    """Raised instead of falling back when a neighbor graph exceeds capacity."""

    def __init__(self, required_edges: int, edge_capacity: int) -> None:
        self.required_edges = int(required_edges)
        self.edge_capacity = int(edge_capacity)
        super().__init__(
            f'SevenNet CUDA Graph edge capacity exceeded: '
            f'required={required_edges}, capacity={edge_capacity}'
        )


class CUDAGraphValidationError(RuntimeError):
    """Raised when fixed-buffer replay does not match eager SevenNet."""


def edge_capacity_from_probe(
    maximum_edges: int,
    *,
    margin: float = 0.25,
    edge_step: int = 128,
) -> int:
    """Round a probed edge count to a conservative fixed capacity."""

    if maximum_edges < 1:
        raise ValueError('maximum_edges must be positive')
    if margin < 0:
        raise ValueError('margin must be non-negative')
    if edge_step < 1:
        raise ValueError('edge_step must be positive')
    required = max(maximum_edges + 1, math.ceil(maximum_edges * (1 + margin)))
    return int(math.ceil(required / edge_step) * edge_step)


def _maximum_neighbors_per_atom(
    edge_index: torch.Tensor,
    *,
    num_atoms: int,
) -> int:
    """Return the largest SevenNet centre degree during a setup probe."""
    if num_atoms < 1:
        raise ValueError('num_atoms must be positive')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, num_edges]')
    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[0], minlength=num_atoms)[:num_atoms]
    return int(counts.max().item())


@torch.no_grad()
def staticize_edges_(
    static_edge_index: torch.Tensor,
    static_edge_vec: torch.Tensor,
    static_cell_shifts: torch.Tensor,
    real_edge_index: torch.Tensor,
    real_edge_vec: torch.Tensor,
    real_cell_shifts: torch.Tensor,
    *,
    dummy_index: int,
    padding_edge_vec: torch.Tensor,
) -> int:
    """Copy one ragged graph into fixed-address model input buffers."""

    if static_edge_index.ndim != 2 or static_edge_index.shape[0] != 2:
        raise ValueError('static_edge_index must have shape [2, capacity]')
    capacity = int(static_edge_index.shape[1])
    num_edges = int(real_edge_index.shape[1])
    if real_edge_index.shape != (2, num_edges):
        raise ValueError('real_edge_index must have shape [2, num_edges]')
    if static_edge_vec.shape != (capacity, 3):
        raise ValueError('static_edge_vec must have shape [capacity, 3]')
    if static_cell_shifts.shape != (capacity, 3):
        raise ValueError('static_cell_shifts must have shape [capacity, 3]')
    if real_edge_vec.shape != (num_edges, 3):
        raise ValueError('real_edge_vec must have shape [num_edges, 3]')
    if real_cell_shifts.shape != (num_edges, 3):
        raise ValueError('real_cell_shifts must have shape [num_edges, 3]')
    if padding_edge_vec.shape != (3,):
        raise ValueError('padding_edge_vec must have shape [3]')
    if num_edges > capacity:
        raise CUDAGraphCapacityError(num_edges, capacity)

    if num_edges:
        static_edge_index[:, :num_edges].copy_(real_edge_index)
        static_edge_vec[:num_edges].copy_(real_edge_vec)
        static_cell_shifts[:num_edges].copy_(real_cell_shifts)
    if num_edges < capacity:
        static_edge_index[:, num_edges:].fill_(dummy_index)
        static_edge_vec[num_edges:].copy_(padding_edge_vec)
        static_cell_shifts[num_edges:].zero_()
    return num_edges


class _RealAtomReduce(nn.Module):
    """Sum atomic energies over real atoms while excluding the dummy sink."""

    def __init__(
        self,
        n_real: int,
        *,
        data_key_in: str,
        data_key_out: str,
        constant: float,
    ) -> None:
        super().__init__()
        self.n_real = int(n_real)
        self.key_input = data_key_in
        self.key_output = data_key_out
        self.constant = float(constant)
        self._is_batch_data = False

    def forward(self, data):
        data[self.key_output] = (
            torch.sum(data[self.key_input][: self.n_real]) * self.constant
        )
        return data


class _ModelOnlyCUDAGraphPotential(_SingleSystemPotential):
    """Fixed-capacity SevenNet evaluator with eager neighbor construction."""

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        modal: str | None,
        profiler: CudaPhaseProfiler,
        requested_edge_capacity: int | None,
        edge_margin: float,
        edge_step: int,
        track_neighbor_capacity: bool,
        capture_warmup: int,
        energy_atol: float,
        force_atol: float,
    ) -> None:
        super().__init__(
            model_path,
            device=device,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            modal=modal,
            compute_stress=False,
            profiler=profiler,
        )
        if requested_edge_capacity is not None and requested_edge_capacity < 1:
            raise ValueError('cuda_graph_edge_capacity must be positive')
        if capture_warmup < 0:
            raise ValueError('cuda_graph_capture_warmup cannot be negative')
        if energy_atol < 0 or force_atol < 0:
            raise ValueError('CUDA Graph validation tolerances cannot be negative')

        self.n_real = int(atomic_numbers.shape[0])
        self.dummy_index = self.n_real
        self.requested_edge_capacity = requested_edge_capacity
        self.edge_margin = float(edge_margin)
        self.edge_step = int(edge_step)
        self.track_neighbor_capacity = bool(track_neighbor_capacity)
        self.capture_warmup = int(capture_warmup)
        self.energy_atol = float(energy_atol)
        self.force_atol = float(force_atol)

        reducer = self.model._modules.get('reduce_total_enegy')
        if reducer is None or not all(
            hasattr(reducer, name)
            for name in ('key_input', 'key_output', 'constant')
        ):
            raise RuntimeError(
                'SevenNet Opt2 requires the released reduce_total_enegy module'
            )
        self.model.replace_module(
            'reduce_total_enegy',
            _RealAtomReduce(
                self.n_real,
                data_key_in=reducer.key_input,
                data_key_out=reducer.key_output,
                constant=float(reducer.constant),
            ),
        )

        self.edge_capacity = 0
        self.static_edge_index: torch.Tensor | None = None
        self.static_edge_vec: torch.Tensor | None = None
        self.static_cell_shifts: torch.Tensor | None = None
        self.padding_edge_vec: torch.Tensor | None = None
        self.static_type_indices: torch.Tensor | None = None
        self.static_atomic_numbers: torch.Tensor | None = None
        self.static_model_positions: torch.Tensor | None = None
        self.static_graph: AtomGraphData | None = None
        self.cuda_graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.static_energy: torch.Tensor | None = None
        self.static_forces: torch.Tensor | None = None
        self.captured = False

        self.capture_count = 0
        self.capture_wall_time_s = 0.0
        self.total_replays = 0
        self.production_calls = 0
        self.production_replays = 0
        self.capacity_misses = 0
        self.min_real_edges: int | None = None
        self.max_real_edges: int | None = None
        self.initial_max_neighbors_per_atom: int | None = None
        self.max_neighbors_per_atom: int | None = None
        self.initial_neighbors_by_atom: list[int] | None = None
        self.max_neighbors_by_atom: list[int] | None = None
        self.output_addresses_stable = False
        self.input_addresses_stable = False
        self._capture_input_addresses: tuple[int, ...] | None = None
        self.replay_stability_passed = False
        self.replay_energy_abs_error = 0.0
        self.replay_force_max_abs_error = 0.0
        self.validation_energy_abs_error = 0.0
        self.validation_force_max_abs_error = 0.0

    def _build_real_inputs(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.profiler.phase('neighbor_list'):
            edge_index, _mapping_system, unit_shifts = self.neighbor_list_fn(
                positions,
                self.cell_batch,
                self.pbc_batch,
                self.cutoff,
                self.system_idx,
            )
        with self.profiler.phase('model_input'):
            model_positions = positions.to(torch.float32)
            model_shifts = unit_shifts.to(torch.float32)
            shifts = torch.mm(model_shifts, self.model_cell)
            edge_vec = (
                model_positions[edge_index[1]]
                - model_positions[edge_index[0]]
                + shifts
            )
        return edge_index, edge_vec, model_shifts

    def _initialize_static_graph(self, edge_capacity: int) -> None:
        self.edge_capacity = int(edge_capacity)
        self.static_edge_index = torch.empty(
            (2, edge_capacity), dtype=torch.long, device=self.device
        )
        edge_vec = torch.empty(
            (edge_capacity, 3), dtype=torch.float32, device=self.device
        )
        edge_vec.requires_grad_(True)
        self.static_edge_vec = edge_vec
        self.static_cell_shifts = torch.zeros(
            (edge_capacity, 3), dtype=torch.float32, device=self.device
        )
        self.padding_edge_vec = torch.tensor(
            [self.cutoff, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        self.static_type_indices = torch.cat(
            (self.type_indices, self.type_indices[:1]), dim=0
        )
        self.static_atomic_numbers = torch.cat(
            (self.atomic_numbers, self.atomic_numbers[:1]), dim=0
        )
        self.static_model_positions = torch.zeros(
            (self.n_real + 1, 3), dtype=torch.float32, device=self.device
        )
        self.static_graph = AtomGraphData(
            x=self.static_type_indices,
            edge_index=self.static_edge_index,
            pos=self.static_model_positions,
            **{
                key.ATOMIC_NUMBERS: self.static_atomic_numbers,
                key.EDGE_VEC: self.static_edge_vec,
                key.CELL: self.model_cell,
                key.CELL_SHIFT: self.static_cell_shifts,
                key.CELL_VOLUME: self.cell_volume,
                key.NUM_ATOMS: torch.tensor(
                    self.n_real + 1, dtype=torch.long, device=self.device
                ),
                key.DATA_MODALITY: self.modal,
                key.INFO: {},
            },
        )

    def _staticize(
        self,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        cell_shifts: torch.Tensor,
    ) -> int:
        assert self.static_edge_index is not None
        assert self.static_edge_vec is not None
        assert self.static_cell_shifts is not None
        assert self.padding_edge_vec is not None
        return staticize_edges_(
            self.static_edge_index,
            self.static_edge_vec,
            self.static_cell_shifts,
            edge_index,
            edge_vec,
            cell_shifts,
            dummy_index=self.dummy_index,
            padding_edge_vec=self.padding_edge_vec,
        )

    def _static_forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.static_graph is not None
        assert self.static_type_indices is not None
        assert self.static_edge_index is not None
        assert self.static_edge_vec is not None
        assert self.static_model_positions is not None
        # Warmup forwards mutate the graph dictionary.  Rebind every external
        # input so capture records kernels reading only the static allocations.
        self.static_graph[key.NODE_FEATURE] = self.static_type_indices
        self.static_graph[key.EDGE_IDX] = self.static_edge_index
        self.static_graph[key.EDGE_VEC] = self.static_edge_vec
        self.static_graph[key.POS] = self.static_model_positions
        with torch.enable_grad():
            output = self.model(self.static_graph)
        return (
            output[key.PRED_FORCE][: self.n_real].detach(),
            output[key.PRED_TOTAL_ENERGY].sum().detach(),
        )

    def _input_addresses(self) -> tuple[int, ...]:
        assert self.static_edge_index is not None
        assert self.static_edge_vec is not None
        assert self.static_cell_shifts is not None
        assert self.static_type_indices is not None
        assert self.static_model_positions is not None
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.static_edge_index,
                self.static_edge_vec,
                self.static_cell_shifts,
                self.static_type_indices,
                self.static_model_positions,
            )
        )

    def capture(self, positions: torch.Tensor) -> None:
        """Probe, capture once, then validate against the eager Opt1 path."""

        if self.captured:
            raise RuntimeError('SevenNet CUDA Graph has already been captured')
        # This eager reference uses the original ragged graph before the fixed
        # graph is installed.  It is setup-only and excluded from MD timing.
        eager_reference = _SingleSystemPotential.__call__(self, positions)
        edge_index, edge_vec, cell_shifts = self._build_real_inputs(positions)
        probed_edges = int(edge_index.shape[1])
        if self.track_neighbor_capacity:
            self.initial_max_neighbors_per_atom = _maximum_neighbors_per_atom(
                edge_index,
                num_atoms=self.n_real,
            )
            self.max_neighbors_per_atom = self.initial_max_neighbors_per_atom
            counts = torch.bincount(edge_index[0], minlength=self.n_real)[: self.n_real]
            self.initial_neighbors_by_atom = counts.detach().cpu().tolist()
            self.max_neighbors_by_atom = list(self.initial_neighbors_by_atom)
        capacity = self.requested_edge_capacity or edge_capacity_from_probe(
            probed_edges,
            margin=self.edge_margin,
            edge_step=self.edge_step,
        )
        if capacity < probed_edges:
            raise CUDAGraphCapacityError(probed_edges, capacity)
        self._initialize_static_graph(capacity)
        self._staticize(edge_index, edge_vec, cell_shifts)

        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = side_stream
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            assert self.static_edge_vec is not None
            capture_edge_vec = self.static_edge_vec.detach().clone()
            capture_edge_vec.requires_grad_(True)
            self.static_edge_vec = capture_edge_vec
            assert self.static_graph is not None
            self.static_graph[key.EDGE_VEC] = capture_edge_vec
            self._capture_input_addresses = self._input_addresses()
            for _ in range(self.capture_warmup):
                self._static_forward()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        capture_started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            static_forces, static_energy = self._static_forward()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - capture_started
        self.cuda_graph = graph
        self.static_forces = static_forces
        self.static_energy = static_energy
        self.capture_count = 1
        self.captured = True

        force_address = static_forces.data_ptr()
        energy_address = static_energy.data_ptr()
        graph.replay()
        replay_forces = static_forces.clone()
        replay_energy = static_energy.clone()
        graph.replay()
        second_forces = static_forces.clone()
        second_energy = static_energy.clone()
        torch.cuda.synchronize(self.device)
        self.total_replays += 2
        self.output_addresses_stable = (
            static_forces.data_ptr() == force_address
            and static_energy.data_ptr() == energy_address
        )
        self.input_addresses_stable = (
            self._input_addresses() == self._capture_input_addresses
        )
        self.replay_energy_abs_error = float(
            (replay_energy - second_energy).abs().max().item()
        )
        self.replay_force_max_abs_error = float(
            (replay_forces - second_forces).abs().max().item()
        )
        self.replay_stability_passed = (
            self.replay_energy_abs_error <= self.energy_atol
            and self.replay_force_max_abs_error <= self.force_atol
        )
        self.validation_energy_abs_error = float(
            (replay_energy.to(torch.float64) - eager_reference.energy)
            .abs()
            .item()
        )
        self.validation_force_max_abs_error = float(
            (
                replay_forces.to(torch.float64) - eager_reference.forces
            ).abs().max().item()
        )
        if not self.output_addresses_stable:
            raise CUDAGraphValidationError(
                'SevenNet CUDA Graph output addresses changed between replays'
            )
        if not self.input_addresses_stable:
            raise CUDAGraphValidationError(
                'SevenNet CUDA Graph input addresses changed between replays'
            )
        for name, value in (
            ('replay energy', replay_energy),
            ('replay forces', replay_forces),
            ('second replay energy', second_energy),
            ('second replay forces', second_forces),
        ):
            if not bool(torch.isfinite(value).all()):
                raise CUDAGraphValidationError(
                    f'SevenNet CUDA Graph produced non-finite {name}'
                )
        self.numerical_validation_within_tolerance = (
            self.replay_stability_passed
            and self.validation_energy_abs_error <= self.energy_atol
            and self.validation_force_max_abs_error <= self.force_atol
        )
    def reset_production_stats(self) -> None:
        self.production_calls = 0
        self.production_replays = 0
        self.capacity_misses = 0
        self.min_real_edges = None
        self.max_real_edges = None
        self.max_neighbors_per_atom = self.initial_max_neighbors_per_atom
        self.max_neighbors_by_atom = (
            None
            if self.initial_neighbors_by_atom is None
            else list(self.initial_neighbors_by_atom)
        )

    def __call__(self, positions: torch.Tensor) -> _ModelOutput:
        if not self.captured or self.cuda_graph is None:
            raise RuntimeError('SevenNet CUDA Graph must be captured before replay')
        edge_index, edge_vec, cell_shifts = self._build_real_inputs(positions)
        num_edges = int(edge_index.shape[1])
        if self.track_neighbor_capacity:
            maximum = _maximum_neighbors_per_atom(
                edge_index,
                num_atoms=self.n_real,
            )
            self.max_neighbors_per_atom = (
                maximum
                if self.max_neighbors_per_atom is None
                else max(self.max_neighbors_per_atom, maximum)
            )
            counts = torch.bincount(edge_index[0], minlength=self.n_real)[: self.n_real]
            values = counts.detach().cpu().tolist()
            self.max_neighbors_by_atom = [
                max(old, new)
                for old, new in zip(self.max_neighbors_by_atom or values, values)
            ]
        self.production_calls += 1
        if num_edges > self.edge_capacity:
            self.capacity_misses += 1
            raise CUDAGraphCapacityError(num_edges, self.edge_capacity)
        self._staticize(edge_index, edge_vec, cell_shifts)
        if self._input_addresses() != self._capture_input_addresses:
            raise CUDAGraphValidationError(
                'SevenNet CUDA Graph static input address changed'
            )
        with self.profiler.phase('model_energy_force'):
            self.cuda_graph.replay()
        self.total_replays += 1
        self.production_replays += 1
        self.min_real_edges = (
            num_edges
            if self.min_real_edges is None
            else min(self.min_real_edges, num_edges)
        )
        self.max_real_edges = (
            num_edges
            if self.max_real_edges is None
            else max(self.max_real_edges, num_edges)
        )
        assert self.static_forces is not None
        assert self.static_energy is not None
        return _ModelOutput(
            energy=self.static_energy.to(torch.float64),
            forces=self.static_forces.to(torch.float64),
            stress=None,
        )

    def stats(self) -> dict[str, int | float | bool | None]:
        hit_rate = (
            self.production_replays / self.production_calls
            if self.production_calls
            else 0.0
        )
        return {
            'cuda_graph_capture_count': self.capture_count,
            'cuda_graph_production_capture_count': 0,
            'cuda_graph_total_replays': self.total_replays,
            'cuda_graph_production_calls': self.production_calls,
            'cuda_graph_production_replays': self.production_replays,
            'cuda_graph_capacity_misses': self.capacity_misses,
            'cuda_graph_hit_rate': hit_rate,
            'cuda_graph_edge_capacity': self.edge_capacity,
            'cuda_graph_min_real_edges': self.min_real_edges,
            'cuda_graph_max_real_edges': self.max_real_edges,
            'cuda_graph_max_neighbors_per_atom': self.max_neighbors_per_atom,
            'cuda_graph_maximum_neighbors_by_atom': self.max_neighbors_by_atom,
            'capacity_probe_collect_per_atom': self.track_neighbor_capacity,
            'cuda_graph_dummy_atoms': 1,
            'cuda_graph_capture_warmup': self.capture_warmup,
            'cuda_graph_capture_wall_time_s': self.capture_wall_time_s,
            'cuda_graph_replay_output_addresses_stable': (
                self.output_addresses_stable
            ),
            'cuda_graph_input_addresses_stable': self.input_addresses_stable,
            'cuda_graph_replay_stability_pass': self.replay_stability_passed,
            'cuda_graph_replay_stability_energy_abs_error_eV': (
                self.replay_energy_abs_error
            ),
            'cuda_graph_replay_stability_force_max_abs_error_eV_per_A': (
                self.replay_force_max_abs_error
            ),
            'cuda_graph_validation_energy_abs_error_eV': (
                self.validation_energy_abs_error
            ),
            'cuda_graph_validation_force_max_abs_error_eV_per_A': (
                self.validation_force_max_abs_error
            ),
            'cuda_graph_validation_energy_atol_eV': self.energy_atol,
            'cuda_graph_validation_force_atol_eV_per_A': self.force_atol,
            'cuda_graph_numerical_validation_failure_policy': 'report_only',
            'cuda_graph_numerical_validation_within_tolerance': (
                self.numerical_validation_within_tolerance
            ),
        }


def run_md(request):
    """Run SevenNet Opt2 under the permanent shared MD contract."""

    from md_benchmark.md_route import MDObservation, MDRunResult, validate_result

    if request.model != 'sevennet' or request.stage != 'opt2':
        raise ValueError(
            f'sevenn.md_stages.opt2 owns sevennet/opt2, got '
            f'{request.model}/{request.stage}'
        )
    if request.backend not in {'model-only-cuda-graph', 'gpu-resident'}:
        raise ValueError(
            "SevenNet opt2 backend must be 'model-only-cuda-graph'"
        )
    if request.config.dtype != 'float64':
        raise ValueError('SevenNet opt2 requires --dtype float64 for the MD state')
    if request.config.device.split(':', maxsplit=1)[0] != 'cuda':
        raise ValueError('SevenNet opt2 is CUDA-only; CPU fallback is forbidden')
    if not torch.cuda.is_available():
        raise RuntimeError('SevenNet opt2 requested CUDA, but CUDA is unavailable')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise RuntimeError(
            'SevenNet opt2 forbids TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1'
        )
    if request.atoms.constraints:
        raise NotImplementedError('SevenNet opt2 does not support ASE constraints')
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError(
            'SevenNet opt2 currently supports force-only MD; trajectory output '
            'requires stress and is deliberately not captured'
        )
    if request.options.get('compute_stress', False):
        raise NotImplementedError(
            'SevenNet opt2 model-only CUDA Graph does not capture stress'
        )

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(request.config.device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    config = request.config
    atoms = request.atoms.copy()
    _configure_output(request)

    positions0 = torch.tensor(
        np.asarray(atoms.positions), device=device, dtype=torch.float64
    )
    momenta0 = torch.tensor(
        _initial_momenta(atoms, config.temperature_k, config.seed),
        device=device,
        dtype=torch.float64,
    )
    masses = torch.tensor(
        atoms.get_masses(), device=device, dtype=torch.float64
    ).unsqueeze(-1)
    cell = torch.tensor(
        np.asarray(atoms.cell), device=device, dtype=torch.float64
    )
    pbc = torch.tensor(np.asarray(atoms.pbc), device=device, dtype=torch.bool)
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(), device=device, dtype=torch.long
    )
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options),
        device=device,
    )
    requested_capacity = request.options.get('cuda_graph_edge_capacity')
    potential = _ModelOnlyCUDAGraphPotential(
        request.model_path,
        device=device,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        modal=request.options.get('modal'),
        profiler=profiler,
        requested_edge_capacity=(
            int(requested_capacity) if requested_capacity is not None else None
        ),
        edge_margin=float(request.options.get('cuda_graph_edge_margin', 0.25)),
        edge_step=int(request.options.get('cuda_graph_edge_step', 128)),
        track_neighbor_capacity=bool(
            request.options.get('capacity_probe_collect_per_atom', False)
        ),
        capture_warmup=int(
            request.options.get('cuda_graph_capture_warmup', 3)
        ),
        energy_atol=float(
            request.options.get('cuda_graph_energy_atol_ev', 2e-4)
        ),
        force_atol=float(
            request.options.get('cuda_graph_force_atol_ev_per_a', 2e-4)
        ),
    )
    potential.capture(positions0)

    dt = config.timestep_fs * units.fs
    tau = config.thermostat_time_fs * units.fs

    def make_thermostat() -> _NoseHooverChain | None:
        if config.integrator == 'berendsen':
            return None
        thermostat = _NoseHooverChain(
            n_atoms=len(atoms),
            temperature_k=config.temperature_k,
            damping=tau,
            device=device,
            dtype=torch.float64,
        )
        thermostat.set_masses(masses)
        return thermostat

    def advance(positions, momenta, output, thermostat):
        if config.integrator == 'berendsen':
            return _berendsen_step(
                positions,
                momenta,
                output.forces,
                masses,
                dt=dt,
                target_temperature_k=config.temperature_k,
                tau=tau,
                force_fn=potential,
            )
        assert thermostat is not None
        return _nhc_step(
            positions,
            momenta,
            output.forces,
            masses,
            dt=dt,
            thermostat=thermostat,
            force_fn=potential,
        )

    if config.warmup_steps:
        warm_positions = positions0.clone()
        warm_momenta = momenta0.clone()
        warm_output = potential(warm_positions)
        warm_thermostat = make_thermostat()
        for _ in range(config.warmup_steps):
            warm_positions, warm_momenta, warm_output = advance(
                warm_positions, warm_momenta, warm_output, warm_thermostat
            )
        torch.cuda.synchronize(device)

    positions = positions0.clone()
    momenta = momenta0.clone()
    thermostat = make_thermostat()
    observations = []
    observation_steps = set(config.observation_steps)
    potential.reset_production_stats()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase('initial_force'):
        output = potential(positions)
    if config.collect_statistics and 0 in observation_steps:
        observations.append(
            MDObservation(
                step=0,
                potential_energy_ev=float(output.energy.cpu()),
                kinetic_energy_ev=float(
                    torch.sum(momenta.square() / (2 * masses)).cpu()
                ),
                forces_ev_per_a=output.forces.cpu().numpy().copy(),
                positions_a=positions.cpu().numpy().copy(),
            )
        )
    for step in range(1, config.steps + 1):
        with profiler.phase('md_step'):
            positions, momenta, output = advance(
                positions, momenta, output, thermostat
            )
        if config.collect_statistics and step in observation_steps:
            observations.append(
                MDObservation(
                    step=step,
                    potential_energy_ev=float(output.energy.cpu()),
                    kinetic_energy_ev=float(
                        torch.sum(momenta.square() / (2 * masses)).cpu()
                    ),
                    forces_ev_per_a=output.forces.cpu().numpy().copy(),
                    positions_a=positions.cpu().numpy().copy(),
                )
            )
    torch.cuda.synchronize(device)
    profiler.stop()
    elapsed = time.perf_counter() - started
    performance_profile = profiler.summary(synchronize=False)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9

    final_atoms = _frame(
        atoms,
        positions=positions,
        momenta=momenta,
        output=output,
        step=config.steps,
    )
    graph_stats = potential.stats()
    expected_replays = config.steps + 1
    if graph_stats['cuda_graph_production_replays'] != expected_replays:
        raise RuntimeError(
            'SevenNet Opt2 production replay count mismatch: '
            f"expected={expected_replays}, "
            f"actual={graph_stats['cuda_graph_production_replays']}"
        )
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory,
        final_atoms=final_atoms,
        observations=observations,
        metadata={
            'engine': 'sevennet_gpu_resident_model_cuda_graph',
            'backend': 'model-only-cuda-graph',
            'requested_backend': request.backend,
            'model_path': str(Path(request.model_path).expanduser().resolve()),
            'gpu_resident': True,
            'md_state_dtype': 'float64',
            'model_input_dtype': 'float32',
            'checkpoint_parameter_dtypes': potential.parameter_dtypes,
            'neighbor_list': 'torch_sim.neighbors.torchsim_nl_cuda',
            'integrator': config.integrator,
            'warmup_steps': config.warmup_steps,
            'tf32': False,
            'amp': False,
            'torch_compile': False,
            'cuda_graph': True,
            'cuda_graph_scope': 'model_only',
            'cuda_graph_neighbor_build_outside': True,
            'fixed_edge_capacity': True,
            'dummy_padding': True,
            'tensor_product_accelerator': None,
            'model_specific_fusion': False,
            'compute_stress': False,
            'performance_profile': performance_profile,
            'modal': request.options.get('modal'),
            **graph_stats,
        },
    )
    validate_result(request, result)
    return result
