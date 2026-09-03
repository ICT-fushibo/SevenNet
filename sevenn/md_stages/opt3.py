"""SevenNet Opt3: whole-step CUDA Graph with fixed-shape PBC topology.

One replay contains the NVT integrator, fixed-shape CUDA neighbor construction,
SevenNet forward/conservative-force path, and the persistent MD-state update.
The implementation uses eSEN's per-centre CAP policy and distributed
dummy sink padding.  Capacity overflow is recorded on device and fails closed
at the next reporting/final synchronization point.  Transaction rollback and
multiple graph buckets are intentionally deferred.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from ase import units
from md_benchmark.neighbor_utils import (
    capacities_from_counts,
    normalize_neighbor_capacities,
)
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)

import sevenn._keys as key
from sevenn.atom_graph_data import AtomGraphData
from sevenn.md_stages.fixed_neighbor import (
    FixedShapeSevenNetNeighborBuilder,
    neighbor_capacity_from_probe,
)
from sevenn.md_stages.opt1 import (
    _configure_output,
    _frame,
    _initial_momenta,
    _ModelOutput,
    _NoseHooverChain,
    _SingleSystemPotential,
)
from sevenn.md_stages.opt2 import (
    CUDAGraphCapacityError,
    CUDAGraphValidationError,
    _ModelOnlyCUDAGraphPotential,
)


def _maximum_degree(edge_index: torch.Tensor, num_atoms: int) -> int:
    """Return the largest SevenNet centre degree during setup only."""

    if edge_index.shape[1] == 0:
        return 0
    counts = torch.bincount(edge_index[0], minlength=num_atoms)[:num_atoms]
    return int(counts.max().item())


def _guarded_uniform_capacity_from_total(
    total_capacity: int,
    num_atoms: int,
    *,
    slot_step: int = 8,
    guard_slots: int = 0,
) -> int:
    """Convert an already guarded total-edge CAP to aligned centre slots."""

    if total_capacity < 1 or num_atoms < 1:
        raise ValueError('total_capacity and num_atoms must be positive')
    if slot_step < 1 or guard_slots < 0:
        raise ValueError('slot_step must be positive and guard_slots non-negative')
    average_ceiling = math.ceil(total_capacity / num_atoms)
    aligned = math.ceil(average_ceiling / slot_step) * slot_step
    return aligned + guard_slots * slot_step


class _WholeStepPotential(_ModelOnlyCUDAGraphPotential):
    """SevenNet fixed-buffer model input owned by the whole-step graph."""

    def __init__(self, *args, dummy_atoms: int, **kwargs) -> None:
        if dummy_atoms < 1:
            raise ValueError('cuda_graph_dummy_atoms must be positive')
        self.dummy_atoms = int(dummy_atoms)
        super().__init__(*args, **kwargs)

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
        self.padding_edge_vec = None
        self.static_type_indices = torch.cat(
            (
                self.type_indices,
                self.type_indices[:1].expand(self.dummy_atoms),
            ),
            dim=0,
        )
        self.static_atomic_numbers = torch.cat(
            (
                self.atomic_numbers,
                self.atomic_numbers[:1].expand(self.dummy_atoms),
            ),
            dim=0,
        )
        self.static_model_positions = torch.zeros(
            (self.n_real + self.dummy_atoms, 3),
            dtype=torch.float32,
            device=self.device,
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
                    self.n_real + self.dummy_atoms,
                    dtype=torch.long,
                    device=self.device,
                ),
                key.DATA_MODALITY: self.modal,
                key.INFO: {},
            },
        )

    @torch.no_grad()
    def write_geometry_(
        self,
        positions: torch.Tensor,
        builder: FixedShapeSevenNetNeighborBuilder,
        *,
        step: torch.Tensor,
    ) -> None:
        """Build topology and update fixed model geometry inside capture."""

        assert self.static_model_positions is not None
        assert self.static_edge_index is not None
        assert self.static_cell_shifts is not None
        assert self.static_edge_vec is not None
        self.static_model_positions[: self.n_real].copy_(positions)
        builder.build(positions, step=step)
        shifts = torch.mm(self.static_cell_shifts, self.model_cell)
        edge_vec = (
            self.static_model_positions[self.static_edge_index[1]]
            - self.static_model_positions[self.static_edge_index[0]]
            + shifts
        )
        self.static_edge_vec.copy_(edge_vec)


def _integrate_nhc_pure(
    momenta: torch.Tensor,
    eta: torch.Tensor,
    p_eta: torch.Tensor,
    thermostat: _NoseHooverChain,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture-safe NHC update that does not mutate persistent state."""

    def update_one(
        p: torch.Tensor,
        values: list[torch.Tensor],
        index: int,
        delta2: float,
        delta4: float,
    ) -> None:
        if index < len(values) - 1:
            values[index] = values[index] * torch.exp(
                -delta4 * values[index + 1] / thermostat.q[index + 1]
            )
        if index == 0:
            g_j = (
                torch.sum(p.square() / thermostat.masses)
                - 3 * thermostat.n_atoms * thermostat.k_t
            )
        else:
            g_j = (
                values[index - 1].square() / thermostat.q[index - 1]
                - thermostat.k_t
            )
        values[index] = values[index] + delta2 * g_j
        if index < len(values) - 1:
            values[index] = values[index] * torch.exp(
                -delta4 * values[index + 1] / thermostat.q[index + 1]
            )

    p = momenta
    eta_out = eta
    p_eta_out = p_eta
    chain_length = int(p_eta.shape[0])
    for _ in range(thermostat.loop_count):
        for coefficient in (
            1.3512071919596578,
            -1.7024143839193153,
            1.3512071919596578,
        ):
            sub_delta = coefficient * delta / thermostat.loop_count
            delta2 = sub_delta / 2
            delta4 = sub_delta / 4
            values = [p_eta_out[index] for index in range(chain_length)]
            for index in range(chain_length - 1, -1, -1):
                update_one(p, values, index, delta2, delta4)
            stacked = torch.stack(values)
            eta_out = eta_out + sub_delta * stacked / thermostat.q
            p = p * torch.exp(-sub_delta * values[0] / thermostat.q[0])
            for index in range(chain_length):
                update_one(p, values, index, delta2, delta4)
            p_eta_out = torch.stack(values)
    return p, eta_out, p_eta_out


class _SevenNetWholeStepGraph:
    """Persistent SevenNet MD state and exactly one whole-step CUDA Graph."""

    def __init__(
        self,
        potential: _WholeStepPotential,
        builder: FixedShapeSevenNetNeighborBuilder,
        *,
        positions: torch.Tensor,
        momenta: torch.Tensor,
        masses: torch.Tensor,
        integrator: str,
        temperature_k: float,
        dt: float,
        tau: float,
        capture_warmup: int,
        verlet_rebuild_interval: int,
        eager_reference: _ModelOutput,
        energy_atol: float,
        force_atol: float,
        state_atol: float,
    ) -> None:
        self.potential = potential
        self.builder = builder
        self.device = positions.device
        self.num_atoms = int(positions.shape[0])
        self.positions = positions.detach().clone()
        self.momenta = momenta.detach().clone()
        self.forces = torch.zeros_like(self.positions)
        self.energy = torch.zeros((), device=self.device, dtype=torch.float64)
        self.masses = masses
        self.integrator_name = integrator
        self.temperature_k = float(temperature_k)
        self.dt = float(dt)
        self.tau = float(tau)
        self.capture_warmup = int(capture_warmup)
        self.verlet_rebuild_interval = int(verlet_rebuild_interval)
        if self.verlet_rebuild_interval < 0:
            raise ValueError('verlet_rebuild_interval must be non-negative')
        if self.builder.verlet_skin <= 0:
            self.verlet_rebuild_interval = 0
        self.advance = torch.zeros((), device=self.device, dtype=torch.float64)
        self.step_counter = torch.zeros((), device=self.device, dtype=torch.long)
        self.thermostat: _NoseHooverChain | None = None
        if integrator == 'nose_hoover_chain':
            self.thermostat = _NoseHooverChain(
                n_atoms=self.num_atoms,
                temperature_k=temperature_k,
                damping=tau,
                device=self.device,
                dtype=torch.float64,
            )
            self.thermostat.set_masses(masses)
        self.initial_positions = self.positions.clone()
        self.initial_momenta = self.momenta.clone()
        self.initial_eta = (
            None if self.thermostat is None else self.thermostat.eta.clone()
        )
        self.initial_p_eta = (
            None if self.thermostat is None else self.thermostat.p_eta.clone()
        )
        self.eager_reference = eager_reference
        self.energy_atol = float(energy_atol)
        self.force_atol = float(force_atol)
        self.state_atol = float(state_atol)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.capture_count = 0
        self.total_replays = 0
        self.production_replays = 0
        self.capture_wall_time_s = 0.0
        self.output_addresses_stable = False
        self.validation_energy_abs_error = 0.0
        self.validation_force_max_abs_error = 0.0
        self.validation_within_tolerance = False
        self.step_validation_errors: dict[str, float] = {}
        self.step_validation_within_tolerance = False

    @torch.no_grad()
    def restore_initial_(self) -> None:
        self.positions.copy_(self.initial_positions)
        self.momenta.copy_(self.initial_momenta)
        self.forces.zero_()
        self.energy.zero_()
        self.advance.zero_()
        self.step_counter.zero_()
        if self.thermostat is not None:
            assert self.initial_eta is not None
            assert self.initial_p_eta is not None
            self.thermostat.eta.copy_(self.initial_eta)
            self.thermostat.p_eta.copy_(self.initial_p_eta)

    def _berendsen_proposal(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kinetic = torch.sum(self.momenta.square() / (2 * self.masses))
        temperature = 2 * kinetic / (3 * self.num_atoms * units.kB)
        scale = torch.sqrt(
            1 + (self.temperature_k / temperature - 1) * self.dt / self.tau
        )
        scale = torch.clamp(scale, min=0.9, max=1.1)
        half = self.momenta * scale + 0.5 * self.dt * self.forces
        half = half - half.sum(dim=0) / self.num_atoms
        positions = self.positions + self.dt * half / self.masses
        return half, positions

    def _graph_body(self) -> None:
        old_positions = self.positions
        old_momenta = self.momenta
        if self.integrator_name == 'berendsen':
            half_momenta, advanced_positions = self._berendsen_proposal()
            eta_final = p_eta_final = None
        else:
            assert self.thermostat is not None
            half_momenta, eta_half, p_eta_half = _integrate_nhc_pure(
                old_momenta,
                self.thermostat.eta,
                self.thermostat.p_eta,
                self.thermostat,
                self.dt / 2,
            )
            half_momenta = half_momenta + 0.5 * self.dt * self.forces
            advanced_positions = (
                old_positions + self.dt * half_momenta / self.masses
            )
        evaluation_positions = old_positions + self.advance * (
            advanced_positions - old_positions
        )
        graph_step = self.step_counter + self.advance.to(torch.long)
        self.potential.write_geometry_(
            evaluation_positions,
            self.builder,
            step=graph_step,
        )
        model_forces, model_energy = self.potential._static_forward()
        forces = model_forces.to(torch.float64)
        half_momenta = half_momenta + 0.5 * self.dt * forces
        if self.integrator_name == 'berendsen':
            advanced_momenta = half_momenta
        else:
            assert self.thermostat is not None
            advanced_momenta, eta_final, p_eta_final = _integrate_nhc_pure(
                half_momenta,
                eta_half,
                p_eta_half,
                self.thermostat,
                self.dt / 2,
            )
        final_momenta = old_momenta + self.advance * (
            advanced_momenta - old_momenta
        )
        with torch.no_grad():
            self.positions.copy_(evaluation_positions)
            self.momenta.copy_(final_momenta)
            self.forces.copy_(forces)
            self.energy.copy_(model_energy)
            if self.thermostat is not None:
                assert eta_final is not None and p_eta_final is not None
                self.thermostat.eta.copy_(
                    self.thermostat.eta
                    + self.advance * (eta_final - self.thermostat.eta)
                )
                self.thermostat.p_eta.copy_(
                    self.thermostat.p_eta
                    + self.advance * (p_eta_final - self.thermostat.p_eta)
                )
            self.step_counter.add_(self.advance.to(torch.long))

    def _persistent_addresses(self) -> tuple[int, ...]:
        tensors = [
            self.positions,
            self.momenta,
            self.forces,
            self.energy,
            self.advance,
            self.step_counter,
            self.builder.edge_index,
            self.builder.cell_offsets,
        ]
        if self.thermostat is not None:
            tensors.extend((self.thermostat.eta, self.thermostat.p_eta))
        return (
            *(tensor.data_ptr() for tensor in tensors),
            *self.potential._input_addresses(),
        )

    def capture(self) -> None:
        if self.graph is not None:
            raise RuntimeError('SevenNet whole-step graph is already captured')
        assert self.potential.static_edge_vec is not None
        capture_edge_vec = self.potential.static_edge_vec.detach().clone()
        capture_edge_vec.requires_grad_(True)
        self.potential.static_edge_vec = capture_edge_vec
        assert self.potential.static_graph is not None
        self.potential.static_graph[key.EDGE_VEC] = capture_edge_vec

        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = side_stream
        side_stream.wait_stream(current_stream)
        with torch.cuda.stream(side_stream):
            self.restore_initial_()
            self.advance.fill_(1.0)
            for _ in range(self.capture_warmup):
                self._graph_body()
            self.restore_initial_()
            self.advance.fill_(1.0)
            self.builder.reset_stats()
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            self._graph_body()
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - started
        self.graph = graph
        self.capture_count = 1

        addresses = self._persistent_addresses()
        self.restore_initial_()
        self.builder.reset_stats()
        self.advance.zero_()
        graph.replay()
        torch.cuda.synchronize(self.device)
        self.total_replays += 1
        self.validation_energy_abs_error = float(
            (self.energy - self.eager_reference.energy).abs().item()
        )
        self.validation_force_max_abs_error = float(
            (self.forces - self.eager_reference.forces).abs().max().item()
        )
        self.validation_within_tolerance = (
            self.validation_energy_abs_error <= self.energy_atol
            and self.validation_force_max_abs_error <= self.force_atol
        )
        if not bool(torch.isfinite(self.energy)) or not bool(
            torch.isfinite(self.forces).all()
        ):
            raise CUDAGraphValidationError(
                'SevenNet whole-step capture produced non-finite output'
            )
        self.output_addresses_stable = addresses == self._persistent_addresses()
        if not self.output_addresses_stable:
            raise CUDAGraphValidationError(
                'SevenNet whole-step state addresses changed'
            )

        # Validate the actual state transition, not only frame-zero model
        # output.  Both paths start from the same state, evaluate frame zero,
        # then advance exactly one fixed-builder NVT step.  The eager path uses
        # the same fixed-capacity topology but does not replay the CUDA Graph.
        self.restore_initial_()
        self.builder.reset_stats()
        self.advance.zero_()
        self._graph_body()
        self.advance.fill_(1.0)
        self._graph_body()
        torch.cuda.synchronize(self.device)
        eager_state = self._validation_state()

        self.restore_initial_()
        self.builder.reset_stats()
        self.advance.zero_()
        graph.replay()
        self.total_replays += 1
        self.advance.fill_(1.0)
        graph.replay()
        self.total_replays += 1
        torch.cuda.synchronize(self.device)
        if addresses != self._persistent_addresses():
            raise CUDAGraphValidationError(
                'SevenNet whole-step persistent addresses changed during '
                'eager/Graph validation'
            )
        self.raise_for_overflow()
        graph_state = self._validation_state()
        errors = {
            name: float((graph_state[name] - eager_state[name]).abs().max().item())
            for name in eager_state
        }
        self.step_validation_errors = errors
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (*eager_state.values(), *graph_state.values())
        )
        if not finite:
            raise CUDAGraphValidationError(
                'SevenNet eager/Graph one-step validation produced non-finite state'
            )
        tolerances = {
            'positions': self.state_atol,
            'momenta': self.state_atol,
            'forces': self.force_atol,
            'energy': self.energy_atol,
            'thermostat_eta': self.state_atol,
            'thermostat_p_eta': self.state_atol,
        }
        self.step_validation_within_tolerance = all(
            errors[name] <= tolerances[name] for name in errors
        )
        if not self.step_validation_within_tolerance:
            details = ', '.join(
                f'{name}={error:.3e} (atol={tolerances[name]:.3e})'
                for name, error in errors.items()
                if error > tolerances[name]
            )
            warnings.warn(
                f'SevenNet whole-step eager/Graph numerical warning: {details}',
                RuntimeWarning,
                stacklevel=2,
            )
        self.restore_initial_()
        self.builder.reset_stats()

    def _validation_state(self) -> dict[str, torch.Tensor]:
        state = {
            'positions': self.positions.clone(),
            'momenta': self.momenta.clone(),
            'forces': self.forces.clone(),
            'energy': self.energy.clone(),
        }
        if self.thermostat is not None:
            state['thermostat_eta'] = self.thermostat.eta.clone()
            state['thermostat_p_eta'] = self.thermostat.p_eta.clone()
        return state

    def reset_production(self) -> None:
        if self.graph is None:
            raise RuntimeError('Capture must complete before production')
        self.restore_initial_()
        self.builder.reset_stats()
        self.builder.initialize_skin(self.positions)
        self.production_replays = 0

    def evaluate_initial(self) -> _ModelOutput:
        if self.graph is None:
            raise RuntimeError('Capture must complete before replay')
        self.advance.zero_()
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        self.advance.fill_(1.0)
        return self.output()

    def step(self) -> _ModelOutput:
        if self.graph is None:
            raise RuntimeError('Capture must complete before replay')
        if (
            self.verlet_rebuild_interval
            and self.production_replays % self.verlet_rebuild_interval == 0
        ):
            self.builder.initialize_skin(self.positions)
        self.graph.replay()
        self.production_replays += 1
        self.total_replays += 1
        return self.output()

    def output(self) -> _ModelOutput:
        return _ModelOutput(
            energy=self.energy,
            forces=self.forces,
            stress=None,
        )

    def raise_for_overflow(self) -> None:
        stats = self.builder.stats()
        if int(stats['fixed_builder_capacity_misses']):
            raise CUDAGraphCapacityError(
                int(stats['fixed_builder_max_overflow_required']),
                int(stats['fixed_builder_max_overflow_capacity']),
            )

    def stats(self) -> dict[str, object]:
        builder_stats = self.builder.stats()
        calls = int(builder_stats['fixed_builder_build_calls'])
        misses = int(builder_stats['fixed_builder_capacity_misses'])
        return {
            **builder_stats,
            'cuda_graph_capture_count': self.capture_count,
            'cuda_graph_production_capture_count': 0,
            'cuda_graph_total_replays': self.total_replays,
            'cuda_graph_production_calls': calls,
            'cuda_graph_production_replays': self.production_replays,
            'cuda_graph_capacity_misses': misses,
            'cuda_graph_hit_rate': (
                self.production_replays / calls if calls else 0.0
            ),
            'cuda_graph_edge_capacity': self.builder.edge_capacity,
            'cuda_graph_min_real_edges': builder_stats[
                'fixed_builder_min_real_edges'
            ],
            'cuda_graph_max_real_edges': builder_stats[
                'fixed_builder_max_real_edges'
            ],
            'cuda_graph_max_padding_fraction': builder_stats[
                'fixed_builder_max_padding_fraction'
            ],
            'cuda_graph_dummy_atoms': self.potential.dummy_atoms,
            'cuda_graph_capture_warmup': self.capture_warmup,
            'verlet_rebuild_interval': self.verlet_rebuild_interval,
            'cuda_graph_capture_wall_time_s': self.capture_wall_time_s,
            'cuda_graph_replay_output_addresses_stable': (
                self.output_addresses_stable
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
                self.validation_within_tolerance
            ),
            'cuda_graph_whole_step_validation_state_atol': self.state_atol,
            'cuda_graph_whole_step_validation_errors': dict(
                self.step_validation_errors
            ),
            'cuda_graph_whole_step_validation_within_tolerance': (
                self.step_validation_within_tolerance
            ),
        }


def run_md(request):
    """Run SevenNet Opt3 under the shared MD contract."""

    from md_benchmark.md_route import MDObservation, MDRunResult, validate_result

    if request.model != 'sevennet' or request.stage != 'opt3':
        raise ValueError(
            f'sevenn.md_stages.opt3 owns sevennet/opt3, got '
            f'{request.model}/{request.stage}'
        )
    if request.backend != 'whole-step-cuda-graph':
        raise ValueError("SevenNet opt3 backend must be 'whole-step-cuda-graph'")
    if request.config.dtype != 'float64':
        raise ValueError('SevenNet opt3 requires --dtype float64 for MD state')
    if request.config.device.split(':', maxsplit=1)[0] != 'cuda':
        raise ValueError('SevenNet opt3 is CUDA-only; CPU fallback is forbidden')
    if not torch.cuda.is_available():
        raise RuntimeError('SevenNet opt3 requested CUDA, but CUDA is unavailable')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise RuntimeError('SevenNet opt3 forbids TF32 override')
    if request.atoms.constraints:
        raise NotImplementedError('SevenNet opt3 does not support constraints')
    if request.config.collect_trajectory or request.output_path is not None:
        raise NotImplementedError(
            'SevenNet opt3 supports observation statistics but not trajectory '
            'or stress capture'
        )
    if request.options.get('compute_stress', False):
        raise NotImplementedError('SevenNet opt3 does not capture stress')

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
    cell = torch.tensor(np.asarray(atoms.cell), device=device, dtype=torch.float64)
    pbc = torch.tensor(np.asarray(atoms.pbc), device=device, dtype=torch.bool)
    atomic_numbers = torch.tensor(
        atoms.get_atomic_numbers(), device=device, dtype=torch.long
    )
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options), device=device
    )
    requested_total = request.options.get('cuda_graph_edge_capacity')
    capture_warmup = int(request.options.get('cuda_graph_capture_warmup', 3))
    energy_atol = float(request.options.get('cuda_graph_energy_atol_ev', 2e-4))
    force_atol = float(
        request.options.get('cuda_graph_force_atol_ev_per_a', 2e-4)
    )
    state_atol = float(request.options.get('cuda_graph_state_atol', 1e-10))
    potential = _WholeStepPotential(
        request.model_path,
        device=device,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        modal=request.options.get('modal'),
        profiler=profiler,
        requested_edge_capacity=(
            int(requested_total) if requested_total is not None else None
        ),
        edge_margin=float(request.options.get('cuda_graph_edge_margin', 0.10)),
        edge_step=int(request.options.get('cuda_graph_edge_step', 8)),
        track_neighbor_capacity=False,
        capture_warmup=capture_warmup,
        verlet_rebuild_interval=int(
            request.options.get('verlet_rebuild_interval', 0)
        ),
        energy_atol=energy_atol,
        force_atol=force_atol,
        dummy_atoms=int(request.options.get('cuda_graph_dummy_atoms', 32)),
    )

    # Setup-only eager reference and degree probe.  Existing total-edge CAP is
    # accepted for compatibility, but per-centre overflow remains authoritative.
    eager_reference = _SingleSystemPotential.__call__(potential, positions0)
    initial_index, _initial_vec, _initial_offsets = potential._build_real_inputs(
        positions0
    )
    initial_maximum = max(1, _maximum_degree(initial_index, len(atoms)))
    inferred_capacity = neighbor_capacity_from_probe(
        initial_maximum,
        margin=float(request.options.get('cuda_graph_neighbor_margin', 0.10)),
        slot_step=int(request.options.get('cuda_graph_neighbor_step', 8)),
    )
    explicit_neighbors = request.options.get('cuda_graph_neighbors_per_atom')
    if explicit_neighbors is not None:
        neighbors_per_atom = int(explicit_neighbors)
        capacity_source = 'trajectory-per-atom-probe'
    else:
        total_floor = 0
        if requested_total is not None:
            total_floor = _guarded_uniform_capacity_from_total(
                int(requested_total),
                len(atoms),
                slot_step=8,
                guard_slots=0,
            )
        neighbors_per_atom = max(inferred_capacity, total_floor)
        capacity_source = 'total-edge-plus-initial-per-atom'
    initial_counts = torch.bincount(initial_index[0], minlength=len(atoms))[: len(atoms)]
    explicit_caps = request.options.get('neighbor_capacities')
    if explicit_caps is None and request.options.get('per_atom_cap', False):
        capacities = capacities_from_counts(
            initial_counts,
            factor=float(request.options.get('cuda_graph_neighbor_margin', 0.10)) + 1.0,
            headroom=1,
            alignment=int(request.options.get('cuda_graph_neighbor_step', 8)),
        )
    else:
        capacities = normalize_neighbor_capacities(
            explicit_caps,
            num_atoms=len(atoms),
            default=neighbors_per_atom,
        )
    initial_capacity_excess = torch.clamp_min(
        initial_counts - torch.as_tensor(capacities, device=device), 0
    )
    if bool(initial_capacity_excess.max().item() > 0):
        raise CUDAGraphCapacityError(
            int(initial_counts.max().item()), int(max(capacities))
        )
    edge_capacity = int(sum(capacities))
    if explicit_caps is None and request.options.get('per_atom_cap', False):
        capacity_source = 'initial-per-atom-cap-vector'
    potential._initialize_static_graph(edge_capacity)
    assert potential.static_edge_index is not None
    assert potential.static_cell_shifts is not None
    builder = FixedShapeSevenNetNeighborBuilder(
        num_atoms=len(atoms),
        cell=cell,
        pbc=pbc,
        cutoff=potential.cutoff,
        neighbors_per_atom=neighbors_per_atom,
        neighbor_capacities=capacities,
        dummy_atoms=potential.dummy_atoms,
        verlet_skin=float(request.options.get('verlet_skin', 0.0)),
        verlet_candidate_capacity=request.options.get('verlet_candidate_capacity'),
        max_neighbors=int(request.options.get('cuda_graph_max_neighbors', 300)),
        degeneracy_tolerance=float(
            request.options.get('cuda_graph_degeneracy_tolerance', 0.01)
        ),
        output_edge_index=potential.static_edge_index,
        output_cell_offsets=potential.static_cell_shifts,
    )
    builder.initialize_skin(positions0)

    graph_md = _SevenNetWholeStepGraph(
        potential,
        builder,
        positions=positions0,
        momenta=momenta0,
        masses=masses,
        integrator=config.integrator,
        temperature_k=config.temperature_k,
        dt=config.timestep_fs * units.fs,
        tau=config.thermostat_time_fs * units.fs,
        capture_warmup=capture_warmup,
        eager_reference=eager_reference,
        energy_atol=energy_atol,
        force_atol=force_atol,
        state_atol=state_atol,
    )
    graph_md.capture()

    if config.warmup_steps:
        graph_md.reset_production()
        graph_md.evaluate_initial()
        for _ in range(config.warmup_steps):
            graph_md.step()
        torch.cuda.synchronize(device)
        graph_md.raise_for_overflow()

    graph_md.reset_production()
    observations = []
    observation_steps = set(config.observation_steps)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase('initial_force'):
        output = graph_md.evaluate_initial()

    def record(step: int) -> None:
        # This is already a reporting synchronization point; detect capacity
        # failure before exporting a potentially truncated observation.
        graph_md.raise_for_overflow()
        observations.append(
            MDObservation(
                step=step,
                potential_energy_ev=float(output.energy.cpu()),
                kinetic_energy_ev=float(
                    torch.sum(graph_md.momenta.square() / (2 * masses)).cpu()
                ),
                forces_ev_per_a=output.forces.cpu().numpy().copy(),
                positions_a=graph_md.positions.cpu().numpy().copy(),
            )
        )

    if config.collect_statistics and 0 in observation_steps:
        record(0)
    for step in range(1, config.steps + 1):
        with profiler.phase('md_step'):
            output = graph_md.step()
        if config.collect_statistics and step in observation_steps:
            record(step)
    torch.cuda.synchronize(device)
    graph_md.raise_for_overflow()
    profiler.stop()
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9
    performance_profile = profiler.summary(synchronize=False)

    final_atoms = _frame(
        atoms,
        positions=graph_md.positions,
        momenta=graph_md.momenta,
        output=output,
        step=config.steps,
    )
    graph_stats = graph_md.stats()
    expected_replays = config.steps + 1
    if graph_stats['cuda_graph_production_replays'] != expected_replays:
        raise RuntimeError(
            'SevenNet Opt3 replay count mismatch: '
            f'expected={expected_replays}, '
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
            'engine': 'sevennet_gpu_resident_whole_step_cuda_graph',
            'backend': 'whole-step-cuda-graph',
            'requested_backend': request.backend,
            'model_path': str(Path(request.model_path).expanduser().resolve()),
            'gpu_resident': True,
            'md_state_dtype': 'float64',
            'model_input_dtype': 'float32',
            'checkpoint_parameter_dtypes': potential.parameter_dtypes,
            'neighbor_list': 'fixed_shape_pbc_cuda',
            'integrator': config.integrator,
            'warmup_steps': config.warmup_steps,
            'tf32': False,
            'amp': False,
            'torch_compile': False,
            'cuda_graph': True,
            'cuda_graph_scope': 'whole_step',
            'cuda_graph_neighbor_build_inside': True,
            'cuda_graph_neighbor_build_outside': False,
            'fixed_edge_capacity': True,
            'capacity_policy': (
                'esen-per-atom-cap'
                if len(set(capacities)) > 1
                else 'esen-uniform-cap'
            ),
            'neighbor_capacities': capacities,
            'capacity_source': capacity_source,
            'capacity_total_to_per_atom_guard_slots': 0,
            'initial_probe_max_neighbors': initial_maximum,
            'dummy_padding': True,
            'sink_padding': 'distributed-dummy-bank',
            'transactional_recovery': False,
            'transaction_rollback': False,
            'capacity_overflow_policy': 'raise-after-sync-no-fallback',
            'cuda_graph_buckets': 1,
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
