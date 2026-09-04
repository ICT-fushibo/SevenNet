"""SevenNet Opt1: eager model inference inside a GPU-resident NVT loop.

Only observations and trajectory frames cross the CUDA/host boundary.  The MD
state, both supported thermostats, neighbor-list construction, and SevenNet
inference remain on CUDA throughout the hot loop.  This stage deliberately uses
the checkpoint's regular e3nn implementation: tensor-product accelerators,
``torch.compile``, CUDA Graphs, TF32, and model-specific fusion are not enabled.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ase.io
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator

from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)

import sevenn._keys as key
from sevenn.atom_graph_data import AtomGraphData


_FOURTH_ORDER_COEFFS = (
    1.3512071919596578,
    -1.7024143839193153,
    1.3512071919596578,
)


@dataclass
class _ModelOutput:
    energy: torch.Tensor
    forces: torch.Tensor
    stress: torch.Tensor | None


class _SingleSystemPotential:
    """Unaccelerated SevenNet plus TorchSim's CUDA neighbor list."""

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        atomic_numbers: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        modal: str | None,
        compute_stress: bool,
        profiler: CudaPhaseProfiler | None = None,
        enable_cueq: bool = False,
    ) -> None:
        try:
            from torch_sim.neighbors import torchsim_nl
        except ImportError as exc:
            raise RuntimeError(
                'SevenNet opt1 requires torch-sim-atomistic>=0.6.0 for its '
                'CUDA neighbor list; install SevenNet with the torchsim extra'
            ) from exc

        from sevenn.util import load_checkpoint

        checkpoint = load_checkpoint(model_path)
        try:
            model = checkpoint.build_model(
                enable_cueq=bool(enable_cueq),
                enable_flash=False,
                enable_oeq=False,
            )
        except Exception as exc:
            if enable_cueq:
                raise RuntimeError(
                    "SevenNet Opt4 cuEquivariance fusion could not be enabled"
                ) from exc
            raise
        if not model.type_map:
            raise ValueError('SevenNet checkpoint has no type map')
        atomic_numbers_host = atomic_numbers.cpu().tolist()
        unknown = sorted(set(atomic_numbers_host) - set(model.type_map))
        if unknown:
            raise ValueError(
                f'SevenNet checkpoint does not support atomic Z={unknown}'
            )
        if model.modal_map:
            if modal is None:
                raise ValueError(
                    f'SevenNet checkpoint requires modal; choose one of '
                    f'{sorted(model.modal_map)}'
                )
            if modal not in model.modal_map:
                raise ValueError(
                    f'unknown SevenNet modal {modal!r}; choose one of '
                    f'{sorted(model.modal_map)}'
                )

        # A single-system graph avoids PyG Collater and its per-step container
        # construction.  False here is model semantics, not an accelerator.
        model.set_is_batch_data(False)
        model.eval_type_map = False
        force_output = model._modules.get('force_output')
        if force_output is None or not hasattr(force_output, 'compute_stress'):
            raise RuntimeError(
                'SevenNet Opt1 requires a force_output module with the '
                'compute_stress switch'
            )
        force_output.compute_stress = compute_stress
        self.model = model.to(device).eval()
        self.device = device
        self.atomic_numbers = atomic_numbers
        self.type_indices = torch.tensor(
            [model.type_map[z] for z in atomic_numbers_host],
            dtype=torch.long,
            device=device,
        )
        self.modal = modal
        self.compute_stress = compute_stress
        self.enable_cueq = bool(enable_cueq)
        self.profiler = profiler or CudaPhaseProfiler(
            enabled=False,
            device=device,
        )
        self.cutoff = float(model.cutoff)
        self.neighbor_list_fn = torchsim_nl
        self.cell_batch = cell.unsqueeze(0)
        self.pbc_batch = pbc.unsqueeze(0)
        self.system_idx = torch.zeros(
            atomic_numbers.shape[0], dtype=torch.long, device=device
        )
        self.model_positions = torch.empty(
            atomic_numbers.shape[0], 3, dtype=torch.float32, device=device
        )
        self.model_cell = cell.to(torch.float32)
        self.cell_volume = torch.det(self.model_cell)
        self.num_atoms = torch.tensor(
            atomic_numbers.shape[0], dtype=torch.long, device=device
        )
        empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        empty_shifts = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.graph = AtomGraphData(
            x=self.type_indices,
            edge_index=empty_edge_index,
            pos=self.model_positions,
            **{
                key.ATOMIC_NUMBERS: atomic_numbers,
                key.EDGE_VEC: empty_shifts,
                key.CELL: self.model_cell,
                key.CELL_SHIFT: empty_shifts,
                key.CELL_VOLUME: self.cell_volume,
                key.NUM_ATOMS: self.num_atoms,
                key.DATA_MODALITY: modal,
                key.INFO: {},
            },
        )
        self.parameter_dtypes = sorted(
            {
                str(parameter.dtype).removeprefix('torch.')
                for parameter in model.parameters()
            }
        )

    def __call__(
        self,
        positions: torch.Tensor,
    ) -> _ModelOutput:
        # Neighbor geometry is evaluated from the FP64 MD state.  SevenNet's
        # regular checkpoint interface consumes float32 geometry, matching its
        # ASE calculator; forces are converted back to FP64 for integration.
        with self.profiler.phase('neighbor_list'):
            edge_index, _mapping_system, unit_shifts = self.neighbor_list_fn(
                positions,
                self.cell_batch,
                self.pbc_batch,
                self.cutoff,
                self.system_idx,
            )
        with self.profiler.phase('model_input'):
            self.model_positions.copy_(positions)
            model_shifts = unit_shifts.to(torch.float32)
            shifts = torch.mm(model_shifts, self.model_cell)
            edge_vec = (
                self.model_positions[edge_index[1]]
                - self.model_positions[edge_index[0]]
                + shifts
            )
        # Reuse the single-system graph container and every static tensor.  The
        # SevenNet modules overwrite their intermediate fields in order, so
        # retaining one container keeps only the immediately preceding graph.
        self.graph[key.NODE_FEATURE] = self.type_indices
        self.graph[key.EDGE_IDX] = edge_index
        self.graph[key.POS] = self.model_positions
        self.graph[key.EDGE_VEC] = edge_vec
        self.graph[key.CELL_SHIFT] = model_shifts
        with self.profiler.phase('model_energy_force'):
            with torch.enable_grad():
                output = self.model(self.graph)
        required = [key.PRED_TOTAL_ENERGY, key.PRED_FORCE]
        if self.compute_stress:
            required.append(key.PRED_STRESS)
        missing = [
            name
            for name in required
            if name not in output or output[name] is None
        ]
        if missing:
            raise RuntimeError(f'SevenNet model omitted required outputs: {missing}')

        # SevenNet stores stress as [xx, yy, zz, xy, yz, xz] with the opposite
        # sign to ASE.  Preserve SevenNetCalculator's exact conversion.
        ase_stress = None
        if self.compute_stress:
            model_stress = output[key.PRED_STRESS]
            ase_stress = (
                -model_stress.reshape(-1)[[0, 1, 2, 4, 5, 3]]
            ).detach().to(torch.float64)
        return _ModelOutput(
            energy=output[key.PRED_TOTAL_ENERGY].sum().detach().to(torch.float64),
            forces=output[key.PRED_FORCE].detach().to(torch.float64),
            stress=ase_stress,
        )


class _NoseHooverChain:
    """CUDA tensor port of ASE's NoseHooverChainThermostat."""

    def __init__(
        self,
        *,
        n_atoms: int,
        temperature_k: float,
        damping: float,
        device: torch.device,
        dtype: torch.dtype,
        chain_length: int = 3,
        loop_count: int = 1,
    ) -> None:
        if chain_length < 1 or loop_count < 1:
            raise ValueError('NHC chain_length and loop_count must be positive')
        self.n_atoms = n_atoms
        self.k_t = torch.tensor(
            units.kB * temperature_k, device=device, dtype=dtype
        )
        self.loop_count = loop_count
        self.q = torch.empty(chain_length, device=device, dtype=dtype)
        self.q[0] = 3 * n_atoms * self.k_t * damping**2
        self.q[1:] = self.k_t * damping**2
        self.eta = torch.zeros(chain_length, device=device, dtype=dtype)
        self.p_eta = torch.zeros(chain_length, device=device, dtype=dtype)

    def integrate(self, momenta: torch.Tensor, delta: float) -> torch.Tensor:
        for _ in range(self.loop_count):
            for coefficient in _FOURTH_ORDER_COEFFS:
                momenta = self._loop(
                    momenta, coefficient * delta / self.loop_count
                )
        return momenta

    def _integrate_p_eta(
        self,
        momenta: torch.Tensor,
        index: int,
        delta2: float,
        delta4: float,
    ) -> None:
        if index < self.p_eta.shape[0] - 1:
            self.p_eta[index] *= torch.exp(
                -delta4 * self.p_eta[index + 1] / self.q[index + 1]
            )
        if index == 0:
            g_j = (
                torch.sum(momenta.square() / self.masses)
                - 3 * self.n_atoms * self.k_t
            )
        else:
            g_j = self.p_eta[index - 1].square() / self.q[index - 1] - self.k_t
        self.p_eta[index] += delta2 * g_j
        if index < self.p_eta.shape[0] - 1:
            self.p_eta[index] *= torch.exp(
                -delta4 * self.p_eta[index + 1] / self.q[index + 1]
            )

    def _loop(self, momenta: torch.Tensor, delta: float) -> torch.Tensor:
        delta2, delta4 = delta / 2, delta / 4
        for index in range(self.p_eta.shape[0] - 1, -1, -1):
            self._integrate_p_eta(momenta, index, delta2, delta4)
        self.eta += delta * self.p_eta / self.q
        momenta = momenta * torch.exp(-delta * self.p_eta[0] / self.q[0])
        for index in range(self.p_eta.shape[0]):
            self._integrate_p_eta(momenta, index, delta2, delta4)
        return momenta

    def set_masses(self, masses: torch.Tensor) -> None:
        self.masses = masses


def _initial_momenta(atoms: Atoms, temperature_k: float, seed: int) -> np.ndarray:
    """Match MaxwellBoltzmannDistribution(..., default force_temp=False)."""
    masses = atoms.get_masses()
    noise = np.random.default_rng(seed).standard_normal((len(atoms), 3))
    return noise * np.sqrt(masses * units.kB * temperature_k)[:, None]


def _berendsen_step(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    target_temperature_k: float,
    tau: float,
    force_fn: Callable[[torch.Tensor], _ModelOutput],
) -> tuple[torch.Tensor, torch.Tensor, _ModelOutput]:
    kinetic_energy = torch.sum(momenta.square() / (2 * masses))
    temperature = 2 * kinetic_energy / (3 * positions.shape[0] * units.kB)
    scale = torch.sqrt(1 + (target_temperature_k / temperature - 1) * dt / tau)
    scale = torch.clamp(scale, min=0.9, max=1.1)
    momenta = momenta * scale
    momenta = momenta + 0.5 * dt * forces
    momenta = momenta - momenta.sum(dim=0) / positions.shape[0]
    positions = positions + dt * momenta / masses
    output = force_fn(positions)
    momenta = momenta + 0.5 * dt * output.forces
    return positions, momenta, output


def _nhc_step(
    positions: torch.Tensor,
    momenta: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    *,
    dt: float,
    thermostat: _NoseHooverChain,
    force_fn: Callable[[torch.Tensor], _ModelOutput],
) -> tuple[torch.Tensor, torch.Tensor, _ModelOutput]:
    dt2 = dt / 2
    momenta = thermostat.integrate(momenta, dt2)
    momenta = momenta + dt2 * forces
    positions = positions + dt * momenta / masses
    output = force_fn(positions)
    momenta = momenta + dt2 * output.forces
    momenta = thermostat.integrate(momenta, dt2)
    return positions, momenta, output


def _frame(
    template: Atoms,
    *,
    positions: torch.Tensor,
    momenta: torch.Tensor,
    output: _ModelOutput,
    step: int,
) -> Atoms:
    frame = template.copy()
    frame.positions = positions.detach().cpu().numpy()
    frame.set_momenta(momenta.detach().cpu().numpy())
    frame.info['md_step'] = step
    results = {
        'energy': float(output.energy.cpu()),
        'forces': output.forces.cpu().numpy(),
    }
    if output.stress is not None:
        results['stress'] = output.stress.cpu().numpy()
    frame.calc = SinglePointCalculator(frame, **results)
    return frame


def _configure_output(request) -> tuple[Path | None, Path | None]:
    config = request.config
    if config.collect_trajectory and config.record_interval < 1:
        raise ValueError('collect_trajectory requires record_interval >= 1')
    if request.output_path is None:
        return None, None
    if not config.collect_trajectory:
        raise ValueError('output_path requires collect_trajectory=True')
    target = Path(request.output_path).expanduser().resolve()
    partial = target.with_name(f'{target.stem}.part.extxyz')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not request.options.get('overwrite', False):
        raise FileExistsError(f'Refusing to overwrite {target}')
    for stale in (target, partial):
        if stale.exists():
            stale.unlink()
    return target, partial


def run_md(request):
    """Run the shared MD contract with the SevenNet Opt1 implementation."""
    from md_benchmark.md_route import MDObservation, MDRunResult, validate_result

    if request.model != 'sevennet' or request.stage != 'opt1':
        raise ValueError(
            f'sevenn.md_stages.opt1 owns sevennet/opt1, got '
            f'{request.model}/{request.stage}'
        )
    if request.backend != 'gpu-resident':
        raise ValueError("SevenNet opt1 backend must be 'gpu-resident'")
    if request.config.dtype != 'float64':
        raise ValueError('SevenNet opt1 requires --dtype float64 for the MD state')
    if request.config.device.split(':', maxsplit=1)[0] != 'cuda':
        raise ValueError('SevenNet opt1 is CUDA-only; CPU fallback is forbidden')
    if not torch.cuda.is_available():
        raise RuntimeError('SevenNet opt1 requested CUDA, but CUDA is unavailable')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
        raise RuntimeError(
            'SevenNet opt1 forbids TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1'
        )
    if request.atoms.constraints:
        raise NotImplementedError('SevenNet opt1 does not support ASE constraints')

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(request.config.device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    config = request.config
    atoms = request.atoms.copy()
    target_path, partial_path = _configure_output(request)

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
    compute_stress = bool(
        config.collect_trajectory or request.options.get('compute_stress', False)
    )
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options),
        device=device,
    )
    potential = _SingleSystemPotential(
        request.model_path,
        device=device,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        modal=request.options.get('modal'),
        compute_stress=compute_stress,
        profiler=profiler,
    )

    def force_fn(positions: torch.Tensor) -> _ModelOutput:
        return potential(positions)

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

    def advance(
        positions: torch.Tensor,
        momenta: torch.Tensor,
        output: _ModelOutput,
        thermostat: _NoseHooverChain | None,
    ) -> tuple[torch.Tensor, torch.Tensor, _ModelOutput]:
        if config.integrator == 'berendsen':
            return _berendsen_step(
                positions,
                momenta,
                output.forces,
                masses,
                dt=dt,
                target_temperature_k=config.temperature_k,
                tau=tau,
                force_fn=force_fn,
            )
        assert thermostat is not None
        return _nhc_step(
            positions,
            momenta,
            output.forces,
            masses,
            dt=dt,
            thermostat=thermostat,
            force_fn=force_fn,
        )

    # Warmup is intentionally excluded from timing, then the full MD and NHC
    # state is restored exactly as run_ase_baseline rebuilds its dynamics object.
    if config.warmup_steps:
        warm_positions = positions0.clone()
        warm_momenta = momenta0.clone()
        warm_output = force_fn(warm_positions)
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
    in_memory_trajectory = (
        [] if config.collect_trajectory and target_path is None else None
    )
    observation_steps = set(config.observation_steps)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    profiler.start()
    started = time.perf_counter()
    with profiler.phase('initial_force'):
        output = force_fn(positions)

    def record_frame(step: int) -> None:
        frame = _frame(
            atoms,
            positions=positions,
            momenta=momenta,
            output=output,
            step=step,
        )
        if partial_path is not None:
            ase.io.write(partial_path, frame, append=True, format='extxyz')
        else:
            assert in_memory_trajectory is not None
            in_memory_trajectory.append(frame)

    # ASE Dynamics invokes attached trajectory observers at nsteps=0.
    if config.collect_trajectory:
        record_frame(0)
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
        if config.collect_trajectory and step % config.record_interval == 0:
            record_frame(step)
    torch.cuda.synchronize(device)
    profiler.stop()
    elapsed = time.perf_counter() - started
    performance_profile = profiler.summary(synchronize=False)
    peak_memory = torch.cuda.max_memory_allocated(device) / 1e9

    if target_path is not None:
        assert partial_path is not None
        os.replace(partial_path, target_path)
    final_atoms = _frame(
        atoms,
        positions=positions,
        momenta=momenta,
        output=output,
        step=config.steps,
    )
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=peak_memory,
        final_atoms=final_atoms,
        observations=observations,
        trajectory=in_memory_trajectory,
        trajectory_path=str(target_path) if target_path is not None else None,
        metadata={
            'engine': 'sevennet_gpu_resident',
            'backend': request.backend,
            'model_path': str(Path(request.model_path).expanduser().resolve()),
            'gpu_resident': True,
            'md_state_dtype': 'float64',
            'requested_dtype': config.dtype,
            'model_input_dtype': 'float32',
            'checkpoint_parameter_dtypes': potential.parameter_dtypes,
            'neighbor_list': 'torch_sim.neighbors.torchsim_nl_cuda',
            'integrator': config.integrator,
            'warmup_steps': config.warmup_steps,
            'tf32': False,
            'torch_compile': False,
            'cuda_graph': False,
            'tensor_product_accelerator': None,
            'model_specific_fusion': False,
            'compute_stress': compute_stress,
            'performance_profile': performance_profile,
            'modal': request.options.get('modal'),
            'trajectory_includes_step_zero': config.collect_trajectory,
        },
    )
    validate_result(request, result)
    return result
