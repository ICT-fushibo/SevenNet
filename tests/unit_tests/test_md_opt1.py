"""Contract and ASE-alignment tests for the GPU-resident Opt1 stage."""

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

import sevenn._keys as key
from sevenn.atom_graph_data import AtomGraphData
from sevenn.md_stages.opt1 import (
    _ModelOutput,
    _NoseHooverChain,
    _SingleSystemPotential,
    _berendsen_step,
    _initial_momenta,
    _nhc_step,
    _frame,
    run_md,
)
from sevenn.nn.force_output import ForceStressOutputFromEdge


class _ConstantForceCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, forces: np.ndarray) -> None:
        super().__init__()
        self.forces = forces

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 0.0, 'forces': self.forces.copy()}


def test_initial_momenta_is_seeded_and_float64() -> None:
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2, 0, 0]])
    first = _initial_momenta(atoms, 300.0, 42)
    second = _initial_momenta(atoms, 300.0, 42)
    assert first.dtype == second.dtype == float
    assert torch.equal(torch.from_numpy(first), torch.from_numpy(second))


def test_berendsen_step_keeps_tensor_state_on_device() -> None:
    positions = torch.zeros((2, 3), dtype=torch.float64)
    momenta = torch.ones((2, 3), dtype=torch.float64) * 0.1
    forces = torch.zeros_like(positions)
    masses = torch.ones((2, 1), dtype=torch.float64) * 63.546

    def harmonic(pos: torch.Tensor) -> _ModelOutput:
        return _ModelOutput(
            energy=pos.square().sum(),
            forces=-2 * pos,
            stress=torch.zeros(6, dtype=pos.dtype, device=pos.device),
        )

    new_positions, new_momenta, output = _berendsen_step(
        positions,
        momenta,
        forces,
        masses,
        dt=0.01,
        target_temperature_k=300,
        tau=1.0,
        force_fn=harmonic,
    )
    assert new_positions.device == positions.device
    assert new_momenta.dtype == torch.float64
    assert output.forces.shape == (2, 3)
    assert torch.isfinite(new_positions).all()


def test_nhc_tensor_port_is_finite() -> None:
    masses = torch.full((2, 1), 63.546, dtype=torch.float64)
    thermostat = _NoseHooverChain(
        n_atoms=2,
        temperature_k=300,
        damping=10.0,
        device=torch.device('cpu'),
        dtype=torch.float64,
    )
    thermostat.set_masses(masses)
    momenta = torch.full((2, 3), 0.1, dtype=torch.float64)
    updated = thermostat.integrate(momenta, 0.01)
    assert updated.dtype == torch.float64
    assert torch.isfinite(updated).all()


@pytest.mark.parametrize('integrator_name', ['berendsen', 'nose_hoover_chain'])
def test_gpu_integrators_match_one_ase_step_on_cpu(integrator_name: str) -> None:
    positions_np = np.array([[0.1, 0.2, 0.3], [1.1, 0.7, 0.4]])
    momenta_np = np.array([[0.22, -0.13, 0.31], [-0.19, 0.17, -0.28]])
    forces_np = np.array([[0.03, -0.02, 0.01], [-0.04, 0.02, -0.01]])
    masses_np = np.array([12.0, 16.0])
    atoms = Atoms('CO', positions=positions_np, masses=masses_np)
    atoms.set_momenta(momenta_np)
    atoms.calc = _ConstantForceCalculator(forces_np)

    positions = torch.tensor(positions_np, dtype=torch.float64)
    momenta = torch.tensor(momenta_np, dtype=torch.float64)
    forces = torch.tensor(forces_np, dtype=torch.float64)
    masses = torch.tensor(masses_np, dtype=torch.float64).unsqueeze(-1)

    def constant_force(pos: torch.Tensor) -> _ModelOutput:
        return _ModelOutput(
            energy=pos.new_zeros(()),
            forces=forces,
            stress=None,
        )

    if integrator_name == 'berendsen':
        ase_md = NVTBerendsen(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            taut=100.0 * units.fs,
            fixcm=True,
        )
        new_positions, new_momenta, _ = _berendsen_step(
            positions,
            momenta,
            forces,
            masses,
            dt=units.fs,
            target_temperature_k=300.0,
            tau=100.0 * units.fs,
            force_fn=constant_force,
        )
    else:
        ase_md = NoseHooverChainNVT(
            atoms,
            timestep=units.fs,
            temperature_K=300.0,
            tdamp=100.0 * units.fs,
        )
        thermostat = _NoseHooverChain(
            n_atoms=2,
            temperature_k=300.0,
            damping=100.0 * units.fs,
            device=torch.device('cpu'),
            dtype=torch.float64,
        )
        thermostat.set_masses(masses)
        new_positions, new_momenta, _ = _nhc_step(
            positions,
            momenta,
            forces,
            masses,
            dt=units.fs,
            thermostat=thermostat,
            force_fn=constant_force,
        )

    ase_md.run(1)
    np.testing.assert_allclose(
        new_positions.numpy(), atoms.positions, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        new_momenta.numpy(), atoms.get_momenta(), rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize('compute_stress', [False, True])
def test_force_output_computes_stress_only_when_requested(
    compute_stress: bool,
) -> None:
    module = ForceStressOutputFromEdge(compute_stress=compute_stress)
    module._is_batch_data = False
    edge_vec = torch.tensor(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad=True
    )
    graph = {
        key.ATOMIC_NUMBERS: torch.tensor([1, 1]),
        key.EDGE_VEC: edge_vec,
        key.EDGE_IDX: torch.tensor([[0, 1], [1, 0]]),
        key.PRED_TOTAL_ENERGY: edge_vec.square().sum(),
        key.CELL_VOLUME: torch.tensor(8.0),
    }
    output = module(graph)
    assert key.PRED_FORCE in output
    assert (key.PRED_STRESS in output) is compute_stress


def test_force_output_hot_path_has_no_explicit_host_transfer() -> None:
    sources = (
        inspect.getsource(ForceStressOutputFromEdge.forward),
        inspect.getsource(_SingleSystemPotential.__call__),
    )
    for source in sources:
        assert '.cpu()' not in source
        assert '.item()' not in source


def test_single_system_potential_reuses_graph_and_static_tensors() -> None:
    atomic_numbers = torch.tensor([29, 29], dtype=torch.long)
    model_positions = torch.empty((2, 3), dtype=torch.float32)
    model_cell = torch.eye(3, dtype=torch.float32) * 4.0
    cell_volume = torch.det(model_cell)
    num_atoms = torch.tensor(2, dtype=torch.long)
    graph = AtomGraphData(
        x=atomic_numbers,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        pos=model_positions,
        **{
            key.ATOMIC_NUMBERS: atomic_numbers,
            key.EDGE_VEC: torch.empty((0, 3)),
            key.CELL: model_cell,
            key.CELL_SHIFT: torch.empty((0, 3)),
            key.CELL_VOLUME: cell_volume,
            key.NUM_ATOMS: num_atoms,
            key.DATA_MODALITY: None,
            key.INFO: {},
        },
    )

    class _FakeModel:
        def __init__(self) -> None:
            self.graph_ids = []

        def __call__(self, actual_graph):
            self.graph_ids.append(id(actual_graph))
            actual_graph[key.PRED_TOTAL_ENERGY] = actual_graph[
                key.EDGE_VEC
            ].square().sum()
            actual_graph[key.PRED_FORCE] = torch.zeros((2, 3))
            return actual_graph

    potential = object.__new__(_SingleSystemPotential)
    potential.atomic_numbers = atomic_numbers
    potential.type_indices = atomic_numbers
    potential.compute_stress = False
    potential.cutoff = 3.0
    potential.cell_batch = (model_cell.to(torch.float64)).unsqueeze(0)
    potential.pbc_batch = torch.ones((1, 3), dtype=torch.bool)
    potential.system_idx = torch.zeros(2, dtype=torch.long)
    potential.model_positions = model_positions
    potential.model_cell = model_cell
    potential.cell_volume = cell_volume
    potential.num_atoms = num_atoms
    potential.graph = graph
    potential.model = _FakeModel()

    def neighbor_list(*_args):
        return (
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            torch.zeros(2, dtype=torch.long),
            torch.zeros((2, 3), dtype=torch.float64),
        )

    potential.neighbor_list_fn = neighbor_list
    static_ids = {
        name: id(potential.graph[name])
        for name in (
            key.ATOMIC_NUMBERS,
            key.CELL,
            key.CELL_VOLUME,
            key.NUM_ATOMS,
        )
    }
    potential(torch.zeros((2, 3), dtype=torch.float64))
    potential(torch.ones((2, 3), dtype=torch.float64))

    assert potential.model.graph_ids == [id(graph), id(graph)]
    assert static_ids == {
        name: id(potential.graph[name]) for name in static_ids
    }


def test_frame_omits_unrequested_stress() -> None:
    atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.7]])
    frame = _frame(
        atoms,
        positions=torch.tensor(atoms.positions, dtype=torch.float64),
        momenta=torch.zeros((2, 3), dtype=torch.float64),
        output=_ModelOutput(
            energy=torch.tensor(-1.0),
            forces=torch.zeros((2, 3)),
            stress=None,
        ),
        step=1,
    )
    assert 'stress' not in frame.calc.results


def test_opt1_rejects_wrong_backend_before_cuda() -> None:
    request = SimpleNamespace(
        model='sevennet',
        stage='opt1',
        backend='eager',
        config=SimpleNamespace(device='cuda:0'),
    )
    with pytest.raises(ValueError, match='gpu-resident'):
        run_md(request)


def test_opt1_rejects_wrong_owner_before_cuda() -> None:
    request = SimpleNamespace(
        model='other',
        stage='opt1',
        backend='gpu-resident',
    )
    with pytest.raises(ValueError, match='owns sevennet/opt1'):
        run_md(request)
