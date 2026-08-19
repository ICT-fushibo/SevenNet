"""Lightweight contract tests for the GPU-resident Opt1 stage."""

from types import SimpleNamespace

import pytest
import torch
from ase import Atoms

from sevenn.md_stages.opt1 import (
    _ModelOutput,
    _NoseHooverChain,
    _berendsen_step,
    _initial_momenta,
    run_md,
)


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

