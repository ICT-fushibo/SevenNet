"""CPU contracts for SevenNet whole-step CUDA Graph MD."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import sevenn.md_route as public_route
from sevenn.md_stages.fixed_neighbor import (
    FixedShapeSevenNetNeighborBuilder,
    neighbor_capacity_from_probe,
)
from sevenn.md_stages.opt1 import _NoseHooverChain
from sevenn.md_stages.opt3 import (
    _guarded_uniform_capacity_from_total,
    _integrate_nhc_pure,
    _SevenNetWholeStepGraph,
    run_md,
)


def test_neighbor_capacity_uses_esen_cap_rounding() -> None:
    assert neighbor_capacity_from_probe(40, margin=0.10, slot_step=8) == 48
    assert neighbor_capacity_from_probe(48, margin=0.0, slot_step=8) == 56


def test_total_capacity_conversion_adds_aligned_guard_bucket() -> None:
    # ceil(2816 / 32) = 88, already aligned, plus one 8-neighbor guard.
    assert _guarded_uniform_capacity_from_total(2816, 32) == 96
    # ceil(100 / 3) = 34, align to 40, then add the guard => 48.
    assert _guarded_uniform_capacity_from_total(100, 3) == 48


def test_fixed_builder_preserves_real_edges_and_distributes_sinks() -> None:
    builder = FixedShapeSevenNetNeighborBuilder(
        num_atoms=2,
        cell=torch.eye(3) * 10,
        pbc=torch.zeros(3, dtype=torch.bool),
        cutoff=1.0,
        neighbors_per_atom=2,
        dummy_atoms=2,
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    edge_index, offsets = builder.build(positions)

    assert edge_index.shape == (2, 4)
    assert torch.equal(edge_index[:, 0], torch.tensor([0, 1]))
    assert torch.equal(edge_index[:, 2], torch.tensor([1, 0]))
    padding = torch.stack((edge_index[:, 1], edge_index[:, 3]))
    assert torch.equal(padding[:, 0], padding[:, 1])
    assert set(padding[:, 0].tolist()) == {2, 3}
    assert bool((offsets[[1, 3]].abs().sum(dim=1) > 0).all())


def test_fixed_builder_records_per_centre_overflow_without_host_branch() -> None:
    builder = FixedShapeSevenNetNeighborBuilder(
        num_atoms=3,
        cell=torch.eye(3) * 10,
        pbc=torch.zeros(3, dtype=torch.bool),
        cutoff=1.0,
        neighbors_per_atom=1,
        dummy_atoms=2,
    )
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]]
    )
    builder.build(positions, step=torch.tensor(7))
    stats = builder.stats()

    assert stats['fixed_builder_capacity_misses'] == 1
    assert stats['fixed_builder_first_overflow_step'] == 7
    assert stats['fixed_builder_max_overflow_required'] == 2
    assert stats['fixed_builder_max_overflow_capacity'] == 1


def test_nhc_pure_update_does_not_mutate_persistent_state() -> None:
    thermostat = _NoseHooverChain(
        n_atoms=2,
        temperature_k=1000,
        damping=2.0,
        device=torch.device('cpu'),
        dtype=torch.float64,
    )
    masses = torch.ones((2, 1), dtype=torch.float64)
    thermostat.set_masses(masses)
    momenta = torch.ones((2, 3), dtype=torch.float64) * 0.1
    eta = thermostat.eta.clone()
    p_eta = thermostat.p_eta.clone()

    new_momenta, new_eta, new_p_eta = _integrate_nhc_pure(
        momenta, eta, p_eta, thermostat, 0.01
    )

    assert torch.equal(thermostat.eta, torch.zeros_like(thermostat.eta))
    assert torch.equal(thermostat.p_eta, torch.zeros_like(thermostat.p_eta))
    assert bool(torch.isfinite(new_momenta).all())
    assert bool(torch.isfinite(new_eta).all())
    assert bool(torch.isfinite(new_p_eta).all())


def test_whole_step_body_contains_builder_model_and_state_update() -> None:
    source = inspect.getsource(_SevenNetWholeStepGraph._graph_body)
    assert 'write_geometry_' in source
    assert '_static_forward' in source
    assert 'self.positions.copy_' in source
    assert 'self.momenta.copy_' in source
    assert 'self.thermostat.eta.copy_' in source


def test_fixed_builder_hot_path_has_no_host_transfer() -> None:
    source = inspect.getsource(FixedShapeSevenNetNeighborBuilder.build)
    assert '.cpu()' not in source
    assert '.item()' not in source


def test_opt3_rejects_wrong_backend_before_cuda() -> None:
    request = SimpleNamespace(
        model='sevennet',
        stage='opt3',
        backend='model-only-cuda-graph',
        config=SimpleNamespace(device='cuda:0'),
    )
    with pytest.raises(ValueError, match='whole-step-cuda-graph'):
        run_md(request)


def test_opt3_rejects_opt1_backend_before_cuda() -> None:
    request = SimpleNamespace(
        model='sevennet',
        stage='opt3',
        backend='gpu-resident',
        config=SimpleNamespace(device='cuda:0'),
    )
    with pytest.raises(ValueError, match='whole-step-cuda-graph'):
        run_md(request)


def test_opt3_rejects_wrong_owner_before_cuda() -> None:
    request = SimpleNamespace(
        model='other',
        stage='opt3',
        backend='whole-step-cuda-graph',
    )
    with pytest.raises(ValueError, match='owns sevennet/opt3'):
        run_md(request)


def test_public_route_dispatches_opt3(monkeypatch) -> None:
    sentinel = object()
    call = {}

    def fake_dispatch(request, *, module_prefix):
        call['request'] = request
        call['module_prefix'] = module_prefix
        return sentinel

    monkeypatch.setattr(public_route, 'run_optimized_stage', fake_dispatch)
    request = SimpleNamespace(model='sevennet', stage='opt3')

    assert public_route.run_md(request) is sentinel
    assert call == {
        'request': request,
        'module_prefix': 'sevenn.md_stages',
    }
