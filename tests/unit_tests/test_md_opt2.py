"""Source-level and CPU contract tests for SevenNet model-only CUDA Graph."""

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from sevenn.md_stages.opt2 import (
    CUDAGraphCapacityError,
    _ModelOnlyCUDAGraphPotential,
    _RealAtomReduce,
    _maximum_neighbors_per_atom,
    edge_capacity_from_probe,
    run_md,
    staticize_edges_,
)


def test_edge_capacity_rounds_with_margin() -> None:
    assert edge_capacity_from_probe(100, margin=0.25, edge_step=64) == 128
    assert edge_capacity_from_probe(128, margin=0.0, edge_step=128) == 256


def test_probe_maximum_neighbors_uses_sevennet_centre_axis() -> None:
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2, 2], [1, 2, 0, 0, 1, 3]], dtype=torch.long
    )
    assert _maximum_neighbors_per_atom(edge_index, num_atoms=4) == 3


def test_staticize_edges_keeps_fixed_addresses_and_isolates_padding() -> None:
    static_index = torch.empty((2, 4), dtype=torch.long)
    static_vec = torch.empty((4, 3))
    static_shift = torch.empty((4, 3))
    addresses = (
        static_index.data_ptr(),
        static_vec.data_ptr(),
        static_shift.data_ptr(),
    )
    real_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    real_vec = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]])
    real_shift = torch.zeros((2, 3))

    count = staticize_edges_(
        static_index,
        static_vec,
        static_shift,
        real_index,
        real_vec,
        real_shift,
        dummy_index=2,
        padding_edge_vec=torch.tensor([4.5, 0.0, 0.0]),
    )

    assert count == 2
    assert addresses == (
        static_index.data_ptr(),
        static_vec.data_ptr(),
        static_shift.data_ptr(),
    )
    assert torch.equal(static_index[:, :2], real_index)
    assert torch.equal(static_index[:, 2:], torch.full((2, 2), 2))
    assert torch.equal(static_vec[2:], torch.tensor([[4.5, 0, 0]]).expand(2, 3))


def test_staticize_edges_fails_closed_on_capacity_overflow() -> None:
    with pytest.raises(CUDAGraphCapacityError, match='required=3, capacity=2'):
        staticize_edges_(
            torch.empty((2, 2), dtype=torch.long),
            torch.empty((2, 3)),
            torch.empty((2, 3)),
            torch.zeros((2, 3), dtype=torch.long),
            torch.zeros((3, 3)),
            torch.zeros((3, 3)),
            dummy_index=2,
            padding_edge_vec=torch.tensor([4.5, 0.0, 0.0]),
        )


def test_real_atom_reduce_excludes_dummy_energy() -> None:
    reducer = _RealAtomReduce(
        2,
        data_key_in='atomic',
        data_key_out='total',
        constant=1.0,
    )
    graph = {'atomic': torch.tensor([[1.0], [2.0], [1000.0]])}
    reducer(graph)
    assert graph['total'].item() == 3.0


def test_model_only_replay_reuses_static_outputs() -> None:
    class _FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    potential = object.__new__(_ModelOnlyCUDAGraphPotential)
    potential.captured = True
    potential.cuda_graph = _FakeGraph()
    potential.edge_capacity = 4
    potential.production_calls = 0
    potential.production_replays = 0
    potential.capacity_misses = 0
    potential.total_replays = 0
    potential.min_real_edges = None
    potential.max_real_edges = None
    potential.initial_max_neighbors_per_atom = None
    potential.max_neighbors_per_atom = None
    potential.track_neighbor_capacity = False
    potential.static_forces = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    potential.static_energy = torch.tensor(-3.0)
    potential._build_real_inputs = lambda _positions: (
        torch.tensor([[0, 1], [1, 0]]),
        torch.zeros((2, 3)),
        torch.zeros((2, 3)),
    )
    potential._staticize = lambda *_args: 2
    potential._input_addresses = lambda: (11, 12, 13, 14, 15)
    potential._capture_input_addresses = (11, 12, 13, 14, 15)
    potential.profiler = SimpleNamespace(
        phase=lambda _name: contextlib.nullcontext()
    )
    force_address = potential.static_forces.data_ptr()
    energy_address = potential.static_energy.data_ptr()

    first = potential(torch.zeros((2, 3)))
    second = potential(torch.ones((2, 3)))

    assert potential.cuda_graph.replays == 2
    assert potential.production_replays == 2
    assert potential.static_forces.data_ptr() == force_address
    assert potential.static_energy.data_ptr() == energy_address
    assert torch.equal(first.forces, second.forces)


def test_model_only_hot_path_has_no_host_transfer_or_recapture() -> None:
    source = inspect.getsource(_ModelOnlyCUDAGraphPotential.__call__)
    assert '.cpu()' not in source
    assert '.item()' not in source
    assert '.capture(' not in source
    assert '.replay()' in source


def test_model_only_path_asserts_fixed_input_addresses() -> None:
    source = inspect.getsource(_ModelOnlyCUDAGraphPotential.__call__)
    capture_source = inspect.getsource(_ModelOnlyCUDAGraphPotential.capture)

    assert '_input_addresses()' in source
    assert '_capture_input_addresses' in capture_source


def test_opt2_rejects_wrong_backend_before_cuda() -> None:
    request = SimpleNamespace(
        model='sevennet',
        stage='opt2',
        backend='eager',
        config=SimpleNamespace(device='cuda:0'),
    )
    with pytest.raises(ValueError, match='model-only-cuda-graph'):
        run_md(request)


def test_opt2_rejects_wrong_owner_before_cuda() -> None:
    request = SimpleNamespace(
        model='other',
        stage='opt2',
        backend='model-only-cuda-graph',
    )
    with pytest.raises(ValueError, match='owns sevennet/opt2'):
        run_md(request)
