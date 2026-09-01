"""Capture-safe fixed-shape PBC neighbor construction for SevenNet MD.

The production ``torchsim_nl`` path returns ragged tensors and therefore
cannot be placed inside one whole-step CUDA Graph.  This builder enumerates a
fixed candidate universe once and writes a fixed number of neighbor slots per
real atom.  Unused slots are distributed over a bank of dummy sink atoms.

The edge convention is SevenNet's convention: ``edge_index[0]`` is the centre
atom, ``edge_index[1]`` is the neighbor image, and ``cell_offsets`` is added to
the neighbor when constructing ``pos[neighbor] - pos[centre] + shift``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def neighbor_capacity_from_probe(
    maximum_neighbors: int,
    *,
    margin: float = 0.10,
    slot_step: int = 8,
) -> int:
    """Apply eSEN CAP headroom and round the per-atom capacity upward."""

    if maximum_neighbors < 1:
        raise ValueError('maximum_neighbors must be positive')
    if margin < 0:
        raise ValueError('margin must be non-negative')
    if slot_step < 1:
        raise ValueError('slot_step must be positive')
    required = max(
        maximum_neighbors + 1,
        math.ceil(maximum_neighbors * (1.0 + margin)),
    )
    return int(math.ceil(required / slot_step) * slot_step)


def _pbc_repetitions(
    cell: Tensor,
    cutoff: float,
    pbc: Tensor,
) -> tuple[int, int, int]:
    """Compute the periodic-image range from reciprocal plane distances."""

    cell64 = cell.detach().to(device='cpu', dtype=torch.float64).reshape(3, 3)
    pbc_cpu = pbc.detach().to(device='cpu', dtype=torch.bool).reshape(3)
    cross_a2a3 = torch.cross(cell64[1], cell64[2], dim=0)
    volume = torch.dot(cell64[0], cross_a2a3)
    if not bool(torch.isfinite(volume)) or float(volume.abs()) == 0.0:
        raise ValueError('Cannot enumerate PBC images for a singular cell')
    reciprocal = (
        cross_a2a3,
        torch.cross(cell64[2], cell64[0], dim=0),
        torch.cross(cell64[0], cell64[1], dim=0),
    )
    repetitions = []
    for axis in range(3):
        if bool(pbc_cpu[axis]):
            inverse_distance = torch.linalg.vector_norm(
                reciprocal[axis] / volume
            )
            repetitions.append(
                int(torch.ceil(cutoff * inverse_distance).item())
            )
        else:
            repetitions.append(0)
    return tuple(repetitions)  # type: ignore[return-value]


class FixedShapeSevenNetNeighborBuilder:
    """Build a uniform-CAP SevenNet graph using fixed-address output tensors."""

    def __init__(
        self,
        *,
        num_atoms: int,
        cell: Tensor,
        pbc: Tensor,
        cutoff: float,
        neighbors_per_atom: int,
        dummy_atoms: int,
        max_neighbors: int = 300,
        degeneracy_tolerance: float = 0.01,
        output_edge_index: Tensor | None = None,
        output_cell_offsets: Tensor | None = None,
    ) -> None:
        if num_atoms < 1:
            raise ValueError('num_atoms must be positive')
        if cutoff <= 0:
            raise ValueError('cutoff must be positive')
        if neighbors_per_atom < 1:
            raise ValueError('neighbors_per_atom must be positive')
        if dummy_atoms < 1:
            raise ValueError('dummy_atoms must be positive')
        if max_neighbors < 1:
            raise ValueError('max_neighbors must be positive')
        if degeneracy_tolerance < 0:
            raise ValueError('degeneracy_tolerance must be non-negative')

        self.num_atoms = int(num_atoms)
        self.cutoff = float(cutoff)
        self.neighbors_per_atom = int(neighbors_per_atom)
        self.dummy_atoms = int(dummy_atoms)
        self.max_neighbors = int(max_neighbors)
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.edge_capacity = self.num_atoms * self.neighbors_per_atom
        self.repetitions = _pbc_repetitions(cell, cutoff, pbc)
        self.cell = cell.detach().reshape(3, 3)

        axes = [
            torch.arange(
                -repetition,
                repetition + 1,
                device=self.device,
                dtype=self.position_dtype,
            )
            for repetition in self.repetitions
        ]
        self.unit_cell_offsets = torch.cartesian_prod(*axes).reshape(-1, 3)
        self.num_cells = int(self.unit_cell_offsets.shape[0])
        self.candidates_per_atom = self.num_atoms * self.num_cells
        self.candidate_neighbors = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.num_cells)
        self.candidate_cell_offsets = self.unit_cell_offsets.repeat(
            self.num_atoms, 1
        )
        self.candidate_ids = torch.arange(
            self.candidates_per_atom, device=self.device, dtype=torch.long
        ).reshape(1, -1)

        if output_edge_index is None:
            output_edge_index = torch.empty(
                (2, self.edge_capacity),
                device=self.device,
                dtype=torch.long,
            )
        if output_cell_offsets is None:
            output_cell_offsets = torch.empty(
                (self.edge_capacity, 3),
                device=self.device,
                dtype=torch.float32,
            )
        if output_edge_index.shape != (2, self.edge_capacity):
            raise ValueError('output_edge_index has the wrong shape')
        if output_cell_offsets.shape != (self.edge_capacity, 3):
            raise ValueError('output_cell_offsets has the wrong shape')
        self.edge_index = output_edge_index
        self.cell_offsets = output_cell_offsets

        slots = torch.arange(
            self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.slot_centres = torch.arange(
            self.num_atoms, device=self.device, dtype=torch.long
        ).repeat_interleave(self.neighbors_per_atom)
        slot_ranks = slots.remainder(self.neighbors_per_atom)
        self.dummy_sinks = (
            (self.slot_centres + slot_ranks).remainder(self.dummy_atoms)
            + self.num_atoms
        )

        cell_norms = torch.linalg.vector_norm(self.cell, dim=1)
        far_axis = int(torch.argmax(cell_norms).item())
        axis_norm = float(cell_norms[far_axis].item())
        if not math.isfinite(axis_norm) or axis_norm <= 0:
            raise ValueError('Cannot construct dummy shift from invalid cell')
        far_shift = max(2, math.ceil((self.cutoff + 1.0) / axis_norm) + 1)
        self.padding_cell_offsets = self.cell_offsets.new_zeros(
            self.edge_capacity, 3
        )
        self.padding_cell_offsets[:, far_axis] = far_shift

        # Capture-safe diagnostics.  Host reads happen only after production.
        self.build_calls = torch.zeros((), device=self.device, dtype=torch.long)
        self.capacity_misses = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.first_overflow_step = torch.full(
            (), -1, device=self.device, dtype=torch.long
        )
        self.minimum_real_edges = torch.full(
            (), self.edge_capacity, device=self.device, dtype=torch.long
        )
        self.maximum_real_edges = torch.zeros(
            (), device=self.device, dtype=torch.long
        )
        self.maximum_neighbors_seen = torch.zeros(
            (), device=self.device, dtype=torch.long
        )

    @torch.no_grad()
    def reset_stats(self) -> None:
        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_neighbors_seen.zero_()

    @torch.no_grad()
    def build(
        self,
        positions: Tensor,
        *,
        step: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Write one fixed-capacity graph without host synchronization."""

        if positions.shape != (self.num_atoms, 3):
            raise ValueError(
                f'Expected positions {(self.num_atoms, 3)}, got '
                f'{tuple(positions.shape)}'
            )
        shifted_neighbors = (
            positions.index_select(0, self.candidate_neighbors)
            + torch.mm(
                self.candidate_cell_offsets.to(dtype=positions.dtype),
                self.cell.to(dtype=positions.dtype),
            )
        )
        delta = shifted_neighbors.unsqueeze(0) - positions.unsqueeze(1)
        distance_sqr = delta.square().sum(dim=-1)
        cutoff_sqr = self.cutoff * self.cutoff
        valid = (distance_sqr <= cutoff_sqr) & (distance_sqr > 0.0001)
        raw_counts = valid.sum(dim=1)

        # Match torch-sim's maximum-neighbor behavior, including degenerate
        # neighbors at the boundary of the selected maximum.
        selection_k = min(self.max_neighbors + 1, self.candidates_per_atom)
        nearest = torch.topk(
            torch.where(
                valid,
                distance_sqr,
                torch.full_like(distance_sqr, torch.inf),
            ),
            k=selection_k,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        if self.candidates_per_atom > self.max_neighbors:
            limited_cutoff = nearest[:, self.max_neighbors]
            limited_cutoff = limited_cutoff + self.degeneracy_tolerance
        else:
            limited_cutoff = torch.full_like(
                raw_counts, cutoff_sqr, dtype=distance_sqr.dtype
            )
        effective_cutoff = torch.where(
            raw_counts > self.max_neighbors,
            limited_cutoff,
            torch.full_like(limited_cutoff, cutoff_sqr),
        )
        included = valid & (distance_sqr <= effective_cutoff.unsqueeze(1))
        included_counts = included.sum(dim=1)

        candidates = self.candidate_ids.expand(self.num_atoms, -1)
        ordered = torch.where(
            included,
            candidates,
            torch.full_like(candidates, self.candidates_per_atom),
        )
        select_k = min(self.neighbors_per_atom, self.candidates_per_atom)
        selected = torch.topk(
            ordered,
            k=select_k,
            dim=1,
            largest=False,
            sorted=True,
        ).values
        if select_k < self.neighbors_per_atom:
            selected = torch.nn.functional.pad(
                selected,
                (0, self.neighbors_per_atom - select_k),
                value=self.candidates_per_atom,
            )
        flat_selected = selected.reshape(-1)
        flat_valid = flat_selected < self.candidates_per_atom
        safe_selected = flat_selected.clamp_max(self.candidates_per_atom - 1)
        neighbors = self.candidate_neighbors.index_select(0, safe_selected)
        offsets = self.candidate_cell_offsets.index_select(0, safe_selected)

        # SevenNet direction: centre -> neighbor.  Padding never touches a
        # real atom and rotates across sinks to avoid one contended reduction.
        self.edge_index[0].copy_(
            torch.where(flat_valid, self.slot_centres, self.dummy_sinks)
        )
        self.edge_index[1].copy_(
            torch.where(flat_valid, neighbors, self.dummy_sinks)
        )
        self.cell_offsets.copy_(
            torch.where(
                flat_valid.unsqueeze(1),
                offsets.to(dtype=self.cell_offsets.dtype),
                self.padding_cell_offsets,
            )
        )

        real_edges = flat_valid.sum()
        maximum_neighbors = included_counts.max()
        overflow = maximum_neighbors > self.neighbors_per_atom
        call_step = self.build_calls if step is None else step
        self.minimum_real_edges.copy_(
            torch.minimum(self.minimum_real_edges, real_edges)
        )
        self.maximum_real_edges.copy_(
            torch.maximum(self.maximum_real_edges, real_edges)
        )
        self.maximum_neighbors_seen.copy_(
            torch.maximum(self.maximum_neighbors_seen, maximum_neighbors)
        )
        self.capacity_misses.add_(overflow.to(torch.long))
        first = (self.first_overflow_step < 0) & overflow
        self.first_overflow_step.copy_(
            torch.where(first, call_step, self.first_overflow_step)
        )
        self.build_calls.add_(1)
        return self.edge_index, self.cell_offsets

    def stats(self) -> dict[str, Any]:
        """Synchronize once after production and expose capacity telemetry."""

        calls = int(self.build_calls.item())
        minimum = int(self.minimum_real_edges.item()) if calls else None
        maximum = int(self.maximum_real_edges.item()) if calls else None
        misses = int(self.capacity_misses.item())
        first_overflow = int(self.first_overflow_step.item())
        maximum_neighbors = int(self.maximum_neighbors_seen.item())
        return {
            'fixed_builder_build_calls': calls,
            'fixed_builder_capacity_misses': misses,
            'fixed_builder_first_overflow_step': (
                first_overflow if first_overflow >= 0 else None
            ),
            'fixed_builder_edge_capacity': self.edge_capacity,
            'fixed_builder_neighbors_per_atom': self.neighbors_per_atom,
            'fixed_builder_capacity_policy': 'uniform-cap',
            'fixed_builder_min_real_edges': minimum,
            'fixed_builder_max_real_edges': maximum,
            'fixed_builder_max_padding_fraction': (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            'fixed_builder_max_neighbors_seen': maximum_neighbors,
            'fixed_builder_max_overflow_required': maximum_neighbors,
            'fixed_builder_max_overflow_capacity': self.neighbors_per_atom,
            'fixed_builder_candidate_universe_size': (
                self.num_atoms * self.candidates_per_atom
            ),
            'fixed_builder_candidates_per_atom': self.candidates_per_atom,
            'fixed_builder_num_pbc_cells': self.num_cells,
            'fixed_builder_pbc_repetitions': list(self.repetitions),
            'fixed_builder_max_neighbors': self.max_neighbors,
            'fixed_builder_degeneracy_tolerance': self.degeneracy_tolerance,
        }
