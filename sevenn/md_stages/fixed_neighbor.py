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

from md_benchmark.neighbor_utils import (
    displacement_exceeds_skin,
    make_slot_layout,
    normalize_neighbor_capacities,
    select_skin_candidates,
)


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
        neighbor_capacities: list[int] | Tensor | None = None,
        dummy_atoms: int,
        verlet_skin: float = 0.0,
        verlet_candidate_capacity: int | None = None,
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
        capacities = normalize_neighbor_capacities(
            neighbor_capacities,
            num_atoms=num_atoms,
            default=int(neighbors_per_atom),
        )
        (
            self.slot_centres,
            self.slot_ranks,
            self.selection_indices,
            self.neighbors_per_atom,
            self.edge_capacity,
        ) = make_slot_layout(capacities, device=cell.device)
        self.neighbor_capacities = torch.as_tensor(
            capacities, dtype=torch.long, device=cell.device
        )
        if verlet_skin < 0:
            raise ValueError('verlet_skin must be non-negative')
        self.verlet_skin = float(verlet_skin)
        self.verlet_candidate_capacity = verlet_candidate_capacity
        self.skin_candidate_ids: Tensor | None = None
        self.skin_candidate_mask: Tensor | None = None
        self.skin_reference_positions: Tensor | None = None
        self.skin_misses = torch.zeros((), device=cell.device, dtype=torch.long)
        self.skin_rebuilds = 0
        self.dummy_atoms = int(dummy_atoms)
        self.max_neighbors = int(max_neighbors)
        self.degeneracy_tolerance = float(degeneracy_tolerance)
        self.device = cell.device
        self.position_dtype = cell.dtype
        self.pbc = pbc.detach().to(device=cell.device, dtype=torch.bool).reshape(3)
        self.repetitions = _pbc_repetitions(cell, cutoff + self.verlet_skin, pbc)
        self.cell = cell.detach().reshape(3, 3).contiguous()
        self.inverse_cell = torch.linalg.inv(self.cell).contiguous()

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

        # ``slot_centres`` and ``selection_indices`` are setup-only metadata.
        # The latter gathers variable-length per-centre CAP rows from the
        # temporary [N, max(capacity)] top-k matrix into the fixed output axis.
        slot_ranks = self.slot_ranks
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
        self.maximum_neighbors_by_atom = torch.zeros(
            self.num_atoms, device=self.device, dtype=torch.long
        )

    @torch.no_grad()
    def initialize_skin(self, positions: Tensor) -> None:
        if self.verlet_skin <= 0:
            return
        requested = self.verlet_candidate_capacity
        slots = max(self.neighbors_per_atom, int(requested)) if requested is not None else max(
            self.neighbors_per_atom * 2, self.neighbors_per_atom + 32
        )
        slots = min(slots, self.candidates_per_atom)
        selected, counts, selected_valid = select_skin_candidates(
            positions,
            self.candidate_neighbors,
            self.candidate_cell_offsets,
            self.cell,
            cutoff=self.cutoff + self.verlet_skin,
            slots_per_atom=slots,
            min_distance_sqr=1.0e-4,
        )
        torch._assert_async(
            (counts <= slots).all(),
            'SevenNet Opt3 Verlet candidate capacity is smaller than the '
            'cutoff+skin candidate count',
        )
        if self.skin_candidate_ids is None:
            self.skin_candidate_ids = selected
            self.skin_candidate_mask = selected_valid
            self.skin_reference_positions = positions.detach().clone()
        else:
            if self.skin_candidate_ids.shape != selected.shape:
                raise RuntimeError("Verlet candidate shape changed during rebuild")
            self.skin_candidate_ids.copy_(selected)
            assert self.skin_candidate_mask is not None
            self.skin_candidate_mask.copy_(selected_valid)
            assert self.skin_reference_positions is not None
            self.skin_reference_positions.copy_(positions)
        self.verlet_candidate_capacity = slots
        self.skin_rebuilds += 1

    @torch.no_grad()
    def reset_stats(self) -> None:
        self.build_calls.zero_()
        self.capacity_misses.zero_()
        self.first_overflow_step.fill_(-1)
        self.minimum_real_edges.fill_(self.edge_capacity)
        self.maximum_real_edges.zero_()
        self.maximum_neighbors_seen.zero_()
        self.maximum_neighbors_by_atom.zero_()
        self.skin_misses.zero_()
        self.skin_rebuilds = 0

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
        if self.skin_candidate_ids is not None:
            assert self.skin_reference_positions is not None
            assert self.skin_candidate_mask is not None
            skin_miss = displacement_exceeds_skin(
                positions,
                self.skin_reference_positions,
                self.verlet_skin,
                self.cell,
                self.pbc,
                self.inverse_cell,
            )
            self.skin_misses.add_(skin_miss.to(torch.long))
            torch._assert_async(
                ~skin_miss,
                'SevenNet Opt3 Verlet skin exhausted; rebuild the candidate list',
            )
            cached = self.skin_candidate_ids.reshape(-1)
            candidate_neighbors = self.candidate_neighbors.index_select(
                0, cached
            ).reshape(self.num_atoms, -1)
            candidate_cell_offsets = self.candidate_cell_offsets.index_select(
                0, cached
            ).reshape(self.num_atoms, -1, 3)
            candidate_width = int(candidate_neighbors.shape[1])
            candidates = torch.arange(
                candidate_width, device=self.device, dtype=torch.long
            ).reshape(1, -1).expand(self.num_atoms, -1)
            shifted_neighbors = positions.index_select(
                0, candidate_neighbors.reshape(-1)
            ).reshape(self.num_atoms, candidate_width, 3) + torch.mm(
                candidate_cell_offsets.reshape(-1, 3).to(dtype=positions.dtype),
                self.cell.to(dtype=positions.dtype),
            ).reshape(self.num_atoms, candidate_width, 3)
            delta = shifted_neighbors - positions.unsqueeze(1)
            valid_candidates = self.skin_candidate_mask
        else:
            candidate_neighbors = self.candidate_neighbors
            candidate_cell_offsets = self.candidate_cell_offsets
            candidates = self.candidate_ids.expand(self.num_atoms, -1)
            shifted_neighbors = (
                positions.index_select(0, candidate_neighbors)
                + torch.mm(
                    candidate_cell_offsets.to(dtype=positions.dtype),
                    self.cell.to(dtype=positions.dtype),
                )
            )
            delta = shifted_neighbors.unsqueeze(0) - positions.unsqueeze(1)
            valid_candidates = torch.ones_like(candidates, dtype=torch.bool)
        distance_sqr = delta.square().sum(dim=-1)
        cutoff_sqr = self.cutoff * self.cutoff
        valid = (
            valid_candidates
            & (distance_sqr <= cutoff_sqr)
            & (distance_sqr > 0.0001)
        )
        raw_counts = valid.sum(dim=1)

        # Match torch-sim's maximum-neighbor behavior, including degenerate
        # neighbors at the boundary of the selected maximum.
        candidate_width = int(valid.shape[1])
        selection_k = min(self.max_neighbors + 1, candidate_width)
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
        if candidate_width > self.max_neighbors:
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

        ordered = torch.where(
            included,
            candidates,
            torch.full_like(candidates, candidate_width),
        )
        select_k = min(self.neighbors_per_atom, candidate_width)
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
                value=candidate_width,
            )
        safe_selected = selected.clamp_max(candidate_width - 1)
        if self.skin_candidate_ids is not None:
            selected_neighbors = torch.gather(
                candidate_neighbors, 1, safe_selected
            )
            selected_offsets = torch.gather(
                candidate_cell_offsets,
                1,
                safe_selected.unsqueeze(-1).expand(-1, -1, 3),
            )
        else:
            selected_neighbors = self.candidate_neighbors.index_select(
                0, safe_selected.reshape(-1)
            ).reshape(self.num_atoms, -1)
            selected_offsets = self.candidate_cell_offsets.index_select(
                0, safe_selected.reshape(-1)
            ).reshape(self.num_atoms, -1, 3)
        # The temporary top-k matrix is [N, max(capacity)].  Keep only each
        # centre's configured prefix and flatten it into the fixed edge axis.
        flat_selected = selected.reshape(-1).index_select(0, self.selection_indices)
        flat_valid = flat_selected < candidate_width
        neighbors = selected_neighbors.reshape(-1).index_select(
            0, self.selection_indices
        )
        offsets = selected_offsets.reshape(-1, 3).index_select(
            0, self.selection_indices
        )

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
        excess_by_atom = torch.clamp_min(
            included_counts - self.neighbor_capacities, 0
        )
        maximum_excess = excess_by_atom.max()
        overflow = maximum_excess > 0
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
        self.maximum_neighbors_by_atom.copy_(
            torch.maximum(self.maximum_neighbors_by_atom, included_counts)
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
            'fixed_builder_neighbor_capacities': self.neighbor_capacities.detach()
            .to(device='cpu')
            .tolist(),
            'fixed_builder_capacity_policy': (
                'per-atom-cap' if self.neighbor_capacities.unique().numel() > 1
                else 'uniform-cap'
            ),
            'fixed_builder_min_real_edges': minimum,
            'fixed_builder_max_real_edges': maximum,
            'fixed_builder_max_padding_fraction': (
                None
                if minimum is None
                else (self.edge_capacity - minimum) / self.edge_capacity
            ),
            'fixed_builder_max_neighbors_seen': maximum_neighbors,
            'fixed_builder_maximum_neighbors_by_atom': self.maximum_neighbors_by_atom.detach()
            .to(device='cpu')
            .tolist(),
            'fixed_builder_max_overflow_required': maximum_neighbors,
            'fixed_builder_max_overflow_capacity': self.neighbors_per_atom,
            'fixed_builder_verlet_skin': self.verlet_skin,
            'fixed_builder_verlet_candidate_capacity': self.verlet_candidate_capacity,
            'fixed_builder_verlet_skin_misses': int(self.skin_misses.item()),
            'fixed_builder_verlet_rebuilds': self.skin_rebuilds,
            'fixed_builder_verlet_enabled': self.skin_candidate_ids is not None,
            'fixed_builder_active_candidate_slots': self.num_atoms
            * (
                int(self.skin_candidate_ids.shape[1])
                if self.skin_candidate_ids is not None
                else self.candidates_per_atom
            ),
            'fixed_builder_candidate_reduction_fraction': (
                0.0
                if self.skin_candidate_ids is None
                else 1.0
                - int(self.skin_candidate_ids.shape[1]) / self.candidates_per_atom
            ),
            'fixed_builder_candidate_universe_size': (
                self.num_atoms * self.candidates_per_atom
            ),
            'fixed_builder_candidates_per_atom': self.candidates_per_atom,
            'fixed_builder_num_pbc_cells': self.num_cells,
            'fixed_builder_pbc_repetitions': list(self.repetitions),
            'fixed_builder_max_neighbors': self.max_neighbors,
            'fixed_builder_degeneracy_tolerance': self.degeneracy_tolerance,
        }
