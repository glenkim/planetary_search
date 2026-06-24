#!/usr/bin/env python3
"""Search 3-planetary, 5-friction-element transmission kinematics.

The model follows the lever equation for each simple planetary gearset:

    omega_s + rho * omega_r - (1 + rho) * omega_c = 0

Permanent connections collapse the nine planetary members into shared
components. Two applied friction elements plus the fixed input speed define
each candidate state. Standing ratios are always derived from feasible integer
ring/sun tooth counts. Topology generation remains on the CPU; PyTorch CUDA can
batch the numeric tooth-triple solves and score reduction on NVIDIA GPUs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Iterator, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by environments without NumPy.
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:  # pragma: no cover - exercised by environments without PyTorch.
    torch = None  # type: ignore[assignment]

MEMBERS = ("S", "C", "R")
MEMBER_ORDER = {"S": 0, "C": 1, "R": 2}
GEARSET_COUNT = 3
ELEMENT_COUNT = 5
APPLIED_ELEMENT_COUNT = 2
DEFAULT_TARGETS = (3.25, 2.23, 1.61, 1.24, 1.0, 0.63, -2.95)
SUN_TEETH_BOUNDS = (14, 60)
PLANET_TEETH_BOUNDS = (14, 50)
RING_TEETH_BOUNDS = (42, 150)
GEARSET_CARRIER_PLANETS = (3, 3, 4)
STATE_COMBINATIONS = tuple(
    itertools.combinations(range(ELEMENT_COUNT), APPLIED_ELEMENT_COUNT)
)
STATE_MASKS = tuple(
    sum(1 << element_index for element_index in state)
    for state in STATE_COMBINATIONS
)
SYSTEM_ROW_COUNT = GEARSET_COUNT + 1 + APPLIED_ELEMENT_COUNT
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class Node:
    gearset: int
    member: str

    @property
    def label(self) -> str:
        return f"{self.member}{self.gearset + 1}"


NODES = tuple(Node(gearset, member) for gearset in range(GEARSET_COUNT) for member in MEMBERS)
NODE_INDEX = {node: index for index, node in enumerate(NODES)}
NODE_BY_LABEL = {node.label: node for node in NODES}


@dataclass(frozen=True)
class Topology:
    permanent_edges: tuple[tuple[Node, Node], ...]
    components: tuple[tuple[Node, ...], ...]
    component_by_node: tuple[int, ...]
    input_component: int
    output_component: int

    def component_of(self, node: Node) -> int:
        return self.component_by_node[NODE_INDEX[node]]

    def component_label(self, component: int) -> str:
        return "/".join(node.label for node in self.components[component])

    def permanent_groups(self) -> tuple[tuple[Node, ...], ...]:
        return tuple(component for component in self.components if len(component) > 1)


@dataclass(frozen=True)
class FrictionElement:
    kind: str
    a: int
    b: int | None = None

    def label(self, topology: Topology) -> str:
        if self.kind == "brake":
            return f"brake {topology.component_label(self.a)}"

        assert self.b is not None
        return f"clutch {topology.component_label(self.a)} <-> {topology.component_label(self.b)}"


@dataclass(frozen=True)
class StateResult:
    state: tuple[int, int]
    ratio: float
    output_speed: float
    velocities: tuple[float, ...]


@dataclass(frozen=True)
class GearsetTeeth:
    sun: int
    ring: int
    planet: int
    carrier_planets: int

    @property
    def rho(self) -> float:
        return self.ring / self.sun


@dataclass(frozen=True)
class GearsetFeasibility:
    concentricity_pass: bool
    assembly_pass: bool
    undercutting_pass: bool
    is_feasible: bool
    message: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "concentricity_pass": self.concentricity_pass,
            "assembly_pass": self.assembly_pass,
            "undercutting_pass": self.undercutting_pass,
            "is_feasible": self.is_feasible,
            "message": list(self.message),
        }


@dataclass(frozen=True)
class LayoutResult:
    score: float
    forward_mse: float
    shift_penalty: float
    rhos: tuple[float, float, float]
    teeth: tuple[GearsetTeeth, GearsetTeeth, GearsetTeeth]
    topology: Topology
    elements: tuple[FrictionElement, ...]
    forward_sequence: tuple[StateResult, ...]
    reverse_state: StateResult | None
    reverse_error: float | None
    double_shift_indices: tuple[int, ...]
    transition_counts: tuple[int, ...]
    valid_state_count: int


@dataclass(frozen=True)
class LayoutTemplate:
    topology: Topology
    elements: tuple[FrictionElement, ...]
    component_count: int
    constant_rows: tuple[tuple[tuple[float, ...], ...], ...]
    rho_rows: tuple[tuple[float, ...], ...]
    rhs_rows: tuple[tuple[float, ...], ...]
    state_masks: tuple[int, ...]
    transition_counts: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TorchLayoutTensors:
    constant_rows: Any
    rho_rows: Any
    rhs_rows: Any
    state_masks: Any
    transition_counts: Any


@dataclass(frozen=True)
class ToothOptionTensors:
    sun: Any
    ring: Any
    planet: Any
    carrier_planets: Any
    rho: Any


@dataclass(frozen=True)
class TorchScoringTensors:
    forward_targets: Any
    forward_position_combinations: Any
    reverse_target: Any


@dataclass(frozen=True)
class TorchRefineCandidatePool:
    scores: Any
    flat_indices: Any
    capacity: int


@dataclass(frozen=True)
class GpuTuneStats:
    tooth_triples: int = 0
    batches: int = 0
    refined_candidates: int = 0


@dataclass(frozen=True)
class CheckpointState:
    topology_index: int
    element_index: int
    stats: dict[str, Any]
    best_results: list[LayoutResult]


class UnionFind:
    def __init__(self, items: Iterable[Node]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: Node) -> Node:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: Node, right: Node) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def node_sort_key(node: Node) -> tuple[int, int]:
    return (node.gearset, MEMBER_ORDER[node.member])


def node_for(gearset: int, member: str) -> Node:
    return Node(gearset, member)


def parse_float_list(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_link_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    invalid = [count for count in counts if count not in (3, 4)]
    if invalid:
        raise argparse.ArgumentTypeError("permanent link counts must be 3, 4, or 3,4")
    return counts


def integer_range(bounds: Sequence[int]) -> range:
    return range(bounds[0], bounds[1] + 1)


def validate_planetary_gearset(
    z_s: int,
    z_p: int,
    z_r: int,
    n_p: int,
) -> GearsetFeasibility:
    values = {
        "Z_s": z_s,
        "Z_p": z_p,
        "Z_r": z_r,
        "N_p": n_p,
    }
    invalid_types = [name for name, value in values.items() if type(value) is not int]
    if invalid_types:
        raise TypeError("gearset feasibility inputs must be integers: " + ", ".join(invalid_types))

    non_positive = [name for name, value in values.items() if value <= 0]
    if non_positive:
        raise ValueError("gearset feasibility inputs must be positive: " + ", ".join(non_positive))

    if n_p < 3:
        raise ValueError("N_p must be at least 3")

    concentricity_pass = z_r == z_s + (2 * z_p)
    assembly_pass = (z_r + z_s) % n_p == 0
    undercutting_pass = z_s >= 14 and z_p >= 14

    messages: list[str] = []
    if not concentricity_pass:
        messages.append("concentricity failed: Z_r must equal Z_s + 2 * Z_p")
    if not assembly_pass:
        messages.append("assembly failed: Z_r + Z_s must be divisible by N_p")
    if not undercutting_pass:
        messages.append("undercutting failed: Z_s and Z_p must both be at least 14")
    if not messages:
        messages.append("gearset is mechanically feasible")

    return GearsetFeasibility(
        concentricity_pass=concentricity_pass,
        assembly_pass=assembly_pass,
        undercutting_pass=undercutting_pass,
        is_feasible=concentricity_pass and assembly_pass and undercutting_pass,
        message=tuple(messages),
    )


def tooth_sort_key(teeth: GearsetTeeth) -> tuple[float, int, int, int, int]:
    return (
        teeth.rho,
        teeth.carrier_planets,
        teeth.sun + teeth.ring + teeth.planet,
        teeth.sun,
        teeth.ring,
    )


def generate_tooth_options(
    rho_bounds: Sequence[float],
    include_equivalent_teeth: bool,
    carrier_planets: int,
) -> tuple[GearsetTeeth, ...]:
    options: list[GearsetTeeth] = []
    representative_by_rho: dict[tuple[Fraction, int], GearsetTeeth] = {}

    for sun in integer_range(SUN_TEETH_BOUNDS):
        for ring in integer_range(RING_TEETH_BOUNDS):
            if ring <= sun:
                continue

            tooth_delta = ring - sun
            if tooth_delta % 2 != 0:
                continue

            planet = tooth_delta // 2
            if planet < PLANET_TEETH_BOUNDS[0] or planet > PLANET_TEETH_BOUNDS[1]:
                continue

            rho = ring / sun
            if rho < rho_bounds[0] or rho > rho_bounds[1]:
                continue

            feasibility = validate_planetary_gearset(
                z_s=sun,
                z_p=planet,
                z_r=ring,
                n_p=carrier_planets,
            )
            if not feasibility.is_feasible:
                continue

            teeth = GearsetTeeth(
                sun=sun,
                ring=ring,
                planet=planet,
                carrier_planets=carrier_planets,
            )
            if include_equivalent_teeth:
                options.append(teeth)
                continue

            rho_key = (Fraction(ring, sun), carrier_planets)
            current = representative_by_rho.get(rho_key)
            if current is None or tooth_sort_key(teeth) < tooth_sort_key(current):
                representative_by_rho[rho_key] = teeth

    if not include_equivalent_teeth:
        options = list(representative_by_rho.values())

    return tuple(sorted(options, key=tooth_sort_key))


def permanent_pair_candidates(allow_internal: bool) -> tuple[tuple[Node, Node], ...]:
    pairs = []
    for left, right in itertools.combinations(NODES, 2):
        if not allow_internal and left.gearset == right.gearset:
            continue
        pairs.append((left, right))
    return tuple(pairs)


def gearset_graph_is_connected(edges: Sequence[tuple[Node, Node]]) -> bool:
    graph = {gearset: set() for gearset in range(GEARSET_COUNT)}
    for left, right in edges:
        if left.gearset == right.gearset:
            continue
        graph[left.gearset].add(right.gearset)
        graph[right.gearset].add(left.gearset)

    seen = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in graph[current] - seen:
            seen.add(neighbor)
            stack.append(neighbor)

    return len(seen) == GEARSET_COUNT


def build_components(
    edges: Sequence[tuple[Node, Node]],
) -> tuple[tuple[tuple[Node, ...], ...], tuple[int, ...]] | None:
    if not gearset_graph_is_connected(edges):
        return None

    union_find = UnionFind(NODES)
    for left, right in edges:
        if not union_find.union(left, right):
            return None

    groups_by_root: dict[Node, list[Node]] = {}
    for node in NODES:
        groups_by_root.setdefault(union_find.find(node), []).append(node)

    components = tuple(
        tuple(sorted(group, key=node_sort_key))
        for group in sorted(groups_by_root.values(), key=lambda group: node_sort_key(min(group, key=node_sort_key)))
    )

    component_by_node = [0] * len(NODES)
    for component_index, component in enumerate(components):
        for node in component:
            component_by_node[NODE_INDEX[node]] = component_index

    for gearset in range(GEARSET_COUNT):
        gearset_components = {
            component_by_node[NODE_INDEX[node_for(gearset, member)]]
            for member in MEMBERS
        }
        if len(gearset_components) != len(MEMBERS):
            return None

    return components, tuple(component_by_node)


def component_signature(components: Sequence[Sequence[Node]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(node.label for node in component) for component in components)


def generate_topologies(
    link_counts: Sequence[int],
    allow_internal: bool = False,
) -> Iterator[Topology]:
    pairs = permanent_pair_candidates(allow_internal)
    seen_partitions: set[tuple[tuple[str, ...], ...]] = set()

    for link_count in link_counts:
        for edges in itertools.combinations(pairs, link_count):
            built = build_components(edges)
            if built is None:
                continue

            components, component_by_node = built
            signature = component_signature(components)
            if signature in seen_partitions:
                continue
            seen_partitions.add(signature)

            for input_component in range(len(components)):
                for output_component in range(len(components)):
                    if input_component == output_component:
                        continue
                    yield Topology(
                        permanent_edges=tuple(edges),
                        components=components,
                        component_by_node=component_by_node,
                        input_component=input_component,
                        output_component=output_component,
                    )


def generate_element_sets(
    topology: Topology,
    include_output_brakes: bool = False,
) -> Iterator[tuple[FrictionElement, ...]]:
    component_indexes = range(len(topology.components))
    brake_components = [
        component
        for component in component_indexes
        if component != topology.input_component
        and (include_output_brakes or component != topology.output_component)
    ]

    candidates = [FrictionElement("brake", component) for component in brake_components]
    candidates.extend(
        FrictionElement("clutch", left, right)
        for left, right in itertools.combinations(component_indexes, 2)
    )

    if len(candidates) < ELEMENT_COUNT:
        return

    yield from itertools.combinations(candidates, ELEMENT_COUNT)


def friction_constraint_row(element: FrictionElement, component_count: int) -> tuple[float, ...]:
    row = [0.0] * component_count
    if element.kind == "brake":
        row[element.a] = 1.0
    else:
        assert element.b is not None
        row[element.a] = 1.0
        row[element.b] = -1.0
    return tuple(row)


def build_layout_template(
    topology: Topology,
    elements: Sequence[FrictionElement],
) -> LayoutTemplate:
    component_count = len(topology.components)
    constant_rows: list[tuple[tuple[float, ...], ...]] = []
    rhs_rows: list[tuple[float, ...]] = []

    rho_rows = [[0.0] * component_count for _ in range(GEARSET_COUNT)]
    for gearset in range(GEARSET_COUNT):
        ring_component = topology.component_of(node_for(gearset, "R"))
        carrier_component = topology.component_of(node_for(gearset, "C"))
        rho_rows[gearset][ring_component] += 1.0
        rho_rows[gearset][carrier_component] -= 1.0

    for applied in STATE_COMBINATIONS:
        rows = [[0.0] * component_count for _ in range(SYSTEM_ROW_COUNT)]
        rhs = [0.0] * SYSTEM_ROW_COUNT

        for gearset in range(GEARSET_COUNT):
            sun_component = topology.component_of(node_for(gearset, "S"))
            carrier_component = topology.component_of(node_for(gearset, "C"))
            rows[gearset][sun_component] += 1.0
            rows[gearset][carrier_component] -= 1.0

        rows[GEARSET_COUNT][topology.input_component] = 1.0
        rhs[GEARSET_COUNT] = 1.0

        for row_index, element_index in enumerate(applied, start=GEARSET_COUNT + 1):
            rows[row_index] = list(
                friction_constraint_row(elements[element_index], component_count)
            )

        constant_rows.append(tuple(tuple(row) for row in rows))
        rhs_rows.append(tuple(rhs))

    transition_counts = tuple(
        tuple(((left_mask ^ right_mask).bit_count() // 2) for right_mask in STATE_MASKS)
        for left_mask in STATE_MASKS
    )

    return LayoutTemplate(
        topology=topology,
        elements=tuple(elements),
        component_count=component_count,
        constant_rows=tuple(constant_rows),
        rho_rows=tuple(tuple(row) for row in rho_rows),
        rhs_rows=tuple(rhs_rows),
        state_masks=STATE_MASKS,
        transition_counts=transition_counts,
    )


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for CUDA search. Install dependencies from "
            "requirements-planetary.txt."
        )
    return torch


def torch_sort(values: Any, *, dim: int, descending: bool) -> tuple[Any, Any]:
    torch_module = require_torch()
    try:
        return torch_module.sort(values, dim=dim, descending=descending, stable=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility.
        return torch_module.sort(values, dim=dim, descending=descending)


def build_torch_layout_tensors(
    template: LayoutTemplate,
    device: Any,
    dtype: Any,
) -> TorchLayoutTensors:
    torch_module = require_torch()
    return TorchLayoutTensors(
        constant_rows=torch_module.tensor(template.constant_rows, device=device, dtype=dtype),
        rho_rows=torch_module.tensor(template.rho_rows, device=device, dtype=dtype),
        rhs_rows=torch_module.tensor(template.rhs_rows, device=device, dtype=dtype),
        state_masks=torch_module.tensor(template.state_masks, device=device, dtype=torch_module.long),
        transition_counts=torch_module.tensor(
            template.transition_counts,
            device=device,
            dtype=torch_module.long,
        ),
    )


def build_tooth_option_tensors(
    tooth_options: Sequence[GearsetTeeth],
    device: Any,
) -> ToothOptionTensors:
    torch_module = require_torch()
    return ToothOptionTensors(
        sun=torch_module.tensor([teeth.sun for teeth in tooth_options], device=device, dtype=torch_module.long),
        ring=torch_module.tensor([teeth.ring for teeth in tooth_options], device=device, dtype=torch_module.long),
        planet=torch_module.tensor([teeth.planet for teeth in tooth_options], device=device, dtype=torch_module.long),
        carrier_planets=torch_module.tensor(
            [teeth.carrier_planets for teeth in tooth_options],
            device=device,
            dtype=torch_module.long,
        ),
        rho=torch_module.tensor([teeth.rho for teeth in tooth_options], device=device, dtype=torch_module.float64),
    )


def build_torch_scoring_tensors(
    forward_targets: Sequence[float],
    reverse_target: float | None,
    device: Any,
) -> TorchScoringTensors:
    torch_module = require_torch()
    forward_count = len(forward_targets)
    forward_position_combinations = tuple(
        itertools.combinations(range(len(STATE_COMBINATIONS)), forward_count)
    )
    if not forward_position_combinations:
        raise ValueError("not enough apply states for requested forward target count")

    return TorchScoringTensors(
        forward_targets=torch_module.tensor(forward_targets, device=device, dtype=torch_module.float64),
        forward_position_combinations=torch_module.tensor(
            forward_position_combinations,
            device=device,
            dtype=torch_module.long,
        ),
        reverse_target=None
        if reverse_target is None
        else torch_module.tensor(reverse_target, device=device, dtype=torch_module.float64),
    )


def torch_ratios_are_close(left: Any, right: Any, tolerance: float) -> Any:
    torch_module = require_torch()
    scale = torch_module.maximum(
        torch_module.ones_like(left),
        torch_module.maximum(torch_module.abs(left), torch_module.abs(right)),
    )
    return torch_module.abs(left - right) <= tolerance * scale


def unique_sorted_value_mask(sorted_values: Any, finite_mask: Any, tolerance: float) -> Any:
    torch_module = require_torch()
    keep_columns = []
    for column_index in range(sorted_values.shape[1]):
        current = sorted_values[:, column_index]
        keep_current = finite_mask[:, column_index]
        for previous_index, previous_keep in enumerate(keep_columns):
            previous = sorted_values[:, previous_index]
            keep_current = keep_current & ~(
                previous_keep & torch_ratios_are_close(current, previous, tolerance)
            )
        keep_columns.append(keep_current)

    if not keep_columns:
        return torch_module.zeros_like(finite_mask)
    return torch_module.stack(keep_columns, dim=1)


def solve_template_batch_torch(
    template: LayoutTemplate,
    tensors: TorchLayoutTensors,
    rhos: Any,
    ratio_abs_limit: float,
    tolerance: float,
) -> tuple[Any, Any]:
    torch_module = require_torch()
    batch_count = int(rhos.shape[0])
    state_count = len(STATE_COMBINATIONS)
    component_count = template.component_count
    device = rhos.device
    dtype = rhos.dtype

    matrices = tensors.constant_rows.unsqueeze(0).expand(batch_count, -1, -1, -1).clone()
    for gearset in range(GEARSET_COUNT):
        matrices[:, :, gearset, :] += (
            rhos[:, gearset].reshape(batch_count, 1, 1)
            * tensors.rho_rows[gearset].reshape(1, 1, component_count)
        )

    rhs = tensors.rhs_rows.unsqueeze(0).expand(batch_count, -1, -1)
    flat_matrices = matrices.reshape(batch_count * state_count, SYSTEM_ROW_COUNT, component_count)
    flat_rhs = rhs.reshape(batch_count * state_count, SYSTEM_ROW_COUNT)
    if not hasattr(torch_module.linalg, "solve_ex"):  # pragma: no cover
        raise RuntimeError("torch.linalg.solve_ex is required for CUDA search")

    if SYSTEM_ROW_COUNT == component_count:
        solve_matrices = flat_matrices
        solve_rhs = flat_rhs
    else:
        transposed = flat_matrices.transpose(-2, -1)
        solve_matrices = torch_module.matmul(transposed, flat_matrices)
        solve_rhs = torch_module.matmul(
            transposed,
            flat_rhs.unsqueeze(-1),
        ).squeeze(-1)

    solutions, info = torch_module.linalg.solve_ex(solve_matrices, solve_rhs)
    flat_valid = info == 0

    residuals = torch_module.max(
        torch_module.abs(
            torch_module.matmul(flat_matrices, solutions.unsqueeze(-1)).squeeze(-1)
            - flat_rhs
        ),
        dim=1,
    ).values
    allowed_residual = tolerance * max(1.0, float(SYSTEM_ROW_COUNT))
    finite_solutions = torch_module.isfinite(solutions).all(dim=1)
    flat_valid = flat_valid & finite_solutions & (residuals <= allowed_residual)
    flat_solutions = torch_module.where(
        flat_valid.reshape(-1, 1),
        solutions,
        torch_module.full_like(solutions, math.nan),
    )

    velocities = flat_solutions.reshape(batch_count, state_count, component_count)
    state_valid = flat_valid.reshape(batch_count, state_count)
    output_speed = velocities[:, :, template.topology.output_component]
    ratios = torch_module.reciprocal(output_speed)
    ratio_valid = (
        state_valid
        & (torch_module.abs(output_speed) >= tolerance)
        & torch_module.isfinite(ratios)
        & (torch_module.abs(ratios) <= ratio_abs_limit)
    )
    ratios = torch_module.where(
        ratio_valid,
        ratios,
        torch_module.full_like(ratios, math.nan),
    )
    return ratios, ratio_valid


def score_ratios_batch_torch(
    ratios: Any,
    valid: Any,
    tensors: TorchLayoutTensors,
    scoring_tensors: TorchScoringTensors,
    max_double_transitions: int,
    transition_penalty: float,
    ratio_tolerance: float,
    require_reverse: bool,
    reverse_weight: float,
    reject_shift_violations: bool,
) -> Any:
    torch_module = require_torch()
    batch_count = int(ratios.shape[0])
    device = ratios.device
    dtype = ratios.dtype

    neg_inf = torch_module.full_like(ratios, -math.inf)
    pos_inf = torch_module.full_like(ratios, math.inf)

    positive_sort_input = torch_module.where(valid & (ratios > 0.0), ratios, neg_inf)
    positive_values, positive_order = torch_sort(
        positive_sort_input,
        dim=1,
        descending=True,
    )
    positive_finite = torch_module.isfinite(positive_values)
    positive_keep = unique_sorted_value_mask(
        positive_values,
        positive_finite,
        ratio_tolerance,
    )

    positions = scoring_tensors.forward_position_combinations
    combo_values = positive_values[:, positions]
    combo_keep = positive_keep[:, positions].all(dim=2)
    target_errors = combo_values - scoring_tensors.forward_targets.reshape(1, 1, -1)
    forward_mse = torch_module.mean(target_errors * target_errors, dim=2)

    combo_state_indices = positive_order[:, positions]
    transition_counts = tensors.transition_counts[
        combo_state_indices[:, :, :-1],
        combo_state_indices[:, :, 1:],
    ]
    double_transition_counts = (transition_counts > 1).sum(dim=2)
    if reject_shift_violations:
        combo_keep = combo_keep & (double_transition_counts == 0)

    excess_double_transitions = torch_module.clamp(
        double_transition_counts - max_double_transitions,
        min=0,
    ).to(dtype)
    shift_penalty = transition_penalty * excess_double_transitions

    reverse_penalty = torch_module.zeros(batch_count, device=device, dtype=dtype)
    candidate_has_reverse = torch_module.ones(batch_count, device=device, dtype=torch_module.bool)
    if scoring_tensors.reverse_target is not None:
        negative_sort_input = torch_module.where(valid & (ratios < 0.0), ratios, pos_inf)
        negative_values, _negative_order = torch_sort(
            negative_sort_input,
            dim=1,
            descending=False,
        )
        negative_finite = torch_module.isfinite(negative_values)
        negative_keep = unique_sorted_value_mask(
            negative_values,
            negative_finite,
            ratio_tolerance,
        )
        reverse_errors = torch_module.where(
            negative_keep,
            (negative_values - scoring_tensors.reverse_target) ** 2,
            torch_module.full_like(negative_values, math.inf),
        )
        best_reverse_error = torch_module.min(reverse_errors, dim=1).values
        candidate_has_reverse = torch_module.isfinite(best_reverse_error)
        reverse_penalty = torch_module.where(
            candidate_has_reverse,
            reverse_weight * best_reverse_error,
            reverse_penalty,
        )

    combo_scores = forward_mse + shift_penalty + reverse_penalty.reshape(batch_count, 1)
    combo_valid = combo_keep
    if require_reverse:
        combo_valid = combo_valid & candidate_has_reverse.reshape(batch_count, 1)
    combo_scores = torch_module.where(
        combo_valid,
        combo_scores,
        torch_module.full_like(combo_scores, math.inf),
    )
    return torch_module.min(combo_scores, dim=1).values


def gear_equations(topology: Topology, rhos: Sequence[float]) -> tuple[list[np.ndarray], list[float]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    component_count = len(topology.components)

    for gearset, rho in enumerate(rhos):
        row = np.zeros(component_count)
        row[topology.component_of(node_for(gearset, "S"))] += 1.0
        row[topology.component_of(node_for(gearset, "R"))] += float(rho)
        row[topology.component_of(node_for(gearset, "C"))] -= 1.0 + float(rho)
        rows.append(row)
        rhs.append(0.0)

    return rows, rhs


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray, tolerance: float) -> np.ndarray | None:
    if np.linalg.matrix_rank(matrix, tol=tolerance) < matrix.shape[1]:
        return None

    try:
        if matrix.shape[0] == matrix.shape[1]:
            solution = np.linalg.solve(matrix, rhs)
        else:
            solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    residual = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
    allowed_residual = tolerance * max(1.0, np.linalg.norm(rhs, ord=np.inf), matrix.shape[0])
    if residual > allowed_residual:
        return None

    return solution


def solve_state(
    topology: Topology,
    elements: Sequence[FrictionElement],
    applied: Sequence[int],
    rhos: Sequence[float],
    ratio_abs_limit: float,
    tolerance: float = 1e-8,
) -> StateResult | None:
    rows, rhs = gear_equations(topology, rhos)
    component_count = len(topology.components)

    input_row = np.zeros(component_count)
    input_row[topology.input_component] = 1.0
    rows.append(input_row)
    rhs.append(1.0)

    for element_index in applied:
        element = elements[element_index]
        row = np.zeros(component_count)
        if element.kind == "brake":
            row[element.a] = 1.0
        else:
            assert element.b is not None
            row[element.a] = 1.0
            row[element.b] = -1.0
        rows.append(row)
        rhs.append(0.0)

    matrix = np.vstack(rows)
    rhs_vector = np.array(rhs)
    velocities = solve_linear_system(matrix, rhs_vector, tolerance)
    if velocities is None:
        return None

    output_speed = float(velocities[topology.output_component])
    if abs(output_speed) < tolerance:
        return None

    ratio = 1.0 / output_speed
    if not math.isfinite(ratio) or abs(ratio) > ratio_abs_limit:
        return None

    return StateResult(
        state=tuple(sorted(applied)),  # type: ignore[arg-type]
        ratio=float(ratio),
        output_speed=output_speed,
        velocities=tuple(float(velocity) for velocity in velocities),
    )


def evaluate_states(
    topology: Topology,
    elements: Sequence[FrictionElement],
    rhos: Sequence[float],
    ratio_abs_limit: float,
) -> tuple[StateResult, ...]:
    states = []
    for applied in STATE_COMBINATIONS:
        state = solve_state(topology, elements, applied, rhos, ratio_abs_limit)
        if state is not None:
            states.append(state)
    return tuple(states)


def ratios_are_close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def unique_states_by_ratio(
    states: Sequence[StateResult],
    positive: bool,
    tolerance: float,
) -> tuple[StateResult, ...]:
    if positive:
        filtered = [state for state in states if state.ratio > 0.0]
        filtered.sort(key=lambda state: state.ratio, reverse=True)
    else:
        filtered = [state for state in states if state.ratio < 0.0]
        filtered.sort(key=lambda state: state.ratio)

    unique: list[StateResult] = []
    for state in filtered:
        if all(not ratios_are_close(state.ratio, existing.ratio, tolerance) for existing in unique):
            unique.append(state)
    return tuple(unique)


def transition_count(left: StateResult, right: StateResult) -> int:
    changed_elements = set(left.state).symmetric_difference(right.state)
    return len(changed_elements) // 2


def score_candidate(
    topology: Topology,
    elements: Sequence[FrictionElement],
    rhos: Sequence[float],
    teeth: tuple[GearsetTeeth, GearsetTeeth, GearsetTeeth],
    forward_targets: Sequence[float],
    reverse_target: float | None,
    max_double_transitions: int,
    transition_penalty: float,
    ratio_abs_limit: float,
    ratio_tolerance: float,
    require_reverse: bool,
    reverse_weight: float,
    reject_shift_violations: bool,
) -> LayoutResult | None:
    states = evaluate_states(topology, elements, rhos, ratio_abs_limit)
    forward_states = unique_states_by_ratio(states, positive=True, tolerance=ratio_tolerance)
    if len(forward_states) < len(forward_targets):
        return None

    reverse_state = None
    reverse_error = None
    if reverse_target is not None:
        reverse_states = unique_states_by_ratio(states, positive=False, tolerance=ratio_tolerance)
        if not reverse_states:
            if require_reverse:
                return None
        else:
            reverse_state = min(reverse_states, key=lambda state: abs(state.ratio - reverse_target))
            reverse_error = (reverse_state.ratio - reverse_target) ** 2

    best_result: LayoutResult | None = None
    best_key: tuple[float, float, int, float] | None = None

    for sequence in itertools.combinations(forward_states, len(forward_targets)):
        forward_mse = sum(
            (state.ratio - target) ** 2 for state, target in zip(sequence, forward_targets)
        ) / len(forward_targets)

        transition_counts = tuple(
            transition_count(left, right) for left, right in itertools.pairwise(sequence)
        )
        double_shift_indices = tuple(
            shift_index
            for shift_index, count in enumerate(transition_counts, start=1)
            if count > 1
        )
        if reject_shift_violations and double_shift_indices:
            continue

        excess_double_transitions = max(0, len(double_shift_indices) - max_double_transitions)
        shift_penalty = transition_penalty * excess_double_transitions
        reverse_penalty = reverse_weight * reverse_error if reverse_error is not None else 0.0
        score = forward_mse + shift_penalty + reverse_penalty

        result = LayoutResult(
            score=float(score),
            forward_mse=float(forward_mse),
            shift_penalty=float(shift_penalty),
            rhos=tuple(float(rho) for rho in rhos),  # type: ignore[arg-type]
            teeth=teeth,
            topology=topology,
            elements=tuple(elements),
            forward_sequence=tuple(sequence),
            reverse_state=reverse_state,
            reverse_error=float(reverse_error) if reverse_error is not None else None,
            double_shift_indices=double_shift_indices,
            transition_counts=transition_counts,
            valid_state_count=len(states),
        )
        key = (
            result.score,
            result.forward_mse,
            len(result.double_shift_indices),
            result.reverse_error if result.reverse_error is not None else math.inf,
        )
        if best_key is None or key < best_key:
            best_result = result
            best_key = key

    return best_result


def tune_candidate(
    topology: Topology,
    elements: Sequence[FrictionElement],
    tooth_options_by_gearset: Sequence[Sequence[GearsetTeeth]],
    tooth_combination_limit: int | None,
    forward_targets: Sequence[float],
    reverse_target: float | None,
    max_double_transitions: int,
    transition_penalty: float,
    ratio_abs_limit: float,
    ratio_tolerance: float,
    require_reverse: bool,
    reverse_weight: float,
    reject_shift_violations: bool,
) -> LayoutResult | None:
    best_result: LayoutResult | None = None
    best_key: tuple[float, float, int, float] | None = None
    for combination_index, teeth in enumerate(
        itertools.product(*tooth_options_by_gearset),
        start=1,
    ):
        if (
            tooth_combination_limit is not None
            and combination_index > tooth_combination_limit
        ):
            break

        result = score_candidate(
            topology=topology,
            elements=elements,
            rhos=tuple(gearset.rho for gearset in teeth),
            teeth=teeth,  # type: ignore[arg-type]
            forward_targets=forward_targets,
            reverse_target=reverse_target,
            max_double_transitions=max_double_transitions,
            transition_penalty=transition_penalty,
            ratio_abs_limit=ratio_abs_limit,
            ratio_tolerance=ratio_tolerance,
            require_reverse=require_reverse,
            reverse_weight=reverse_weight,
            reject_shift_violations=reject_shift_violations,
        )
        if result is None:
            continue

        key = (
            result.score,
            result.forward_mse,
            len(result.double_shift_indices),
            result.reverse_error if result.reverse_error is not None else math.inf,
        )
        if best_key is None or key < best_key:
            best_result = result
            best_key = key

    return best_result


def tooth_combination_total(option_counts: Sequence[int]) -> int:
    total = 1
    for count in option_counts:
        total *= count
    return total


def flat_tooth_index_to_tuple(flat_index: int, option_counts: Sequence[int]) -> tuple[int, int, int]:
    second_stride = option_counts[1] * option_counts[2]
    first = flat_index // second_stride
    remainder = flat_index % second_stride
    second = remainder // option_counts[2]
    third = remainder % option_counts[2]
    return first, second, third


def tooth_tuple_from_flat_index(
    tooth_options_by_gearset: Sequence[Sequence[GearsetTeeth]],
    flat_index: int,
) -> tuple[GearsetTeeth, GearsetTeeth, GearsetTeeth]:
    option_counts = tuple(len(options) for options in tooth_options_by_gearset)
    indexes = flat_tooth_index_to_tuple(flat_index, option_counts)
    return (
        tooth_options_by_gearset[0][indexes[0]],
        tooth_options_by_gearset[1][indexes[1]],
        tooth_options_by_gearset[2][indexes[2]],
    )


def unravel_tooth_indices(flat_indices: Any, option_counts: Sequence[int]) -> tuple[Any, Any, Any]:
    second_stride = option_counts[1] * option_counts[2]
    first = flat_indices // second_stride
    remainder = flat_indices % second_stride
    second = remainder // option_counts[2]
    third = remainder % option_counts[2]
    return first, second, third


def iter_tooth_index_batches(
    option_counts: Sequence[int],
    batch_size: int,
    tooth_combination_limit: int | None,
    sampling_mode: str,
    random_seed: int,
    device: Any,
) -> Iterator[Any]:
    torch_module = require_torch()
    total = tooth_combination_total(option_counts)
    if sampling_mode == "exhaustive":
        selected_count = total if tooth_combination_limit is None else min(total, tooth_combination_limit)
        for start in range(0, selected_count, batch_size):
            stop = min(start + batch_size, selected_count)
            yield torch_module.arange(start, stop, device=device, dtype=torch_module.long)
        return

    if tooth_combination_limit is None:
        raise ValueError(f"--sampling-mode {sampling_mode} requires --tooth-combination-limit")

    selected_count = min(total, tooth_combination_limit)
    if sampling_mode == "stratified-rho":
        if selected_count == 1:
            yield torch_module.zeros(1, device=device, dtype=torch_module.long)
            return

        denominator = selected_count - 1
        for start in range(0, selected_count, batch_size):
            stop = min(start + batch_size, selected_count)
            positions = torch_module.arange(start, stop, device=device, dtype=torch_module.float64)
            scaled = positions * float(total - 1) / float(denominator)
            yield torch_module.round(scaled).to(torch_module.long)
        return

    if sampling_mode == "random":
        try:
            generator = torch_module.Generator(device=device)
        except TypeError:  # pragma: no cover - older PyTorch compatibility.
            generator = torch_module.Generator()
        generator.manual_seed(random_seed)
        remaining = selected_count
        while remaining > 0:
            current = min(batch_size, remaining)
            remaining -= current
            yield torch_module.randint(
                total,
                (current,),
                device=device,
                dtype=torch_module.long,
                generator=generator,
            )
        return

    raise ValueError(f"unknown sampling mode: {sampling_mode}")


def make_refine_candidate_pool(
    refine_limit: int,
    device: Any,
    score_dtype: Any,
    index_dtype: Any,
) -> TorchRefineCandidatePool:
    torch_module = require_torch()
    capacity = max(refine_limit * 4, refine_limit)
    return TorchRefineCandidatePool(
        scores=torch_module.full(
            (capacity,),
            math.inf,
            device=device,
            dtype=score_dtype,
        ),
        flat_indices=torch_module.full(
            (capacity,),
            -1,
            device=device,
            dtype=index_dtype,
        ),
        capacity=capacity,
    )


def update_refine_candidate_pool(
    pool: TorchRefineCandidatePool,
    scores: Any,
    flat_indices: Any,
) -> TorchRefineCandidatePool:
    torch_module = require_torch()
    if scores.numel() == 0:
        return pool

    local_count = min(pool.capacity, int(scores.shape[0]))
    finite_scores = torch_module.where(
        torch_module.isfinite(scores),
        scores,
        torch_module.full_like(scores, math.inf),
    )
    local_scores, local_positions = torch_module.topk(
        finite_scores,
        k=local_count,
        largest=False,
    )
    local_flat_indices = flat_indices[local_positions].to(pool.flat_indices.dtype)

    combined_scores = torch_module.cat((pool.scores, local_scores))
    combined_flat_indices = torch_module.cat((pool.flat_indices, local_flat_indices))
    best_scores, best_positions = torch_module.topk(
        combined_scores,
        k=pool.capacity,
        largest=False,
    )
    return TorchRefineCandidatePool(
        scores=best_scores,
        flat_indices=combined_flat_indices[best_positions],
        capacity=pool.capacity,
    )


def refine_candidate_pool_to_cpu(
    pool: TorchRefineCandidatePool,
) -> list[tuple[float, int]]:
    torch_module = require_torch()
    finite = torch_module.isfinite(pool.scores)
    scores = pool.scores[finite].detach().cpu().tolist()
    flat_indices = pool.flat_indices[finite].detach().cpu().tolist()
    return sorted(
        (float(score), int(flat_index))
        for score, flat_index in zip(scores, flat_indices)
        if int(flat_index) >= 0
    )


def tune_candidate_gpu(
    topology: Topology,
    elements: Sequence[FrictionElement],
    tooth_options_by_gearset: Sequence[Sequence[GearsetTeeth]],
    tooth_tensors_by_gearset: Sequence[ToothOptionTensors],
    scoring_tensors: TorchScoringTensors,
    tooth_combination_limit: int | None,
    forward_targets: Sequence[float],
    reverse_target: float | None,
    max_double_transitions: int,
    transition_penalty: float,
    ratio_abs_limit: float,
    ratio_tolerance: float,
    require_reverse: bool,
    reverse_weight: float,
    reject_shift_violations: bool,
    batch_size: int,
    sampling_mode: str,
    random_seed: int,
    refine_limit: int,
    device: Any,
) -> tuple[LayoutResult | None, GpuTuneStats]:
    if np is None:
        raise RuntimeError(
            "NumPy is required to reconstruct exact GPU finalists. Install dependencies "
            "from requirements-planetary.txt."
        )

    torch_module = require_torch()
    template = build_layout_template(topology, elements)
    layout_tensors = build_torch_layout_tensors(template, device=device, dtype=torch_module.float64)
    option_counts = tuple(len(options) for options in tooth_options_by_gearset)
    refine_candidate_pool = make_refine_candidate_pool(
        refine_limit=refine_limit,
        device=device,
        score_dtype=torch_module.float64,
        index_dtype=torch_module.long,
    )
    tooth_triples = 0
    batches = 0

    for flat_indices in iter_tooth_index_batches(
        option_counts=option_counts,
        batch_size=batch_size,
        tooth_combination_limit=tooth_combination_limit,
        sampling_mode=sampling_mode,
        random_seed=random_seed,
        device=device,
    ):
        if flat_indices.numel() == 0:
            continue

        first, second, third = unravel_tooth_indices(flat_indices, option_counts)
        rhos = torch_module.stack(
            (
                tooth_tensors_by_gearset[0].rho[first],
                tooth_tensors_by_gearset[1].rho[second],
                tooth_tensors_by_gearset[2].rho[third],
            ),
            dim=1,
        )
        ratios, valid = solve_template_batch_torch(
            template=template,
            tensors=layout_tensors,
            rhos=rhos,
            ratio_abs_limit=ratio_abs_limit,
            tolerance=1e-8,
        )
        scores = score_ratios_batch_torch(
            ratios=ratios,
            valid=valid,
            tensors=layout_tensors,
            scoring_tensors=scoring_tensors,
            max_double_transitions=max_double_transitions,
            transition_penalty=transition_penalty,
            ratio_tolerance=ratio_tolerance,
            require_reverse=require_reverse,
            reverse_weight=reverse_weight,
            reject_shift_violations=reject_shift_violations,
        )
        refine_candidate_pool = update_refine_candidate_pool(
            refine_candidate_pool,
            scores,
            flat_indices,
        )
        tooth_triples += int(flat_indices.numel())
        batches += 1

    best_result: LayoutResult | None = None
    best_key: tuple[float, float, int, float] | None = None
    seen_flat_indexes: set[int] = set()
    refined_candidates = 0
    refine_candidates = refine_candidate_pool_to_cpu(refine_candidate_pool)

    for _score, flat_index in refine_candidates:
        if flat_index in seen_flat_indexes:
            continue
        seen_flat_indexes.add(flat_index)
        teeth = tooth_tuple_from_flat_index(tooth_options_by_gearset, flat_index)
        result = score_candidate(
            topology=topology,
            elements=elements,
            rhos=tuple(gearset.rho for gearset in teeth),
            teeth=teeth,
            forward_targets=forward_targets,
            reverse_target=reverse_target,
            max_double_transitions=max_double_transitions,
            transition_penalty=transition_penalty,
            ratio_abs_limit=ratio_abs_limit,
            ratio_tolerance=ratio_tolerance,
            require_reverse=require_reverse,
            reverse_weight=reverse_weight,
            reject_shift_violations=reject_shift_violations,
        )
        refined_candidates += 1
        if result is None:
            continue

        key = (
            result.score,
            result.forward_mse,
            len(result.double_shift_indices),
            result.reverse_error if result.reverse_error is not None else math.inf,
        )
        if best_key is None or key < best_key:
            best_result = result
            best_key = key

        if refined_candidates >= refine_limit:
            break

    return best_result, GpuTuneStats(
        tooth_triples=tooth_triples,
        batches=batches,
        refined_candidates=refined_candidates,
    )


def add_best_result(best_results: list[LayoutResult], result: LayoutResult, limit: int) -> None:
    best_results.append(result)
    best_results.sort(key=lambda item: (item.score, item.forward_mse, len(item.double_shift_indices)))
    del best_results[limit:]


def node_from_label(label: str) -> Node:
    try:
        return NODE_BY_LABEL[label]
    except KeyError as error:
        raise ValueError(f"unknown node label in checkpoint: {label}") from error


def topology_to_checkpoint_dict(topology: Topology) -> dict[str, object]:
    return {
        "permanent_edges": [
            [left.label, right.label] for left, right in topology.permanent_edges
        ],
        "components": [
            [node.label for node in component] for component in topology.components
        ],
        "input_component": topology.input_component,
        "output_component": topology.output_component,
    }


def topology_from_checkpoint_dict(data: dict[str, object]) -> Topology:
    components_data = data["components"]
    if not isinstance(components_data, list):
        raise ValueError("checkpoint topology components must be a list")
    components = tuple(
        tuple(node_from_label(str(label)) for label in component)
        for component in components_data
    )
    component_by_node = [0] * len(NODES)
    for component_index, component in enumerate(components):
        for node in component:
            component_by_node[NODE_INDEX[node]] = component_index

    edges_data = data["permanent_edges"]
    if not isinstance(edges_data, list):
        raise ValueError("checkpoint topology permanent_edges must be a list")
    permanent_edges = tuple(
        (node_from_label(str(edge[0])), node_from_label(str(edge[1])))
        for edge in edges_data
    )
    return Topology(
        permanent_edges=permanent_edges,
        components=components,
        component_by_node=tuple(component_by_node),
        input_component=int(data["input_component"]),
        output_component=int(data["output_component"]),
    )


def element_to_checkpoint_dict(element: FrictionElement) -> dict[str, object]:
    return {"kind": element.kind, "a": element.a, "b": element.b}


def element_from_checkpoint_dict(data: dict[str, object]) -> FrictionElement:
    b = data.get("b")
    return FrictionElement(
        kind=str(data["kind"]),
        a=int(data["a"]),
        b=None if b is None else int(b),
    )


def teeth_to_checkpoint_dict(teeth: GearsetTeeth) -> dict[str, int]:
    return {
        "sun": teeth.sun,
        "ring": teeth.ring,
        "planet": teeth.planet,
        "carrier_planets": teeth.carrier_planets,
    }


def teeth_from_checkpoint_dict(data: dict[str, object]) -> GearsetTeeth:
    return GearsetTeeth(
        sun=int(data["sun"]),
        ring=int(data["ring"]),
        planet=int(data["planet"]),
        carrier_planets=int(data["carrier_planets"]),
    )


def state_to_checkpoint_dict(state: StateResult | None) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "state": list(state.state),
        "ratio": state.ratio,
        "output_speed": state.output_speed,
        "velocities": list(state.velocities),
    }


def state_from_checkpoint_dict(data: dict[str, object] | None) -> StateResult | None:
    if data is None:
        return None
    return StateResult(
        state=tuple(int(index) for index in data["state"]),  # type: ignore[index]
        ratio=float(data["ratio"]),
        output_speed=float(data["output_speed"]),
        velocities=tuple(float(value) for value in data["velocities"]),  # type: ignore[index]
    )


def layout_result_to_checkpoint_dict(result: LayoutResult) -> dict[str, object]:
    return {
        "score": result.score,
        "forward_mse": result.forward_mse,
        "shift_penalty": result.shift_penalty,
        "rhos": list(result.rhos),
        "teeth": [teeth_to_checkpoint_dict(teeth) for teeth in result.teeth],
        "topology": topology_to_checkpoint_dict(result.topology),
        "elements": [element_to_checkpoint_dict(element) for element in result.elements],
        "forward_sequence": [
            state_to_checkpoint_dict(state) for state in result.forward_sequence
        ],
        "reverse_state": state_to_checkpoint_dict(result.reverse_state),
        "reverse_error": result.reverse_error,
        "double_shift_indices": list(result.double_shift_indices),
        "transition_counts": list(result.transition_counts),
        "valid_state_count": result.valid_state_count,
    }


def layout_result_from_checkpoint_dict(data: dict[str, object]) -> LayoutResult:
    return LayoutResult(
        score=float(data["score"]),
        forward_mse=float(data["forward_mse"]),
        shift_penalty=float(data["shift_penalty"]),
        rhos=tuple(float(rho) for rho in data["rhos"]),  # type: ignore[index]
        teeth=tuple(
            teeth_from_checkpoint_dict(teeth) for teeth in data["teeth"]  # type: ignore[index]
        ),  # type: ignore[arg-type]
        topology=topology_from_checkpoint_dict(data["topology"]),  # type: ignore[arg-type]
        elements=tuple(
            element_from_checkpoint_dict(element) for element in data["elements"]  # type: ignore[index]
        ),
        forward_sequence=tuple(
            state
            for state in (
                state_from_checkpoint_dict(item)
                for item in data["forward_sequence"]  # type: ignore[index]
            )
            if state is not None
        ),
        reverse_state=state_from_checkpoint_dict(data["reverse_state"]),  # type: ignore[arg-type]
        reverse_error=None
        if data["reverse_error"] is None
        else float(data["reverse_error"]),
        double_shift_indices=tuple(
            int(index) for index in data["double_shift_indices"]  # type: ignore[index]
        ),
        transition_counts=tuple(
            int(count) for count in data["transition_counts"]  # type: ignore[index]
        ),
        valid_state_count=int(data["valid_state_count"]),
    )


def checkpoint_args(args: argparse.Namespace) -> dict[str, object]:
    stored: dict[str, object] = {}
    for key, value in vars(args).items():
        if key in {"resume"}:
            continue
        if isinstance(value, tuple):
            stored[key] = list(value)
        else:
            stored[key] = value
    return stored


def write_checkpoint(
    path: str,
    args: argparse.Namespace,
    stats: dict[str, Any],
    best_results: Sequence[LayoutResult],
    next_topology_index: int,
    next_element_index: int,
) -> None:
    payload = {
        "version": CHECKPOINT_VERSION,
        "args": checkpoint_args(args),
        "resume": {
            "topology_index": next_topology_index,
            "element_index": next_element_index,
        },
        "stats": stats,
        "best_results": [
            layout_result_to_checkpoint_dict(result) for result in best_results
        ],
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def load_checkpoint(path: str) -> CheckpointState:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if int(payload.get("version", 0)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint version")

    resume = payload["resume"]
    best_results = [
        layout_result_from_checkpoint_dict(result)
        for result in payload.get("best_results", [])
    ]
    return CheckpointState(
        topology_index=int(resume["topology_index"]),
        element_index=int(resume["element_index"]),
        stats=dict(payload.get("stats", {})),
        best_results=best_results,
    )


def resolve_search_backend(args: argparse.Namespace) -> tuple[str, Any | None]:
    if args.backend == "cpu":
        return "cpu", None

    if args.backend == "cuda":
        torch_module = require_torch()
        if not torch_module.cuda.is_available():
            raise RuntimeError("PyTorch CUDA is not available on this system")
        device = torch_module.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must name a CUDA device when --backend cuda is used")
        return "cuda", device

    if torch is not None and torch.cuda.is_available():
        torch_module = require_torch()
        device = torch_module.device(args.device)
        if device.type != "cuda":
            return "cpu", None
        return "cuda", device

    return "cpu", None


def search(args: argparse.Namespace) -> tuple[list[LayoutResult], dict[str, Any]]:
    forward_targets, reverse_target = split_targets(args.targets)
    max_double_transitions = 0 if args.strict_single_transition else args.max_double_transitions
    backend, device = resolve_search_backend(args)

    tooth_options_by_gearset = tuple(
        generate_tooth_options(
            rho_bounds=args.standing_ratio_bounds,
            include_equivalent_teeth=args.include_equivalent_teeth,
            carrier_planets=carrier_planets,
        )
        for carrier_planets in GEARSET_CARRIER_PLANETS
    )
    if any(not options for options in tooth_options_by_gearset):
        raise ValueError("tooth-count bounds produced no feasible gearsets")

    tooth_tensors_by_gearset: tuple[ToothOptionTensors, ...] | None = None
    scoring_tensors: TorchScoringTensors | None = None
    if backend == "cuda":
        assert device is not None
        tooth_tensors_by_gearset = tuple(
            build_tooth_option_tensors(options, device=device)
            for options in tooth_options_by_gearset
        )
        scoring_tensors = build_torch_scoring_tensors(
            forward_targets=forward_targets,
            reverse_target=reverse_target,
            device=device,
        )

    stats: dict[str, Any] = {
        "backend": backend,
        "topologies": 0,
        "element_sets": 0,
        "candidates": 0,
        "valid_candidates": 0,
        "pruned_candidates": 0,
        "tooth_triples_evaluated": 0,
        "gpu_batches": 0,
        "gpu_refined_candidates": 0,
        "tooth_options_g1": len(tooth_options_by_gearset[0]),
        "tooth_options_g2": len(tooth_options_by_gearset[1]),
        "tooth_options_g3": len(tooth_options_by_gearset[2]),
    }
    best_results: list[LayoutResult] = []
    resume_topology_index = 0
    resume_element_index = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        stats.update(checkpoint.stats)
        stats["backend"] = backend
        best_results = checkpoint.best_results
        best_results.sort(key=lambda item: (item.score, item.forward_mse, len(item.double_shift_indices)))
        del best_results[args.top:]
        resume_topology_index = checkpoint.topology_index
        resume_element_index = checkpoint.element_index

    for topology_index, topology in enumerate(
        generate_topologies(args.permanent_links, args.allow_internal_permanent)
    ):
        if topology_index < resume_topology_index:
            continue

        resuming_mid_topology = (
            topology_index == resume_topology_index and resume_element_index > 0
        )
        if (
            not resuming_mid_topology
            and args.topology_limit is not None
            and stats["topologies"] >= args.topology_limit
        ):
            break
        if not resuming_mid_topology:
            stats["topologies"] += 1

        element_sets_for_topology = resume_element_index if resuming_mid_topology else 0
        for element_index, elements in enumerate(
            generate_element_sets(topology, args.include_output_brakes)
        ):
            if resuming_mid_topology and element_index < resume_element_index:
                continue
            if (
                args.element_limit_per_topology is not None
                and element_sets_for_topology >= args.element_limit_per_topology
            ):
                break

            element_sets_for_topology += 1
            if args.candidate_limit is not None and stats["candidates"] >= args.candidate_limit:
                if args.checkpoint is not None:
                    write_checkpoint(
                        path=args.checkpoint,
                        args=args,
                        stats=stats,
                        best_results=best_results,
                        next_topology_index=topology_index,
                        next_element_index=element_index,
                    )
                return best_results, stats

            stats["element_sets"] += 1
            stats["candidates"] += 1

            if args.probe_tooth_triples:
                probe_result = tune_candidate(
                    topology=topology,
                    elements=elements,
                    tooth_options_by_gearset=tooth_options_by_gearset,
                    tooth_combination_limit=args.probe_tooth_triples,
                    forward_targets=forward_targets,
                    reverse_target=reverse_target,
                    max_double_transitions=max_double_transitions,
                    transition_penalty=args.transition_penalty,
                    ratio_abs_limit=args.ratio_abs_limit,
                    ratio_tolerance=args.ratio_tolerance,
                    require_reverse=not args.allow_missing_reverse,
                    reverse_weight=args.reverse_weight,
                    reject_shift_violations=args.strict_single_transition,
                )
                if probe_result is None:
                    stats["pruned_candidates"] += 1
                    continue

            if backend == "cuda":
                assert device is not None
                assert tooth_tensors_by_gearset is not None
                assert scoring_tensors is not None
                result, gpu_stats = tune_candidate_gpu(
                    topology=topology,
                    elements=elements,
                    tooth_options_by_gearset=tooth_options_by_gearset,
                    tooth_tensors_by_gearset=tooth_tensors_by_gearset,
                    scoring_tensors=scoring_tensors,
                    tooth_combination_limit=args.tooth_combination_limit,
                    forward_targets=forward_targets,
                    reverse_target=reverse_target,
                    max_double_transitions=max_double_transitions,
                    transition_penalty=args.transition_penalty,
                    ratio_abs_limit=args.ratio_abs_limit,
                    ratio_tolerance=args.ratio_tolerance,
                    require_reverse=not args.allow_missing_reverse,
                    reverse_weight=args.reverse_weight,
                    reject_shift_violations=args.strict_single_transition,
                    batch_size=args.batch_size,
                    sampling_mode=args.sampling_mode,
                    random_seed=args.random_seed,
                    refine_limit=args.gpu_refine_top,
                    device=device,
                )
                stats["tooth_triples_evaluated"] += gpu_stats.tooth_triples
                stats["gpu_batches"] += gpu_stats.batches
                stats["gpu_refined_candidates"] += gpu_stats.refined_candidates
            else:
                result = tune_candidate(
                    topology=topology,
                    elements=elements,
                    tooth_options_by_gearset=tooth_options_by_gearset,
                    tooth_combination_limit=args.tooth_combination_limit,
                    forward_targets=forward_targets,
                    reverse_target=reverse_target,
                    max_double_transitions=max_double_transitions,
                    transition_penalty=args.transition_penalty,
                    ratio_abs_limit=args.ratio_abs_limit,
                    ratio_tolerance=args.ratio_tolerance,
                    require_reverse=not args.allow_missing_reverse,
                    reverse_weight=args.reverse_weight,
                    reject_shift_violations=args.strict_single_transition,
                )
            if result is not None:
                stats["valid_candidates"] += 1
                add_best_result(best_results, result, args.top)

            if args.progress_every and stats["candidates"] % args.progress_every == 0:
                print(
                    "searched "
                    f"{stats['candidates']} candidates, "
                    f"{stats['valid_candidates']} valid, "
                    f"{len(best_results)} ranked",
                    file=sys.stderr,
                    flush=True,
                )

            if (
                args.checkpoint is not None
                and args.checkpoint_every
                and stats["candidates"] % args.checkpoint_every == 0
            ):
                write_checkpoint(
                    path=args.checkpoint,
                    args=args,
                    stats=stats,
                    best_results=best_results,
                    next_topology_index=topology_index,
                    next_element_index=element_index + 1,
                )

        resume_element_index = 0
        if args.checkpoint is not None:
            write_checkpoint(
                path=args.checkpoint,
                args=args,
                stats=stats,
                best_results=best_results,
                next_topology_index=topology_index + 1,
                next_element_index=0,
            )

    return best_results, stats


def split_targets(targets: Sequence[float]) -> tuple[tuple[float, ...], float | None]:
    if len(targets) > 0 and targets[-1] < 0.0:
        forward_targets = tuple(targets[:-1])
        reverse_target = targets[-1]
    else:
        forward_targets = tuple(targets)
        reverse_target = None

    if len(forward_targets) not in (6, 7):
        raise ValueError(
            "provide six or seven positive forward targets, optionally followed by "
            "a negative reverse ratio"
        )
    if any(target <= 0.0 for target in forward_targets):
        raise ValueError("forward targets must be positive ratios")
    return forward_targets, reverse_target


def state_element_names(state: StateResult) -> list[str]:
    return [f"E{index + 1}" for index in state.state]


def result_to_dict(result: LayoutResult, forward_targets: Sequence[float]) -> dict[str, object]:
    topology = result.topology

    return {
        "score": result.score,
        "forward_mse": result.forward_mse,
        "shift_penalty": result.shift_penalty,
        "rho": list(result.rhos),
        "teeth": [
            {
                "gearset": index,
                "sun": teeth.sun,
                "ring": teeth.ring,
                "planet": teeth.planet,
                "carrier_planets": teeth.carrier_planets,
                "rho": teeth.rho,
            }
            for index, teeth in enumerate(result.teeth, start=1)
        ],
        "permanent_edges": [
            f"{left.label}-{right.label}" for left, right in result.topology.permanent_edges
        ],
        "permanent_groups": [
            [node.label for node in group] for group in result.topology.permanent_groups()
        ],
        "input": topology.component_label(topology.input_component),
        "output": topology.component_label(topology.output_component),
        "elements": [
            {"name": f"E{index + 1}", "placement": element.label(topology)}
            for index, element in enumerate(result.elements)
        ],
        "forward_gears": [
            {
                "gear": gear_index,
                "ratio": state.ratio,
                "target": forward_targets[gear_index - 1],
                "applied": state_element_names(state),
                "transition_count_to_next": result.transition_counts[gear_index - 1]
                if gear_index <= len(result.transition_counts)
                else None,
            }
            for gear_index, state in enumerate(result.forward_sequence, start=1)
        ],
        "reverse": None
        if result.reverse_state is None
        else {
            "ratio": result.reverse_state.ratio,
            "applied": state_element_names(result.reverse_state),
            "squared_error": result.reverse_error,
        },
        "double_transition_shifts": [
            f"{index}->{index + 1}" for index in result.double_shift_indices
        ],
        "valid_state_count": result.valid_state_count,
    }


def print_text_report(
    results: Sequence[LayoutResult],
    stats: dict[str, Any],
    forward_targets: Sequence[float],
    reverse_target: float | None,
) -> None:
    print(
        "Search complete: "
        f"{stats['candidates']} candidates, "
        f"{stats['valid_candidates']} valid candidates, "
        f"{stats['topologies']} topologies."
    )
    if stats.get("tooth_options_g1"):
        print(
            "Tooth options per gearset: "
            f"G1={stats['tooth_options_g1']}, "
            f"G2={stats['tooth_options_g2']}, "
            f"G3={stats['tooth_options_g3']}."
        )
    print(f"Backend: {stats.get('backend', 'cpu')}.")
    if stats.get("backend") == "cuda":
        print(
            "CUDA batches: "
            f"{stats.get('gpu_batches', 0)}, "
            f"tooth triples evaluated: {stats.get('tooth_triples_evaluated', 0)}, "
            f"finalists refined: {stats.get('gpu_refined_candidates', 0)}."
        )
    if stats.get("pruned_candidates"):
        print(f"Probe-pruned candidates: {stats['pruned_candidates']}.")

    if not results:
        print("No valid configurations found.")
        return

    for rank, result in enumerate(results, start=1):
        topology = result.topology
        print()
        print(f"Rank {rank}")
        print(f"  score: {result.score:.8g}")
        print(f"  forward_mse: {result.forward_mse:.8g}")
        if result.shift_penalty:
            print(f"  shift_penalty: {result.shift_penalty:.8g}")
        print("  rho: " + ", ".join(f"{rho:.6g}" for rho in result.rhos))
        print("  teeth:")
        for index, teeth in enumerate(result.teeth, start=1):
            print(
                f"    G{index}: sun={teeth.sun} ring={teeth.ring} "
                f"planet={teeth.planet} carrier_planets={teeth.carrier_planets} "
                f"rho={teeth.rho:.6g}"
            )
        print(
            "  permanent_edges: "
            + ", ".join(f"{left.label}-{right.label}" for left, right in topology.permanent_edges)
        )
        print(
            "  permanent_groups: "
            + ", ".join(
                "/".join(node.label for node in group) for group in topology.permanent_groups()
            )
        )
        print(f"  input: {topology.component_label(topology.input_component)}")
        print(f"  output: {topology.component_label(topology.output_component)}")
        print("  elements:")
        for index, element in enumerate(result.elements, start=1):
            print(f"    E{index}: {element.label(topology)}")

        print("  apply_chart:")
        print("    gear  ratio       target      error       applied     shift")
        for gear_index, (state, target) in enumerate(
            zip(result.forward_sequence, forward_targets),
            start=1,
        ):
            error = state.ratio - target
            shift = "-"
            if gear_index > 1:
                count = result.transition_counts[gear_index - 2]
                shift = "double" if count > 1 else "single"
            print(
                f"    {gear_index:>4}  {state.ratio:>10.6g}  {target:>10.6g}  "
                f"{error:>10.6g}  {','.join(state_element_names(state)):>10}  {shift}"
            )

        if reverse_target is not None:
            if result.reverse_state is None:
                print(f"  reverse: missing target {reverse_target:.6g}")
            else:
                error = result.reverse_state.ratio - reverse_target
                print(
                    "  reverse: "
                    f"{result.reverse_state.ratio:.6g} target {reverse_target:.6g} "
                    f"error {error:.6g} applied {','.join(state_element_names(result.reverse_state))}"
                )

        if result.double_shift_indices:
            shifts = ", ".join(f"{index}->{index + 1}" for index in result.double_shift_indices)
            print(f"  double_transition_shifts: {shifts}")
        else:
            print("  double_transition_shifts: none")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brute-force and score 3-planetary, 5-friction-element transmission layouts.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--targets",
        type=parse_float_list,
        default=DEFAULT_TARGETS,
        help="Comma-separated ratios: six or seven forward values and optional negative reverse.",
    )
    parser.add_argument(
        "--standing-ratio-bounds",
        type=float,
        nargs=2,
        default=(1.5, 3.5),
        metavar=("MIN", "MAX"),
        help="Bounds used to filter tooth-derived standing ratios.",
    )
    parser.add_argument(
        "--permanent-links",
        type=parse_link_counts,
        default=(3, 4),
        help="Permanent connection counts to search: 3, 4, or 3,4.",
    )
    parser.add_argument(
        "--strict-single-transition",
        action="store_true",
        help="Require every adjacent shift to share one applied element.",
    )
    parser.add_argument(
        "--max-double-transitions",
        type=int,
        default=1,
        help="Allowed double-transition shifts before penalty.",
    )
    parser.add_argument(
        "--transition-penalty",
        type=float,
        default=1_000_000.0,
        help="Penalty per double-transition shift beyond the allowed count.",
    )
    parser.add_argument(
        "--reverse-weight",
        type=float,
        default=0.0,
        help="Optional ranking weight for reverse squared error.",
    )
    parser.add_argument(
        "--allow-missing-reverse",
        action="store_true",
        help="Keep candidates even when no reverse state exists.",
    )
    parser.add_argument(
        "--ratio-abs-limit",
        type=float,
        default=50.0,
        help="Discard states with absolute ratio above this limit.",
    )
    parser.add_argument(
        "--ratio-tolerance",
        type=float,
        default=1e-4,
        help="Relative tolerance for treating ratios as redundant.",
    )
    parser.add_argument(
        "--include-equivalent-teeth",
        action="store_true",
        help="Keep tooth pairs with duplicate reduced rho values.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Search backend. auto uses CUDA when PyTorch reports an NVIDIA GPU.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch CUDA device, for example cuda or cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32768,
        help="Tooth triples per CUDA batch.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("exhaustive", "random", "stratified-rho"),
        default="exhaustive",
        help="Tooth-triple enumeration mode.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used by random tooth-triple sampling.",
    )
    parser.add_argument(
        "--gpu-refine-top",
        type=int,
        default=16,
        help="GPU-selected tooth triples refined exactly on CPU per layout.",
    )
    parser.add_argument(
        "--tooth-combination-limit",
        type=int,
        help="Maximum tooth triples evaluated per topology/friction-element candidate.",
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--topology-limit", type=int)
    parser.add_argument("--element-limit-per-topology", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--checkpoint",
        help="Write resumable search progress to this JSON file.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Candidate interval for checkpoint writes when --checkpoint is set.",
    )
    parser.add_argument(
        "--resume",
        help="Resume from a checkpoint JSON file.",
    )
    parser.add_argument(
        "--probe-tooth-triples",
        type=int,
        default=0,
        help="CPU-probe this many early tooth triples before full backend evaluation.",
    )
    parser.add_argument("--include-output-brakes", action="store_true")
    parser.add_argument("--allow-internal-permanent", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--self-test", action="store_true", help="Run embedded unit checks.")
    return parser


def normalize_args(args: argparse.Namespace) -> None:
    args.targets = tuple(args.targets)
    args.standing_ratio_bounds = tuple(args.standing_ratio_bounds)

    if args.standing_ratio_bounds[0] >= args.standing_ratio_bounds[1]:
        raise ValueError("--standing-ratio-bounds MIN must be less than MAX")

    if args.tooth_combination_limit is not None and args.tooth_combination_limit < 1:
        raise ValueError("--tooth-combination-limit must be positive")

    if args.top < 1:
        raise ValueError("--top must be positive")
    if args.max_double_transitions < 0:
        raise ValueError("--max-double-transitions cannot be negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.gpu_refine_top < 1:
        raise ValueError("--gpu-refine-top must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")
    if args.probe_tooth_triples < 0:
        raise ValueError("--probe-tooth-triples cannot be negative")
    if args.sampling_mode != "exhaustive" and args.tooth_combination_limit is None:
        raise ValueError(f"--sampling-mode {args.sampling_mode} requires --tooth-combination-limit")


def topology_from_components(
    components: Sequence[Sequence[Node]],
    input_component: int,
    output_component: int,
) -> Topology:
    component_by_node = [0] * len(NODES)
    normalized_components = tuple(
        tuple(sorted(component, key=node_sort_key)) for component in components
    )
    for component_index, component in enumerate(normalized_components):
        for node in component:
            component_by_node[NODE_INDEX[node]] = component_index
    return Topology(
        permanent_edges=(),
        components=normalized_components,
        component_by_node=tuple(component_by_node),
        input_component=input_component,
        output_component=output_component,
    )


def run_self_tests() -> None:
    six_forward = (4.0, 2.5, 1.6, 1.2, 1.0, 0.75)
    assert split_targets(six_forward) == (six_forward, None)
    assert split_targets(six_forward + (-3.0,)) == (six_forward, -3.0)
    assert split_targets(DEFAULT_TARGETS) == (DEFAULT_TARGETS[:-1], DEFAULT_TARGETS[-1])

    for invalid_targets in (
        (4.0, 2.5, 1.6, 1.2, 1.0),
        six_forward + (0.6, 0.5),
        six_forward + (0.0,),
        six_forward + (0.6, -3.0, -2.0),
    ):
        try:
            split_targets(invalid_targets)
            raise AssertionError(f"invalid targets accepted: {invalid_targets}")
        except ValueError:
            pass

    left = StateResult((0, 1), 1.0, 1.0, ())
    right_single = StateResult((1, 2), 1.0, 1.0, ())
    right_double = StateResult((2, 3), 1.0, 1.0, ())
    assert transition_count(left, right_single) == 1
    assert transition_count(left, right_double) == 2

    feasible = validate_planetary_gearset(z_s=20, z_p=20, z_r=60, n_p=4)
    assert feasible.concentricity_pass
    assert feasible.assembly_pass
    assert feasible.undercutting_pass
    assert feasible.is_feasible
    assert feasible.as_dict()["is_feasible"] is True

    concentricity_failure = validate_planetary_gearset(z_s=20, z_p=19, z_r=60, n_p=4)
    assert not concentricity_failure.concentricity_pass
    assert not concentricity_failure.is_feasible
    assert "concentricity failed" in concentricity_failure.message[0]

    assembly_failure = validate_planetary_gearset(z_s=20, z_p=20, z_r=60, n_p=3)
    assert not assembly_failure.assembly_pass
    assert not assembly_failure.is_feasible

    undercutting_failure = validate_planetary_gearset(z_s=13, z_p=20, z_r=53, n_p=3)
    assert not undercutting_failure.undercutting_pass
    assert not undercutting_failure.is_feasible

    try:
        validate_planetary_gearset(z_s=20.0, z_p=20, z_r=60, n_p=4)  # type: ignore[arg-type]
        raise AssertionError("non-integer tooth counts must be rejected")
    except TypeError:
        pass

    try:
        validate_planetary_gearset(z_s=0, z_p=20, z_r=60, n_p=4)
        raise AssertionError("non-positive tooth counts must be rejected")
    except ValueError:
        pass

    try:
        validate_planetary_gearset(z_s=20, z_p=20, z_r=60, n_p=2)
        raise AssertionError("planet counts below three must be rejected")
    except ValueError:
        pass

    tooth_options_g1 = generate_tooth_options(
        rho_bounds=(2.0, 3.0),
        include_equivalent_teeth=False,
        carrier_planets=GEARSET_CARRIER_PLANETS[0],
    )
    tooth_options_g3 = generate_tooth_options(
        rho_bounds=(2.0, 3.0),
        include_equivalent_teeth=False,
        carrier_planets=GEARSET_CARRIER_PLANETS[2],
    )
    assert GEARSET_CARRIER_PLANETS == (3, 3, 4)
    assert any(abs(teeth.rho - 3.0) < 1e-12 for teeth in tooth_options_g1)
    assert any(abs(teeth.rho - 3.0) < 1e-12 for teeth in tooth_options_g3)
    assert all(teeth.carrier_planets == 3 for teeth in tooth_options_g1)
    assert all(teeth.carrier_planets == 4 for teeth in tooth_options_g3)
    assert all((teeth.ring - teeth.sun) % 2 == 0 for teeth in tooth_options_g1)
    assert all(SUN_TEETH_BOUNDS[0] <= teeth.sun <= SUN_TEETH_BOUNDS[1] for teeth in tooth_options_g1)
    assert all(
        PLANET_TEETH_BOUNDS[0] <= teeth.planet <= PLANET_TEETH_BOUNDS[1]
        for teeth in tooth_options_g1
    )
    assert all(RING_TEETH_BOUNDS[0] <= teeth.ring <= RING_TEETH_BOUNDS[1] for teeth in tooth_options_g1)
    assert all((teeth.ring + teeth.sun) % teeth.carrier_planets == 0 for teeth in tooth_options_g1)
    assert all((teeth.ring + teeth.sun) % teeth.carrier_planets == 0 for teeth in tooth_options_g3)
    assert abs(GearsetTeeth(sun=20, ring=60, planet=20, carrier_planets=4).rho - 3.0) < 1e-12

    first_topology = next(generate_topologies((3,)))
    assert first_topology.input_component != first_topology.output_component
    for gearset in range(GEARSET_COUNT):
        assert len(
            {first_topology.component_of(node_for(gearset, member)) for member in MEMBERS}
        ) == len(MEMBERS)

    topology = topology_from_components(
        (
            (
                node_for(0, "S"),
                node_for(1, "S"),
                node_for(1, "C"),
                node_for(1, "R"),
                node_for(2, "S"),
                node_for(2, "C"),
                node_for(2, "R"),
            ),
            (node_for(0, "R"),),
            (node_for(0, "C"),),
        ),
        input_component=0,
        output_component=2,
    )
    duplicate_brakes = (FrictionElement("brake", 1), FrictionElement("brake", 1))
    solved = solve_state(
        topology,
        duplicate_brakes,
        (0, 1),
        (2.0, 2.0, 2.0),
        ratio_abs_limit=50.0,
    )
    assert solved is not None
    assert abs(solved.ratio - 3.0) < 1e-8

    ratio_states = (
        StateResult((0, 1), 3.0, 1 / 3.0, ()),
        StateResult((0, 2), 3.00001, 1 / 3.00001, ()),
        StateResult((1, 2), 2.0, 1 / 2.0, ()),
    )
    assert len(unique_states_by_ratio(ratio_states, positive=True, tolerance=1e-4)) == 2

    known_elements = (
        FrictionElement("brake", 1),
        FrictionElement("brake", 1),
        FrictionElement("brake", 2),
        FrictionElement("clutch", 0, 1),
        FrictionElement("clutch", 0, 2),
    )
    known_template = build_layout_template(topology, known_elements)
    assert known_template.component_count == 3
    assert len(known_template.constant_rows) == len(STATE_COMBINATIONS)
    assert known_template.state_masks == STATE_MASKS

    known_rhos = (2.0, 2.0, 2.0)
    known_states = evaluate_states(
        topology=topology,
        elements=known_elements,
        rhos=known_rhos,
        ratio_abs_limit=50.0,
    )
    assert any(state.state == (0, 1) and abs(state.ratio - 3.0) < 1e-8 for state in known_states)
    known_forward = unique_states_by_ratio(known_states, positive=True, tolerance=1e-4)
    assert known_forward
    known_apply_chart = tuple((state.state, round(state.ratio, 8)) for state in known_forward)
    assert known_apply_chart[0][1] >= known_apply_chart[-1][1]

    if torch is not None and torch.cuda.is_available():
        torch_module = require_torch()
        device = torch_module.device("cuda")
        layout_tensors = build_torch_layout_tensors(
            known_template,
            device=device,
            dtype=torch_module.float64,
        )
        ratios, valid = solve_template_batch_torch(
            template=known_template,
            tensors=layout_tensors,
            rhos=torch_module.tensor([known_rhos], device=device, dtype=torch_module.float64),
            ratio_abs_limit=50.0,
            tolerance=1e-8,
        )
        gpu_ratios = ratios.detach().cpu().tolist()[0]
        gpu_valid = valid.detach().cpu().tolist()[0]
        for state_index, applied in enumerate(STATE_COMBINATIONS):
            cpu_state = solve_state(
                topology=topology,
                elements=known_elements,
                applied=applied,
                rhos=known_rhos,
                ratio_abs_limit=50.0,
            )
            assert gpu_valid[state_index] == (cpu_state is not None)
            if cpu_state is not None:
                assert abs(gpu_ratios[state_index] - cpu_state.ratio) < 1e-7

        gpu_teeth_options = (
            (GearsetTeeth(sun=21, ring=63, planet=21, carrier_planets=3),),
            (GearsetTeeth(sun=21, ring=63, planet=21, carrier_planets=3),),
            (GearsetTeeth(sun=20, ring=60, planet=20, carrier_planets=4),),
        )
        cpu_result = tune_candidate(
            topology=topology,
            elements=known_elements,
            tooth_options_by_gearset=gpu_teeth_options,
            tooth_combination_limit=1,
            forward_targets=(3.0,),
            reverse_target=None,
            max_double_transitions=1,
            transition_penalty=1_000_000.0,
            ratio_abs_limit=50.0,
            ratio_tolerance=1e-4,
            require_reverse=False,
            reverse_weight=0.0,
            reject_shift_violations=False,
        )
        tooth_tensors = tuple(
            build_tooth_option_tensors(options, device=device)
            for options in gpu_teeth_options
        )
        scoring_tensors = build_torch_scoring_tensors(
            forward_targets=(3.0,),
            reverse_target=None,
            device=device,
        )
        gpu_result, gpu_stats = tune_candidate_gpu(
            topology=topology,
            elements=known_elements,
            tooth_options_by_gearset=gpu_teeth_options,
            tooth_tensors_by_gearset=tooth_tensors,
            scoring_tensors=scoring_tensors,
            tooth_combination_limit=1,
            forward_targets=(3.0,),
            reverse_target=None,
            max_double_transitions=1,
            transition_penalty=1_000_000.0,
            ratio_abs_limit=50.0,
            ratio_tolerance=1e-4,
            require_reverse=False,
            reverse_weight=0.0,
            reject_shift_violations=False,
            batch_size=1,
            sampling_mode="exhaustive",
            random_seed=0,
            refine_limit=1,
            device=device,
        )
        assert gpu_stats.tooth_triples == 1
        assert (cpu_result is None) == (gpu_result is None)
        if cpu_result is not None and gpu_result is not None:
            assert abs(cpu_result.score - gpu_result.score) < 1e-7
            assert cpu_result.forward_sequence[0].state == gpu_result.forward_sequence[0].state


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        normalize_args(args)
        if np is None:
            parser.error("NumPy is required. Install dependencies from requirements-planetary.txt.")
        if args.self_test:
            run_self_tests()
            print("self-tests passed")
            return 0

        forward_targets, reverse_target = split_targets(args.targets)

        results, stats = search(args)
        if args.json:
            print(
                json.dumps(
                    {
                        "stats": stats,
                        "results": [
                            result_to_dict(result, forward_targets) for result in results
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print_text_report(results, stats, forward_targets, reverse_target)

        return 0 if results else 1
    except ValueError as error:
        parser.error(str(error))
        return 2
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
