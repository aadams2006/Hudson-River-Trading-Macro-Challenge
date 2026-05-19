"""
Greedy macro placer with a few row-ordering heuristics.

It tries several simple packings and keeps the one with the lowest
estimated wirelength.

Usage:
    uv run evaluate submissions/sa_placer.py
    uv run evaluate submissions/sa_placer.py --all
    uv run evaluate submissions/sa_placer.py -b ibm03
"""

import torch
import random

from macro_place.benchmark import Benchmark


class SAPlacerCompetitive:
    """Greedy macro placer with multiple row-ordering heuristics."""

    def __init__(self, seed: int = 42):
        """Initialize the placer with a fixed random seed."""
        self.seed = seed
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

    def _estimate_wirelength(
        self, placement: torch.Tensor, benchmark: Benchmark
    ) -> float:
        """Quick HPWL estimate on the hard-macro nets."""
        cost = 0.0
        for net_nodes in benchmark.net_nodes:
            if len(net_nodes) < 2:
                continue
            # Estimate HPWL on hard macros only.
            hard_nodes = [int(n) for n in net_nodes if int(n) < benchmark.num_hard_macros]
            if len(hard_nodes) < 2:
                continue

            positions = placement[hard_nodes]
            min_x = positions[:, 0].min().item()
            max_x = positions[:, 0].max().item()
            min_y = positions[:, 1].min().item()
            max_y = positions[:, 1].max().item()

            hpwl = (max_x - min_x) + (max_y - min_y)
            cost += hpwl

        return cost

    def _place_greedy(
        self, benchmark: Benchmark, sort_key: callable
    ) -> torch.Tensor:
        """
        Greedy row packing with a caller-supplied sort key.

        Args:
            benchmark: Benchmark object
            sort_key: Function that takes an index and returns a sort value

        Returns:
            Placement tensor
        """
        placement = benchmark.macro_positions.clone()

        # Work on movable hard macros only.
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_indices = torch.where(movable)[0].tolist()

        if not movable_indices:
            return placement

        sizes = benchmark.macro_sizes
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        # Order macros with the requested heuristic.
        movable_indices.sort(key=sort_key)

        gap = 0.001
        cursor_x = 0.0
        cursor_y = 0.0
        row_height = 0.0

        for idx in movable_indices:
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            # Start a new row when the current one runs out of width.
            if cursor_x + w > canvas_w:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0

            # Fall back to the origin if we run out of height.
            if cursor_y + h > canvas_h:
                placement[idx, 0] = w / 2
                placement[idx, 1] = h / 2
                continue

            # Store the macro center.
            placement[idx, 0] = cursor_x + w / 2
            placement[idx, 1] = cursor_y + h / 2

            cursor_x += w + gap
            row_height = max(row_height, h)

        return placement

    def _get_net_degree(self, benchmark: Benchmark) -> torch.Tensor:
        """Compute the number of nets attached to each macro."""
        degree = torch.zeros(benchmark.num_macros, dtype=torch.long)
        for net_nodes in benchmark.net_nodes:
            for node in net_nodes:
                node_idx = int(node)
                if node_idx < len(degree):
                    degree[node_idx] += 1
        return degree

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """
        Generate a placement by picking the best row-ordering heuristic.

        Args:
            benchmark: Benchmark object with circuit data

        Returns:
            placement: [num_macros, 2] tensor of (x, y) center positions
        """
        sizes = benchmark.macro_sizes
        net_degree = self._get_net_degree(benchmark)

        # Candidate row-ordering heuristics.
        strategies = [
            # Height first.
            lambda i: -sizes[i, 1].item(),
            # Width first.
            lambda i: -sizes[i, 0].item(),
            # Area first.
            lambda i: -(sizes[i, 0] * sizes[i, 1]).item(),
            # Prefer taller blocks before wider ones.
            lambda i: -(sizes[i, 1] / (sizes[i, 0] + 1e-6)).item(),
            # Higher net degree first.
            lambda i: -net_degree[i].item(),
        ]

        # Try each ordering and keep the lowest HPWL estimate.
        best_placement = None
        best_cost = float('inf')

        for strategy in strategies:
            try:
                placement = self._place_greedy(benchmark, strategy)
                cost = self._estimate_wirelength(placement, benchmark)

                if cost < best_cost:
                    best_cost = cost
                    best_placement = placement.clone()
            except Exception:
                continue

        # Default to the input placement if every heuristic fails.
        if best_placement is None:
            best_placement = benchmark.macro_positions.clone()

        # Fixed macros stay where the benchmark puts them.
        fixed_mask = benchmark.macro_fixed
        best_placement[fixed_mask] = benchmark.macro_positions[fixed_mask]

        return best_placement
