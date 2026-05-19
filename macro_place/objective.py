"""
Proxy cost computation using PlacementCost's ground-truth evaluator.

Wraps PlacementCost methods to compute wirelength, density, and congestion
costs. Also computes overlap metrics for validation and analysis.
"""

import torch
import math
from typing import Dict, Optional

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark


# Patch PlacementCost so boundary lookups clamp into the grid.
_original_get_grid_cell_location = PlacementCost._PlacementCost__get_grid_cell_location


def _patched_get_grid_cell_location(self, x_pos, y_pos):
    """Version of the grid lookup with bounds clamping."""
    self.grid_width = float(self.width / self.grid_col)
    self.grid_height = float(self.height / self.grid_row)
    row = math.floor(y_pos / self.grid_height)
    col = math.floor(x_pos / self.grid_width)

    # Clamp the lookup into the valid grid range.
    row = max(0, min(row, self.grid_row - 1))
    col = max(0, min(col, self.grid_col - 1))

    return row, col


PlacementCost._PlacementCost__get_grid_cell_location = _patched_get_grid_cell_location


def compute_overlap_metrics(
    placement: torch.Tensor, benchmark: Benchmark
) -> Dict[str, float]:
    """
    Compute overlap metrics for macro placement.

    Borrowed from intern_challenge placement.py and adapted for macro placement.

    Args:
        placement: [num_macros, 2] tensor of (x, y) center positions
        benchmark: Benchmark object with macro sizes

    Returns:
        Dictionary with:
            - overlap_count: Number of overlapping macro pairs
            - total_overlap_area: Total area of all overlaps (um^2)
            - max_overlap_area: Largest single overlap area (um^2)
            - num_macros_with_overlaps: Number of macros involved in at least one overlap
            - overlap_ratio: Fraction of macros with overlaps
    """
    num_macros = placement.shape[0]

    if num_macros <= 1:
        return {
            "overlap_count": 0,
            "total_overlap_area": 0.0,
            "max_overlap_area": 0.0,
            "num_macros_with_overlaps": 0,
            "overlap_ratio": 0.0,
        }

    # Pull positions and sizes into NumPy.
    positions = placement.cpu().detach().numpy()
    widths = benchmark.macro_sizes[:, 0].cpu().numpy()
    heights = benchmark.macro_sizes[:, 1].cpu().numpy()

    overlap_count = 0
    total_overlap_area = 0.0
    max_overlap_area = 0.0
    macros_with_overlaps = set()

    # Only check hard-macro pairs; soft macros can overlap by design.
    num_hard = getattr(benchmark, "num_hard_macros", num_macros)
    for i in range(num_hard):
        for j in range(i + 1, num_hard):
            # Center-to-center distances.
            dx = abs(positions[i, 0] - positions[j, 0])
            dy = abs(positions[i, 1] - positions[j, 1])

            # Minimum separation for a legal placement.
            min_sep_x = (widths[i] + widths[j]) / 2.0
            min_sep_y = (heights[i] + heights[j]) / 2.0

            # Overlap depth in each dimension.
            overlap_x = max(0.0, min_sep_x - dx)
            overlap_y = max(0.0, min_sep_y - dy)

            # A pair overlaps only when both dimensions overlap.
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                overlap_count += 1
                total_overlap_area += overlap_area
                max_overlap_area = max(max_overlap_area, overlap_area)
                macros_with_overlaps.add(i)
                macros_with_overlaps.add(j)

    num_macros_with_overlaps = len(macros_with_overlaps)
    overlap_ratio = num_macros_with_overlaps / num_macros if num_macros > 0 else 0.0

    return {
        "overlap_count": overlap_count,
        "total_overlap_area": total_overlap_area,
        "max_overlap_area": max_overlap_area,
        "num_macros_with_overlaps": num_macros_with_overlaps,
        "overlap_ratio": overlap_ratio,
    }


def compute_proxy_cost(
    placement: torch.Tensor,
    benchmark: Benchmark,
    plc: PlacementCost,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute proxy cost using PlacementCost's ground-truth evaluator.

    Args:
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with circuit data
        plc: PlacementCost object
        weights: Optional cost weights

    Returns:
        Dictionary with the proxy cost terms and overlap metrics.
    """
    if weights is None:
        weights = {"wirelength": 1.0, "density": 0.5, "congestion": 0.5}

    # Push the candidate placement into PlacementCost.
    _set_placement(plc, placement, benchmark)

    # Read the proxy cost terms from PlacementCost.
    wirelength_cost = plc.get_cost()
    density_cost = plc.get_density_cost()
    congestion_cost = plc.get_congestion_cost()  # Uses the clamped lookup above.

    # Weighted sum used by the proxy objective.
    proxy = (
        weights["wirelength"] * wirelength_cost
        + weights["density"] * density_cost
        + weights["congestion"] * congestion_cost
    )

    # Collect overlap stats for validation and reporting.
    overlap_metrics = compute_overlap_metrics(placement, benchmark)

    return {
        "proxy_cost": proxy,
        "wirelength_cost": wirelength_cost,
        "density_cost": density_cost,
        "congestion_cost": congestion_cost,
        **overlap_metrics,
    }


def _set_placement(plc: PlacementCost, placement: torch.Tensor, benchmark: Benchmark):
    """
    Set macro positions in PlacementCost.

    Args:
        plc: PlacementCost object
        placement: [num_macros, 2] tensor of (x, y) positions
        benchmark: Benchmark object with macro index mappings
    """
    # Convert the tensor once for the PlacementCost API.
    placement_np = placement.cpu().numpy()

    # Cache macro_name -> [pin_indices] on the plc object.
    if not hasattr(plc, "_macro_pin_map"):
        pin_map = {}
        for idx, mod in enumerate(plc.modules_w_pins):
            if mod.get_type() == "MACRO_PIN" and hasattr(mod, "get_macro_name"):
                name = mod.get_macro_name()
                if name not in pin_map:
                    pin_map[name] = []
                pin_map[name].append(idx)
        plc._macro_pin_map = pin_map

    # Update hard macro positions first.
    for i, macro_idx in enumerate(benchmark.hard_macro_indices):
        x, y = placement_np[i]
        node = plc.modules_w_pins[macro_idx]
        node.set_pos(x, y)
        # Refresh pin positions because pin.get_pos() can cache stale values.
        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    # Then update the soft macro positions.
    num_hard = benchmark.num_hard_macros
    for i, macro_idx in enumerate(benchmark.soft_macro_indices):
        x, y = placement_np[num_hard + i]
        node = plc.modules_w_pins[macro_idx]
        node.set_pos(x, y)
        for pin_idx in plc._macro_pin_map.get(node.get_name(), []):
            pin = plc.modules_w_pins[pin_idx]
            pin.set_pos(x + pin.x_offset, y + pin.y_offset)

    # Resize the congestion arrays if PlacementCost left them stale.
    _ensure_congestion_arrays(plc)

    # Force PlacementCost to recompute all derived costs.
    plc.FLAG_UPDATE_WIRELENGTH = True
    plc.FLAG_UPDATE_DENSITY = True
    plc.FLAG_UPDATE_CONGESTION = True


def _ensure_congestion_arrays(plc: PlacementCost):
    """
    Ensure congestion arrays are sized for the current grid.

    Args:
        plc: PlacementCost object
    """
    expected_size = plc.grid_col * plc.grid_row
    current_size = len(plc.H_routing_cong)

    if current_size != expected_size:
        # Reinitialize with the expected size.
        plc.V_routing_cong = [0] * expected_size
        plc.H_routing_cong = [0] * expected_size
        plc.V_macro_routing_cong = [0] * expected_size
        plc.H_macro_routing_cong = [0] * expected_size
