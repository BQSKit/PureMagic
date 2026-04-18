#!/usr/bin/env python3
"""Analyze WISQ JSON files: count steps, total IDs, and average IDs per step."""

import json
import sys
import os
import math


def dascot_square_sparse_qubit_count(n: int) -> dict:
    """
    Returns the total logical qubit count for the DASCOT Square Sparse
    architecture for a circuit with `n` logical data qubits.
    """
    s = math.ceil(math.sqrt(n))  # ceil(sqrt(N))
    inner_side = 2 * s + 1
    inner_total = inner_side * inner_side
    data_slots = s * s
    final_side = inner_side + 2  # = 2s + 3
    final_total = final_side * final_side
    border_cells = final_total - inner_total
    magic = border_cells // 2
    routing = (inner_total - data_slots) + (border_cells - magic)
    total = data_slots + routing + magic
    assert total == final_total
    return {
        "data": data_slots,
        "routing": routing,
        "magic": magic,
        "total": total,
        "grid_side": final_side,
    }


def analyze_wisq_json(filepath):
    print(f"File: {filepath}")
    with open(filepath) as f:
        data = json.load(f)

    steps = data["steps"]
    num_steps = len(steps)

    all_ids = [entry["id"] for step in steps for entry in step]
    num_ids = len(all_ids)

    avg_ids_per_step = num_ids / num_steps if num_steps > 0 else 0

    # Number of logical data qubits from the qubit map
    n_data = len(data["map"])
    ss = dascot_square_sparse_qubit_count(n_data)
    area = ss["total"]
    wisq_volume = area * num_steps

    # Alternative area: each magic qubit costs 121 logical qubits instead of 1
    area_magic11 = (area - ss["magic"]) + ss["magic"] * 121
    wisq_volume_magic11 = area_magic11 * num_steps

    print(f"  Number of steps:              {num_steps}")
    print(f"  Total number of IDs:          {num_ids}")
    print(f"  Average IDs per step:         {avg_ids_per_step:.2f}")
    print(f"  Logical data qubits (n):      {n_data}")
    print(
        f"  Square sparse total area:     {area}  (grid {ss['grid_side']}x{ss['grid_side']}, magic={ss['magic']})"
    )
    print(f"  WISQ volume (area × steps):   {wisq_volume}")
    print(
        f"  Area (magic×121 qubits):      {area_magic11}  (non-magic={area - ss['magic']}, magic={ss['magic']}×121={ss['magic']*121})"
    )
    print(f"  WISQ volume (magic×121):      {wisq_volume_magic11}")
    print()


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default: all JSON files in results/all_wisq
        directory = os.path.join(os.path.dirname(__file__), "..", "results", "all_wisq")
        files = sorted(
            os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")
        )

    for filepath in files:
        analyze_wisq_json(filepath)


if __name__ == "__main__":
    main()
