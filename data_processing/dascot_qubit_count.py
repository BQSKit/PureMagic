#!/usr/bin/env python3
"""
Calculates the total number of logical qubits required by the DASCOT
Square Sparse and Compact architectures for a circuit with N logical
data qubits.

Derived from wisq/src/wisq/architecture.py with magic_states='all_sides'.

Square Sparse layout:
  1. Inner grid: (2*ceil(sqrt(N)) + 1) x (2*ceil(sqrt(N)) + 1)
     - Data qubits at (x,y) where x%2==1 and y%2==1 → ceil(sqrt(N))^2 slots
     - Routing ancillae fill remaining inner cells
  2. 1-cell magic state border on all sides
     → final grid: (2*ceil(sqrt(N)) + 3)^2
     - Magic states: every other border cell → 4*(ceil(sqrt(N))+1)

Compact layout:
  1. Inner grid: 3 x (2*ceil(N/2) - 1)
     - Data qubits: even columns of top and bottom rows → 2*ceil(N/2) slots
     - Middle row: all routing ancillae
  2. 1-cell magic state border on all sides
     → final grid: 5 x (2*ceil(N/2) + 1)
     - Magic states: every other border cell
"""

import math


def dascot_square_sparse_qubit_count(n: int) -> dict:
    """
    Returns the total logical qubit count for the DASCOT Square Sparse
    architecture for a circuit with `n` logical data qubits.

    Args:
        n: Number of logical data qubits in the circuit.

    Returns:
        A dict with keys:
          'data'     - number of data qubit slots in the grid (>= n)
          'routing'  - number of routing ancilla qubits
          'magic'    - number of magic state qubits (border)
          'total'    - total logical qubits = data + routing + magic
          'grid_side'- side length of the final grid
    """
    s = math.ceil(math.sqrt(n))  # ceil(sqrt(N))

    # Inner grid (before border): (2s+1) x (2s+1)
    inner_side = 2 * s + 1
    inner_total = inner_side * inner_side

    # Data qubit slots: odd-x AND odd-y positions in the inner grid
    data_slots = s * s  # exactly s^2 slots (>= n by construction)

    # Final grid after adding 1-cell border on all sides: (2s+3) x (2s+3)
    final_side = inner_side + 2  # = 2s + 3
    final_total = final_side * final_side

    # Border ring cell count: (2s+3)^2 - (2s+1)^2 = 8s+8
    border_cells = final_total - inner_total  # = 8s + 8

    # Magic states: every other border cell (all_sides() picks alternating cells)
    magic = border_cells // 2  # = 4s + 4 = 4*(s+1)

    # Routing ancillae: all inner non-data cells + non-magic border cells
    routing = (inner_total - data_slots) + (border_cells - magic)

    total = data_slots + routing + magic
    assert total == final_total, f"Sanity check failed: {total} != {final_side}^2 = {final_total}"

    return {
        "data": data_slots,
        "routing": routing,
        "magic": magic,
        "total": total,
        "grid_side": final_side,
    }


def volume_from_json(filepath: str) -> dict:
    """
    Reads a DASCOT JSON output file and computes the space-time volume.

    Volume = total_logical_qubits × time_steps
    where total_logical_qubits = arch['width'] * arch['height']
    and time_steps = len(steps).

    Also computes what the qubit count and volume would be for the
    *other* architecture (square sparse ↔ compact) for the same N,
    assuming the same number of time steps (a lower bound — the other
    architecture would likely need more or fewer steps in practice).

    Args:
        filepath: Path to a DASCOT JSON output file.

    Returns:
        A dict with keys:
          'n_data'         - number of logical data qubits in the circuit
          'arch_qubits'    - total logical qubits used (width * height)
          'arch_width'     - grid width
          'arch_height'    - grid height
          'time_steps'     - number of routing time steps
          'volume'         - arch_qubits * time_steps
          'alt_arch'       - name of the alternative architecture
          'alt_qubits'     - total qubits for the alternative architecture
          'alt_volume_lb'  - alt_qubits * time_steps (lower bound on alt volume)
    """
    import json

    with open(filepath) as f:
        data = json.load(f)

    arch = data["arch"]
    width = arch["width"]
    height = arch["height"]
    n_data = len(arch["alg_qubits"])
    total_qubits = width * height
    time_steps = len(data["steps"])
    volume = total_qubits * time_steps

    # Determine which architecture this is and compute the alternative
    ss = dascot_square_sparse_qubit_count(n_data)
    cp = dascot_compact_qubit_count(n_data)

    if total_qubits == ss["total"]:
        arch_name = "square_sparse"
        alt_name = "compact"
        alt_qubits = cp["total"]
    elif total_qubits == cp["total"]:
        arch_name = "compact"
        alt_name = "square_sparse"
        alt_qubits = ss["total"]
    else:
        # Custom architecture — report both for reference
        arch_name = f"custom ({width}x{height})"
        alt_name = "square_sparse"
        alt_qubits = ss["total"]

    return {
        "n_data": n_data,
        "arch_name": arch_name,
        "arch_qubits": total_qubits,
        "arch_width": width,
        "arch_height": height,
        "time_steps": time_steps,
        "volume": volume,
        "alt_arch": alt_name,
        "alt_qubits": alt_qubits,
        "alt_volume_lb": alt_qubits * time_steps,
    }


def dascot_compact_qubit_count(n: int) -> dict:
    """
    Returns the total logical qubit count for the DASCOT Compact
    architecture for a circuit with `n` logical data qubits.

    Derived from wisq/src/wisq/architecture.py:compact_layout()
    with magic_states='all_sides'.

    Args:
        n: Number of logical data qubits in the circuit.

    Returns:
        A dict with keys:
          'data'        - number of data qubit slots in the grid (>= n)
          'routing'     - number of routing ancilla qubits
          'magic'       - number of magic state qubits (border)
          'total'       - total logical qubits = data + routing + magic
          'grid_width'  - width (columns) of the final grid
          'grid_height' - height (rows) of the final grid
    """
    c = math.ceil(n / 2)  # ceil(N/2)

    # Inner grid (before border): 3 rows x (2c-1) columns
    inner_width = 2 * c - 1
    inner_height = 3
    inner_total = inner_width * inner_height

    # Data qubit slots: even columns (0, 2, 4, ...) of top row (row 0)
    # and bottom row (row 2) → c slots per row × 2 rows = 2c slots
    data_slots = 2 * c  # >= n by construction

    # Final grid after adding 1-cell border on all sides: 5 x (2c+1)
    final_width = inner_width + 2  # = 2c + 1
    final_height = inner_height + 2  # = 5
    final_total = final_width * final_height

    # Border ring cell count: final_total - inner_total
    border_cells = final_total - inner_total

    # Magic states: every other border cell (all_sides() picks alternating cells)
    # Perimeter = 2*(final_width + final_height) - 4
    perimeter = 2 * (final_width + final_height) - 4
    magic = perimeter // 2

    # Routing ancillae: all inner non-data cells + non-magic border cells
    routing = (inner_total - data_slots) + (border_cells - magic)

    total = data_slots + routing + magic
    assert (
        total == final_total
    ), f"Sanity check failed: {total} != {final_width}x{final_height} = {final_total}"

    return {
        "data": data_slots,
        "routing": routing,
        "magic": magic,
        "total": total,
        "grid_width": final_width,
        "grid_height": final_height,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate DASCOT logical qubit counts and space-time volumes."
    )
    parser.add_argument(
        "-n",
        "--n_qubits",
        type=int,
        default=None,
        help="Number of logical data qubits. Prints qubit breakdown for both architectures.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        nargs="+",
        help="Path(s) to DASCOT JSON output file(s). Computes volume and compares architectures.",
    )
    args = parser.parse_args()

    if args.file is not None:
        # Volume analysis from DASCOT JSON output(s)
        hdr = (
            f"{'File':<30}  {'N':>4}  {'Arch':<14}  {'Qubits':>7}  "
            f"{'Steps':>7}  {'Volume':>12}  |  "
            f"{'Alt arch':<14}  {'Alt Q':>6}  {'Alt vol (lb)':>13}"
        )
        print(hdr)
        print("-" * len(hdr))
        for fpath in args.file:
            v = volume_from_json(fpath)
            import os

            fname = os.path.basename(fpath)
            print(
                f"{fname:<30}  {v['n_data']:>4}  {v['arch_name']:<14}  "
                f"{v['arch_qubits']:>7}  {v['time_steps']:>7}  {v['volume']:>12,}  |  "
                f"{v['alt_arch']:<14}  {v['alt_qubits']:>6}  {v['alt_volume_lb']:>13,}"
            )

    elif args.n_qubits is not None:
        n = args.n_qubits
        ss = dascot_square_sparse_qubit_count(n)
        cp = dascot_compact_qubit_count(n)
        s = math.ceil(math.sqrt(n))
        c = math.ceil(n / 2)
        print(f"N = {n} logical data qubits")
        print()
        print(f"  Square Sparse  (s = ceil(√N) = {s})")
        print(f"    Grid:     {ss['grid_side']} x {ss['grid_side']}")
        print(f"    Data:     {ss['data']}")
        print(f"    Routing:  {ss['routing']}")
        print(f"    Magic:    {ss['magic']}")
        print(f"    Total:    {ss['total']}")
        print()
        print(f"  Compact  (c = ceil(N/2) = {c})")
        print(f"    Grid:     {cp['grid_height']} x {cp['grid_width']}")
        print(f"    Data:     {cp['data']}")
        print(f"    Routing:  {cp['routing']}")
        print(f"    Magic:    {cp['magic']}")
        print(f"    Total:    {cp['total']}")
    else:
        hdr = (
            f"{'N':>6}  {'SS grid':>9}  {'SS data':>7}  {'SS rout':>7}  "
            f"{'SS mag':>6}  {'SS tot':>6}  |  "
            f"{'Cmp grid':>9}  {'Cmp data':>8}  {'Cmp rout':>8}  "
            f"{'Cmp mag':>7}  {'Cmp tot':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for n in [1, 4, 9, 16, 25, 36, 64, 100, 144]:
            ss = dascot_square_sparse_qubit_count(n)
            cp = dascot_compact_qubit_count(n)
            g_ss = f"{ss['grid_side']}x{ss['grid_side']}"
            g_cp = f"{cp['grid_height']}x{cp['grid_width']}"
            print(
                f"{n:>6}  {g_ss:>9}  {ss['data']:>7}  {ss['routing']:>7}  "
                f"{ss['magic']:>6}  {ss['total']:>6}  |  "
                f"{g_cp:>9}  {cp['data']:>8}  {cp['routing']:>8}  "
                f"{cp['magic']:>7}  {cp['total']:>7}"
            )
