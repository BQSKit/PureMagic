#!/usr/bin/env python3

"""Generate random sequences of Pauli Products."""
from argparse import ArgumentParser

from random import randint
from random import uniform
from random import choice

import numpy as np
import matplotlib.pyplot as plt


def random_size(max_size: int) -> int:
    """Generate a random size up to max_size."""
    return randint(1, max_size)


def random_targets(
    favored_qubits: list[int],
    product_size: int,
    total_num_qubits: int,
) -> list[int]:
    """Generate a random list of targets of given size."""
    assert product_size <= total_num_qubits
    chosen, pool = [], favored_qubits.copy()
    unfavored_qubits = [q for q in range(total_num_qubits) if q not in favored_qubits]
    for _ in range(product_size):
        if len(pool) < 1:
            pool += unfavored_qubits
        target = choice(pool)
        pool.pop(pool.index(target))
        chosen.append(target)
    return chosen


def random_pauli_product(
    favored_qubits: list[int],
    max_product_size: int,
    num_total_qubits: int,
) -> str:
    """Generate a random Pauli product acting on the given qubits."""
    size = random_size(max_product_size)
    targets = random_targets(favored_qubits, size, num_total_qubits)
    terms = [(target, choice(["X", "Y", "Z"])) for target in targets]
    string = [choice(["+", "-"])] + ["_"] * num_total_qubits
    for target, pauli in terms:
        string[target + 1] = pauli
    return "".join(string) + "<pi/8>"


def pauli_product_sequence(
    num_qubits: int,
    num_products: int,
    max_product_size: int,
    prob_of_qubit_reuse: float,
) -> list[str]:
    """
    Generate a sequence of random Pauli products.

    Args:
        num_qubits (int): The total number of qubits.

        num_products (int): The number of Pauli products to generate.

        max_product_size (int): The maximum size of each Pauli product.

        prob_of_qubit_reuse (float): Already used qubits may be added to
            the pool of possible targets with this probability. Higher
            values of this number will lead to less parallelism.
    """
    qubits = list(range(num_qubits))
    available_qubits = set(qubits.copy())
    used_qubits = set()
    pauli_products = []
    for _ in range(num_products):
        for q in used_qubits:
            if uniform(0, 1) < prob_of_qubit_reuse:
                available_qubits.add(q)
        pp = random_pauli_product(
            list(available_qubits),
            max_product_size,
            num_qubits,
        )
        pauli_products.append(pp)
        for i, char in enumerate(pp[1:-6]):
            if char != "_":
                used_qubits.add(i)
    return pauli_products


def overlaps_with(pauli_a: str, pauli_b: str) -> bool:
    """Return True if pauli_a and pauli_b overlap on any qubit."""
    for a, b in zip(pauli_a[1:-6], pauli_b[1:-6]):
        if a != "_" and b != "_":
            return True
    return False


def measure_parallelism(pauli_products: list[str]) -> float:
    """Return the average number of products per lcycle."""
    depth = 0
    layer = []
    for pp in pauli_products:
        if not any(overlaps_with(pp, p) for p in layer):
            layer.append(pp)
        else:
            depth += 1
            layer = [pp]
    return len(pauli_products) / (depth + 1)


def generate_circuits_for_thresholds(num_qubits, num_products_per_threshold, thresholds):
    """Generate circuits until each threshold has the required number of products."""
    circuits = [[] for _ in range(len(thresholds))]
    target_counts = [num_products_per_threshold] * len(thresholds)

    batch_size = 1000  # Generate products in batches
    iteration = 0
    thresholds_done = [False] * len(thresholds)

    while any(len(circuit) < target for circuit, target in zip(circuits, target_counts)):
        iteration += 1
        if iteration % 10 == 0:
            print(f"Generation iteration {iteration}...")

        # Generate a batch of products with
        # varying parameters
        for s in range(2, min(num_qubits, 36), 2):  # Limit max size for efficiency
            for p_int in range(0, 10):
                p = p_int / 10
                pps = pauli_product_sequence(num_qubits, batch_size, s, p)
                para = measure_parallelism(pps)

                # Assign products to appropriate threshold bucket
                for i, thresh in enumerate(thresholds):
                    if para <= thresh:
                        if len(circuits[i]) < target_counts[i]:
                            needed = target_counts[i] - len(circuits[i])
                            circuits[i].extend(pps[:needed])
                            if len(circuits[i]) >= target_counts[i] and not thresholds_done[i]:
                                print(
                                    f"Reached required number for threshold {thresh} in"
                                    f" {iteration} iterations"
                                )
                                thresholds_done[i] = True
                        break

        # Print progress
        # for i, (circuit, target) in enumerate(zip(circuits, target_counts)):
        #    print(f"  Threshold {thresholds[i]}: {len(circuit)}/{target} products")

    # Trim to exact counts if any went over
    for i in range(len(circuits)):
        circuits[i] = circuits[i][: target_counts[i]]

    return circuits


def parse_thresholds(threshold_str):
    """Parse threshold string into list of floats."""
    try:
        # Split by comma and convert to floats
        thresholds = [float(x.strip()) for x in threshold_str.split(",")]
        # Sort thresholds to ensure proper ordering
        thresholds.sort()
        return thresholds
    except ValueError as e:
        raise ValueError(f"Invalid threshold format: {e}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "-n",
        "--num_qubits",
        type=int,
        default=64,
        help="Total number of qubits",
    )
    parser.add_argument(
        "-m",
        "--num_products",
        type=int,
        default=10000,
        help="Number of Pauli products to generate",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot parallelism heatmap",
    )
    parser.add_argument(
        "-t",
        "--thresholds",
        type=str,
        default="1.2,1.5,2.0,2.5",
        help="Comma-separated list of parallelism thresholds (default: 1.2,1.5,2.0,2.5)",
    )
    parser.add_argument(
        "--add_max_threshold",
        action="store_true",
        default=True,
        help="Automatically add num_qubits as the final threshold (default: True)",
    )

    args = parser.parse_args()

    num_qubits = args.num_qubits
    num_products = args.num_products

    if args.plot:
        parallelism, products = [], []
        for s in range(2, num_qubits, 2):
            for p in range(0, 10):
                p = p / 10
                pps = pauli_product_sequence(num_qubits, num_products, s, p)
                para = measure_parallelism(pps)
                # print(f"size={s}, p_reuse={p/10} -> parallelism={para}")
                parallelism.append((s, p, para))
                products.append((para, pps))

        # Build grid of s (rows) and p (cols) from collected data
        s_vals = sorted({s for s, p, para in parallelism})
        p_vals = sorted({p for s, p, para in parallelism})

        Z = np.zeros((len(s_vals), len(p_vals)))
        for s, p, para in parallelism:
            i = s_vals.index(s)
            j = p_vals.index(p)
            Z[i, j] = para

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            Z,
            origin="lower",
            aspect="auto",
            extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)),
            cmap="viridis",
        )
        plt.colorbar(im, label="Average parallelism")
        plt.xlabel("p (prob of qubit reuse)")
        plt.ylabel("s (max product size)")
        plt.title("Parallelism vs s and p")
        plt.xticks(p_vals)
        plt.yticks(s_vals)
        plt.tight_layout()
        plt.show()

    try:
        thresholds = parse_thresholds(args.thresholds)
        if args.add_max_threshold:
            thresholds.append(num_qubits)
        print(f"Using thresholds: {thresholds}")
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    print(f"Generating {num_products} products for each of {len(thresholds)} thresholds...")
    circuits = generate_circuits_for_thresholds(num_qubits, num_products, thresholds)

    prefix = f"rnd_n{num_qubits}_m{num_products}"
    for i, circuit in enumerate(circuits):
        print(f"Circuit {thresholds[i]} has {len(circuit)} products")
        threshold_str = str(thresholds[i]).replace(".", "_")
        fname = f"{prefix}.{threshold_str}.txt"
        with open(fname, "w") as f:
            for pp in circuit:
                f.write(f"{pp}\n")
        print(f"Saved {fname}")
