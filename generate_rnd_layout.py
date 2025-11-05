#!/usr/bin/env python3

"""Generate random sequences of Pauli Products."""
from argparse import ArgumentParser

from random import randint
from random import uniform
from random import choice

from pickle import dump

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
    unfavored_qubits = [
        q for q in range(total_num_qubits)
        if q not in favored_qubits
    ]
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
    terms = [(target, choice(['X', 'Y', 'Z'])) for target in targets]
    string = [choice(['+', '-'])] + ['_'] * num_total_qubits
    for target, pauli in terms:
        string[target + 1] = pauli
    return ''.join(string) + '<pi/8>'


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
            if char != '_':
                used_qubits.add(i)
    return pauli_products


def overlaps_with(pauli_a: str, pauli_b: str) -> bool:
    """Return True if pauli_a and pauli_b overlap on any qubit."""
    for a, b in zip(pauli_a[1:-6], pauli_b[1:-6]):
        if a != '_' and b != '_':
            return True
    return False


def measure_parallelism(pauli_products: list[str]) -> float:
    """Return the average number of products per time step."""
    depth = 0
    layer = []
    for pp in pauli_products:
        if not any(overlaps_with(pp, p) for p in layer):
            layer.append(pp)
        else:
            depth += 1
            layer = [pp]
    return len(pauli_products) / (depth + 1)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--num_qubits', type=int, default=64,
        help='Total number of qubits',
    )
    parser.add_argument(
        '--num_products', type=int, default=1000,
        help='Number of Pauli products to generate',
    )
    parser.add_argument(
        '--plot', action='store_true', default=False,
        help='Plot parallelism heatmap',
    )
    args = parser.parse_args()

    num_qubits = args.num_qubits
    num_products = args.num_products

    parallelism, products = [], []
    for s in range(2, num_qubits, 2):
        for p in range(0, 10):
            p = p / 10
            pps = pauli_product_sequence(num_qubits, num_products, s, p)
            para = measure_parallelism(pps)
            print(f'size={s}, p_reuse={p/10} -> parallelism={para}')
            parallelism.append((s, p, para))
            products.append((para, pps))

    if args.plot:
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
            origin='lower',
            aspect='auto',
            extent=(min(p_vals), max(p_vals), min(s_vals), max(s_vals)),
            cmap='viridis',
        )
        plt.colorbar(im, label='Average parallelism')
        plt.xlabel('p (prob of qubit reuse)')
        plt.ylabel('s (max product size)')
        plt.title('Parallelism vs s and p')
        plt.xticks(p_vals)
        plt.yticks(s_vals)
        plt.tight_layout()
        plt.show()

    low_thresh = 1.2         # below this is low parallelism
    moderate_thresh = 2.0    # below this is moderate parallelism
    high_thresh = 3.0        # below this is high parallelism
    low, moderate, high, very_high = [], [], [], []
    for para, pps in products:
        if para < low_thresh:
            low.append((para, pps))
        elif para < moderate_thresh:
            moderate.append((para, pps))
        elif para < high_thresh:
            high.append((para, pps))
        else:
            very_high.append((para, pps))

    # Print summary
    print(f'Low parallelism: {len(low)}')
    print(f'Moderate parallelism: {len(moderate)}')
    print(f'High parallelism: {len(high)}')
    print(f'Very high parallelism: {len(very_high)}')

    prefix = f'q{num_qubits}_n{num_products}_'
    with open(f'{prefix}low.pkl', 'wb') as f:
        dump(low, f)
    with open(f'{prefix}moderate.pkl', 'wb') as f:
        dump(moderate, f)
    with open(f'{prefix}high.pkl', 'wb') as f:
        dump(high, f)
    with open(f'{prefix}very-high.pkl', 'wb') as f:
        dump(very_high, f)
