#!/usr/bin/env -S python -u

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import sys
from utils import timer


class PauliProduct:

    def __init__(self, rng, num_qubits, pauli_product_qubits, start_qubit):
        self.basis_options = ["X", "Z", "Y"]
        # self.basis_options = ["X", "Z"]
        self.operators = [" "] * num_qubits
        self.start_qubit = start_qubit
        self.qubits_used = pauli_product_qubits
        for i in range(start_qubit, start_qubit + pauli_product_qubits):
            self.operators[i] = self.basis_options[int(np.floor(rng.uniform(0, len(self.basis_options))))]

    def __str__(self):
        s = ""
        for i in range(len(self.operators)):
            if self.operators[i] != " ":
                s += str(i) + self.operators[i] + " "
        return s.strip()


def print_circuit(circuit):
    for i, pauli_product in enumerate(circuit):
        print(i, pauli_product)


@timer
def plot_circuit(circuit):
    circuit_fname = "lssp-circuit"
    print("Drawing circuit...", circuit_fname)

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_rows = len(circuit[0][0].operators)
    # scale the fontsize
    fs_slope = 10.0 / (56.0 - 4.0)
    fontsize = int(np.ceil(16.0 - (num_rows - 4.0) * fs_slope))
    for i in range(num_rows):
        ax.text(0 - 1.5, i, "|q" + str(i) + ">", va="center", fontsize=fontsize)
    for col, circuit_cycle in enumerate(circuit):
        for pauli_product in circuit_cycle:
            for start_pos in range(num_rows):
                if pauli_product.operators[start_pos] != " ":
                    break
            ry_start = None
            for i in range(start_pos, num_rows):
                if pauli_product.operators[i] == " ":
                    break
                ax.text(col, i, pauli_product.operators[i], va="center", fontsize=fontsize)
                if ry_start == None:
                    ry_start = i
                ry_end = i
            rect_height = ry_end - ry_start
            top_shift = 0.11 * math.sqrt(num_rows)
            height_shift = 0.08 * math.sqrt(num_rows) + top_shift
            ax.add_patch(
                patches.Rectangle(
                    (col - 0.1, ry_start - top_shift), 0.45, rect_height + height_shift, edgecolor="black", facecolor="lightgreen"
                )
            )
    plt.xlim(-1.8, len(circuit))
    plt.ylim(num_rows, -1)
    plt.tick_params(axis="y", left=False, labelleft=False)
    plt.tick_params(axis="x", bottom=False, labelbottom=False)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(circuit_fname + ".pdf")
    plt.savefig(circuit_fname + ".png")


def gen_rnd_circuit_cycle(args, rng, num_qubits, mean_qubits, sigma_qubits):
    pauli_products = []
    start_qubit = 0
    # print("Pauli products to schedule:")
    for tries in range(10000):
        # this is a hack to ensure only positive numbers for the normal sampling
        for _ in range(100):
            pauli_product_qubits = int(np.floor(rng.normal(mean_qubits, sigma_qubits)))
            if pauli_product_qubits > 0 and pauli_product_qubits <= num_qubits:
                break
        else:
            print("Couldn't generate a random number in range [0, %d], using %d" % (num_qubits, mean_qubits), file=sys.stderr)
            pauli_product_qubits = mean_qubits

        if start_qubit + pauli_product_qubits > num_qubits:
            # retry to generate a smaller Pauli product
            continue
        pauli_products.append(PauliProduct(rng, num_qubits, pauli_product_qubits, start_qubit))
        # print(" ", pauli_products[-1])
        start_qubit += pauli_product_qubits
        gap_prob = args.gap_prob
        while rng.uniform(0, 1) < gap_prob:
            start_qubit += 1
            if start_qubit >= num_qubits:
                break
            gap_prob /= 2.0

    if args.verbose:
        print("Generated", len(pauli_products), "Pauli products in cycle")
    return pauli_products


@timer
def gen_rnd_circuit(args, rng, num_qubits):
    mean_qubits = float(num_qubits) * args.qubits_per_pauli_product
    sigma_qubits = 2.0
    circuit = []
    num_pauli_products = 0
    counts = []
    for i in range(args.circuit_depth):
        circuit.append(gen_rnd_circuit_cycle(args, rng, num_qubits, mean_qubits, sigma_qubits))
        num_pauli_products += len(circuit[-1])
        for pp in circuit[-1]:
            counts.append(pp.qubits_used)
    print(
        "Generated",
        num_pauli_products,
        "Pauli products, an average of %.3f per cycle" % (float(num_pauli_products) / args.circuit_depth),
    )

    if args.plot in ["freqs", "all"]:
        hist_fname = "lssp-operator-freqs"
        print("Plotting circuit histogram to", hist_fname, "...")
        plt.close()
        plt.rcParams.update({"font.size": 10})
        plt.xlabel("number of qubits")
        plt.ylabel("Frequency")
        bins = range(max(counts) + 1)
        _, bins, _ = plt.hist(counts, bins, density=True, align="right")
        density = 1.0 / (sigma_qubits * np.sqrt(2 * np.pi)) * np.exp(-((bins - mean_qubits) ** 2) / (2 * sigma_qubits**2))
        plt.plot(bins, density)
        plt.grid()
        plt.tight_layout()
        plt.savefig(hist_fname + ".pdf")
        plt.savefig(hist_fname + ".png")

    return circuit
