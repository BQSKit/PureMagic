#!/usr/bin/env -S python -u

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import sys
from utils import timer
import pauliproduct


class RndCircuit(list):
    def __init__(self, args, rng, num_qubits):
        list.__init__(self)
        self.args = args
        self.mean_qubits = float(num_qubits) * args.qubits_per_pauli_product
        self.sigma_qubits = 2.0
        self.num_pauli_products = 0
        self.counts = []
        self.rng = rng
        self.num_qubits = num_qubits
        self.gen_rnd_circuit()

    @timer
    def gen_rnd_circuit(self):
        for i in range(self.args.circuit_depth):
            self.append(self.gen_rnd_circuit_cycle())
            self.num_pauli_products += len(self[-1])
            for pp in self[-1]:
                self.counts.append(pp.qubits_used)
        print(
            "Generated",
            self.num_pauli_products,
            "Pauli products, an average of %.3f per cycle" % (float(self.num_pauli_products) / self.args.circuit_depth),
        )

    def gen_rnd_circuit_cycle(self):
        pauli_products = []
        start_qubit = 0
        # print("Pauli products to schedule:")
        for _ in range(10000):
            # this is a hack to ensure only positive numbers for the normal sampling
            for _ in range(100):
                pauli_product_qubits = int(np.floor(self.rng.normal(self.mean_qubits, self.sigma_qubits)))
                if pauli_product_qubits > 0 and pauli_product_qubits <= self.num_qubits:
                    break
            else:
                print(
                    "Couldn't generate a random number in range [0, %d], using %d" % (self.num_qubits, self.mean_qubits),
                    file=sys.stderr,
                )
                pauli_product_qubits = self.mean_qubits

            if start_qubit + pauli_product_qubits > self.num_qubits:
                # retry to generate a smaller Pauli product
                continue
            pauli_products.append(pauliproduct.PauliProduct(self.rng, self.num_qubits, pauli_product_qubits, start_qubit))
            # print(" ", pauli_products[-1])
            start_qubit += pauli_product_qubits
            gap_prob = self.args.gap_prob
            while self.rng.uniform(0, 1) < gap_prob:
                start_qubit += 1
                if start_qubit >= self.num_qubits:
                    break
                gap_prob /= 2.0

        if self.args.verbose:
            print("Generated", len(pauli_products), "Pauli products in cycle")
        return pauli_products

    def __str__(self):
        s = ""
        for i, pauli_product in enumerate(self):
            s = str(i) + " " + pauli_product.__str__() + "\n"
        return s

    @timer
    def plot(self):
        circuit_fname = "lssp-circuit"
        print("Drawing circuit...", circuit_fname)

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        num_rows = len(self[0][0].operators)
        # scale the fontsize
        fs_slope = 10.0 / (56.0 - 4.0)
        fontsize = int(np.ceil(16.0 - (num_rows - 4.0) * fs_slope))
        for i in range(num_rows):
            ax.text(0 - 1.5, i, "|q" + str(i) + ">", va="center", fontsize=fontsize)
        for col, circuit_cycle in enumerate(self):
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
                        (col - 0.1, ry_start - top_shift),
                        0.45,
                        rect_height + height_shift,
                        edgecolor="black",
                        facecolor="lightgreen",
                    )
                )
        plt.xlim(-1.8, len(self))
        plt.ylim(num_rows, -1)
        plt.tick_params(axis="y", left=False, labelleft=False)
        plt.tick_params(axis="x", bottom=False, labelbottom=False)
        plt.box(False)
        plt.tight_layout()
        plt.savefig(circuit_fname + ".pdf")
        plt.savefig(circuit_fname + ".png")

    def plot_freqs(self):
        hist_fname = "lssp-operator-freqs"
        print("Plotting circuit histogram to", hist_fname, "...")
        plt.close()
        plt.rcParams.update({"font.size": 10})
        plt.xlabel("number of qubits")
        plt.ylabel("Frequency")
        bins = range(max(self.counts) + 1)
        _, bins, _ = plt.hist(self.counts, bins, density=True, align="right")
        density = (
            1.0
            / (self.sigma_qubits * np.sqrt(2 * np.pi))
            * np.exp(-((bins - self.mean_qubits) ** 2) / (2 * self.sigma_qubits**2))
        )
        plt.plot(bins, density)
        plt.grid()
        plt.tight_layout()
        plt.savefig(hist_fname + ".pdf")
        plt.savefig(hist_fname + ".png")
