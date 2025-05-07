#!/usr/bin/env -S python -u

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import pickle
from utils import timer
import pauliproduct

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/../../quilt")
import quilt


class RealCircuit(list):
    def __init__(self, args):
        list.__init__(self)
        self.args = args
        self.num_pauli_products = 0
        self.counts = []
        self.num_qubits = 0
        self.load_circuit()

    def load_circuit(self):
        f = open(self.args.circuit, "rb")
        dag = pickle.load(f)
        dag.print(self.args.circuit + ".txt")

        self.num_qubits = 0
        for node in dag.nodes.values():
            self.num_qubits = max(self.num_qubits, node.product.qubits[-1])
        self.num_qubits += 1
        for node in dag.nodes.values():
            self.append(pauliproduct.PauliProduct(self.num_qubits))
            self[-1].set(node.id, node.product, dag.parents(node), dag.children(node))
            # print(self[-1])

    def draw_graph(self):
        g = nx.DiGraph()
        for node in self:
            for child in node.children:
                g.add_edge(node.id, child)

        layer_i = 0
        nodes_used = set()
        nodes_left = set()
        pos = {}
        max_qubit = 0
        for node in self:
            nodes_left.add(node.id)
        while nodes_left:
            layer = []
            nodes_left_copy = nodes_left.copy()
            nodes_used_copy = nodes_used.copy()
            for node_id in nodes_left_copy:
                node = self[node_id]
                for parent in node.parents:
                    if parent not in nodes_used_copy:
                        break
                else:
                    g.nodes[node.id]["layer"] = layer_i
                    layer.append((node.id, node.get_product_str(), node.get_qubits()))
                    pos[node.id] = [layer_i, max_qubit - node.get_qubits()[0]]
                    nodes_used.add(node.id)
                    nodes_left.remove(node.id)
            layer_i += 1
            print(layer)
        print("Number of layers", layer_i)

        print("Number of nodes", g.number_of_nodes())
        nx.draw_networkx(g, pos=pos)
        plt.show()

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
