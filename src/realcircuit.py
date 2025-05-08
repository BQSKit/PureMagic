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


class RealCircuit(list):
    def __init__(self, args):
        list.__init__(self)
        self.args = args
        self.num_pauli_products = 0
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

    def get_layers(self):
        layer_i = 0
        nodes_used = set()
        nodes_left = set()
        for node in self:
            nodes_left.add(node.id)
        layers = []
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
                    layer.append(node)
                    nodes_used.add(node.id)
                    nodes_left.remove(node.id)
            layer_i += 1
            layers.append(layer)
        return layers

    def draw_graph(self):
        g = nx.DiGraph()
        for node in self:
            g.add_node(node.id)
            for child in node.children:
                g.add_edge(node.id, child)

        layers = self.get_layers()
        print("Number of layers", len(layers))
        pos = {}
        for layer_i, layer in enumerate(layers):
            for node in layer:
                g.nodes[node.id]["layer"] = layer_i
                pos[node.id] = [layer_i, self.num_qubits - node.get_qubits()[0]]
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
        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(111)
        num_rows = self.num_qubits
        # scale the fontsize
        fs_slope = 10.0 / (56.0 - 4.0)
        fontsize = int(np.ceil(16.0 - (num_rows - 4.0) * fs_slope))
        # for i in range(num_rows):
        #    ax.text(0 - 2.5, i, "|q" + str(i) + ">", va="center", fontsize=fontsize)
        layers = self.get_layers()
        min_layer = 0
        max_layer = len(layers)
        if len(self.args.plot_circuit_range) > 0:
            min_layer, max_layer = [int(s) for s in self.args.plot_circuit_range.split(":")]

        for col, layer in enumerate(layers):
            if col < min_layer:
                continue
            if col == max_layer:
                break
            for pauli_product in layer:
                for start_pos in range(num_rows):
                    if pauli_product.operators[start_pos] != " ":
                        ax.text(
                            col,
                            start_pos - 0.15,
                            pauli_product.id,
                            va="center",
                            fontsize=fontsize * 0.8,
                            stretch="condensed",
                            rotation="vertical",
                        )
                        break
                for i in range(start_pos, num_rows):
                    if pauli_product.operators[i] != " ":
                        end_pos = i
                for i in range(start_pos, end_pos + 1):
                    if pauli_product.operators[i] == " ":
                        continue
                    ax.text(col, i, pauli_product.operators[i], va="center", fontsize=fontsize)
                rect_height = end_pos - start_pos
                top_shift = 0.11 * math.sqrt(num_rows)
                height_shift = 0.08 * math.sqrt(num_rows) + top_shift
                ax.add_patch(
                    patches.Rectangle(
                        (col - 0.1, start_pos - top_shift),
                        0.8,
                        rect_height + height_shift,
                        edgecolor="black",
                        facecolor="#ffff99" if pauli_product.is_pi_over_four() else "#99ff99",
                    )
                )
        plt.xlim(min_layer, max_layer)
        plt.ylim(num_rows - 0.5, -0.5)
        plt.xlabel("Time Steps")
        plt.ylabel("Qubits")
        plt.tight_layout()
        plt.savefig(circuit_fname + ".pdf")
        plt.savefig(circuit_fname + ".png")
        plt.show()

    def plot_freqs(self):
        hist_fname = "lssp-operator-freqs"
        print("Plotting circuit histogram to", hist_fname, "...")
        plt.close()
        plt.rcParams.update({"font.size": 10})
        plt.xlabel("number of qubits")
        plt.ylabel("Frequency")
        counts = []
        for node in self:
            counts.append(node.qubits_used)
        bins = range(max(counts) + 1)
        _, bins, _ = plt.hist(counts, bins, density=True, align="right")
        # density = (
        #    1.0
        #    / (self.sigma_qubits * np.sqrt(2 * np.pi))
        #    * np.exp(-((bins - self.mean_qubits) ** 2) / (2 * self.sigma_qubits**2))
        # )
        # plt.plot(bins, density)
        plt.grid()
        plt.tight_layout()
        plt.savefig(hist_fname + ".pdf")
        plt.savefig(hist_fname + ".png")
