#!/usr/bin/env -S python -u

import os
import sys
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once")
    import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import pickle
from pathlib import Path
from utils import timer
from pauliproduct import PauliProduct, Operator


class RealCircuit(list):
    def __init__(self, args):
        list.__init__(self)
        self.args = args
        self.num_pauli_products = 0
        self.num_qubits = 0
        self.load_circuit()

    @timer
    def load_circuit(self):
        dag_df = pd.read_csv(self.args.circuit, sep="\t")
        for i, row in dag_df.iterrows():
            self.append(PauliProduct())
            self[-1].set_vals(row["id"], row["product"], row["parents"], row["children"])
        self.num_qubits = max(pp.max_qubit for pp in self)
        print(f"Loaded circuit with {len(self)} products and {self.num_qubits} qubits")

    def get_layers(self):
        layer_i = 0
        pps_used = set()
        pps_left = set()
        for pp in self:
            pps_left.add(pp.id)
        layers = []
        while pps_left:
            layer = []
            pps_left_copy = pps_left.copy()
            pps_used_copy = pps_used.copy()
            for pp_id in pps_left_copy:
                pp = self[pp_id]
                for parent in pp.parents:
                    if parent not in pps_used_copy:
                        break
                else:
                    layer.append(pp)
                    pps_used.add(pp.id)
                    pps_left.remove(pp.id)
            layer_i += 1
            layers.append(layer)
        return layers

    def get_statistics(self):
        layers = self.get_layers()
        num_noncliffords = [0] * len(layers)
        num_odd_ys = [0] * len(layers)
        num_ys = [0] * len(layers)
        num_nonclifford_layers = 0
        for i, layer in enumerate(layers):
            nonclifford_layer = False
            for pp in layer:
                if not pp.is_clifford():
                    num_noncliffords[i] += 1
                    nonclifford_layer = True
                if pp.num_ys > 0:
                    num_ys[i] += 1
                    if pp.num_ys % 2 == 1:
                        num_odd_ys[i] += 1
            if nonclifford_layer:
                num_nonclifford_layers += 1
        return (
            len(layers),
            max(num_noncliffords),
            np.mean(num_noncliffords),
            max(num_odd_ys),
            np.mean(num_odd_ys),
            max(num_ys),
            np.mean(num_ys),
            num_nonclifford_layers,
        )

    def check_clifford_relations(self):
        for node in self:
            # a clifford shouldn't have any non-clifford children
            if node.is_clifford():
                for child_id in node.children:
                    if not self[child_id].is_clifford():
                        raise RuntimeError(
                            f"Node {node.id} is a clifford but has a non-clifford child {child_id}"
                        )

    def split_ys(self):
        new_pps = []
        new_pp_id = len(self)
        for pp in self:
            if pp.num_ys > 0:
                new_pp = PauliProduct()
                new_pp.id = new_pp_id
                new_pp_id += 1
                new_pp.angle = pp.angle
                new_pp.num_ys = pp.num_ys
                new_pp.need_estabilizer = True
                new_pp.operators = []
                pp_operators_updated = []
                # convert product to X only
                for i, op in enumerate(pp.operators):
                    if op.basis == "X":
                        pp_operators_updated.append(op)
                    elif op.basis == "Z":
                        new_pp.operators.append(op)
                    elif op.basis == "Y":
                        pp_operators_updated.append(Operator(op.qubit, "x"))
                        new_pp.operators.append(Operator(op.qubit, "z"))
                pp.operators = pp_operators_updated
                # now set the parents and children appropriately
                new_pp.children = pp.children.copy()
                new_pp.parents = [pp.id]
                pp.children = [new_pp.id]
                new_pps.append(new_pp)
                for child_id in new_pp.children:
                    self[child_id].parents[self[child_id].parents.index(pp.id)] = new_pp.id
        self.extend(new_pps)
        print(
            f"After splitting {len(new_pps)} Y products there are "
            f"{len(self)} products in the circuit"
        )

    def draw_graph(self):
        g = nx.DiGraph()
        for node in self:
            g.add_node(node.id)
            for child in node.children:
                g.add_edge(node.id, child)

        layers = self.get_layers()
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

    def print(self, fname):
        f = open(fname, "w")
        print("id product Ys ES children parents", file=f)
        for i, pauli_product in enumerate(self):
            print(f"{pauli_product.__str__()}", file=f)
        f.close()

    @timer
    def plot(self, show_product_ids):
        plt.rcParams["font.size"] = 20
        circuit_fname = Path(self.args.circuit).stem + ".circuit"
        print("Drawing circuit...", circuit_fname)
        self.print(circuit_fname + ".txt")

        plt.close()
        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(111)
        num_rows = self.num_qubits
        # scale the fontsize
        fs_slope = 10.0 / (56.0 - 4.0)
        fontsize = max(int(np.ceil(16.0 - (num_rows - 4.0) * fs_slope)), 6)
        # for i in range(num_rows):
        #    ax.text(0 - 2.5, i, "|q" + str(i) + ">", va="center", fontsize=fontsize)
        layers = self.get_layers()
        print("Number of layers", len(layers))
        min_layer = 0
        max_layer = len(layers)
        if len(self.args.plot_circuit_range) > 0:
            min_layer, max_layer = [int(s) for s in self.args.plot_circuit_range.split(":")]
            max_layer = min(len(layers), max_layer)
            min_layer = min(max_layer, min_layer)
        num_layers = max_layer - min_layer

        for col, layer in enumerate(layers):
            if col < min_layer:
                continue
            if col == max_layer:
                break
            for pauli_product in layer:
                for op in pauli_product.operators:
                    if show_product_ids:
                        ax.text(
                            col,
                            op.qubit - 0.15,
                            pauli_product.id,
                            va="center",
                            fontsize=fontsize * 0.8,
                            stretch="condensed",
                            rotation="vertical",
                        )
                    break
                for op in pauli_product.operators:
                    if num_layers <= 100 and not show_product_ids:
                        ax.text(col, op.qubit, op.basis, va="center", fontsize=fontsize)
                start_pos = pauli_product.operators[0].qubit
                end_pos = pauli_product.operators[-1].qubit
                rect_height = end_pos - start_pos
                top_shift = 0.11 * math.sqrt(num_rows)
                height_shift = 0.08 * math.sqrt(num_rows) + top_shift
                ax.add_patch(
                    patches.Rectangle(
                        (col - 0.1, start_pos - top_shift),
                        0.8,
                        rect_height + height_shift,
                        # edgecolor="black",
                        lw=0.2,
                        edgecolor="none",
                        facecolor="#cccc22" if pauli_product.is_clifford() else "#22ff22",
                    )
                )
        plt.xlim(min_layer, max_layer)
        plt.ylim(num_rows - 0.5, -0.5)
        plt.xlabel("Time Steps")
        plt.ylabel("Qubits")
        plt.title(Path(self.args.circuit).stem)
        plt.tight_layout()
        plt.savefig(circuit_fname + ".pdf")
        plt.savefig(circuit_fname + ".png")
        # plt.show()

    def plot_freqs(self):
        hist_fname = Path(self.args.circuit).stem + ".freqs"
        print("Plotting circuit histogram to", hist_fname, "...")
        plt.close()
        plt.rcParams.update({"font.size": 10})
        plt.xlabel("number of qubits")
        plt.ylabel("Frequency")
        # plt.yscale("log")
        counts = []
        for node in self:
            counts.append(node.qubits_used)
        bins = range(max(counts) + 1)
        _, bins, _ = plt.hist(counts, bins, density=True, align="left")
        plt.grid()
        plt.tight_layout()
        plt.savefig(hist_fname + ".pdf")
        # plt.savefig(hist_fname + ".png")
