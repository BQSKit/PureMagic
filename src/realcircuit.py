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
        layers = self.get_layers()
        min_layer = 0
        max_layer = len(layers)
        if len(self.args.plot_circuit_range) > 0:
            min_layer, max_layer = [int(s) for s in self.args.plot_circuit_range.split(":")]
            max_layer = min(len(layers), max_layer)
            min_layer = min(max_layer, min_layer)
        self.plot_range(layers[min_layer:max_layer], min_layer, show_product_ids)

    def plot_range(self, layers, min_layer, show_product_ids):
        plt.rcParams["font.size"] = 10
        circuit_fname = Path(self.args.circuit).stem + ".circuit"
        print("Printing circuit to ", circuit_fname + ".txt")
        circuit_ofile = open(circuit_fname + ".txt", "w")

        plt.close()
        num_rows = self.num_qubits
        num_layers = len(layers)
        max_layer = min_layer + num_layers
        fig = plt.figure(figsize=(0.17 * float(num_layers), 0.22 * float(num_rows)))
        ax = fig.add_subplot(111)

        for col, layer in enumerate(layers):
            col += min_layer
            for pauli_product in layer:
                print(f"{col}: {str(pauli_product)}", file=circuit_ofile)
                if show_product_ids:
                    ax.text(
                        col,
                        pauli_product.operators[0].qubit - 0.15,
                        pauli_product.id,
                        va="center",
                        fontsize=8,
                        stretch="condensed",
                        rotation="vertical",
                    )
                else:
                    for op in pauli_product.operators:
                        ax.text(col, op.qubit, op.basis, va="center", fontsize=8)
                start_pos = pauli_product.operators[0].qubit
                end_pos = pauli_product.operators[-1].qubit
                rect_height = (end_pos - start_pos) + 0.8
                ax.add_patch(
                    patches.Rectangle(
                        (col - 0.1, start_pos - 0.4),
                        0.8,
                        rect_height,
                        lw=0.2,
                        edgecolor="none",
                        facecolor="#cccc22" if pauli_product.is_clifford() else "#22ff22",
                    )
                )
        circuit_ofile.close()
        plt.xlim(min_layer, max_layer)
        plt.ylim(num_rows + 0.5, -0.5)
        plt.xticks(range(min_layer, max_layer, 5))
        ax.tick_params(axis="x", which="both", top=True, labeltop=True)
        plt.yticks(range(0, num_rows + 1))
        plt.xlabel("Time Steps")
        plt.ylabel("Qubits")
        plt.title(Path(self.args.circuit).stem)
        plt.tight_layout()
        # plt.savefig(circuit_fname + ".pdf")
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
