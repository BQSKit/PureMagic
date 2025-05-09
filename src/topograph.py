#!/usr/bin/env -S python -u

import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from pathlib import Path
from utils import timer


def is_magic_node(node):
    assert node[0] in ["m", "b", "d"]
    return node[0] == "m"


def is_data_node(node):
    assert node[0] in ["m", "b", "d"]
    return node[0] == "d"


def is_bus_node(node):
    assert node[0] in ["m", "b", "d"]
    return node[0] == "b"


def get_node_label(label, col, row):
    return label + str(math.ceil(col)) + "-" + str(math.ceil(row))


class TopoGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def get_topo_dims(self):
        sq_dim = int(np.floor(np.sqrt(self.args.min_num_qubits)))
        patch_rows = int(sq_dim / 2) + sq_dim % 2
        qubits_per_col = 2 * patch_rows
        num_cols = 2 * int(np.ceil(self.args.min_num_qubits / qubits_per_col)) + 1
        # 2 rows for magic, 1 always for bus, 2 per patch row, rows for bus qubits
        num_rows = 2 + 1 + 2 * patch_rows + int(patch_rows / self.args.bus_ratio)
        print("Layout dimensions:", num_cols, num_rows)
        return num_cols, num_rows

    def set_dims(self, args, rng):
        self.args = args
        self.rng = rng
        self.num_cols, self.num_rows = self.get_topo_dims()
        if self.num_cols > 0 and self.num_rows > 0:
            self.gen_topo()

    def gen_magic_columns(self):
        for col in range(self.num_cols):
            if col % 2 != 0:
                continue
            node_label = self.add_labeled_node("m", col, 0)
            self.add_edge(node_label, get_node_label("b", col, 1))
            for row in range(1, self.num_rows - 2):
                node_label = self.add_labeled_node("b", col, row)
                self.add_edge(node_label, get_node_label("b", col, row + 1))
            prev_node_label = self.add_labeled_node("b", col, self.num_rows - 2)
            node_label = self.add_labeled_node("m", col, self.num_rows - 1)
            self.add_edge(node_label, prev_node_label, other="")

    def add_bus_row(self, col, row):
        node_label = self.add_labeled_node("b", col, row)
        self.add_edge(node_label, get_node_label("b", col - 1, row))
        self.add_edge(node_label, get_node_label("b", col + 1, row))

    def add_data_row(self, qi, col, row, op):
        q = int(qi / 2) if op == "X" else int(qi / 2) - 1
        node_label1 = "d" + str(q) + op
        node_label2 = "d" + str(q + 1) + op
        other = None
        self.add_node(node_label1, pos=[float(col) - 0.35, self.num_rows - 1 - row], color="#9999FF", other=other)
        self.add_node(node_label2, pos=[float(col) + 0.35, self.num_rows - 1 - row], color="#9999FF", other=other)
        self.add_edge(get_node_label("b", col - 1, row), node_label1)
        self.add_edge(get_node_label("b", col + 1, row), node_label2)

    def shuffle_qubits(self):
        qubit_order = list(range(self.num_data_qubits))
        self.rng.shuffle(qubit_order)
        # very good for grover 5 node
        # qubit_order = [4, 1, 0, 5, 3, 2]
        # very bad for grover 5 node
        # qubit_order = [1, 0, 4, 2, 5, 3]
        print("Using qubit order:", qubit_order)
        qubit_map = {}
        for i, new_i in enumerate(qubit_order):
            qubit_map["d" + str(i) + "X"] = "dd" + str(new_i) + "X"
            qubit_map["d" + str(i) + "Z"] = "dd" + str(new_i) + "Z"
        nx.relabel_nodes(self, qubit_map, copy=False)
        qubit_map = {}
        for i in range(self.num_data_qubits):
            qubit_map["dd" + str(i) + "X"] = "d" + str(i) + "X"
            qubit_map["dd" + str(i) + "Z"] = "d" + str(i) + "Z"
        nx.relabel_nodes(self, qubit_map, copy=False)

    @timer
    def gen_topo(self):
        self.gen_magic_columns()
        qi = 0
        spacing = self.args.bus_ratio * 2 + 1
        print("spacing", spacing)
        for col in range(self.num_cols):
            if col % 2 == 0:
                continue
            bus_rows = 0
            for row in range(1, self.num_rows - 1):
                offset = row - 1
                if offset % spacing == 0:
                    self.add_bus_row(col, row)
                    bus_rows += 1
                else:
                    self.add_data_row(qi, col, row, "X" if (row - bus_rows) % 2 == 1 else "Z")
                    qi += 2
        num_data_qubits = int(sum([is_data_node(node) for node in self.nodes]) / 2)
        num_magic_qubits = sum([is_magic_node(node) for node in self.nodes])
        num_bus_qubits = sum([is_bus_node(node) for node in self.nodes])
        print("Number of qubits:")
        print("  magic:", num_magic_qubits)
        print("  data: ", num_data_qubits)
        print("  bus:  ", num_bus_qubits)
        print("Space efficiency: %.2f" % (float(num_data_qubits) / (num_data_qubits + num_bus_qubits)))
        print("Magic state ratio: %.2f" % (float(num_magic_qubits) / (num_data_qubits + num_magic_qubits)))
        self.num_data_qubits = num_data_qubits
        self.num_magic_qubits = num_magic_qubits
        self.num_bus_qubits = num_bus_qubits
        if self.args.rnd_order:
            self.shuffle_qubits()

    def add_labeled_node(self, label, col, row):
        node_colors = {"m": "#FFBB99", "b": "#cccccc", "d": "#9999FF"}
        node_label = get_node_label(label, col, row)
        self.add_node(node_label, pos=[col, self.num_rows - 1 - row], color=node_colors[label])
        return node_label

    @timer
    def plot(self, fname_added="", pauli_product_paths=[], title_str=""):
        topo_fname = Path(self.args.circuit).stem + fname_added
        print("Plotting topology to", topo_fname, title_str, "...")
        # print("Generated topology with", num_qubits, "data qubits and ")
        plt.close()
        plt.rc("figure", figsize=[self.num_cols, self.num_rows])
        node_pos = nx.get_node_attributes(self, "pos")
        node_colors = nx.get_node_attributes(self, "color").values()
        edge_labels = nx.get_edge_attributes(self, "label")
        edge_colors = ["black"] * self.number_of_edges()
        edge_width = [1] * self.number_of_edges()
        node_edge_colors = ["white"] * self.number_of_nodes()
        node_line_widths = [1] * self.number_of_nodes()
        node_labels = {}
        for i, node in enumerate(self.nodes()):
            node_labels[node] = "" if is_bus_node(node) else node
            node_labels[node] = node
        cmap = plt.get_cmap("hsv", len(pauli_product_paths) + 1)
        for pi, pauli_path in enumerate(pauli_product_paths):
            pauli_product, pauli_product_graph = pauli_path
            for ei, edge in enumerate(self.edges):
                if pauli_product_graph.has_edge(*edge):
                    edge_colors[ei] = cmap(pi)
                    edge_width[ei] = 6
            root_node = None
            # print(pauli_product_graph.nodes)
            for ni, node in enumerate(self.nodes):
                if pauli_product_graph.has_node(node):
                    node_edge_colors[ni] = cmap(pi)
                    node_line_widths[ni] = 3
                    if is_magic_node(node):
                        root_node = node
            col, row = root_node[1:].split("-")
            col = float(col) - 0.2
            if row == "0":
                row = float(self.num_rows) - 0.5
            else:
                row = -0.5
            t = plt.text(col, row, str(pauli_product.id) + ":" + pauli_product.get_product_str(), color="black")
            t.set_bbox(dict(facecolor=cmap(pi), alpha=0.2, edgecolor=cmap(pi)))
        nx.draw_networkx(
            self,
            pos=node_pos,
            node_size=1000,
            node_color=node_colors,
            font_size=10,
            edge_color=edge_colors,
            width=edge_width,
            edgecolors=node_edge_colors,
            linewidths=node_line_widths,
            labels=node_labels,
            connectionstyle="angle3,angleA=90,angleB=0",
            arrows=True,
        )
        nx.draw_networkx_edge_labels(self, node_pos, edge_labels, rotate=False)
        plt.box(False)
        plt.title(title_str).set_fontsize(6 * math.sqrt(self.num_rows))
        plt.tight_layout()
        plt.savefig(topo_fname + ".pdf")
        # plt.savefig(topo_fname + ".png")
