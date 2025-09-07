#!/usr/bin/env -S python -u

import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once")
    import networkx as nx
import sys
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from utils import timer


def is_magic_node(node):
    assert node[0] in ["m", "b", "d", "a"]
    return node[0] == "m"


def is_data_node(node):
    assert node[0] in ["m", "b", "d", "a"]
    return node[0] == "d"


def is_bus_node(node):
    assert node[0] in ["m", "b", "d", "a"]
    return node[0] == "b"


def is_ancilla_node(node):
    assert node[0] in ["m", "b", "d", "a"]
    return node[0] == "a"


def get_node_label(label, col, row):
    return label + str(math.ceil(col)) + "-" + str(math.ceil(row))


class TopoGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

    def get_topo_dims(self, bus_ratio):
        sq_dim = int(np.floor(np.sqrt(self.args.min_num_qubits)))
        patch_rows = int(sq_dim / 2) + sq_dim % 2
        bus_rows = int(patch_rows / bus_ratio) + 1
        print(f"patch rows {patch_rows}, bus rows {bus_rows}")
        qubits_per_col = 2 * patch_rows
        num_data_cols = int(np.ceil(self.args.min_num_qubits / qubits_per_col))
        num_cols = 2 * num_data_cols + 1
        # 2 rows for magic, 2 per patch row, rows for bus qubits
        num_rows = 2 + 2 * patch_rows + bus_rows
        return num_cols, num_rows

    def set_dims(self, args, rng):
        self.args = args
        self.rng = rng
        self.num_cols, self.num_rows = self.get_topo_dims(args.bus_ratio)
        if self.num_cols > 0 and self.num_rows > 0:
            self.gen_topo()

    def is_magic_column(self, col):
        return col % 2 != 1

    def gen_magic_columns(self):
        for col in range(self.num_cols):
            if not self.is_magic_column(col):
                continue
            node_label = self.add_labeled_node("m", col, 0)
            self.add_edge(node_label, get_node_label("b", col, 1))
            for row in range(1, self.num_rows - 2):
                node_label = self.add_labeled_node("b", col, row)
                self.add_edge(node_label, get_node_label("b", col, row + 1))
            prev_node_label = self.add_labeled_node("b", col, self.num_rows - 2)
            node_label = self.add_labeled_node("m", col, self.num_rows - 1)
            self.add_edge(node_label, prev_node_label, other="")

    def add_bus_qubit(self, col, row):
        node_label = self.add_labeled_node("b", col, row)
        if col > 0:
            ch = (
                "m"
                if self.is_magic_column(col - 1) and (row == 0 or row == self.num_rows - 1)
                else "b"
            )
            self.add_edge(node_label, get_node_label(ch, col - 1, row))
        if col < self.num_cols - 1:
            ch = (
                "m"
                if self.is_magic_column(col + 1) and (row == 0 or row == self.num_rows - 1)
                else "b"
            )
            self.add_edge(node_label, get_node_label(ch, col + 1, row))

    def add_data_qubit(self, qi, col, row, op):
        q = int(qi / 2) if op == "X" else int(qi / 2) - 1
        node_label1 = "d" + str(q) + op
        node_label2 = "d" + str(q + 1) + op
        other = None
        self.add_node(
            node_label1,
            pos=[float(col) - 0.25, self.num_rows - 1 - row],
            color="#9999FF",
            other=other,
        )
        self.add_node(
            node_label2,
            pos=[float(col) + 0.25, self.num_rows - 1 - row],
            color="#9999FF",
            other=other,
        )
        self.add_edge(get_node_label("b", col - 1, row), node_label1)
        self.add_edge(get_node_label("b", col + 1, row), node_label2)

    def add_ancilla_qubit(self, col, row):
        node_label = self.add_labeled_node("a", col, row)
        if col > 0:
            if not (self.is_magic_column(col - 1) and (row == 0 or row == self.num_rows - 1)):
                self.add_edge(node_label, get_node_label("b", col - 1, row))
        if col < self.num_cols - 1:
            if not (self.is_magic_column(col + 1) and (row == 0 or row == self.num_rows - 1)):
                self.add_edge(node_label, get_node_label("b", col + 1, row))
        # assume fixed ancilla in first and last row
        if row == 0:
            self.add_edge(node_label, get_node_label("b", col, row + 1))
        elif row == self.num_rows - 1:
            self.add_edge(node_label, get_node_label("b", col, row - 1))

    def shuffle_qubits(self):
        qubit_order = list(range(self.num_data_qubits))
        self.rng.shuffle(qubit_order)
        # very good for grover 5 node
        # qubit_order = [4, 1, 0, 5, 3, 2]
        # very bad for grover 5 node
        # qubit_order = [1, 0, 4, 2, 5, 3]
        # qubit_order = list(range(0, self.num_data_qubits, 2))
        # qubit_order.extend(list(range(1, self.num_data_qubits, 2)))
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
        for col in range(1, self.num_cols, 1):
            if self.is_magic_column(col):
                continue
            bus_rows = 0
            data_rows = 0
            for row in range(0, self.num_rows):
                offset = row - 1
                if row == 0 or row == self.num_rows - 1:
                    if not self.is_magic_column(col):
                        self.add_ancilla_qubit(col, row)
                elif row == self.num_rows - 1 or offset % spacing == 0:
                    if not self.is_magic_column(col) or (row != 0 and row == self.num_rows - 1):
                        self.add_bus_qubit(col, row)
                        bus_rows += 1
                else:
                    self.add_data_qubit(qi, col, row, "X" if data_rows % 2 == 0 else "Z")
                    qi += 2
                    data_rows += 1
        num_data_qubits = int(sum([is_data_node(node) for node in self.nodes]) / 2)
        num_magic_qubits = sum([is_magic_node(node) for node in self.nodes])
        num_bus_qubits = sum([is_bus_node(node) for node in self.nodes])
        num_ancilla_qubits = sum([is_ancilla_node(node) for node in self.nodes])
        print("Number of qubits:")
        print("  magic:   ", num_magic_qubits)
        print("  data:    ", num_data_qubits)
        print("  bus:     ", num_bus_qubits)
        print("  ancilla: ", num_ancilla_qubits)
        print(
            "Space efficiency: %.2f"
            % (float(num_data_qubits) / (num_data_qubits + num_bus_qubits + num_ancilla_qubits))
        )
        print(
            "Magic state ratio: %.2f"
            % (float(num_magic_qubits) / (num_data_qubits + num_magic_qubits))
        )
        print(
            "Ancilla ratio: %.2f"
            % (float(num_ancilla_qubits) / (num_data_qubits + num_ancilla_qubits))
        )
        self.num_data_qubits = num_data_qubits
        self.num_magic_qubits = num_magic_qubits
        self.num_bus_qubits = num_bus_qubits
        if self.args.rnd_order:
            self.shuffle_qubits()

    def add_labeled_node(self, label, col, row):
        node_colors = {"m": "#FFBB99", "b": "#aaaaaa", "d": "#9999FF", "a": "#FF88AA"}
        node_label = get_node_label(label, col, row)
        self.add_node(node_label, pos=[col, self.num_rows - 1 - row], color=node_colors[label])
        if is_magic_node(node_label):
            self.nodes[node_label]["busy_count"] = 0
        return node_label

    @timer
    def plot(self, fname_added="", pauli_product_paths=[], title_str=""):
        topo_fname = Path(self.args.circuit).stem + fname_added
        print("Plotting topology to", topo_fname, title_str, "...")
        # print("Generated topology with", num_qubits, "data qubits and ")
        plt.close()
        plt.rc("figure", figsize=[self.num_cols, self.num_rows])
        fig, ax = plt.subplots()
        bg_color = "#dddddd"
        ax.add_patch(Rectangle((-0.5, -0.5), self.num_cols, self.num_rows, facecolor=bg_color))
        node_pos = nx.get_node_attributes(self, "pos")
        node_colors = nx.get_node_attributes(self, "color").values()
        edge_labels = nx.get_edge_attributes(self, "label")
        edge_colors = [bg_color] * self.number_of_edges()
        edge_width = [1] * self.number_of_edges()
        node_edge_colors = [bg_color] * self.number_of_nodes()
        node_line_widths = [1] * self.number_of_nodes()
        node_labels = {}
        for _, node in enumerate(self.nodes()):
            node_labels[node] = node
        cmap = plt.get_cmap("hsv", len(pauli_product_paths) + 1)
        label_col = -1.5
        label_row = self.num_rows
        for pi, pauli_path in enumerate(pauli_product_paths):
            pauli_product, pauli_product_graph = pauli_path
            for ei, edge in enumerate(self.edges):
                if pauli_product_graph.has_edge(*edge):
                    edge_colors[ei] = cmap(pi)  # type: ignore
                    edge_width[ei] = 6
            root_node = None
            # print(pauli_product_graph.nodes)
            for ni, node in enumerate(self.nodes):
                if pauli_product_graph.has_node(node):
                    node_edge_colors[ni] = cmap(pi)  # type: ignore
                    node_line_widths[ni] = 3
                    if is_magic_node(node):
                        root_node = node
            if root_node != None:
                col, row = root_node[1:].split("-")
                if row == "0":
                    row = float(self.num_rows) - 0.5
                else:
                    row = -0.5
                col = float(col)
                row = float(row)
                col = float(col) - 0.2
            else:
                col = label_col
                row = label_row
                label_row -= 0.35
            t = plt.text(col, row, pauli_product.get_product_str(), color="black")
            t.set_bbox(dict(facecolor=cmap(pi), alpha=0.2, edgecolor=cmap(pi)))
        for row in range(self.num_rows + 1):
            plt.axhline(
                row - 0.5,
                xmin=0.5 / self.num_cols,
                xmax=1.0 - 0.5 / self.num_cols,
                ls="-",
                lw=2,
                c="white",
                alpha=0.5,
            )
        for col in range(self.num_cols + 1):
            plt.axvline(
                col - 0.5,
                ymin=0.5 / self.num_rows,
                ymax=1.0 - 0.5 / self.num_rows,
                ls="-",
                lw=2,
                c="white",
                alpha=0.5,
            )
        plt.plot(0, 0, 1, 1, ls="-", c="black", lw=10)
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
        plt.savefig(topo_fname + ".png")
