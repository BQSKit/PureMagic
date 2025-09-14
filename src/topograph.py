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
    assert node[0] in ["m", "b", "d", "a", "e"]
    return node[0] == "m"


def is_data_node(node):
    assert node[0] in ["m", "b", "d", "a", "e"]
    return node[0] == "d"


def is_bus_node(node):
    assert node[0] in ["m", "b", "d", "a", "e"]
    return node[0] == "b"


def is_ancilla_node(node):
    assert node[0] in ["m", "b", "d", "a", "e"]
    return node[0] == "a"


def is_estabilizer_node(node):
    assert node[0] in ["m", "b", "d", "a", "e"]
    return node[0] == "e"


def get_node_label(label, col, row):
    return label + str(math.ceil(col)) + "-" + str(math.ceil(row))


class TopoGraph(nx.Graph):
    def __init__(self):
        nx.Graph.__init__(self)

        self.num_data_qubits = 0
        self.num_bus_qubits = 0
        self.num_magic_qubits = 0
        self.num_ancilla_qubits = 0
        self.num_estabilizer_qubits = 0
        self.num_qubits = 0

    @timer
    def set_topo(self, args, min_num_qubits, rng):
        self.args = args
        self.rng = rng
        sq_dim = int(np.floor(np.sqrt(min_num_qubits)))
        patch_rows = int(sq_dim / 2) + sq_dim % 2
        bus_rows = patch_rows + 1
        # print(f"patch rows {patch_rows}, bus rows {bus_rows}")
        qubits_per_col = 2 * patch_rows
        num_data_cols = int(np.ceil(min_num_qubits / qubits_per_col))
        # a bus column on both sides of the data columns plus 2 extra for side columns for magic
        self.num_cols = 2 * num_data_cols + 3
        # 2 rows for magic, 2 per patch row, rows for bus qubits
        self.num_rows = 2 + 2 * patch_rows + bus_rows
        if self.num_cols > 0 and self.num_rows > 0:
            print("Layout dimensions:", self.num_cols, self.num_rows)
            self.gen_topo()
            frac_data = float(self.num_data_qubits) / self.num_qubits
            frac_bus = float(self.num_bus_qubits) / self.num_qubits
            frac_magic = float(self.num_magic_qubits) / self.num_qubits
            frac_ancilla = float(self.num_ancilla_qubits) / self.num_qubits
            frac_estabilizer = float(self.num_estabilizer_qubits) / self.num_qubits
            print("Number of qubits:")
            print(f"  data:         {self.num_data_qubits} ({frac_data:.3f})")
            print(f"  bus:          {self.num_bus_qubits} ({frac_bus:.3f})")
            print(f"  magic:        {self.num_magic_qubits} ({frac_magic:.3f})")
            print(f"  ancilla:      {self.num_ancilla_qubits} ({frac_ancilla:.3f})")
            print(f"  e-stabilizer: {self.num_estabilizer_qubits} ({frac_estabilizer:.3f})")
            print(f"  total:        {self.num_qubits}")

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

    def add_labeled_node(self, label, col, row):
        node_colors = {
            "m": "#FFBB99",
            "b": "#aaaaaa",
            "d": "#9999FF",
            "a": "#FF88AA",
            "e": "#99CC99",
        }
        node_label = get_node_label(label, col, row)
        self.add_node(node_label, pos=[col, self.num_rows - 1 - row], color=node_colors[label])
        if is_magic_node(node_label):
            self.nodes[node_label]["busy_count"] = 0
        return node_label

    def add_data_qubit(self, qi, col, row, op, node_grid):
        q = int(qi / 2) if op == "X" else int(qi / 2) - 1
        node_label1 = "d" + str(q) + op
        node_label2 = "d" + str(q + 1) + op
        self.add_node(
            node_label1,
            pos=[float(col) - 0.25, self.num_rows - 1 - row],
            color="#9999FF",
        )
        self.add_node(
            node_label2,
            pos=[float(col) + 0.25, self.num_rows - 1 - row],
            color="#9999FF",
        )
        node_grid[col][row] = f"d{str(q)}/{str(q+1)}{op}"

    def add_border_row(self, row, node_grid):
        node_grid[0][row] = self.add_labeled_node("b", 0, row)
        node_grid[self.num_cols - 1][row] = self.add_labeled_node("b", self.num_cols - 1, row)
        for col in range(1, self.num_cols - 1):
            if self.num_cols > 9:
                node_grid[col][row] = self.add_labeled_node("m", col, row)
            else:
                if col % 2 == 1:
                    node_grid[col][row] = self.add_labeled_node("m", col, row)
                elif col % 4 == 0:
                    node_grid[col][row] = self.add_labeled_node("a", col, row)
                else:
                    node_grid[col][row] = self.add_labeled_node("m", col, row)

    def add_border_column(self, col, node_grid):
        for row in range(1, self.num_rows - 1):
            node_grid[col][row] = self.add_labeled_node("m", col, row)

    def link_nbs(self, node1, node2):
        if is_magic_node(node1) and is_magic_node(node2):
            return
        if is_data_node(node1):
            node1_left, node1_right = node1.split("/")
            node1 = node1_left + node1_right[-1]
        elif is_data_node(node2):
            _, node2_right = node2.split("/")
            node2 = "d" + node2_right
        self.add_edge(node1, node2)

    def gen_topo(self):
        # first add all the nodes
        node_grid = [[]] * self.num_cols
        for i in range(len(node_grid)):
            node_grid[i] = [""] * self.num_rows
        # add top row
        self.add_border_row(0, node_grid)
        # add left edge column
        self.add_border_column(0, node_grid)
        # add center nodes
        qi = 0
        for col in range(1, self.num_cols - 1):
            if col % 2 == 0:  # data column
                for row in range(1, self.num_rows - 1):
                    if row % 3 + 1 == 2:
                        if row % 6 + 1 == 5 and row != self.num_rows - 2 and col % 4 == 0:
                            if self.num_cols <= 9 or col % 8 == 0:
                                node_grid[col][row] = self.add_labeled_node("e", col, row)
                            else:
                                node_grid[col][row] = self.add_labeled_node("a", col, row)
                        else:
                            node_grid[col][row] = self.add_labeled_node("b", col, row)
                    else:
                        self.add_data_qubit(
                            qi, col, row, "X" if row % 3 + 1 == 3 else "Z", node_grid
                        )
                        qi += 2
            else:  # bus column
                for row in range(1, self.num_rows - 1):
                    node_grid[col][row] = self.add_labeled_node("b", col, row)
        # add right edge column
        self.add_border_column(self.num_cols - 1, node_grid)
        # add bottom row
        self.add_border_row(self.num_rows - 1, node_grid)
        # summary statistics
        self.num_data_qubits = int(sum([is_data_node(node) for node in self.nodes]) / 2)
        self.num_magic_qubits = sum([is_magic_node(node) for node in self.nodes])
        self.num_bus_qubits = sum([is_bus_node(node) for node in self.nodes])
        self.num_ancilla_qubits = sum([is_ancilla_node(node) for node in self.nodes])
        self.num_estabilizer_qubits = sum([is_estabilizer_node(node) for node in self.nodes])
        self.num_qubits = (
            self.num_data_qubits
            + self.num_bus_qubits
            + self.num_magic_qubits
            + self.num_ancilla_qubits
            + self.num_estabilizer_qubits
        )
        # now add all the edges
        print("Topology:")
        for row in range(0, self.num_rows):
            print("  ", end="")
            for col in range(0, self.num_cols):
                node = node_grid[col][row]
                if self.num_data_qubits <= 100:
                    print(f"{node:8}", end=" ")
                else:
                    print(f"{node:9}", end=" ")
                if col > 0:
                    self.link_nbs(node, node_grid[col - 1][row])
                nb_node = node_grid[col][row - 1]
                if row > 0 and not is_data_node(node) and not is_data_node(nb_node):
                    self.link_nbs(node, nb_node)
            print("")

        if self.args.rnd_order:
            self.shuffle_qubits()

    @timer
    def plot(self, fname_added="", pauli_product_paths=[], title_str="", node_labels=None):
        topo_fname = Path(self.args.circuit).stem + fname_added
        print(f"Plotting topology to {topo_fname}...")
        # print("Generated topology with", num_qubits, "data qubits and ")
        plt.close()
        plt.rc("figure", figsize=[self.num_cols, self.num_rows])
        fig, ax = plt.subplots()
        bg_color = "#dddddd"
        ax.add_patch(Rectangle((-0.5, -0.5), self.num_cols, self.num_rows, facecolor=bg_color))
        node_pos = nx.get_node_attributes(self, "pos")
        node_colors = nx.get_node_attributes(self, "color").values()
        edge_labels = nx.get_edge_attributes(self, "label")
        # edge_colors = [bg_color] * self.number_of_edges()
        edge_colors = ["#999999"] * self.number_of_edges()
        edge_width = [1] * self.number_of_edges()
        node_edge_colors = [bg_color] * self.number_of_nodes()
        node_line_widths = [1] * self.number_of_nodes()
        if node_labels is None:
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
                col = int(col)
                row = int(row)
                if row == 0:
                    row -= 0.9
                row = float(self.num_rows) - row - 1.5
                col = float(col) - 0.1
            else:
                col = label_col
                row = label_row
                label_row -= 0.35
            t = plt.text(col, row, pauli_product.get_product_str(), color="black", fontsize=11)
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
        try:
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
        except nx.NetworkXError as err:
            # for node in self.nodes():
            #    print(f"{node} {self.nodes[node]}")
            raise err
        plt.box(False)
        plt.title(title_str).set_fontsize(6 * math.sqrt(self.num_rows))
        plt.tight_layout()
        # plt.savefig(topo_fname + ".pdf")
        plt.savefig(topo_fname + ".png")
