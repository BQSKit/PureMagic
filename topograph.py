#!/usr/bin/env -S python -u

import networkx as nx
import math
import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"[{func.__name__}: {elapsed_time:0.4f} s]", flush=True)
        return value

    return wrapper_timer


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
    def __init__(self, num_cols=0, num_rows=0):
        nx.Graph.__init__(self)
        self.num_cols = num_cols
        self.num_rows = num_rows
        if num_cols > 0 and num_rows > 0:
            self.gen_topo()

    @timer
    def gen_topo(self):
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
            self.add_edge(node_label, prev_node_label)
        qi = 0
        for col in range(self.num_cols):
            if col % 2 == 0:
                continue
            for row in range(1, self.num_rows - 1):
                if row % 3 == 1:
                    node_label = self.add_labeled_node("b", col, row)
                    self.add_edge(node_label, get_node_label("b", col - 1, row))
                    self.add_edge(node_label, get_node_label("b", col + 1, row))
                else:
                    if row % 3 == 2:
                        node_label1 = "d" + str(int(qi / 2)) + "X"
                        node_label2 = "d" + str(int(qi / 2) + 1) + "X"
                        self.add_edge(get_node_label("b", col, row + 2), node_label1)
                        self.add_edge(get_node_label("b", col, row + 2), node_label2)
                    else:
                        node_label1 = "d" + str(int(qi / 2) - 1) + "Z"
                        node_label2 = "d" + str(int(qi / 2)) + "Z"
                        self.add_edge(get_node_label("b", col, row - 2), node_label1)
                        self.add_edge(get_node_label("b", col, row - 2), node_label2)
                    self.add_node(node_label1, pos=[float(col) - 0.35, self.num_rows - 1 - row], color="#9999FF")
                    self.add_edge(get_node_label("b", col - 1, row), node_label1)
                    self.add_node(node_label2, pos=[float(col) + 0.35, self.num_rows - 1 - row], color="#9999FF")
                    self.add_edge(get_node_label("b", col + 1, row), node_label2)
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

    def add_labeled_node(self, label, col, row):
        # node_colors = {"m": "#FFBB99", "b": "#B3FFBF", "d": "#9999FF"}
        node_colors = {"m": "#FFBB99", "b": "#cccccc", "d": "#9999FF"}
        node_label = get_node_label(label, col, row)
        self.add_node(node_label, pos=[col, self.num_rows - 1 - row], color=node_colors[label])
        return node_label
