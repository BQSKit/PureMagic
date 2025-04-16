#!/usr/bin/env -S python -u

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
import sys


def get_node_label(label, col, row):
    return label + str(math.ceil(col)) + "-" + str(math.ceil(row))


def add_node(label, col, row, num_rows, node_pos, node_colors):
    node_label = get_node_label(label, col, row)
    node_pos[node_label] = [col, num_rows - 1 - row]
    if label == "m":
        node_colors.append("#FFBB99")
    elif label == "b":
        node_colors.append("#B3FFBF")
    elif label.startswith("d"):
        node_colors.append("#9999FF")
    return node_label


def build_parallel_topo(num_cols, num_rows):
    topo_graph = nx.Graph()
    node_pos = {}
    node_colors = []
    edge_labels = {}
    for col in range(num_cols):
        if col % 2 != 0:
            continue
        node_label = add_node("m", col, 0, num_rows, node_pos, node_colors)
        topo_graph.add_edge(node_label, get_node_label("b", col, 1))
        for row in range(1, num_rows - 2):
            node_label = add_node("b", col, row, num_rows, node_pos, node_colors)
            topo_graph.add_edge(node_label, get_node_label("b", col, row + 1))
        prev_node_label = add_node("b", col, num_rows - 2, num_rows, node_pos, node_colors)
        node_label = add_node("m", col, num_rows - 1, num_rows, node_pos, node_colors)
        topo_graph.add_edge(node_label, prev_node_label)
    for col in range(num_cols):
        if col % 2 == 0:
            continue
        for row in range(1, num_rows - 1):
            if row % 3 == 1:
                node_label = add_node("b", col, row, num_rows, node_pos, node_colors)
                topo_graph.add_edge(node_label, get_node_label("b", col - 1, row))
                topo_graph.add_edge(node_label, get_node_label("b", col + 1, row))
            else:
                node_label = add_node("d", col, row, num_rows, node_pos, node_colors)
                if row % 3 == 2:
                    topo_graph.add_edge(node_label, get_node_label("b", col - 1, row))
                    edge_labels[(node_label, get_node_label("b", col - 1, row))] = "X"
                    topo_graph.add_edge(node_label, get_node_label("b", col + 1, row))
                    edge_labels[(node_label, get_node_label("b", col + 1, row))] = "X"
                    topo_graph.add_edge(node_label, get_node_label("b", col, row - 1))
                    edge_labels[(node_label, get_node_label("b", col, row - 1))] = "Z"
                else:
                    topo_graph.add_edge(node_label, get_node_label("b", col - 1, row))
                    edge_labels[(node_label, get_node_label("b", col - 1, row))] = "Z"
                    topo_graph.add_edge(node_label, get_node_label("b", col + 1, row))
                    edge_labels[(node_label, get_node_label("b", col + 1, row))] = "Z"
                    topo_graph.add_edge(node_label, get_node_label("b", col, row + 1))
                    edge_labels[(node_label, get_node_label("b", col, row + 1))] = "X"
    # print(topo_graph.nodes)
    # print(topo_graph.edges)

    num_data_qubits = sum([node[0] == "d" for node in topo_graph.nodes])
    num_magic_qubits = sum([node[0] == "m" for node in topo_graph.nodes])
    num_bus_qubits = sum([node[0] == "b" for node in topo_graph.nodes])
    print("Number of qubits:")
    print("  magic:", num_magic_qubits)
    print("  data: ", num_data_qubits)
    print("  bus:  ", num_bus_qubits)
    print("Space efficiency: %.2f" % (float(num_data_qubits) / (num_data_qubits + num_bus_qubits)))
    # print("Generated topology with", num_qubits, "data qubits and ")
    plt.rc("figure", figsize=[num_cols, num_rows])
    nx.draw_networkx(topo_graph, pos=node_pos, node_size=1000, node_color=node_colors, font_size=10)
    nx.draw_networkx_edge_labels(topo_graph, node_pos, edge_labels, rotate=False)
    plt.tight_layout()
    plt.savefig("lssp.pdf")
    plt.savefig("lssp.png")


def gen_rnd_circuit(rseed, num_qubits, qubits_per_operator, num_operators):
    rng = np.random.default_rng(seed=rseed)
    mean_qubits = float(num_qubits) * qubits_per_operator
    sigma_qubits = 2.0
    operators = []
    basis_options = ["X", "Z", "Y"]
    counts = []
    for i in range(num_operators):
        operator = []
        # this is a hack to ensure only positive numbers for the normal sampling
        for _ in range(100):
            operator_qubits = int(np.floor(rng.normal(mean_qubits, sigma_qubits)))
            if operator_qubits > 0 and operator_qubits <= 10:
                break
        else:
            print("Couldn't generate a random number in range [0, %d], using %d" % (num_qubits, mean_qubits), file=sys.stderr)
            operator_qubits = 1
        for _ in range(operator_qubits):
            operator.append(basis_options[int(np.floor(rng.uniform(0, 3)))])
        operators.append(operator)
        counts.append(len(operator))

    for operator in operators:
        print(operator)
    # _, bins, _ = plt.hist(counts, 10, density=True)
    # plt.plot(bins, 1 / (sigma_qubits * np.sqrt(2 * np.pi)) * np.exp(-((bins - mean_qubits) ** 2) / (2 * sigma_qubits**2)))
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental scheduler for the LSSP")
    parser.add_argument("--min-num-qubits", "-n", type=int, default=10, help="Minimum number of data qubits")
    parser.add_argument(
        "--qubits-per-operator",
        "-q",
        type=float,
        default=0.1,
        help="Mean fraction data qubits per operator (normal distribution)",
    )
    parser.add_argument("--num-operators", "-m", type=int, default=50, help="Number of operators to generate")
    parser.add_argument("--rnd-seed", "-r", type=int, default=29, help="Random seed")
    args = parser.parse_args()
    print("Arguments:\n ", "\n  ".join(f"{k}={v}" for k, v in vars(args).items()))

    # gen_rnd_circuit(args.rnd_seed, args.num_qubits, args.qubits_per_operator, args.num_operators)
    # the rows dimension needs to be a multiple of 3, and a minimum of 6
    # the columns dimension needs to be a multiple of 2, with 1 added (so 3, 5, 7, 9, ...)
    sq_dim = int(np.floor(np.sqrt(args.min_num_qubits)))
    patch_rows = int(sq_dim / 2) + sq_dim % 2
    num_rows = 3 * patch_rows + 3
    qubits_per_col = 2 * patch_rows
    num_cols = 2 * int(np.ceil(args.min_num_qubits / qubits_per_col)) + 1
    print("Layout dimensions:", num_cols, num_rows)
    build_parallel_topo(num_cols, num_rows)
