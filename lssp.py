#!/usr/bin/env -S python -u

import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import argparse
import sys
import copy


def get_args():
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
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
    args = parser.parse_args()
    print("Arguments:\n ", "\n  ".join(f"{k}={v}" for k, v in vars(args).items()))
    return args


def get_topo_dims(min_num_qubits):
    # the rows dimension needs to be a multiple of 3, and a minimum of 6
    # the columns dimension needs to be a multiple of 2, with 1 added (so 3, 5, 7, 9, ...)
    sq_dim = int(np.floor(np.sqrt(min_num_qubits)))
    patch_rows = int(sq_dim / 2) + sq_dim % 2
    num_rows = 3 * patch_rows + 3
    qubits_per_col = 2 * patch_rows
    num_cols = 2 * int(np.ceil(min_num_qubits / qubits_per_col)) + 1
    print("Layout dimensions:", num_cols, num_rows)
    return num_cols, num_rows


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


def operator_str(operator):
    return "".join(operator)


def print_circuit(circuit):
    for i, operator in enumerate(circuit):
        print(i, operator_str(operator))


def add_node(topo_graph, label, col, row, num_rows):
    node_colors = {"m": "#FFBB99", "b": "#B3FFBF", "d": "#9999FF"}
    node_label = get_node_label(label, col, row)
    topo_graph.add_node(node_label, pos=[col, num_rows - 1 - row], color=node_colors[label])
    return node_label


def plot_steiner_tree(topo_graph):
    terminal_nodes = ["m0-0"]
    for node in topo_graph.nodes:
        if is_data_node(node):
            terminal_nodes.append(node)
    steiner_graph = nx.algorithms.approximation.steiner_tree(topo_graph, terminal_nodes)
    stree_fname = "lssp-steiner"
    plt.close()
    node_pos = nx.get_node_attributes(topo_graph, "pos")
    node_colors = nx.get_node_attributes(topo_graph, "color").values()
    nx.draw_networkx(topo_graph, pos=node_pos, node_size=1000, node_color=node_colors, font_size=10)
    edge_labels = nx.get_edge_attributes(topo_graph, "label")
    nx.draw_networkx(steiner_graph, pos=node_pos, node_size=1000, font_size=10)
    nx.draw_networkx_edge_labels(steiner_graph, node_pos, edge_labels, rotate=False)
    plt.tight_layout()
    plt.savefig(stree_fname + ".pdf")


def plot_topology(topo_graph, topo_fname, num_cols, num_rows, operator_paths=[]):
    print("Plotting topology to", topo_fname, "...")
    # print("Generated topology with", num_qubits, "data qubits and ")
    plt.close()
    plt.rc("figure", figsize=[num_cols, num_rows])
    node_pos = nx.get_node_attributes(topo_graph, "pos")
    node_colors = nx.get_node_attributes(topo_graph, "color").values()
    edge_labels = nx.get_edge_attributes(topo_graph, "label")
    edge_colors = ["black"] * topo_graph.number_of_edges()
    edge_width = [1] * topo_graph.number_of_edges()
    node_edge_colors = ["white"] * topo_graph.number_of_nodes()
    colors = ["red", "magenta", "blue", "orange", "cyan"]
    for pi, operator_path in enumerate(operator_paths):
        ops = operator_path[0]
        path_graph = operator_path[1]
        for ei, edge in enumerate(topo_graph.edges):
            if path_graph.has_edge(*edge):
                edge_colors[ei] = colors[pi]
                edge_width[ei] = 3
        root_node = None
        for ni, node in enumerate(topo_graph.nodes):
            if path_graph.has_node(node):
                node_edge_colors[ni] = colors[pi]
                if is_magic_node(node):
                    root_node = node
        col, row = root_node[1:].split("-")
        col = float(col) - 0.2
        if row == "0":
            row = float(num_rows) - 0.5
        else:
            row = -0.5
        plt.text(col, row, ops, color=colors[pi])
    nx.draw_networkx(
        topo_graph,
        pos=node_pos,
        node_size=1000,
        node_color=node_colors,
        font_size=10,
        edge_color=edge_colors,
        width=edge_width,
        edgecolors=node_edge_colors,
    )
    nx.draw_networkx_edge_labels(topo_graph, node_pos, edge_labels, rotate=False)
    plt.tight_layout()
    plt.savefig(topo_fname + ".pdf")
    plt.savefig(topo_fname + ".png")


def build_parallel_topo(num_cols, num_rows):
    topo_graph = nx.Graph()
    for col in range(num_cols):
        if col % 2 != 0:
            continue
        node_label = add_node(topo_graph, "m", col, 0, num_rows)
        topo_graph.add_edge(node_label, get_node_label("b", col, 1))
        for row in range(1, num_rows - 2):
            node_label = add_node(topo_graph, "b", col, row, num_rows)
            topo_graph.add_edge(node_label, get_node_label("b", col, row + 1))
        prev_node_label = add_node(topo_graph, "b", col, num_rows - 2, num_rows)
        node_label = add_node(topo_graph, "m", col, num_rows - 1, num_rows)
        topo_graph.add_edge(node_label, prev_node_label)
    for col in range(num_cols):
        if col % 2 == 0:
            continue
        for row in range(1, num_rows - 1):
            if row % 3 == 1:
                node_label = add_node(topo_graph, "b", col, row, num_rows)
                topo_graph.add_edge(node_label, get_node_label("b", col - 1, row))
                topo_graph.add_edge(node_label, get_node_label("b", col + 1, row))
            else:
                node_label = add_node(topo_graph, "d", col, row, num_rows)
                if row % 3 == 2:
                    topo_graph.add_edge(node_label, get_node_label("b", col - 1, row), label="X")
                    topo_graph.add_edge(node_label, get_node_label("b", col + 1, row), label="X")
                    topo_graph.add_edge(node_label, get_node_label("b", col, row - 1), label="Z")
                else:
                    topo_graph.add_edge(node_label, get_node_label("b", col - 1, row), label="Z")
                    topo_graph.add_edge(node_label, get_node_label("b", col + 1, row), label="Z")
                    topo_graph.add_edge(node_label, get_node_label("b", col, row + 1), label="X")

    # print(topo_graph.nodes)
    # print(topo_graph.edges)

    num_data_qubits = sum([is_data_node(node) for node in topo_graph.nodes])
    num_magic_qubits = sum([is_magic_node(node) for node in topo_graph.nodes])
    num_bus_qubits = sum([is_bus_node(node) for node in topo_graph.nodes])
    print("Number of qubits:")
    print("  magic:", num_magic_qubits)
    print("  data: ", num_data_qubits)
    print("  bus:  ", num_bus_qubits)
    print("Space efficiency: %.2f" % (float(num_data_qubits) / (num_data_qubits + num_bus_qubits)))
    print("Magic state ratio: %.2f" % (float(num_magic_qubits) / (num_data_qubits + num_magic_qubits)))

    return num_data_qubits, topo_graph


def gen_rnd_circuit(rng, num_qubits, qubits_per_operator, num_operators):
    mean_qubits = float(num_qubits) * qubits_per_operator
    sigma_qubits = 2.0
    operators = []
    # basis_options = ["X", "Z", "Y"]
    basis_options = ["X", "Z"]
    counts = []
    for _ in range(num_operators):
        operator = []
        # this is a hack to ensure only positive numbers for the normal sampling
        for _ in range(100):
            operator_qubits = int(np.floor(rng.normal(mean_qubits, sigma_qubits)))
            if operator_qubits > 0 and operator_qubits <= num_qubits:
                break
        else:
            print("Couldn't generate a random number in range [0, %d], using %d" % (num_qubits, mean_qubits), file=sys.stderr)
            operator_qubits = mean_qubits
        for _ in range(operator_qubits):
            operator.append(basis_options[int(np.floor(rng.uniform(0, len(basis_options))))])
        operators.append(operator)
        counts.append(len(operator))

    plot_circuit_histogram = False
    if plot_circuit_histogram:
        hist_fname = "lssp-operator-freqs"
        print("Plotting circuit histogram to", hist_fname, "...")
        plt.close()
        plt.rcParams.update({"font.size": 20})
        plt.xlabel("number of qubits")
        plt.ylabel("Frequency")
        _, bins, _ = plt.hist(counts, num_qubits, density=True)
        # counts, bins = np.histogram(counts, 20)
        density = 1 / (sigma_qubits * np.sqrt(2 * np.pi)) * np.exp(-((bins - mean_qubits) ** 2) / (2 * sigma_qubits**2))
        plt.plot(bins, density)
        plt.tight_layout()
        plt.savefig(hist_fname + ".pdf")
        plt.savefig(hist_fname + ".png")
    return operators


def schedule_operator_dfs(topo_graph, operator):
    ops = operator_str(operator)
    print("Trying to schedule operator", ops)
    magic_nodes = []
    for node in topo_graph.nodes:
        if is_magic_node(node):
            magic_nodes.append(node)
    if len(magic_nodes) == 0:
        print("Could not find starting node for operator", ops)
        return None
    for root_node in magic_nodes:
        print("Starting at node", root_node)
        visited = {root_node}
        for node in topo_graph.nodes():
            if is_magic_node(node):
                visited.add(node)
        queue = [root_node]
        operator_graph = nx.Graph()
        while len(queue):
            node = queue.pop(0)
            operator_graph.add_node(node)
            # look for data nodes first
            for nb in topo_graph[node]:
                if is_data_node(nb):
                    qubit_basis = topo_graph[node][nb]["label"]
                    nb_with_basis = nb + qubit_basis
                    if nb_with_basis not in visited:
                        visited.add(nb_with_basis)
                        if qubit_basis in operator:
                            # print("Found basis", qubit_basis, "at node", nb)
                            operator_graph.add_edge(node, nb)
                            # we have used this, so ensure that this node cannot be visited again in another basis
                            visited.add(nb + ("X" if qubit_basis == "Z" else "Z"))
                            operator.remove(qubit_basis)
                            if len(operator) == 0:
                                queue = []
                                break
            # now extend along the bus
            for nb in topo_graph[node]:
                if not is_data_node(nb) and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
                    operator_graph.add_edge(node, nb)
        break

    if len(operator) > 0:
        # could not schedule all components
        return None

    # print(operator_graph.edges)
    while True:
        dangling_nodes = []
        for node in operator_graph.nodes:
            if is_bus_node(node) and operator_graph.degree(node) == 1:
                dangling_nodes.append(node)
        if len(dangling_nodes) == 0:
            break
        # print("Dangling nodes:", dangling_nodes)
        operator_graph.remove_nodes_from(dangling_nodes)

    return (ops, operator_graph)


def schedule_circuit(rng, topo_graph, circuit):
    # How do we choose the order in which to process the operators?
    # random, shortest first, longest first
    # How do we allocate qubits to operators?
    # random walk, dfs, bfs
    # print_circuit(circuit)
    rnd_order_circuit = rng.permutation(np.array(circuit, dtype="object"))
    # print_circuit(rnd_order_circuit)
    operator_paths = []
    working_topo_graph = copy.deepcopy(topo_graph)
    for operator in rnd_order_circuit:
        if len(rnd_order_circuit) == 1:
            operator = operator.tolist()
        # operator_path = schedule_operator_rnd(rng, working_topo_graph, operator.tolist())
        operator_path = schedule_operator_dfs(working_topo_graph, operator)
        if operator_path == None:
            print("Could not schedule operator", operator_str(operator))
            break
        operator_paths.append(operator_path)
        # now remove the operator path from the graph
        working_topo_graph.remove_nodes_from(operator_path[1].nodes)
        orphaned_nodes = []
        for node in working_topo_graph.nodes:
            if working_topo_graph.degree(node) == 0:
                orphaned_nodes.append(node)
        working_topo_graph.remove_nodes_from(orphaned_nodes)

    if len(operator_paths) > 0:
        plot_topology(topo_graph, "lssp-topo-path", num_cols, num_rows, operator_paths=operator_paths)
        plot_topology(working_topo_graph, "lssp-working-topo", num_cols, num_rows)


if __name__ == "__main__":
    args = get_args()
    rng = np.random.default_rng(seed=args.rseed)
    num_cols, num_rows = get_topo_dims(args.min_num_qubits)
    num_data_qubits, topo_graph = build_parallel_topo(num_cols, num_rows)
    plot_topology(topo_graph, "lssp-topo", num_cols, num_rows)
    # plot_steiner_tree(topo_graph)
    if num_data_qubits != args.min_num_qubits:
        print("Adjusted number of data qubits from", args.min_num_qubits, "to", num_data_qubits)
    circuit = gen_rnd_circuit(rng, num_data_qubits, args.qubits_per_operator, args.num_operators)
    schedule_circuit(rng, topo_graph, circuit)
