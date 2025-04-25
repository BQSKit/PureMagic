#!/usr/bin/env -S python -u

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import argparse
import sys
import copy
import time
import functools
import multiprocessing as mp
import topograph
from topograph import is_bus_node, is_data_node, is_magic_node
import rndcircuit


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


def get_args():
    parser = argparse.ArgumentParser(description="Experimental scheduler for the LSSP")
    parser.add_argument("--min-num-qubits", "-n", type=int, default=10, help="Minimum number of data qubits")
    parser.add_argument(
        "--qubits-per-pauli-product",
        "-q",
        type=float,
        default=0.1,
        help="Mean fraction data qubits per Pauli product (normal distribution)",
    )
    parser.add_argument("--circuit-depth", "-d", type=int, default=1, help="Depth of the circuit")
    parser.add_argument("--rseed", "-r", type=int, default=29, help="Random seed")
    parser.add_argument("--gap-prob", "-g", type=float, default=0.5, help="Probability of a gap in the circuit at a qubit")
    path_methods = ["steiner", "bfs", "shortestpaths"]
    parser.add_argument(
        "--path-method",
        type=str,
        default="bfs",
        choices=path_methods,
        help="Method to use for finding paths: " + ", ".join(path_methods),
    )
    sort_orders = ["none", "random", "ascending", "descending"]
    parser.add_argument(
        "--sort-order",
        "-s",
        type=str,
        default="none",
        choices=sort_orders,
        help="Sorting Pauli products before scheduling: " + ", ".join(sort_orders),
    )
    parser.add_argument("--threads", "-t", type=int, default=0, help="Number of processes for multiprocessing")
    plot_options = ["none", "all", "circuit", "paths", "freqs"]
    parser.add_argument("--plot", "-p", type=str, default="none", choices=plot_options, help="Plot: " + ", ".join(plot_options))
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    print("Arguments:\n ", "\n  ".join(f"{k}={v}" for k, v in vars(args).items()))
    return args


def get_topo_dims():
    # the rows dimension needs to be a multiple of 3, and a minimum of 6
    # the columns dimension needs to be a multiple of 2, with 1 added (so 3, 5, 7, 9, ...)
    sq_dim = int(np.floor(np.sqrt(args.min_num_qubits)))
    patch_rows = int(sq_dim / 2) + sq_dim % 2
    num_rows = 3 * patch_rows + 3
    qubits_per_col = 2 * patch_rows
    num_cols = 2 * int(np.ceil(args.min_num_qubits / qubits_per_col)) + 1
    print("Layout dimensions:", num_cols, num_rows)
    return num_cols, num_rows


class PauliProduct:

    def __init__(self, rng, num_qubits, pauli_product_qubits, start_qubit):
        self.basis_options = ["X", "Z", "Y"]
        # self.basis_options = ["X", "Z"]
        self.operators = [" "] * num_qubits
        self.start_qubit = start_qubit
        self.qubits_used = pauli_product_qubits
        for i in range(start_qubit, start_qubit + pauli_product_qubits):
            self.operators[i] = self.basis_options[int(np.floor(rng.uniform(0, len(self.basis_options))))]

    def __str__(self):
        s = ""
        for i in range(len(self.operators)):
            if self.operators[i] != " ":
                s += str(i) + self.operators[i] + " "
        return s.strip()


def print_circuit(circuit):
    for i, pauli_product in enumerate(circuit):
        print(i, pauli_product)


@timer
def plot_topology(topo_graph, topo_fname, pauli_product_paths=[], title_str=""):
    print("Plotting topology to", topo_fname, "...")
    # print("Generated topology with", num_qubits, "data qubits and ")
    plt.close()
    plt.rc("figure", figsize=[topo_graph.num_cols, topo_graph.num_rows])
    node_pos = nx.get_node_attributes(topo_graph, "pos")
    node_colors = nx.get_node_attributes(topo_graph, "color").values()
    edge_labels = nx.get_edge_attributes(topo_graph, "label")
    edge_colors = ["black"] * topo_graph.number_of_edges()
    edge_width = [1] * topo_graph.number_of_edges()
    node_edge_colors = ["white"] * topo_graph.number_of_nodes()
    node_line_widths = [1] * topo_graph.number_of_nodes()
    node_labels = {}
    for i, node in enumerate(topo_graph.nodes()):
        node_labels[node] = "" if is_bus_node(node) else node
        node_labels[node] = node
    cmap = plt.get_cmap("hsv", len(pauli_product_paths) + 1)
    for pi, pauli_path in enumerate(pauli_product_paths):
        pauli_product, pauli_product_graph = pauli_path
        for ei, edge in enumerate(topo_graph.edges):
            if pauli_product_graph.has_edge(*edge):
                edge_colors[ei] = cmap(pi)
                edge_width[ei] = 6
        root_node = None
        # print(pauli_product_graph.nodes)
        for ni, node in enumerate(topo_graph.nodes):
            if pauli_product_graph.has_node(node):
                node_edge_colors[ni] = cmap(pi)
                node_line_widths[ni] = 3
                if is_magic_node(node):
                    root_node = node
        col, row = root_node[1:].split("-")
        col = float(col) - 0.2
        if row == "0":
            row = float(topo_graph.num_rows) - 0.5
        else:
            row = -0.5
        t = plt.text(col, row, pauli_product, color="black")
        t.set_bbox(dict(facecolor=cmap(pi), alpha=0.2, edgecolor=cmap(pi)))
    nx.draw_networkx(
        topo_graph,
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
    nx.draw_networkx_edge_labels(topo_graph, node_pos, edge_labels, rotate=False)
    plt.box(False)
    plt.title(title_str).set_fontsize(6 * math.sqrt(topo_graph.num_rows))
    plt.tight_layout()
    plt.savefig(topo_fname + ".pdf")
    plt.savefig(topo_fname + ".png")


def trim_dangling_nodes(g):
    while True:
        dangling_nodes = []
        for node in g.nodes:
            if is_bus_node(node) and g.degree(node) == 1:
                dangling_nodes.append(node)
        if len(dangling_nodes) == 0:
            break
        g.remove_nodes_from(dangling_nodes)


def schedule_pauli_product_bfs(topo_graph, pauli_product, root_node):
    visited = {root_node}
    num_found_operators = 0
    for node in topo_graph.nodes():
        if is_magic_node(node):
            visited.add(node)
    queue = [root_node]
    pauli_product_graph = nx.Graph()
    num_expected_ops = pauli_product.qubits_used
    for op in pauli_product.operators:
        if op == "Y":
            num_expected_ops += 1
    while len(queue):
        node = queue.pop(0)
        pauli_product_graph.add_node(node)
        # look for data nodes first
        for nb in topo_graph[node]:
            if nb not in visited and is_data_node(nb):
                visited.add(nb)
                qubit_index = int(nb[1:-1])
                qubit_basis = nb[-1]
                qbs = [pauli_product.operators[qubit_index]]
                if qbs[0] == "Y":
                    qbs = ["X", "Z"]
                if qubit_basis in qbs:
                    # print("Found basis", qubit_basis, "at node", nb)
                    pauli_product_graph.add_edge(node, nb)
                    num_found_operators += 1
                    if num_found_operators == num_expected_ops:
                        trim_dangling_nodes(pauli_product_graph)
                        return pauli_product_graph
        # now extend along the bus
        for nb in topo_graph[node]:
            if not is_data_node(nb) and nb not in visited:
                visited.add(nb)
                queue.append(nb)
                pauli_product_graph.add_edge(node, nb)
    return None


def get_topo_digraph(topo_graph):
    topo_digraph = topo_graph.to_directed()
    # now strip the directed edges coming out of the data nodes, to prevent paths that go into then out of data nodes
    edges_to_remove = []
    for edge in topo_digraph.edges():
        if is_data_node(edge[0]):
            edges_to_remove.append(edge)
    topo_digraph.remove_edges_from(edges_to_remove)
    return topo_digraph


def schedule_pauli_product_shortest_paths(topo_graph, pauli_product, root_node):
    topo_digraph = get_topo_digraph(topo_graph)

    while True:
        terminal_nodes = []
        for oi, operator in enumerate(pauli_product.operators):
            if operator != " ":
                ops = ["X", "Z"] if operator == "Y" else [operator]
                for op in ops:
                    node = "d" + str(oi) + op
                    if node not in topo_graph:
                        return None
                    terminal_nodes.append(node)
        paths = nx.multi_source_dijkstra_path(topo_digraph, terminal_nodes)
        tree_g = nx.Graph()
        for terminal_node in terminal_nodes:
            try:
                path_nodes = nx.shortest_path(topo_digraph, root_node, terminal_node)
            except nx.NetworkXNoPath as err:
                return None
            tree_g.add_edge(root_node, path_nodes[0])
            for i in range(len(path_nodes) - 1):
                tree_g.add_edge(path_nodes[i], path_nodes[i + 1])
        return tree_g


def mehlhorn_steiner_tree(topo_graph, terminal_nodes):
    # this is exactly like the steiner tree computation in the networkx liblary, except that for the dijkstra path calculation
    # and the shortest path, we use a digraph with the edges that go from the data nodes outwards removed. This prevents trees
    # that pass through the data nodes, instead of just terminating at the data nodes
    topo_digraph = get_topo_digraph(topo_graph)
    paths = nx.multi_source_dijkstra_path(topo_digraph, terminal_nodes)

    d_1 = {}
    s = {}
    for v in topo_graph.nodes():
        s[v] = paths[v][0]
        d_1[(v, s[v])] = len(paths[v]) - 1

    # G1-G4 names match those from the Mehlhorn 1988 paper.
    G_1_prime = nx.Graph()
    for u, v, data in topo_graph.edges(data=True):
        su, sv = s[u], s[v]
        weight_here = d_1[(u, su)] + data.get("weight", 1) + d_1[(v, sv)]
        if not G_1_prime.has_edge(su, sv):
            G_1_prime.add_edge(su, sv, weight=weight_here)
        else:
            new_weight = min(weight_here, G_1_prime[su][sv]["weight"])
            G_1_prime.add_edge(su, sv, weight=new_weight)

    G_2 = nx.minimum_spanning_edges(G_1_prime, data=True)

    G_3 = nx.Graph()
    for u, v, d in G_2:
        path = nx.shortest_path(topo_digraph, u, v, "weight")
        for n1, n2 in nx.utils.pairwise(path):
            G_3.add_edge(n1, n2)

    G_3_mst = list(nx.minimum_spanning_edges(G_3, data=False))
    G_4 = topo_graph.edge_subgraph(G_3_mst).copy()
    nx.approximation.steinertree._remove_nonterminal_leaves(G_4, terminal_nodes)
    edges = G_4.edges()
    T = topo_graph.edge_subgraph(edges)
    for node in T.nodes():
        if is_data_node(node) and T.degree(node) > 1:
            print("Failure in tree construction: data node", node, "has degree", T.degree(node))
    return T


def schedule_pauli_product_steiner(topo_graph, pauli_product, root_node):
    working_graph = copy.deepcopy(topo_graph)
    # print("trying steiner tree from root", root_node, "for", pauli_product.__str__(), "terminals", terminal_nodes)
    while True:
        terminal_nodes = [root_node]
        for oi, operator in enumerate(pauli_product.operators):
            if operator != " ":
                ops = ["X", "Z"] if operator == "Y" else [operator]
                for op in ops:
                    node = "d" + str(oi) + op
                    if node not in working_graph:
                        return None
                    terminal_nodes.append(node)
        try:
            # g = nx.algorithms.approximation.steiner_tree(topo_graph, terminal_nodes)
            g = mehlhorn_steiner_tree(working_graph, terminal_nodes)
            if not all([node in g for node in terminal_nodes]):
                return None
            return g
        except KeyError as err:
            # we have a disconnected node, so we need to reschedule without that node
            missing_node = err.args[0]
            # print("Key error", missing_node, "found?", missing_node in working_graph)
            working_graph.remove_nodes_from([missing_node])


def schedule_pauli_product(topo_graph, pauli_product):
    magic_nodes = []
    for node in topo_graph.nodes:
        if is_magic_node(node):
            magic_nodes.append(node)
    if len(magic_nodes) == 0:
        # print("Could not find starting node for Pauli product", pauli_product.__str__())
        return None
    # schedule from each available magic node in turn, and take the one that uses the fewest nodes
    pauli_product_graph = None
    for root_node in magic_nodes:
        if args.path_method == "bfs":
            g = schedule_pauli_product_bfs(topo_graph, pauli_product, root_node)
        elif args.path_method == "steiner":
            g = schedule_pauli_product_steiner(topo_graph, pauli_product, root_node)
        elif args.path_method == "shortestpaths":
            g = schedule_pauli_product_shortest_paths(topo_graph, pauli_product, root_node)
        else:
            raise ValueError("Unknown path method " + args.path_method)
        if g == None:
            continue
        # print("Found path with", g.number_of_nodes(), "nodes")
        if pauli_product_graph == None or g.number_of_nodes() < pauli_product_graph.number_of_nodes():
            # print("Found new best graph with nodes", g.number_of_nodes())
            pauli_product_graph = copy.deepcopy(g)

    if pauli_product_graph == None:
        # could not schedule all components
        return None

    return pauli_product_graph


def schedule_cycle(rng, topo_graph, circuit):
    # How do we choose the order in which to process the Pauli products?
    # We start with the given order. Other mappings are possible.
    if args.sort_order == "none":
        ordered_circuit = circuit
    elif args.sort_order == "random":
        ordered_circuit = rng.permutation(np.array(circuit, dtype="object"))
    elif args.sort_order == "descending":
        ordered_circuit = sorted(circuit, key=lambda x: x.qubits_used, reverse=True)
    elif args.sort_order == "ascending":
        ordered_circuit = sorted(circuit, key=lambda x: x.qubits_used, reverse=False)

    pauli_product_paths = []
    working_topo_graph = copy.deepcopy(topo_graph)
    num_qubits_scheduled = 0
    num_bus_qubits_scheduled = 0
    remaining_circuit = []
    for pauli_product in ordered_circuit:
        pauli_product_graph = schedule_pauli_product(working_topo_graph, pauli_product)
        if pauli_product_graph == None:
            # print("* Could not schedule Pauli product", pauli_product)
            remaining_circuit.append(pauli_product)
            continue
        # print("Scheduled Pauli product", pauli_product.__str__(), "with", pauli_product_graph.number_of_nodes(), "nodes")
        pauli_product_paths.append((pauli_product, pauli_product_graph))
        num_qubits_scheduled += pauli_product.qubits_used
        num_bus_qubits_scheduled += pauli_product_graph.number_of_nodes() - pauli_product.qubits_used - 1
        # now remove the Pauli product path from the graph
        working_topo_graph.remove_nodes_from(pauli_product_graph.nodes)
        orphaned_nodes = []
        for node in working_topo_graph.nodes:
            if working_topo_graph.degree(node) == 0:
                orphaned_nodes.append(node)
        working_topo_graph.remove_nodes_from(orphaned_nodes)

    if args.verbose:
        print("Scheduling results:")
    frac_paths = float(len(pauli_product_paths)) / len(circuit)
    frac_data_qubits = float(num_qubits_scheduled) / topo_graph.num_data_qubits
    frac_bus_qubits = float(num_bus_qubits_scheduled) / topo_graph.num_bus_qubits
    if args.verbose:
        print("  Pauli products:  %d/%d (%.2f)" % (len(pauli_product_paths), len(circuit), frac_paths))
        print("  data qubits:     %d/%d (%.2f)" % (num_qubits_scheduled, topo_graph.num_data_qubits, frac_data_qubits))
        print("  bus qubits:     %d/%d (%.2f)" % (num_bus_qubits_scheduled, topo_graph.num_bus_qubits, frac_bus_qubits))

    if len(pauli_product_paths) > 0:
        title_str = args.path_method + " (pps %.2f, data %.2f, bus %.2f)" % (frac_paths, frac_data_qubits, frac_bus_qubits)
        return title_str, pauli_product_paths, remaining_circuit
        # plot_topology(working_topo_graph, "lssp-working-topo", num_cols, num_rows)
    return None, None, remaining_circuit


def schedule_circuit_cycle(rng, topo_graph, circuit_cycle, cycle_i):
    remaining_circuit_cycle = circuit_cycle
    for i in range(100):
        title_str, pauli_product_paths, remaining_circuit_cycle = schedule_cycle(rng, topo_graph, circuit_cycle)
        if title_str is not None and args.plot in ["paths", "all"]:
            fname = "lssp-topo-path-" + str(i) + "-" + str(cycle_i) + "-" + args.path_method
            plot_topology(topo_graph, fname, pauli_product_paths, title_str)
        circuit_cycle = remaining_circuit_cycle
        if len(circuit_cycle) == 0:
            break
    if args.verbose:
        print("Scheduled full circuit cycle in", i + 1, "time steps")
    return i + 1


def schedule_circuit(rank, num_ranks, rng, topo_graph, circuit):
    num_steps = 0
    for ci, circuit_cycle in enumerate(circuit):
        if ci % num_ranks == rank:
            num_steps += schedule_circuit_cycle(rng, topo_graph, circuit_cycle, ci)
    return num_steps


class ScheduleProcess(mp.Process):
    def __init__(self, rank, num_ranks, rng, topo_graph, circuit):
        mp.Process.__init__(self)
        self.num_steps = mp.Value("i", 0)
        self.rank = rank
        self.num_ranks = num_ranks
        self.rng = rng
        self.topo_graph = topo_graph
        self.circuit = circuit

    def run(self):
        self.num_steps.value = schedule_circuit(self.rank, self.num_ranks, self.rng, self.topo_graph, self.circuit)


@timer
def schedule_multiprocessing(num_ranks, rng, topo_graph, circuit):
    proc = [None] * num_ranks
    for rank in range(num_ranks):
        proc[rank] = ScheduleProcess(0, num_ranks, rng, topo_graph, circuit)
        proc[rank].start()
    for rank in range(num_ranks):
        proc[rank].join()
    tot_num_steps = 0
    for rank in range(num_ranks):
        tot_num_steps += proc[rank].num_steps.value
    return tot_num_steps


@timer
def main():
    num_ranks = mp.cpu_count() if args.threads == 0 else args.threads
    print("Running on", num_ranks, "cores")
    rng = np.random.default_rng(seed=args.rseed)
    num_cols, num_rows = get_topo_dims()
    topo_graph = topograph.TopoGraph(num_cols, num_rows)
    if topo_graph.num_data_qubits != args.min_num_qubits:
        print("Adjusted number of data qubits from", args.min_num_qubits, "to", topo_graph.num_data_qubits)

    circuit = rndcircuit.gen_rnd_circuit(args, rng, topo_graph.num_data_qubits)
    if args.plot in ["circuit", "all"]:
        rndcircuit.plot_circuit(circuit)
    # tot_num_steps = schedule_circuit(0, 1, rng, topo_graph, circuit)
    tot_num_steps = schedule_multiprocessing(num_ranks, rng, topo_graph, circuit)
    print("Scheduled full circuit in", tot_num_steps, "(%.2f efficiency)" % (float(args.circuit_depth) / tot_num_steps))


if __name__ == "__main__":
    args = get_args()
    main()
