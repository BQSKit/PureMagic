#!/usr/bin/env -S python -u

import networkx as nx
import numpy as np
import copy
from topograph import is_bus_node, is_data_node, is_magic_node


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


def schedule_pauli_product(args, topo_graph, pauli_product):
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


class Scheduler:
    def __init__(self, args, rank, num_ranks, rng, topo_graph):
        self.args = args
        self.rank = rank
        self.num_ranks = num_ranks
        self.rng = rng
        self.topo_graph = topo_graph

    def schedule_circuit(self, circuit):
        num_steps = 0
        for ci, circuit_cycle in enumerate(circuit):
            if ci % self.num_ranks == self.rank:
                num_steps += self.schedule_circuit_cycle(circuit_cycle, ci)
        return num_steps

    def schedule_circuit_cycle(self, circuit_cycle, cycle_i):
        remaining_circuit_cycle = circuit_cycle
        for i in range(100):
            title_str, pauli_product_paths, remaining_circuit_cycle = self.schedule_cycle(circuit_cycle)
            if title_str is not None and "paths" in self.args.plot:
                fname = "lssp-topo-path-" + str(i) + "-" + str(cycle_i) + "-" + self.args.path_method
                self.topo_graph.plot(fname, pauli_product_paths, title_str)
            circuit_cycle = remaining_circuit_cycle
            if len(circuit_cycle) == 0:
                break
        if self.args.verbose:
            print("Scheduled full circuit cycle in", i + 1, "time steps")
        return i + 1

    def schedule_cycle(self, circuit):
        # How do we choose the order in which to process the Pauli products?
        # We start with the given order. Other mappings are possible.
        if self.args.sort_order == "none":
            ordered_circuit = circuit
        elif self.args.sort_order == "random":
            ordered_circuit = self.rng.permutation(np.array(circuit, dtype="object"))
        elif self.args.sort_order == "descending":
            ordered_circuit = sorted(circuit, key=lambda x: x.qubits_used, reverse=True)
        elif self.args.sort_order == "ascending":
            ordered_circuit = sorted(circuit, key=lambda x: x.qubits_used, reverse=False)

        pauli_product_paths = []
        working_topo_graph = copy.deepcopy(self.topo_graph)
        num_qubits_scheduled = 0
        num_bus_qubits_scheduled = 0
        remaining_circuit = []
        for pauli_product in ordered_circuit:
            pauli_product_graph = schedule_pauli_product(self.args, working_topo_graph, pauli_product)
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

        if self.args.verbose:
            print("Scheduling results:")
        frac_paths = float(len(pauli_product_paths)) / len(circuit)
        frac_data_qubits = float(num_qubits_scheduled) / self.topo_graph.num_data_qubits
        frac_bus_qubits = float(num_bus_qubits_scheduled) / self.topo_graph.num_bus_qubits
        if self.args.verbose:
            print("  Pauli products:  %d/%d (%.2f)" % (len(pauli_product_paths), len(circuit), frac_paths))
            print("  data qubits:     %d/%d (%.2f)" % (num_qubits_scheduled, self.topo_graph.num_data_qubits, frac_data_qubits))
            print("  bus qubits:     %d/%d (%.2f)" % (num_bus_qubits_scheduled, self.topo_graph.num_bus_qubits, frac_bus_qubits))

        if len(pauli_product_paths) > 0:
            title_str = self.args.path_method + " (pps %.2f, data %.2f, bus %.2f)" % (
                frac_paths,
                frac_data_qubits,
                frac_bus_qubits,
            )
            return title_str, pauli_product_paths, remaining_circuit
            # working_top_graph.plot("lssp-working-topo", num_cols, num_rows)
        return None, None, remaining_circuit
